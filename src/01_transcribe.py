"""Step 1: transcribe every clip with Whisper, keeping the speaker's mistakes.

Why this step matters
---------------------
The target is a *grammar* score, so the words actually spoken carry most of the signal.
Whisper is trained to output clean, well-formed text and will quietly "fix" errors like
"he don't" -> "he doesn't". To counter that we give it a verbatim-style prompt (fillers,
repetitions, wrong verb forms), which pushes the decoder toward literal transcription.

Besides the text, we also record ASR confidence features:
  * token log-probabilities: low values often mean unclear or non-fluent speech
  * no-speech probability: high values flag silent / unusable clips

Output: artifacts/transcripts_<tag>.parquet, one row per clip.

Usage (run from the project root with the GPU venv):
  .venv/Scripts/python src/01_transcribe.py                              # all train + test clips
  .venv/Scripts/python src/01_transcribe.py --limit 20 --tag pilot       # quick pilot on 20 train clips
  .venv/Scripts/python src/01_transcribe.py --limit 20 --no-prompt       # pilot without the verbatim prompt
"""
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import ART, DATA, DTYPE, SR   # paths + precision, shared with Kaggle runs

MAX_CHUNK = 30 * SR   # Whisper's encoder sees at most 30 s of audio
MODEL = "openai/whisper-large-v3-turbo"   # ~3.2 GB peak in fp32 (local GTX 1650), ~1.7 GB in fp16 (Kaggle T4)

# Verbatim-style prompt: deliberately full of fillers, repetitions and grammar errors so
# Whisper keeps the speaker's mistakes instead of correcting them.
PROMPT = ("Umm, so I goes to the, uh, the market yesterday and, like, he don't have "
          "no apples. Hmm, I mean... I was, I was think that it are good.")


def split_chunks(wav):
    """Split a waveform into pieces of at most 30 s for Whisper.

    Cutting at a fixed 30 s mark could slice a word in half, so the cut is placed at
    the quietest 100 ms frame between 20 s and 30 s into the current window
    (most likely a pause between words).
    """
    chunks, start = [], 0
    while len(wav) - start > MAX_CHUNK:
        lo, hi = start + 20 * SR, start + MAX_CHUNK   # search window for the cut point
        frame = SR // 10                              # 100 ms frames
        seg = wav[lo:hi]
        n = len(seg) // frame
        energy = (seg[: n * frame].reshape(n, frame) ** 2).mean(1)   # mean power per frame
        cut = lo + int(energy.argmin()) * frame + frame // 2         # middle of the quietest frame
        chunks.append(wav[start:cut])
        start = cut
    chunks.append(wav[start:])
    # Drop tiny leftovers (< 0.25 s): they only produce hallucinated text.
    return [c for c in chunks if len(c) > SR // 4]


@torch.inference_mode()
def confidence(model, feats, prefix, tokens, nospeech_id):
    """Score the decoded text with one teacher-forced forward pass.

    We feed [prefix + decoded tokens] back into the decoder and read off:
      * the log-probability Whisper assigns to each decoded token, and
      * the probability of <|nospeech|> right after <|startoftranscript|>
        (this is how Whisper itself decides a segment is silent).
    This is done separately from generate() because beam search makes per-token
    scores awkward to recover, and the forward pass is cheap.
    """
    ids = torch.tensor([prefix + tokens], device=model.device)
    logits = model(input_features=feats, decoder_input_ids=ids).logits[0].float()
    # Fail loudly: NaN logits mean broken numerics (e.g. fp16 on a GTX 16xx), not a quiet clip.
    assert torch.isfinite(logits).all(), "non-finite Whisper logits: check model dtype"
    logp = logits.log_softmax(-1)
    p_nospeech = logp[0, nospeech_id].exp().item()    # position 0 = prediction after SOT
    if not tokens:
        return [], p_nospeech
    tok = torch.tensor(tokens, device=model.device)
    # The logit at position i predicts token i+1, so decoded token k is predicted
    # from position len(prefix)-1+k.
    tok_lp = logp[len(prefix) - 1: len(prefix) - 1 + len(tokens)].gather(1, tok[:, None])[:, 0]
    return tok_lp.tolist(), p_nospeech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pilot mode: N train clips spread across all labels")
    ap.add_argument("--no-prompt", action="store_true", help="disable the verbatim prompt (for comparison)")
    ap.add_argument("--tag", default="full", help="suffix of the output parquet file")
    args = ap.parse_args()

    # ---- Clip list: all train + test clips, or a label-stratified pilot subset ----
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    te = pd.read_csv(DATA / "test.csv").assign(split="test")
    files = pd.concat([tr, te], ignore_index=True)
    if args.limit:
        # Sort by label and take evenly spaced rows -> the pilot covers low, mid and high scores.
        files = tr.sort_values("label").iloc[np.linspace(0, len(tr) - 1, args.limit).astype(int)]

    # ---- Model: PyTorch SDPA attention, precision from common.DTYPE (fp32 on GTX 16xx, fp16 elsewhere) ----
    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, dtype=DTYPE, attn_implementation="sdpa").to("cuda").eval()
    tokz = proc.tokenizer
    # Decoder prefix Whisper expects for English transcription without timestamps.
    prefix = tokz.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])
    nospeech_id = tokz.convert_tokens_to_ids("<|nospeech|>")
    prompt_ids = None if args.no_prompt else proc.get_prompt_ids(PROMPT, return_tensors="pt").to("cuda")

    rows = []
    for r in tqdm(files.itertuples(), total=len(files)):
        wav, sr = sf.read(DATA / r.split / r.filename, dtype="float32")
        assert sr == SR, f"{r.filename}: unexpected sample rate {sr}"

        texts, lps, nsp = [], [], []     # per-chunk text, all token log-probs, per-chunk no-speech prob
        for ch in split_chunks(wav):
            # Log-mel features, padded to 30 s by the processor.
            feats = proc(ch, sampling_rate=SR, return_tensors="pt").input_features.to("cuda", DTYPE)
            gen = model.generate(feats, language="en", task="transcribe", num_beams=4,
                                 prompt_ids=prompt_ids, max_new_tokens=220)
            # Keep only text tokens: every special token id is >= eos_token_id in Whisper's vocab.
            toks = [t for t in gen[0].tolist() if t < tokz.eos_token_id]
            if prompt_ids is not None:
                # Some transformers versions return the prompt as part of the output: strip it.
                p = prompt_ids.tolist()[1:]          # [1:] skips the <|startofprev|> marker
                if toks[: len(p)] == p:
                    toks = toks[len(p):]
            lp, ns = confidence(model, feats, prefix, toks, nospeech_id)
            texts.append(tokz.decode(toks, skip_special_tokens=True).strip())
            lps += lp
            nsp.append(ns)

        # Aggregate per-clip confidence stats (NaN-safe in case nothing was decoded).
        lps = np.array(lps) if lps else np.array([np.nan])
        rows.append(dict(filename=r.filename, split=r.split, label=r.label,
                         duration=len(wav) / SR, n_chunks=len(nsp), text=" ".join(texts),
                         asr_avg_lp=np.nanmean(lps),                # overall confidence
                         asr_min_lp=np.nanmin(lps),                 # worst single token
                         asr_p10_lp=np.nanpercentile(lps, 10),      # robust "low-confidence tail"
                         asr_n_tok=int(np.isfinite(lps).sum()),     # amount of speech decoded
                         asr_nospeech_max=max(nsp),
                         asr_nospeech_mean=float(np.mean(nsp))))

    out = ART / f"transcripts_{args.tag}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"saved {out} ({len(rows)} clips, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()
