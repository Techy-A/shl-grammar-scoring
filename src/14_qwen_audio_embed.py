"""Step 14: a second audio-LLM view (Qwen2-Audio-7B-Instruct), pooled like 11_voxtral_embed.py.

Voxtral (3B) gave the largest single gain of any model. Qwen2-Audio pairs a Whisper-large encoder with a 7B
language model trained on spoken-language QA, so its hidden states should be a strong but different view;
the aim is to add it to the big Ridge (c4c_ridge_vox_allaudio) as one more feature table.

Differences from Voxtral:
* Qwen2-Audio's feature extractor truncates at 30 s, so each clip is split into <= 30 s pieces that enter the
  prompt as "Audio 1", "Audio 2" (the model's own multi-audio format). The processor expands each <|AUDIO|>
  to exactly as many tokens as that piece has audio (25 per second), so no padding tokens need dropping.
* 7B in fp16 (~16 GB) doesn't fit one T4, so the model is spread over every visible GPU (device_map="auto").
* Only the last position goes through the LM head (the full logits over a 150k vocabulary would cost ~0.5 GB).

Pooling: hidden states of layers 20-60% of depth averaged, then mean and std over the audio tokens
-> 2 x 3584 features. The direct judgement (probabilities of "1".."5" after the rubric) is saved as well.

Output: artifacts/feat_qwenaudio[_crops][_shard<k>].parquet (filename, split, label, a0..)
        artifacts/feat_qwenaudio_judge[_crops][_shard<k>].parquet (filename, split, label, q_mean, q_p1..q_p5)
Usage : python src/14_qwen_audio_embed.py [--crops] [--shard 0 2] [--limit 3]
"""
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from common import ART, DATA, DTYPE, N_CROPS, SR, crop_lengths

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
PIECE = 30 * SR   # the feature extractor's limit
PROMPT = """You heard a spoken answer from an English test (split into consecutive parts above). Score the speaker's
GRAMMAR: sentence structure, syntax, and grammatical accuracy and complexity. Fillers and self-corrections are normal.

1: Struggles with sentence structure and syntax; limited control of simple grammatical structures.
2: Simple structures with consistent basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of either grammar or sentence structure, with errors in the other.
4: Strong control of sentence structure and grammar; occasional minor errors that don't impede understanding.
5: High grammatical accuracy; complex structures handled well; self-corrects when necessary.

Reply with a single digit from 1 to 5."""


def split_pieces(wav):
    """<= 30 s pieces of equal length (two 27 s halves rather than 30 s + 24 s), each at least 1 s."""
    n = int(np.ceil(len(wav) / PIECE))
    return [p for p in np.array_split(wav, n) if len(p) >= SR]


@torch.inference_mode()
def embed_clip(wav, proc, model, layers, digits):
    pieces = split_pieces(wav)
    # Written out instead of apply_chat_template: this is exactly the model's template for audio-then-text.
    audio = "".join(f"Audio {i + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n" for i in range(len(pieces)))
    text = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{audio}{PROMPT}<|im_end|>\n<|im_start|>assistant\n")
    x = proc(text=text, audio=pieces, sampling_rate=SR, return_tensors="pt").to(model.device)
    x["input_features"] = x["input_features"].to(DTYPE)
    x.pop("num_audio_tokens", None)   # processor bookkeeping, not a model input
    out = model.model(**x, output_hidden_states=True, use_cache=False)
    logits = model.lm_head(out.last_hidden_state[0, -1:])[0]
    probs = torch.softmax(logits[digits].float(), 0).cpu().numpy()
    hs = out.hidden_states   # with device_map="auto" the layers sit on different GPUs
    h = torch.stack([hs[l][0].to(hs[0].device).float() for l in layers]).mean(0)
    h = h[(x["input_ids"][0] == model.config.audio_token_id).to(h.device)]   # audio positions only (no padding)
    return torch.cat([h.mean(0), h.std(0)]).cpu().numpy(), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", action="store_true", help="embed test-length crops of the train clips instead")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("K", "N"), help="rows K, K+N, K+2N, ...")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = ap.parse_args()

    # Same row list (and crops) as 05_audio_embed.py and 11_voxtral_embed.py.
    files = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                       pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    files["n"] = 10**9
    if args.crops:
        tr = files[files.split == "train"]
        lens = crop_lengths(len(tr))
        files = pd.concat([tr.assign(split=f"crop{i}", n=lens[i]) for i in range(N_CROPS)], ignore_index=True)
    k, n = args.shard
    files = files.iloc[k::n].reset_index(drop=True)
    if args.limit:
        files = files.head(args.limit)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, dtype=DTYPE, device_map="auto").eval()
    L = model.config.text_config.num_hidden_layers
    layers = range(int(0.2 * L), int(0.6 * L) + 1)   # hidden_states[0] is the embedding layer
    digits = [proc.tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]
    assert len(set(digits)) == 5, f"digit token ids not distinct: {digits}"

    folder = lambda split: "test" if split == "test" else "train"   # crops are cut from train clips
    emb, probs = map(np.stack, zip(*[embed_clip(sf.read(DATA / folder(r.split) / r.filename, dtype="float32")[0][:r.n],
                                                proc, model, layers, digits)
                                     for r in tqdm(files.itertuples(), total=len(files), mininterval=60)]))
    assert np.isfinite(emb).all() and np.isfinite(probs).all(), "non-finite outputs (fp16 overflow?)"
    ids = files[["filename", "split", "label"]]
    sfx = ("_crops" if args.crops else "") + (f"_shard{k}" if n > 1 else "")
    pd.concat([ids, pd.DataFrame(emb, columns=[f"a{i}" for i in range(emb.shape[1])])], axis=1).to_parquet(
        ART / f"feat_qwenaudio{sfx}.parquet", index=False)
    ids.assign(q_mean=probs @ np.arange(1, 6), **{f"q_p{d}": probs[:, d - 1] for d in range(1, 6)}).to_parquet(
        ART / f"feat_qwenaudio_judge{sfx}.parquet", index=False)
    print(f"saved feat_qwenaudio{sfx} {emb.shape}, layers {layers.start}-{layers.stop - 1} of {L}, "
          f"judge mean {(probs @ np.arange(1, 6)).mean():.2f}, "
          f"peak VRAM {[round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())]} GB")


if __name__ == "__main__":
    main()
