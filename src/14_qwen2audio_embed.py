"""Step 14: a second audio-LLM view (Qwen2-Audio-7B-Instruct).

Voxtral's hidden states (11_voxtral_embed.py) were the biggest gain; Qwen2-Audio is a different audio language model
(Whisper-large-v3 encoder + Qwen-7B), so its hidden states are a second, partly independent view of the same speech.

Qwen2-Audio's feature extractor takes at most 30 s, so each clip is cut into 30 s windows (a last piece under 1 s is
dropped), each window is embedded on its own, and the windows are averaged weighted by length (as 05_audio_embed.py
does with 20 s windows). The processor expands <|AUDIO|> to exactly as many tokens as the window has real audio.
Two views come from the same forward pass:
* audio: hidden states of the lower-middle layers (20-60% of depth) averaged, then mean and std over the audio tokens
  -> 2 x 4096 features. The audio comes before the text, so the prompt doesn't change them.
* question: the rubric follows the audio, so the hidden states of those text tokens have read the audio *and* the
  question about its grammar, a question-conditioned summary. Mean over the tokens after the audio at the middle and
  upper layers (40-80% of depth, averaged), plus the last position (where the answer starts) -> 2 x 4096 features.

The fp16 weights (~16 GB) don't fit one T4, so the model spans both (device_map="auto") and runs in one process.
Output: artifacts/feat_qwen2audio[_crops].parquet, artifacts/feat_qwen2audio_q[_crops].parquet (filename, split, label, a0..)
Usage : python src/14_qwen2audio_embed.py [--crops] [--limit 3]
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
WINDOW = 30 * SR
RUBRIC = """Score the GRAMMAR of this spoken answer from an English test: sentence structure, syntax, and grammatical
accuracy and complexity. Fillers and self-corrections are normal in speech.
1: Struggles with sentence structure and syntax; limited control of simple grammatical structures.
2: Simple structures with consistent basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of either grammar or sentence structure, with errors in the other.
4: Strong control of sentence structure and grammar; occasional minor errors that don't impede understanding.
5: High grammatical accuracy; complex structures handled well; self-corrects when necessary.
Reply with a single digit from 1 to 5."""
PROMPT = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
          f"Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{RUBRIC}<|im_end|>\n<|im_start|>assistant\n")


@torch.inference_mode()
def embed_clip(wav, proc, model, layers, qlayers):
    vecs, qvecs, weights = [], [], []
    for start in range(0, len(wav), WINDOW):
        w = wav[start:start + WINDOW]
        if len(w) < SR:   # skip a trailing piece under 1 s
            continue
        x = proc(text=PROMPT, audio=[w], sampling_rate=SR, return_tensors="pt").to(model.device)
        x["input_features"] = x["input_features"].to(DTYPE)
        hs = model(**x, output_hidden_states=True).hidden_states
        ids = x["input_ids"][0]
        audio = ids == model.config.audio_token_id
        h = torch.stack([hs[l][0].to(ids.device) for l in layers]).float().mean(0)[audio]   # (audio tokens, hidden)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
        after = torch.arange(len(ids), device=ids.device) > audio.nonzero().max()          # the rubric and answer start
        q = torch.stack([hs[l][0].to(ids.device) for l in qlayers]).float().mean(0)
        qvecs.append(torch.cat([q[after].mean(0), q[-1]]).cpu().numpy())
        weights.append(len(w))
    return np.average(vecs, axis=0, weights=weights), np.average(qvecs, axis=0, weights=weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", action="store_true", help="embed test-length crops of the train clips instead")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = ap.parse_args()

    # Same row list as 05_audio_embed.py, so crops line up with the other audio tables.
    files = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                       pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    files["n"] = 10**9
    if args.crops:
        tr = files[files.split == "train"]
        lens = crop_lengths(len(tr))
        files = pd.concat([tr.assign(split=f"crop{i}", n=lens[i]) for i in range(N_CROPS)], ignore_index=True)
    if args.limit:
        files = files.head(args.limit)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, dtype=DTYPE, device_map="auto").eval()
    L = model.config.text_config.num_hidden_layers
    layers = range(int(0.2 * L), int(0.6 * L) + 1)    # hidden_states[0] is the embedding layer
    qlayers = range(int(0.4 * L), int(0.8 * L) + 1)

    folder = lambda split: "test" if split == "test" else "train"   # crops are cut from train clips
    emb, qemb = map(np.stack, zip(*[embed_clip(sf.read(DATA / folder(r.split) / r.filename, dtype="float32")[0][:r.n],
                                               proc, model, layers, qlayers)
                                    for r in tqdm(files.itertuples(), total=len(files), mininterval=60)]))
    assert np.isfinite(emb).all() and np.isfinite(qemb).all(), "non-finite embeddings (fp16 overflow?)"
    sfx = "_crops" if args.crops else ""
    for name, e in [(f"feat_qwen2audio{sfx}", emb), (f"feat_qwen2audio_q{sfx}", qemb)]:
        pd.concat([files[["filename", "split", "label"]], pd.DataFrame(e, columns=[f"a{i}" for i in range(e.shape[1])])],
                  axis=1).to_parquet(ART / f"{name}.parquet", index=False)
    print(f"saved feat_qwen2audio{sfx} {emb.shape} + feat_qwen2audio_q{sfx} {qemb.shape}, "
          f"layers {layers.start}-{layers.stop - 1} / {qlayers.start}-{qlayers.stop - 1} of {L}, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB on the first GPU")


if __name__ == "__main__":
    main()
