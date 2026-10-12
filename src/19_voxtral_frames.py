"""Step 19: cache Voxtral's frame-level states for every full clip (input for 20_attn_pool.py).

11_voxtral_embed.py pools the audio tokens to mean + std once per clip / crop. Here the frames themselves are kept, the
same layers (20-60% of depth, averaged) and only tokens that cover real audio, so a pooling head can be trained on them
with any number of random test-length crops. A crop of the train clips is a prefix of the audio and the language model's
attention is causal, so a crop's frames are (nearly) a prefix of the full clip's frames; only the encoder's last,
partial 30 s chunk differs.

Frames are reduced to 256 dimensions by PCA (fitted on 100k random frames) to keep the cache small.
--shard k n embeds every n-th clip; --merge fits the PCA on all shards and writes the cache.
Output: artifacts/voxtral_frames.npz (frames float16 [n_frames, 256], start offsets, filename, split, label, pca mean/basis)
Usage : python src/19_voxtral_frames.py --shard 0 2 --raw /kaggle/tmp ; ... ; python src/19_voxtral_frames.py --merge --raw /kaggle/tmp
"""
import argparse
import importlib
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, VoxtralForConditionalGeneration

from common import ART, DATA, DTYPE, SR

vox = importlib.import_module("11_voxtral_embed")
DIM = 256


@torch.inference_mode()
def frames(wav, proc, model, layers, tmp):
    sf.write(tmp, wav, SR)
    conv = [{"role": "user", "content": [{"type": "audio", "path": str(tmp)}, {"type": "text", "text": vox.PROMPT}]}]
    x = proc.apply_chat_template(conv).to("cuda", dtype=DTYPE)
    hs = model(**x, output_hidden_states=True).hidden_states
    h = torch.stack([hs[l][0] for l in layers]).float().mean(0)
    pos = (x["input_ids"][0] == model.config.audio_token_id).nonzero().squeeze(-1)
    dur = len(wav) / SR   # keep the audio tokens that cover real audio (as 11_voxtral_embed.embed_clip)
    keep = [p for i, p in enumerate(pos.tolist())
            if i % vox.CHUNK_TOKENS < math.ceil(min(max(dur - 30 * (i // vox.CHUNK_TOKENS), 0), 30) * vox.TOKENS_PER_SEC)]
    return h[keep].half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("K", "N"))
    ap.add_argument("--raw", required=True, help="folder for the per-shard raw frames")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    raw = Path(args.raw)

    if args.merge:
        parts = [np.load(f, allow_pickle=True) for f in sorted(raw.glob("frames_shard*.npz"))]
        ids = pd.concat([pd.DataFrame({c: p[c] for c in ("filename", "split", "label")}) for p in parts], ignore_index=True)
        lens = np.concatenate([p["lens"] for p in parts])
        X = np.concatenate([p["frames"] for p in parts])   # fp16 on the CPU: ~4 GB, twice that in fp32 doesn't fit a T4
        sample = torch.from_numpy(X[np.random.default_rng(0).permutation(len(X))[:100_000]]).cuda().float()
        mu = sample.mean(0)
        _, _, V = torch.pca_lowrank(sample - mu, q=DIM, center=False)
        Z = np.concatenate([((torch.from_numpy(X[i:i + 50_000]).cuda().float() - mu) @ V).half().cpu().numpy()
                            for i in range(0, len(X), 50_000)])
        np.savez(ART / "voxtral_frames.npz", frames=Z, start=np.r_[0, np.cumsum(lens)[:-1]], lens=lens,
                 filename=ids.filename.values, split=ids.split.values, label=ids.label.values,
                 mu=mu.cpu().numpy(), basis=V.cpu().numpy())
        return print(f"saved voxtral_frames.npz: {len(ids)} clips, {len(Z)} frames x {DIM}")

    files = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                       pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    k, n = args.shard
    files = files.iloc[k::n].reset_index(drop=True)
    proc = AutoProcessor.from_pretrained(vox.MODEL)
    model = VoxtralForConditionalGeneration.from_pretrained(vox.MODEL, dtype=DTYPE).to("cuda").eval()
    L = model.config.text_config.num_hidden_layers
    layers = range(int(0.2 * L), int(0.6 * L) + 1)
    tmp = Path(tempfile.gettempdir()) / f"voxframes_{k}.wav"
    out = [frames(sf.read(DATA / r.split / r.filename, dtype="float32")[0], proc, model, layers, tmp)
           for r in tqdm(files.itertuples(), total=len(files), mininterval=60)]
    assert all(np.isfinite(f).all() for f in out), "non-finite frames (fp16 overflow?)"
    raw.mkdir(parents=True, exist_ok=True)
    np.savez(raw / f"frames_shard{k}.npz", frames=np.concatenate(out), lens=np.array([len(f) for f in out]),
             filename=files.filename.values, split=files.split.values, label=files.label.values)
    print(f"shard {k}: {len(out)} clips, {sum(len(f) for f in out)} frames")


if __name__ == "__main__":
    main()
