"""Step 7: fine-tune WavLM to regress the score directly from the audio.

Frozen WavLM + Ridge (05_audio_embed.py) was the strongest single model, so letting the audio model itself
adapt to the task is the most promising next step (the audio counterpart of 06_finetune_text.py).

Design choices:
* Learnable softmax weights over all hidden layers (different layers carry different cues),
  mean over time -> linear head. MSE on label / 5.
* The CNN feature encoder stays frozen (standard for small-data wav2vec2/WavLM fine-tuning).
* Training on random 15 s crops: augmentation (a new crop every epoch) and test-like lengths
  (test clips are shorter than train clips).
* Prediction = average over consecutive 15 s windows of the whole clip, weighted by window length.
* Fixed epochs and gfold_s0 from artifacts/folds.csv, same as 06_finetune_text.py, so the OOF can be blended.

  train : python src/07_finetune_audio.py --folds 0 1 2     -> artifacts/s4b_fold{k}.parquet
  merge : python src/07_finetune_audio.py --merge           -> artifacts/pred_<name>.parquet + results row
"""
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from transformers import WavLMModel

from common import AMP, DATA, SR, finetune_fold, fold_of, merge_folds

CROP = 15 * SR
MODEL, NAME = "microsoft/wavlm-base-plus", "s4b_wavlm_ft"
EPOCHS, LR, BS = 8, 3e-5, 8


class AudioRegressor(nn.Module):
    def __init__(self, name):
        super().__init__()
        # fp32 master weights. layerdrop=0: WavLM randomly skips layers in training by default, which
        # would change the number of hidden states per step and break the per-layer weights below.
        self.body = WavLMModel.from_pretrained(name, dtype=torch.float32, layerdrop=0.0)
        self.body.feature_extractor._freeze_parameters()
        n_layers = self.body.config.num_hidden_layers + 1   # + the CNN/embedding output
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Linear(self.body.config.hidden_size, 1)

    def forward(self, x):
        hs = torch.stack(self.body(x, output_hidden_states=True).hidden_states)   # (layers, batch, frames, dim)
        h = (self.layer_w.softmax(0)[:, None, None, None] * hs).sum(0)
        return self.head(h.mean(1)).squeeze(-1)


def normalise(w):
    """Zero-mean / unit-variance per clip, as WavLM's feature extractor does."""
    return (w - w.mean()) / (w.std() + 1e-7)


def random_crop(wav, rng):
    if len(wav) <= CROP:
        return np.pad(wav, (0, CROP - len(wav)))
    s = rng.integers(0, len(wav) - CROP)
    return wav[s:s + CROP]


@torch.inference_mode()
def predict(model, wavs):
    model.eval()
    preds = []
    for wav in wavs:
        windows = [wav[s:s + CROP] for s in range(0, len(wav), CROP)]
        windows = [w for w in windows if len(w) >= SR] or [wav]   # drop <1 s tails
        ps = []
        for w in windows:
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                ps.append(model(torch.from_numpy(normalise(w))[None].cuda()).float().item())
        preds.append(np.average(ps, weights=[len(w) for w in windows]) * 5)
    return np.array(preds)


def make_model():
    model = AudioRegressor(MODEL).to("cuda")
    model.body.gradient_checkpointing_enable()
    return model, [{"params": [p for p in model.body.parameters() if p.requires_grad], "lr": LR},
                   {"params": [model.layer_w, *model.head.parameters()], "lr": 1e-3}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge_folds("s4b", NAME, MODEL)

    df = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                    pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    folds = fold_of(df.filename, "gfold_s0")
    # All audio in RAM (~3.8 GB float32): avoids re-reading WAVs every epoch.
    wavs = [sf.read(DATA / r.split / r.filename, dtype="float32")[0] for r in df.itertuples()]

    def train_batches(idx, y, rng):   # a fresh random 15 s crop of every clip, every epoch
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            x = np.stack([normalise(random_crop(wavs[j], rng)) for j in b])
            yield torch.from_numpy(x).cuda(), torch.from_numpy(y[b]).cuda()

    for k in args.folds:
        finetune_fold(k, df, folds, make_model, train_batches, lambda m, idx: predict(m, [wavs[j] for j in idx]),
                      EPOCHS, BS, "s4b")


if __name__ == "__main__":
    main()
