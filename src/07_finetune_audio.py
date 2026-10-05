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
* Fixed epochs and fold_s0 from artifacts/folds.csv, same as 06_finetune_text.py, so the OOF can be blended.

  train : python src/07_finetune_audio.py --folds 0 1 2     -> artifacts/s4b_fold{k}.parquet
  merge : python src/07_finetune_audio.py --merge           -> artifacts/pred_<name>.parquet + results row
"""
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from transformers import WavLMModel, get_linear_schedule_with_warmup

from common import ART, DATA, DTYPE, SR, merge_folds

CROP = 15 * SR
AMP = DTYPE == torch.float16   # mixed precision on Kaggle T4 only (GTX 1650 NaNs in fp16)


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


def train_fold(k, df, wavs, folds, args):
    torch.manual_seed(k)
    rng = np.random.default_rng(k)
    is_tr = (df.split == "train").values
    tr_idx = np.where(is_tr & (folds != k))[0]
    va_idx = np.where(is_tr & (folds == k))[0]
    te_idx = np.where(~is_tr)[0]
    y = (df.label.fillna(0).values / 5).astype(np.float32)

    model = AudioRegressor(args.model).to("cuda")
    model.body.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([{"params": [p for p in model.body.parameters() if p.requires_grad], "lr": args.lr},
                             {"params": [model.layer_w, *model.head.parameters()], "lr": 1e-3}], weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(tr_idx) / args.bs))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=AMP)

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr_idx)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            x = torch.from_numpy(np.stack([normalise(random_crop(wavs[j], rng)) for j in b])).cuda()
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = nn.functional.mse_loss(model(x).float(), torch.from_numpy(y[b]).cuda())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        # Monitoring only; not used to pick an epoch (keeps the OOF honest).
        p = np.clip(predict(model, [wavs[j] for j in va_idx]), 0, 5)
        print(f"fold {k} epoch {ep}: val RMSE {np.sqrt(np.mean((p - df.label.values[va_idx]) ** 2)):.4f}", flush=True)

    idx = np.r_[va_idx, te_idx]
    pd.DataFrame({"filename": df.filename.values[idx], "split": df.split.values[idx],
                  "pred": predict(model, [wavs[j] for j in idx])}).to_parquet(ART / f"s4b_fold{k}.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--name", default="s4b_wavlm_ft")
    args = ap.parse_args()
    if args.merge:
        return merge_folds("s4b", args.name, args.model)

    df = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                    pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    folds = df[["filename"]].merge(pd.read_csv(ART / "folds.csv"), how="left")["fold_s0"].fillna(-1).values
    # All audio in RAM (~3.8 GB float32): avoids re-reading WAVs every epoch.
    wavs = [sf.read(DATA / r.split / r.filename, dtype="float32")[0] for r in df.itertuples()]
    for k in args.folds:
        train_fold(k, df, wavs, folds, args)


if __name__ == "__main__":
    main()
