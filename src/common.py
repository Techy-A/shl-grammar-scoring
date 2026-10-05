"""Settings shared by every step, so the same scripts run locally and on Kaggle unchanged.

Pipeline (each script caches its output in artifacts/ for the next):
  01_transcribe.py      Whisper transcripts + ASR confidence
  02_text_features.py   length-robust text stats from the transcripts
  03_train.py           CV models (mean / LightGBM / Ridge) on any cached feature tables
  04_text_embed.py      frozen DeBERTa embeddings of the transcripts
  05_audio_embed.py     frozen WavLM embeddings of the audio
  06_finetune_text.py   DeBERTa fine-tuned to predict the score from the transcript
  07_finetune_audio.py  WavLM fine-tuned to predict the score from the audio
  08_blend.py           score-0 gate + non-negative blend of all predictions -> submission.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    # Competition data is mounted read-only; outputs must go to /kaggle/working to be downloadable.
    # The mount path differs between kernels (/kaggle/input/competitions/<slug>/... or /kaggle/input/<slug>/...).
    DATA = next(Path("/kaggle/input").glob("**/Dataset_Final"))
    ART = Path("/kaggle/working/artifacts")
else:
    ROOT = Path(__file__).resolve().parents[1]   # project root (D:\SHLHIRINGASS)
    DATA = ROOT / "data" / "Dataset_Final"       # train.csv, test.csv, train/, test/
    ART = ROOT / "artifacts"                     # cached outputs shared by later steps
ART.mkdir(parents=True, exist_ok=True)

SR = 16000   # all clips are 16 kHz mono (checked during EDA)

# fp16 halves memory and is much faster on the Kaggle T4, but GTX 16xx cards (the local
# GTX 1650) produce NaNs in fp16 (verified: Whisper's encoder output was all NaN), so they use fp32.
DTYPE = torch.float32 if "GTX 16" in torch.cuda.get_device_name() else torch.float16
AMP = DTYPE == torch.float16   # mixed-precision training only where fp16 works (Kaggle T4)


def finetune_fold(k, df, folds, make_model, batches, predict, epochs, bs, prefix):
    """Training loop shared by 06_finetune_text.py and 07_finetune_audio.py, for fold k.

    make_model() -> (model, AdamW param groups); built after seeding, so each fold is reproducible.
    batches(idx, y, rng) yields (model input, target) for one epoch over the train rows idx.
    predict(model, idx) -> scores on the 0-5 scale.
    Targets are label / 5 (in [0, 1]), MSE loss. A fixed number of epochs, no early stopping, so the
    out-of-fold predictions stay honest. Writes artifacts/<prefix>_fold{k}.parquet (val + test rows).
    """
    torch.manual_seed(k)
    rng = np.random.default_rng(k)
    is_tr = (df.split == "train").values
    tr_idx = np.where(is_tr & (folds != k))[0]
    va_idx = np.where(is_tr & (folds == k))[0]
    te_idx = np.where(~is_tr)[0]
    y = (df.label.fillna(0).values / 5).astype(np.float32)

    model, groups = make_model()
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr_idx) / bs))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=AMP)   # loss scaling for fp16; weights stay fp32
    for ep in range(epochs):
        model.train()
        for x, yb in batches(rng.permutation(tr_idx), y, rng):
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = nn.functional.mse_loss(model(x).float(), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        # Monitoring only; not used to pick an epoch (keeps the OOF honest).
        p = np.clip(predict(model, va_idx), 0, 5)
        print(f"fold {k} epoch {ep}: val RMSE {np.sqrt(np.mean((p - df.label.values[va_idx]) ** 2)):.4f}", flush=True)

    idx = np.r_[va_idx, te_idx]
    pd.DataFrame({"filename": df.filename.values[idx], "split": df.split.values[idx],
                  "pred": predict(model, idx)}).to_parquet(ART / f"{prefix}_fold{k}.parquet", index=False)


def fold_of(filenames, col):
    """Fold number of each clip in artifacts/folds.csv column col (written by 10_speaker_folds.py).
    folds.csv lists train clips only; other names get -1. Test rows can share a train filename and
    then get that clip's fold, so callers select rows by split as well (they all do)."""
    return pd.read_csv(ART / "folds.csv").set_index("filename")[col].reindex(filenames).fillna(-1).astype(int).values


def merge_folds(prefix, name, desc):
    """Combine per-fold outputs of a fine-tuning run (artifacts/<prefix>_fold{k}.parquet).

    Train clips appear in exactly one validation fold -> honest out-of-fold (OOF) predictions.
    Test clips are predicted by all 5 fold models -> averaged. Writes artifacts/pred_<name>.parquet
    in the same format as 03_train.py and appends a row to the results table.
    """
    parts = pd.concat([pd.read_parquet(ART / f"{prefix}_fold{k}.parquet") for k in range(5)])
    pred = parts.groupby(["filename", "split"], as_index=False).pred.mean()
    pred["pred"] = pred.pred.clip(0, 5)
    # Train and test reuse filenames, so labels are joined on (filename, split): test rows stay NaN.
    labels = pd.read_csv(DATA / "train.csv").assign(split="train")
    pred = pred.merge(labels, on=["filename", "split"], how="left")[["filename", "split", "label", "pred"]]
    tr = pred[pred.split == "train"]
    rmse = float(np.sqrt(np.mean((tr.pred - tr.label) ** 2)))
    r = float(np.corrcoef(tr.pred, tr.label)[0, 1])
    print(f"{name}: CV RMSE {rmse:.4f} | Pearson {r:.4f}")
    pred.to_parquet(ART / f"pred_{name}.parquet", index=False)
    pd.DataFrame([dict(name=name, model="finetune", feats=desc, n_feats=0, cv_rmse=round(rmse, 4),
                       cv_pearson=round(r, 4))]).to_csv(ART / "results.csv", mode="a", header=False, index=False)
