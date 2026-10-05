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

ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    # Competition data is mounted read-only; outputs must go to /kaggle/working to be downloadable.
    DATA = Path("/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final")
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
