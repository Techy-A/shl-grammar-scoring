# SHL Hiring Assessment 2026: Spoken Grammar Scoring

Predicts a 0–5 grammar proficiency score for spoken-English audio clips
(Kaggle competition `shl-hiring-assessment-2026`).

## Approach

Two views of each clip, combined at the end:

- **What was said:** Whisper-large-v3-turbo transcripts, decoded with a verbatim-style prompt so Whisper
  keeps the speaker's grammar mistakes instead of fixing them. Used through hand-crafted text stats and
  ASR confidence, frozen DeBERTa-v3-large embeddings, and a fine-tuned DeBERTa-v3-large regressor.
- **How it was said:** frozen WavLM embeddings (20 s windows, mean + std pooling of middle layers),
  fed to Ridge and SVR models.
- **Blend:** clips that the WavLM model scores near 0 are set to 0 (silent / unusable recordings; it
  catches all 37 in train with no false positives). The rest get an intercept plus non-negative
  weights over all models' out-of-fold predictions.

All models share the same stratified folds (`artifacts/folds.csv`), so their out-of-fold predictions
can be blended without leakage.

## Pipeline

| Script | Step |
|---|---|
| `src/01_transcribe.py` | Whisper transcripts + ASR confidence |
| `src/02_text_features.py` | length-robust text stats from the transcripts |
| `src/03_train.py` | CV models (mean / LightGBM / Ridge / SVR) on cached feature tables |
| `src/04_text_embed.py` | frozen DeBERTa embeddings of the transcripts |
| `src/05_audio_embed.py` | frozen WavLM / Whisper-encoder embeddings of the audio |
| `src/06_finetune_text.py` | DeBERTa fine-tuned to predict the score from the transcript |
| `src/07_finetune_audio.py` | WavLM fine-tuned to predict the score from the audio |
| `src/08_blend.py` | score-0 gate + non-negative blend -> `artifacts/submission.csv` |

`src/common.py` holds paths and precision settings, so the same scripts run locally and on Kaggle.
Each script caches its output in `artifacts/` for the next step.

## Results (5-fold CV on 769 train clips)

| Model | CV RMSE | CV MAE |
|---|---|---|
| Train mean (baseline) | 1.238 | |
| Text stats + LightGBM | 0.771 | |
| DeBERTa embeddings + Ridge | 0.756 | |
| DeBERTa fine-tuned | 0.776 | |
| WavLM embeddings + Ridge | 0.564 | |
| WavLM embeddings + SVR | 0.535 | 0.397 |
| WavLM + DeBERTa embeddings + SVR | 0.519 | 0.396 |
| **Blend** | **0.483** | **0.349** |

Best public leaderboard score: **0.3454** (blend, unrounded). The leaderboard tracks CV MAE.
Rounding predictions to the 0.5 label grid improved CV but made the public score about 0.02 worse.

## Running

Data: put the competition's `Dataset_Final/` under `data/`.

```bash
pip install -r requirements.txt
python src/01_transcribe.py --tag prompt
python src/02_text_features.py --tag prompt
python src/04_text_embed.py --tag prompt
python src/05_audio_embed.py
python src/03_train.py --name s3_svr_wavlm --model svr --feats feat_wavlm-base-plus
# ... remaining models, see each script's docstring
python src/08_blend.py s1_lgbm_prompt s2_ridge_deberta s3_ridge_wavlm s4_deberta_ft \
    s3c_ridge_wavlm_deberta s3_svr_wavlm s3c_svr_wavlm_deberta
```

GPU-heavy steps run on Kaggle (2× T4): `bash kaggle/run_job.sh <job>` uploads `src/` as a private
dataset, runs `kaggle/jobs/<job>/run.py` as a kernel, waits, and downloads its outputs into
`artifacts/`. GTX 16xx cards produce NaNs in fp16, so `common.py` switches them to fp32.
