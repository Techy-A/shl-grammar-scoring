# Leaderboard submissions

Each file is a `filename,label` prediction for the 216 `test.csv` clips, produced by `src/08_blend.py`.
Lower is better; the public leaderboard behaves like MAE.

| File | Kaggle ref | Blend | Rounded to 0.5 | CV MAE | Public score |
|---|---|---|---|---|---|
| `01_blend_v1.csv` | 56860586 | score-0 gate + LightGBM, DeBERTa Ridge, WavLM Ridge, DeBERTa fine-tune, WavLM+DeBERTa Ridge | no | 0.370 | 0.3463 |
| `02_blend_v2_rounded.csv` | 56860724 | v1 + SVR on WavLM and on WavLM+DeBERTa | yes | 0.335* | 0.3678 |
| `03_blend_v2.csv` | 56860768 | v1 + SVR on WavLM and on WavLM+DeBERTa | no | 0.349 | **0.3454** |
| `04_blend_v1_rounded.csv` | 56860770 | v1 | yes | 0.351* | 0.3654 |

\* Rounding improves CV MAE (train labels sit on a 0.5 grid) but made the public score about 0.02
worse in both paired submissions, so the test labels are probably not on the grid. Rounding was dropped.
