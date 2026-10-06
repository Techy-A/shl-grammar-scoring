# Leaderboard submissions

Each file is a `filename,label` prediction for the 216 `test.csv` clips, produced by `src/08_blend.py`.
Lower is better; the competition evaluates with RMSE and Pearson correlation.

| File | Kaggle ref | Blend | Rounded to 0.5 | CV MAE | Public score |
|---|---|---|---|---|---|
| `01_blend_v1.csv` | 56860586 | score-0 gate + LightGBM, DeBERTa Ridge, WavLM Ridge, DeBERTa fine-tune, WavLM+DeBERTa Ridge | no | 0.370 | 0.3463 |
| `02_blend_v2_rounded.csv` | 56860724 | v1 + SVR on WavLM and on WavLM+DeBERTa | yes | 0.335* | 0.3678 |
| `03_blend_v2.csv` | 56860768 | v1 + SVR on WavLM and on WavLM+DeBERTa | no | 0.349 | **0.3454** |
| `04_blend_v1_rounded.csv` | 56860770 | v1 | yes | 0.351* | 0.3654 |
| `05_blend_v3.csv` | 56862170 | v2 + SVR on WavLM-large, SVR on WavLM-large+Whisper+DeBERTa, Whisper Ridge, all-audio SVR | no | **0.340** | 0.3550 |
| `06_blend_v2_lengthfix.csv` | 56862646 | v2 + 0.14 for test clips of 45-58 s (length-bias test) | no | — | 0.3556 |
| `07_blend_v4.csv` | 56869082 | v3 with its SVR models trained on full clips + test-length crops (`--crops`) | no | 0.388 (crop-CV) | 0.3384 |
| `08_blend_v6.csv` | 56875361 | every model trained on full clips + 4 test-length crops (Ridge alpha picked with clip-grouped CV) | no | 0.387 (crop-CV) | 0.3373 |
| `09_blend_v7.csv` | 56879050 | text models trained and validated on transcripts of the crops (not the full clip) + 4-crop audio models | no | 0.392 (crop-CV, honest text) | **0.3350** |
| `10_blend_v8.csv` | 56879454 | v7 + the two text SVRs on crop transcripts | no | 0.3885 (crop-CV, honest text) | 0.3382 |

\* Rounding improves CV MAE (train labels sit on a 0.5 grid) but made the public score about 0.02
worse in both paired submissions, so the test labels are probably not on the grid. Rounding was dropped.

v3 has the best CV MAE but a worse public score than v2. With 216 public clips the score's standard error
is about 0.02, so public differences this size are mostly noise; the private leaderboard decides.

06 tested a length bias: cutting full-length train clips to 48 s lowers the WavLM model's predictions
by 0.14. Adding that back to the 45-58 s test clips made v2 worse, so the blend itself isn't
under-predicting short test clips.

v7's crop-CV looks worse than v6's only because v6 validated crops with text from the *full* clip
(words a short test clip never has). With honest validation the text models score lower, but the
leaderboard confirms v7 is better.

**Metric probes** (`probe_v7_plus0.2.csv` 0.3849, `probe_v7_minus0.2.csv` 0.3305). Adding a constant leaves
Pearson unchanged. For a pure-RMSE score the two results would satisfy LB(+)^2 + LB(-)^2 = 2*LB(v7)^2 + 2*0.2^2
(0.3045); they give 0.2574, so the leaderboard score also contains a part shifting can't change (consistent with
the stated RMSE + Pearson evaluation). Under every form tested, v7's test predictions are about 0.12 too high on average.
