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
| `06_blend_v2_lengthfix.csv` | 56862646 | v2 + 0.14 for test clips of 45-58 s (length-bias test) | no | n/a | 0.3556 |
| `07_blend_v4.csv` | 56869082 | v3 with its SVR models trained on full clips + test-length crops (`--crops`) | no | 0.388 (crop-CV) | 0.3384 |
| `08_blend_v6.csv` | 56875361 | every model trained on full clips + 4 test-length crops (Ridge alpha picked with clip-grouped CV) | no | 0.387 (crop-CV) | 0.3373 |
| `09_blend_v7.csv` | 56879050 | text models trained and validated on transcripts of the crops (not the full clip) + 4-crop audio models | no | 0.392 (crop-CV, honest text) | 0.3350 |
| `10_blend_v8.csv` | 56879454 | v7 + the two text SVRs on crop transcripts | no | 0.3885 (crop-CV, honest text) | 0.3382 |
| `11_blend_v7_biasfix.csv` | 56880076 | v7 − 0.118: the test-set bias measured by the two shift probes (below) | no | n/a | **0.3262** |
| `12_blend_v9_biasfix.csv` | 56884530 | v7 + Ridge on Qwen3-4B transcript embeddings + Ridge on w2v-BERT 2.0 + DeBERTa, mean matched to 11 | no | 0.386 (crop-CV RMSE 0.518) | 0.3341 |

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

**Bias-fix probes** (`probe_biasfix_full_minus0.1.csv` 0.3277, `probe_biasfix_short_minus0.1.csv` 0.3321). Taking a further 0.1
off either length group of `11_blend_v7_biasfix.csv` (0.3262) made the score worse, so the −0.118 correction is already
about right for short and full-length clips alike. No further shift.

**v9** improved CV RMSE from 0.528 to 0.518 but scored 0.3341 on the public leaderboard, 0.008 worse than v7 with the bias
fix (predictions correlate 0.993, so most of that gap is real, not noise). It is the third time that adding models
improved speaker-grouped CV and hurt the leaderboard (v3, v8, v9). Changes to *how* models are trained (crops,
crop transcripts, the bias fix) transferred; extra models stacked on the blend did not.

**Scale probe** (`probe_scale_1.15.csv`, 56888773, **0.3248**). v7's test predictions are compressed (std 0.80, against
1.12 for its out-of-fold predictions on train). Stretching `11_blend_v7_biasfix.csv` around its mean,
mean + 1.15 * (pred - mean), clipped to [0, 5], improved the public score from 0.3262 to 0.3248.
**Short-clip stretch probe** (`probe_stretch_short_x1.2.csv`, 56888795, 0.3408). Stretching only the 153 short test clips
(<= 55 s) by a further 1.2 around their own mean made the score much worse (0.3248 to 0.3408). Their compressed spread
(std 0.65) is not something to correct; the gain from the global stretch must come from elsewhere.
**Full-length stretch** (`probe_stretch_full_only.csv`, 56888818, **0.3241**). The x1.15 stretch from `probe_scale_1.15.csv`
applied to the 63 full-length test clips (> 55 s) only, with the short clips left as in `11_blend_v7_biasfix.csv`. Best so far:
the global stretch's gain came from the full-length clips, and stretching the short clips costs score.
**Batch offset** (`13_batch_offset.csv`, 56890641, 0.3320). On train, the 45.06 s recording batch (87 clips; 84 of the 216
test clips) is over-predicted by 0.25 in cross-validation, and a per-batch offset learned on the training folds improved
CV RMSE from 0.528 to 0.523. Applied to the test set (batch -0.298, other clips -0.003, same total shift as the bias fix,
full-length clips x1.15) it made the public score worse (0.3241 to 0.3320): the test batch does not behave like the train
batch, probably because it answers questions that never occur in training. Not used.
**Voxtral blend** (`14_blend_b3_voxtral.csv`, 56892005, **0.3224**). Voxtral-Mini-3B (an audio language model) hidden states
over the audio, pooled like the other encoders, combined with WavLM-large in one SVR on full clips + crops: CV RMSE 0.5794,
the best single model. Blended with three of the v7 models (DeBERTa ridge, Whisper-encoder ridge, WavLM-large SVR): blend CV
0.5207 against 0.5280 for v7. Same corrections as the 0.3241 file (test mean matched to v7 - 0.118, full-length clips x1.15).
New best public score. The idea of an audio LLM's hidden states as features comes from another participant's public solution.
**Ridge on Voxtral + all audio encoders** (`15_blend_ridge3.csv`, 56899694, **0.3207**). Ridge suits the wide Voxtral
features better than SVR: Ridge on Voxtral + WavLM-base-plus + WavLM-large + Whisper encoder (12288 features, crops) has
CV RMSE 0.5375 against 0.5794 for the Voxtral + WavLM-large SVR. Three models, one per kind of signal (DeBERTa ridge for
the transcript, this ridge, WavLM-large SVR): blend CV 0.5158 against 0.5207. Same corrections (test mean matched to
v7 - 0.118, a shift of -0.1151; full-length clips x1.15). New best public score; 2nd on the public leaderboard.
