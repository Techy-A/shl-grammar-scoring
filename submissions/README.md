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
**Kernel ridge with per-block weights** (`16_blend_kernel.csv`, 56908476, 0.3290, worse). One kernel ridge on a weighted sum of
per-table kernels (Voxtral 1, WavLM 1, WavLM-large linear 0.25 and RBF 2, Whisper encoder 0.25, DeBERTa 0.5; weights from a
coordinate search on seed 0) plus the DeBERTa ridge: blend CV 0.5044 against 0.5158, but the public score got worse. A fourth
case of a CV gain from the blend's composition that does not reach the leaderboard; the weight search probably fitted the
CV folds. Same corrections (shift -0.081 to v7 - 0.118, full-length clips x1.15). Not used.
**Equal-weight kernel ridge** (`17_blend_kernel_equal.csv`, 0.3286, worse). The Voxtral + all-audio Ridge replaced by a kernel
ridge over the same four tables with equal per-table weight (no weight search): blend CV 0.5085 against 0.5158, public worse,
like 16. Both kernel versions lose about 0.008 on the public score for reasons CV does not show. Qwen2-Audio-7B hidden states
(CV 0.573 alone, the best single table) add nothing next to Voxtral. Voxtral QLoRA stopped (fold 0: 0.764 then 0.789).
**Short clips compressed x0.89** (`18_ridge3_short_x0.89.csv`, 0.3255, worse). The 0.3207 file with the short test clips
(<= 55 s) scaled x0.89 around their mean, the optimum of a quadratic fitted through the x1.0 and x1.2 short-stretch probes.
Worse: the short clips' spread is already about right, and the two-point RMSE model of the score does not hold. 15 stays best.

## Identifying the leaderboard metric

The competition uses a private custom metric notebook (`metric_template`); its pages only say "Pearson Correlation and RMSE".
Five probes rescale the 0.3207 file around its mean, `m + s (p - m)`, unclipped. For s > 0 Pearson is unchanged and only RMSE
moves; s = -1 flips the sign of Pearson.

| s | 0.05 | 0.5 | 1 | 1.5 | 2 | -1 |
|---|---|---|---|---|---|---|
| public | 0.6225 | 0.4251 | 0.3207 | 0.4403 | 0.6659 | 1.8714 |

`a * sqrt(R^2 + C (s - s0)^2) + K` fits the five s > 0 scores to 1e-5 (a squared-error form misses by 0.019; a free power
comes out at 0.4997, so the score is linear in RMSE). The flip then separates the Pearson part: with `K = w (1 - r)` and
`K' = w (1 + r)`, w = 0.40 and r = 0.89. So the public score is **0.6 RMSE + 0.4 (1 - Pearson)** on the public 60% of the test
clips; for the 0.3207 file the public RMSE is about 0.46 and the Pearson about 0.89. The best scale is s0 = 0.98, so the spread is
already calibrated (worth 0.0002). Once calibrated, RMSE moves with Pearson as well: +0.01 Pearson is worth about -0.016.

**Later experiments, no submission.** CrisperWhisper verbatim transcripts for the text model: better under speaker-grouped
folds, no gain under topic-grouped folds (`folds_topic.csv`, which match the test set's topic distance). Co-training (each
model also learns from the blend's predictions on unlabelled held-out clips, simulated inside CV): worse under every fold
scheme (topic-grouped 0.3268 vs 0.330 after one round, 0.340 after two). Calibration is exhausted too: re-fitting the shift
probes with the decoded metric gives an optimal shift of -0.1179 (used: -0.118), and the best rescaling is 0.98.

**Overnight search** (`21_kernel_crisper_svr.csv`, 0.3390, worse). 18 candidate models trained without the score-0 clips,
scored on topic-grouped, strict-topic and speaker-grouped folds (3 seeds each); a greedy blend search on the topic folds
picked the tuned kernel + a CrisperWhisper text ridge + an all-audio SVR (topic 0.3175 vs 0.3242 for the 0.3207 structure,
better on the other two schemes too, bootstrap P = 0.99). Public 0.3390: the third public failure of the kernel models
(16, 17, 21) although they win every CV scheme, so they are excluded from candidates. Without them, the 0.3207 structure is a
local optimum: adding any of 13 ridge/SVR models or the transcript statistics changes the topic-fold score by less than 0.001.
Importance weighting for the covariate shift (each training clip weighted by how test-like it looks): better only on a
test-weighted CV score (-0.006), worse on every plain fold scheme; not submitted.

**Voxtral judgement states** (`22_ridge3_voxjudge.csv`, 0.3247, worse). `11_voxtral_embed.py --last` saves the hidden states
that come after the audio and therefore also see the scoring rubric (last token before the answer, mid and upper layers;
mean over the rubric's tokens), computed on a free Colab T4. They are less domain-shifted than the audio states (test-vs-train
AUC 0.79 vs 0.83), and adding them to the main ridge improves every CV scheme in both training variants (topic-grouped
-0.0026, strict-topic -0.0024, speaker-grouped -0.0033). Submitted as a minimal change of the 0.3207 file (its raw predictions
plus the blend difference, same corrections; correlation 0.9987): public 0.3247. The seventh change in a row that is better
or equal in validation and worse on the public 60%; the 0.3207 file is a lucky local optimum of the public clips.

**Ensemble of three blends** (`23_avg_15_14_20.csv`, **0.3195**, new best). Equal-weight average of the 3-model blend
(0.3207), the 4-model Voxtral blend (0.3224) and the 3-model blend trained without score-0 clips (0.3242). For squared error,
mean((a + b)/2 - y)^2 = (MSE_a + MSE_b)/2 - mean((a - b)^2)/4, so the score of an average follows from the components' public
scores and their (label-free) differences; it was predicted at 0.3196 (and an earlier average, 19, at 0.3228 vs 0.3227 actual).

**Fourth blend on other encoders** (`24_avg4_w2v_member.csv`, 0.3234, worse). A 3-model blend on encoders the ensemble
does not use (ridge on Qwen3-4B transcript embeddings, ridge on Qwen2-Audio + w2v-BERT 2.0, SVR on w2v-BERT 2.0), built
with the notebook's pipeline (`03_train.py --crops`, `08_blend.py`; CV RMSE 0.529, Pearson 0.904). It differs from the
ensemble by 0.19 per clip (RMS; the members differ from each other by 0.07 to 0.14), and as a fourth equal-weight member it
improves the out-of-fold metric under speaker-grouped folds (0.3414 to 0.3398, bootstrap P = 0.85) and, in kernel form,
under topic-grouped (-0.0015) and strict-topic folds (-0.0031). The averaging identity put the break-even at 0.3376 for the
blend alone, against a CV-calibrated estimate of 0.329 to 0.342. Public 0.3234: by the identity the new blend alone would
score about 0.354, so on the public clips these encoders do about 0.02 worse than validation predicts. The eighth
validation-better change in a row that is worse publicly.

**WavLM-large layer choice** (no submission). A public solution uses WavLM-large layer 20 alone, where our features average
layers 8-16. Per-layer features (mean and std over time, layers 12-24) show the single upper layers are better on their own:
SVR CV RMSE 0.585 at layer 20 (0.575 at 22) against 0.597, ridge 0.590 against 0.614. In the blends they are not: replacing
the WavLM-large SVR with the layer-20 one makes all three blends worse (ensemble OOF metric 0.3414 to 0.3457, bootstrap
P(better) = 0.02), the layer-20 features also make the combined audio ridge and the Voxtral + WavLM SVR worse, and added as an
extra model the layer-20 SVR gets a blend weight of 0. The middle layers carry the acoustic detail that the Voxtral states
lack; layer 20 overlaps with Voxtral. Our component scores, re-measured the way the public solution reports them (full-clip
validation, the 732 clips scored 1-5, trained without the noise clips), match its numbers (Voxtral ridge 0.538 vs 0.535).

**Clip to [2, 5]** (`25_ens_clip2.csv`, 0.3206, worse). A public solution clips its predictions to [2, 5] (only 4 of the 732
scored training clips are below 2). On our out-of-fold predictions it helps every blend (ensemble metric 0.3414 to 0.3402,
bootstrap P = 1.00), and none of the 12 test clips predicted below 2 looks like a noise recording (gate scores 1.8-2.8, train
noise clips at most 0.41). Public 0.3206 against 0.3195: those slow, hesitant answers are scored low on the public clips too.
A separate blend for the 45.06 s batch (90 train / 87 test clips) also helps in CV, mostly through its own offset (-0.0046),
which the batch-offset probe (13) already rejected; with that offset removed the gain is -0.0012. Not submitted.

**Fine-tuned text model, CEFR classifiers, Gemini grader** (no submission). DeBERTa-v3-large fine-tuned on the
transcripts (test-length crops, no score-0 clips, 5 folds on a free Colab T4): RMSE 0.592 on the 732 scored clips, the
same as the frozen DeBERTa ridge; added to the blends, ensemble OOF metric -0.0016 (bootstrap P = 0.81). A document-level
CEFR classifier (ModernBERT, Apache-2.0) correlates 0.06-0.08 with the ensemble's residuals under topic-grouped folds
(a CC BY-NC model was not used: the rules allow only licences that permit commercial use). Gemini 3.5 Flash Lite grading
the audio zero-shot against the rubric (`src/16_gemini_judge.py`, free tier: 500 requests a day): Pearson 0.60 with the
labels on 38 training clips, 0.74 with the ensemble and 0.05 with its residuals, so it adds nothing the ensemble lacks.

**Fine-tuned text model + batch weights** (`26_ft_batchw.csv`, **0.3175**, new best, 3rd on the public leaderboard).
The three blends each get the fine-tuned DeBERTa-v3-large text model (`06_finetune_text.py --crops --drop-zero`) as an extra
model, and separate non-negative weights for the 45.06 s recording batch (90 train / 87 test clips), re-centred to the global
blend's mean on the training batch so the batch gets no extra offset (a batch offset was rejected by probe 13). Out-of-fold
ensemble metric 0.3414 -> 0.3390 (bootstrap P = 0.86). On the batch the blends drop the WavLM-large SVR and lean on the
text models and the combined audio ridge. Same corrections as before (mean 3.228668, full-length clips x1.15).

**Leaderboard-weighted mix** (`27_lb_weighted_mix.csv`, **0.3158**, new best, 2nd on the public leaderboard). Weighted
average of four scored submissions: 26 (0.650), the Qwen2-Audio member `cand_qwen3.csv` (0.178), `11_blend_v7_biasfix.csv`
(0.122) and 15 (0.050). The weights minimise the public MSE predicted from the files' public scores and their label-free
differences (w'Gw with G_ij = (MSE_i + MSE_j - mean((p_i - p_j)^2)) / 2, weights >= 0 summing to 1); predicted 0.3153.
Unlike every earlier average, these weights are tuned on the public leaderboard, so the private score may gain less.

**Weighted mix with the batch-offset file** (`28_lb_weighted_mix2.csv`, **0.3150**, new best). The same public-score
weighting over more scored files picks up the batch-offset submission (13): 0.624 x 26, 0.200 x 13, 0.095 x the
Qwen2-Audio member, 0.058 x 15, 0.023 x 11; predicted 0.3145. In effect a partial offset for the 45.06 s batch, which on
its own (13) had made the score worse. Tuned on the public clips like 27.

**RoBERTa-large instead of the DeBERTa fine-tune** (`29_roberta_batchw.csv`, 0.3225, worse). RoBERTa-large fine-tuned the
same way (Kaggle 2x T4, `06_finetune_text.py --model roberta-large --crops --drop-zero`): RMSE 0.602 on the 732 scored
clips, correlation 0.93 with the DeBERTa fine-tune. In place of it in each blend: out-of-fold metric 0.3390 to 0.3368
(bootstrap P = 0.80), but public 0.3225 against 0.3175, and it adds nothing to the public-score weighting (weight 0.06).

**Nine seeds of the DeBERTa fine-tune** (`30_ft_seed9.csv`, 0.3196, worse). The fine-tune of 26 averaged over 9 seeds
(`06_finetune_text.py --seed`): RMSE 0.572 on the 732 scored clips against 0.592 for one run; out-of-fold metric of the
main model 0.3390 to 0.3371 (bootstrap P = 0.88), but public 0.3196 against 26's 0.3175, so 26's single run was lucky on
the public clips. It gets weight 0 in the public-score weighting (best mix of all scored files predicted at 0.3139).

**Later experiments, no submission** (12 Oct). None improved the out-of-fold metric beyond the nine-seed fine-tune:
Voxtral-Small-24B in 4-bit (ridge RMSE 0.615 against Voxtral-Mini's 0.601); Phi-4-multimodal audio-LLM states
(`17_phi4mm_embed.py`, ridge 0.668, no blend gain); Phi-4 transcripts for the fine-tune (`--tag phi4`: correlation 0.95
with the Whisper version, blend 0.3379 against 0.3371); Qwen2.5-7B with LoRA as a regressor (`--lora`, two pilot folds no
better than DeBERTa, correlation 0.03 with the blend's residuals); batch weights shrunk toward the global weights (worse
than fully separate weights at every strength).
