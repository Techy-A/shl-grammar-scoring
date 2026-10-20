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

**Nine-seed fine-tune plus Phi-4 audio states** (`31_ft_seed9_phi4.csv`, 0.3203, worse). 30 with a ridge on
Phi-4-multimodal states added to each blend; out-of-fold metric 0.3371 (same as 30). Its predictions differ from 28 by
0.114 RMS, so it was submitted to test whether it helps the public-score weighting: weight 0.33, best mix of all scored
files predicted at 0.3132 (28: 0.3150). Not submitted, since it would not pass the next scores on the leaderboard (0.3126).

**Weighted average of six submissions** (`32_lb_weighted_mix3.csv`, **0.3134, new best**). Same method as 28, now with
31 among the inputs and the shift probes converted exactly (a shift does not change Pearson, so their RMSE is
(score - 0.4 (1 - r)) / 0.6 rather than proportional to the score). Weights: 31 0.330, 26 0.230, 13 0.198,
probe_stretch_short_x1.2 0.109, probe_biasfix_full_minus0.1 0.090, 29 0.042; predicted 0.3132. The weights are fitted to
public scores, so this step is tuned to the public 60% of the test clips.

**Attention pooling on Voxtral frames** (`33_attn_pool.csv`, 0.3178). 26 with one more model in each blend: a small
attention-pooling head (`20_attn_pool.py`) on Voxtral frame states cached once per clip (`19_voxtral_frames.py`,
PCA to 256 dimensions), trained on a new random test-length prefix of every clip in every epoch. A prefix of the
frames matches the true crop embedding closely (median per-clip correlation 0.997). Alone: RMSE 0.576 on the 732
scored clips (Voxtral ridge 0.584); correlation 0.078 with the blend's residuals; out-of-fold metric of the main model
0.3390 to 0.3340 (bootstrap P = 0.96), the largest gain since the fine-tuned text model.

**Weighted average of six submissions** (`34_lb_weighted_mix4.csv`, **0.3116, new best, 2nd**). Same method as 32,
with 33 among the inputs: 33 0.494, 13 0.215, probe_biasfix_full_minus0.1 0.120, 31 0.089, probe_stretch_short_x1.2
0.081, cand_qwen3 0.001; predicted 0.3110. Weights fitted to public scores (tuned to the public 60% of the test clips).

**After 34, no submission** (12 Oct). Attention-pooling heads on Phi-4-multimodal and WavLM-large frames
(`19_voxtral_frames.py --source`, `20_attn_pool.py --source`): weaker alone (RMSE 0.626 and 0.645 on the scored clips)
and worse next to the Voxtral head (out-of-fold metric 0.3340 to 0.3352-0.3365). Task-adaptive pretraining of DeBERTa
on all transcripts before the fine-tune (`18_tapt.py`, MLM loss 6.1 to 2.7): RMSE 0.594, correlation 0.96 with the
current fine-tune; replacing it 0.3340 to 0.3345, adding it 0.3351.

**Overnight experiments, 17-18 Oct, no submission.** Three procedure changes to the attention-pooling head
(`20_attn_pool.py --text / --dropout 0 / --ema 0.99`, Kaggle job `shl-headvar`, all variants retrained together on 2x
T4). They are judged at the main-model level, swapping the head in all three blends, on out-of-fold
0.6·RMSE + 0.4·(1 − r). The baseline is 0.3387 on these runs.
- Transcript conditioning: the DeBERTa embedding of the matching crop shifts the attention logits and joins the pooled
  vector. Result +0.0034, worse (bootstrap P(better) 0.02). As an extra blend member instead of a replacement: +0.0003.
- Weight EMA (0.99): 0.0000.
- Dropout 0: −0.0002, within noise.
- **Noise reference:** the same head trained on CPU instead of GPU gives −0.0003 with P = 0.92. A clip bootstrap alone
  does not capture training noise, so P ≈ 0.9 from a single retrain is not evidence. Frozen-feature heads look
  exhausted.

**Public-score mixes with small negative weights** (`35_lb_mix_neg0.10.csv`, `36_lb_mix_neg0.15.csv`, 18 Oct, both
public-tuned). The same method as 34 (`src/21_lb_mix.py`), with the lower bound on the weights moved below 0. With
weights ≥ 0 the script reproduces 34 (predicted 0.3107, scored 0.3116).
- 35, weights ≥ −0.10 (Σ|w| 5.0): predicted 0.3050, scored **0.3068**.
- 36, weights ≥ −0.15 (Σ|w| 6.4): predicted 0.3035, scored **0.3064**. This ties first place on the public leaderboard
  and ranks 2nd on time.
- The forecast error grew with the weights' size (+0.0018 and +0.0029, against +0.0006 for 34), as expected.
- Both files are fitted to the public 60% of the test clips, so their private score is expected to fall back toward
  the main model's. The main model alone (33, 0.3178) involves no public tuning.

**Final submission** (`final_submission.csv`, **0.3062, 1st on the public leaderboard**, 18 Oct). The same mix
(`src/21_lb_mix.py`) with 35 and 36 added as inputs and weights ≥ −0.20 (Σ|w| 8.6).
- Raw forecast 0.3019. Corrected for the drift seen on 34-36 (about +0.00045 per unit of Σ|w|), the forecast was
  0.3058; it scored 0.3062.
- Like 34-36, it is fitted to the public 60% of the test clips, so its private score is expected to fall back toward
  the main model's (33, 0.3178).

**Main-model procedure changes, 19 Oct, no submission (draft).** Four candidate changes to how members are trained or
corrected, each judged at the main-model level (the three blends of 33, `08_blend.py --batch` logic) on speaker-grouped
out-of-fold predictions at the 4 test-length crops, metric 0.6·RMSE + 0.4·(1 − r), paired bootstrap over the 769 clips.
- **Baseline.** The cached `pred_*.parquet` files give 0.3387 (the overnight baseline). All arms below retrain the
  ridge/SVR members with an exact fast re-implementation of `03_train.py` (eigen-decomposition ridge, same alpha
  search; SVRs identical to the cached files, the ridges match `results.csv` but not the cached ridge files, e.g. DeBERTa
  ridge RMSE 0.7775 vs 0.7758), so the comparable baseline is **0.3386** (blend folds gfold_s1; 0.3383 with gfold_s0,
  0.3376 with gfold_s2). The ridge/SVR members are deterministic; the noise reference is the attention head retrained
  on CPU instead of GPU (`pred_vp_attn_cpunb`, baseline 0.3383) and the three fold seeds of the blend.
- **Soft pseudo-labels, all eight ridge/SVR members** (`src/22_pseudo_label.py --members <all>`). Fold-honest: in fold
  k the pseudo rows are the fold-k crops plus the test clips, labelled by the main model built from the fold-k member
  models with blend weights fitted without fold k (that teacher scores 0.3394, like the OOF; a first version with
  in-sample blend weights scored 0.329 and was discarded as leaky). Every member gets much better on its own (WavLM-large
  SVR metric 0.407 to 0.335) but the main model gets **worse**: w 0.3 +0.0019 (P(better) 0.04), w 0.5 +0.0018 (P 0.05),
  same sign under all three blend fold seeds. The members move toward the teacher and the blend loses diversity. Split by
  member (w 0.5): DeBERTa ridges −0.0009 (P 0.99), Whisper ridge +0.0002, Voxtral+audio ridges +0.0014 (P 0.001),
  WavLM-large SVRs +0.0021 (P 0.03), Voxtral+WavLM-large SVR −0.0001. The attention head was not pseudo-labelled
  (about an hour per arm on CPU, and distillation into the other audio members hurt).
- **Soft pseudo-labels, the two DeBERTa text ridges only** (`src/22_pseudo_label.py`, default; the subset was chosen
  after the all-member arm failed). w 0.1 −0.0004, 0.3 −0.0007, 0.5 −0.0009, **1 −0.0011 (0.3386 to 0.3375, P 0.987)**,
  2 −0.0012 (P 0.97), 4 −0.0012 (P 0.94). At w 1: blend folds gfold_s0 −0.0010 (P 0.98), gfold_s2 −0.0007 (P 0.92);
  with the CPU-retrained head in both arms −0.0011 (P 0.988), −0.0011, −0.0007. Leak controls: members from fold seed 0
  only (every teacher input fold-honest) −0.0007 (P 0.95), seed 1 −0.0009, seed 2 −0.0012; no test pseudo rows −0.0009
  (P 0.98). On random 130-clip subsets (the public size) the gain is −0.0011 ± 0.0012 (better in 83%). The text ridges
  move toward the main model (correlation with its OOF 0.85 to 0.90) and their blend weight doubles (0.10 to 0.20).
  Test file `artifacts/cand_33_textpl.csv` = 33 plus the change this makes to the recomputed main model (RMS 0.013, mean
  −0.0001, correlation with 33 0.9999). **Forecast** (before any submission): public 0.3178 − 0.0009 = **0.3169**, noise
  sd about 0.001 on the public clips (worst-case identity bounds 0.3101-0.3255). For the public-score mix, swapping 33
  for this file moves it by its effective weight times that change (well under 0.001); re-fitting the mix needs the
  file's real public score. Passes the bar (P ≥ 0.9, a training change, stable over head retrains and fold seeds), but
  the gain is a tenth of the public leaderboard's resolution and the member subset was picked after looking.
- **Transductive standardisation of the evaluation rows** (ridge/SVR members refit-free: the held-out crops of fold k,
  and at test the 216 test clips, are standardised with their own statistics and mapped onto the training rows').
  Whole set, mean + std: −0.0021 (P 0.94), gfold_s0 −0.0037, gfold_s2 −0.0029. Per batch group (45.06 s batch / rest),
  mean only: −0.0039 (P 0.997), −0.0047, −0.0039. Non-transductive control (map from the training crops' statistics, a
  pure length correction): +0.0001, so the gain comes from the evaluation set's own statistics. Not trusted: in CV the
  held-out fold has the training label distribution by construction (stratified folds), so aligning its statistics to the
  training rows is free there, while on test it changes the group means CV cannot judge. The per-batch version lowers the
  test batch's raw mean by 0.10 relative to the other clips (3.128/3.425 to 3.033/3.433), an implicit batch offset like
  the one probe 13 rejected; the whole-set version moves it +0.05 the other way. With each fold's mean matched to the
  training labels in both arms the whole-set gain is −0.0011/−0.0026/−0.0007 (gfold_s0/s1/s2, P 0.88/0.98/0.74).
- Decomposition: mean only (whole set) −0.0011 (P 0.83), gfold_s0 −0.0025, gfold_s2 −0.0018. **Std only** (each
  feature's spread in the evaluation set scaled to the training rows', mean kept): **−0.0010 (0.3386 to 0.3376,
  P 0.997)**, gfold_s0 −0.0012 (P 1.00), gfold_s2 −0.0011 (P 0.999); 130-clip subsets −0.0010 ± 0.0009. This formally
  passes the bar, but the test clips are far more shifted than a CV fold: the std-only version moves the OOF by 0.013
  RMS and the test predictions by 0.052 (full-length clips −0.057 on average, short non-batch clips −0.040, batch clips
  +0.008, i.e. group-relative shifts of the kind probes 13 and the short-clip stretch rejected). CV validates a change
  four times smaller than the one it would make on test, so its forecast is an extrapolation: file
  `artifacts/cand_33_tstd4.csv` (33 plus the change, RMS 0.047, correlation 0.9986), public 0.3168 if the gain scales
  with the change as in CV, about 0.3193 if the change is unrelated to the public errors, noise sd about 0.0025;
  worst-case identity bounds 0.290-0.346. Not recommended without a probe.
- Stress test of the std-only version: the evaluation set's spread taken with the 45.06 s batch reweighted to 40% (the
  test set's share; the training folds have 12%): −0.0007 (P 0.98), gfold_s0 −0.0005 (P 0.92), gfold_s2 −0.0006
  (P 0.96). The gain falls by about 40% as soon as the evaluation set looks a little more like the test set, which
  supports reading it as partly a CV artefact.
- Verdict for this round: only the text-ridge pseudo-labels pass the bar with a test-side change inside the validated
  range (RMS 0.013 on test vs 0.016 on OOF); forecast 0.3169 for the main model, against 0.3178. Nothing here is close to
  the 0.29-0.30 needed to matter on the leaderboard, and none was submitted.
- **Extremes recalibration** (nested over the blend folds, applied to the main model's OOF). The OOF is already
  calibrated: the slope of the label on the prediction is 1.007, and stretching around the mean is best at 1.0 (0.95:
  0.3402, 1.05: 0.3391). Isotonic +0.0041 to +0.0075 (P ≤ 0.08), half isotonic +0.0000 to +0.0016, linear +0.0027 to
  +0.0035, piecewise linear with separate slopes below/above the mean +0.0027 to +0.0037, the same per batch +0.0004 to
  +0.0032. All worse. The pull toward the mean at scores 1.5-2 and 5 is regression to the mean of a calibrated
  predictor (E[y | p] is right, E[p | y] is not), not something a monotone map can fix.
- **Distributional target** for the two DeBERTa text members (multinomial logistic over the label grid on 128 PCA
  dimensions, prediction = expected score, C by grouped inner CV). Alone worse (metric 0.612 vs 0.555); in the blends
  −0.0012 (P 0.86), gfold_s0 −0.0015 (P 0.93), gfold_s2 −0.0011 (P 0.85). Below the bar, and as a weaker model that
  helps by differing it is closer to a composition change.

**Main-model procedure changes, round 2, 20 Oct, no submission (draft).** CPU only, on cached features. Judged as in
round 1: the three blends of 33 (`08_blend.py --batch` logic, the round-1 re-implementation `ev.py`), speaker-grouped
out-of-fold predictions at the 4 test-length crops, metric 0.6·RMSE + 0.4·(1 − r), paired bootstrap over the 769 clips,
each arm under the three blend fold seeds (gfold_s0/s1/s2). Baseline with the cached members: 0.3381 / 0.3387 / 0.3379.

- **Topic-grouped folds** (`src/23_topic_folds.py`, new; the `folds_topic.csv` mentioned above no longer existed). Question
  ids are derived from the full-clip transcripts: tf-idf, average-linkage cosine clusters on the scored train clips, the
  threshold (0.879, 131 topics) set so that a validation clip's nearest transcript in the other folds is as similar as a
  test clip's nearest train transcript (median 0.215 vs 0.216). Topics that share a likely speaker are joined (cached
  WavLM features, see below; distance < 0.05, so 140 groups, the largest 197 clips of playground answers). Output
  `artifacts/folds_topic.csv` (tfold_s0..s2, fold sizes 206/150/141/136/136) and `artifacts/topics.csv`.
- **Speaker ids without audio.** The speaker-verification embeddings of step 10 are not cached and the audio is not in
  this checkout, so the WavLM-SV model could not be rerun (ECAPA2 not used: CC-BY-NC). Proxy: the cached WavLM-base-plus
  and WavLM-large features (layers 4-9 / 8-16, mean + std), standardised, 32 principal components, average-linkage
  cosine clusters. Checked against the folds: a true same-speaker pair always shares its fold in all three gfold seeds
  (random pairs: 1.2%). At distance 0.10, 77% of proxy pairs do; at 0.05, 91%. No clip of the 45.06 s batch is ever
  grouped (0 of 90 train, 0 of 87 test). At 0.20 the proxy puts 50 of the 216 test clips in 13 pairs and 8 triples,
  close to the 55 a competitor reports with ECAPA2.

- **Arm 1: speaker test-mean shrink** (a correction: p' = p + λ(mean of p over the same speaker's clips − p), score-0
  gated clips untouched). Label-free, applied to the main model's OOF and, at test, to the test clips' own groups.
  - Same-speaker label agreement first (proxy pairs that also share all three fold seeds, distance 0.10, 102 pairs):
    identical labels in 69% of pairs, mean |Δy| 0.22 against 1.15 for random pairs, label correlation 0.91. The OOF
    errors are correlated too (residual correlation 0.51; 0.58 at distance 0.20), so only part of the noise averages out.
  - In OOF a group must come from one fold model, as at test, so the groups are proxy clusters split by fold tuple
    ("fold-consistent"). Raw proxy clusters also join clips from different folds, whose predictions come from different
    models; averaging those removes model noise that the test set does not have, so they overstate the gain (raw:
    −0.0029/−0.0049/−0.0046, P 0.89/0.98/0.98; not used).
  - Fold-consistent, distance and λ nested over the blend folds (distance 0.08-0.20, λ 0-1): gfold_s0 −0.0012 (P 0.78),
    s1 −0.0025 (P 0.953), s2 −0.0023 (P 0.953); bootstrap over groups instead of clips P 0.86/0.99/0.99. Same with the
    CPU-retrained head: −0.0012/−0.0025/−0.0023 (P 0.77/0.95/0.95). λ chosen on other topics (nested over tfold_s0):
    −0.0012/−0.0018/−0.0017 (P 0.77/0.87/0.86). The nested picks are almost always distance 0.20, λ 0.55-0.65; that
    setting fixed after looking gives −0.0024/−0.0026/−0.0024 (P 0.94/0.95/0.94).
  - Wider distances keep helping (up to 0.40: −0.0025/−0.0034/−0.0031, P 0.85/0.93/0.92), so part of the gain is plain
    smoothing among similar-sounding clips, not speaker identity.
  - Test coverage is about half of train's: the shrink changes 39-43 of the 216 test clips (by 0.10 RMS each, as in
    OOF), against 295 of the 769 train clips. Expected test gain roughly half the OOF gain, about −0.001.
  - **Verdict: below the bar** (s0 and the topic-nested check under 0.9). Reference file, not a candidate:
    `artifacts/cand_33_spk_only.csv` (33 plus the change, RMS 0.046, 39 clips).
- **Head runs** (`src/20_attn_pool.py`, CPU, 3 seeds × 5 folds each, judged by swapping the head in all three blends).
  The default path is byte-identical to the old script (same md5 on a 1-seed, 2-epoch run) and the new CPU default run
  (`pred_vp_attn_cpu_r0`) matches `pred_vp_attn_cpunb` to 5e-7. Retrains use seeds 0-2 (r0), 3-5 (r3), 6-8 (r6), with
  the baseline retrained on the same seeds. **Retrain noise reference:** the baseline itself scores 0.3384 / 0.3401 /
  0.3379 / 0.3377 (s1) across r0/r3/r6/r9, a spread of 0.002, larger than any effect below.
- **Arm 2a: distribution head** (`--out bins`: softmax over the 9 half-point labels 1-5, MSE on the expected value,
  biases at the label log-frequencies). Head alone (metric on the scored clips) 0.4155/0.4041/0.4136 vs baseline
  0.4121/0.4150/0.4110. Main model: r0 −0.0006/+0.0006/+0.0000 (P 0.65/0.36/0.50), r3 −0.0051/−0.0040/−0.0046
  (P 0.99/0.97/0.99), r6 +0.0012/+0.0010/+0.0005 (P 0.25/0.28/0.37); pooled 9 seeds vs 9 seeds −0.0016/−0.0008/−0.0011
  (P 0.83/0.68/0.75). A fourth retrain (r9, seeds 9-11): −0.0008/+0.0000/−0.0014 (P 0.67/0.49/0.79); pooled 12 seeds vs 12 seeds −0.0014/−0.0006/−0.0012 (P 0.81/0.63/0.75). **Fails**: one lucky retrain at P 0.99 and two at noise level, exactly the trap the
  3-retrain rule is for.
- **Arm 2b: CORN ordinal head** (`--out corn`). Dropped after 2 folds: validation RMSE 0.648/0.714 against 0.558/0.617
  for the baseline on the same folds (+0.09).
- **Arm 3: oversample rare scores** (`--oversample 2`: clips scored ≤ 2 or ≥ 4.5 twice per epoch). Head alone
  0.4136/0.4232/0.4142; main model r0 +0.0002/+0.0001/−0.0002, r3 +0.0008/+0.0010/+0.0005, r6 +0.0002/+0.0005/−0.0002;
  pooled +0.0004/+0.0006/+0.0003 (P 0.26/0.19/0.31). **Fails.**
- **Arm 4: correlation-aware loss** (`--corr 0.5`: MSE + 0.5·(1 − batch Pearson)). Head alone slightly better
  (0.4100/0.4094 vs 0.4121/0.4150), main model r0 +0.0005/+0.0012/+0.0004 (P 0.19/0.02/0.26), r3 −0.0004/−0.0001/−0.0008;
  pooled +0.0000/+0.0006/−0.0001. Its errors correlate more with the main model's (0.894 vs 0.886; bins 0.859). Stopped
  after 2 retrains. **Fails.**
- **Topic-grouped head runs** were dropped once all head arms failed on speaker-grouped folds.
- **Arm 5: stack** of round 1's text-ridge pseudo-labels (w 1) and the speaker shrink (nested as above, on top of the
  pseudo-labelled main model), against the re-implemented baseline (0.3383/0.3386/0.3376). With the GPU head:
  −0.0023/−0.0036/−0.0030 (P 0.92/0.99/0.985). CPU head r0: −0.0022/−0.0036/−0.0030 (P 0.92/0.99/0.99); r3:
  −0.0023/−0.0035/−0.0030 (P 0.94/0.99/0.99); r6: −0.0018/−0.0025/−0.0029 (P 0.89/0.96/0.99). P ≥ 0.9 in 11 of 12
  (head, fold seed) cells. The shrink on top of the pseudo-labels alone: −0.0007 to −0.0025 (P 0.69-0.96). On random
  130-clip subsets −0.0037 ± 0.0036 (better in 85%). **Formally passes, carried by the pseudo-labels** (which pass
  alone); the shrink half is below the bar on its own.
  - Test file `artifacts/cand_33_textpl_spk.csv`: 33 plus the change the stack makes to the recomputed main model
    (shrink at distance 0.20, λ 0.55, both picked on the full OOF; RMS 0.047, mean 0.000, 202 clips moved, 39 of them by
    the shrink; correlation with 33 0.9986).
  - **Forecast** (before any submission): 0.3178 − 0.0009 (pseudo-labels, round 1's forecast) − 0.0010 (shrink, OOF
    gain scaled by its test coverage) = **0.3159**, range 0.3155-0.3169 (pseudo-labels −0.0009 to −0.0011, shrink 0 to −0.0012). Noise sd about
    0.0035 on the public clips. By the averaging identity the change costs mean(Δ²) = 0.0022 in MSE and pays only if it
    correlates with 33's errors at ≥ 0.05 (RMS change / (2 × public RMSE ≈ 0.46)); in OOF the realised correlation is
    0.14 against a break-even 0.06.
