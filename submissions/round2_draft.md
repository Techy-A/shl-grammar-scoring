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
  0.3379 (s1) across r0/r3/r6, a spread of 0.002, larger than any effect below.
- **Arm 2a: distribution head** (`--out bins`: softmax over the 9 half-point labels 1-5, MSE on the expected value,
  biases at the label log-frequencies). Head alone (metric on the scored clips) 0.4155/0.4041/0.4136 vs baseline
  0.4121/0.4150/0.4110. Main model: r0 −0.0006/+0.0006/+0.0000 (P 0.65/0.36/0.50), r3 −0.0051/−0.0040/−0.0046
  (P 0.99/0.97/0.99), r6 +0.0012/+0.0010/+0.0005 (P 0.25/0.28/0.37); pooled 9 seeds vs 9 seeds −0.0016/−0.0008/−0.0011
  (P 0.83/0.68/0.75). BINS_R9 **Fails**: one lucky retrain at P 0.99 and two at noise level, exactly the trap the
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
