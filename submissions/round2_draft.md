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

ARM_RESULTS
