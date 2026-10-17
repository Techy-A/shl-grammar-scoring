"""Step 8: score-0 gate + non-negative blend of out-of-fold predictions -> submission.csv.

* Gate: score-0 clips are acoustically distinct (most contain ordinary speech, so it's probably the
  recording conditions, not the grammar). Frozen WavLM + Ridge predicts
  them almost perfectly, while the text models miss them by 1.5-2 points. So any clip whose
  WavLM prediction is below GATE is set to 0. The text models never get a vote on those clips.
* Blend: the remaining clips get intercept + non-negative weights over the models' predictions.
  Non-negative weights keep correlated models from cancelling each other out (overfitting).
* No rounding: train labels sit on a 0.5 grid and snapping to it cut CV MAE by 0.02, but it made
  the public leaderboard 0.02 *worse* in two paired submissions, so test labels are probably not on
  the grid (e.g. averaged raters). Rounding also worsens CV RMSE (+0.012) while improving CV MAE, and the
  leaderboard is scored with RMSE and Pearson correlation, so RMSE is the number to optimise.
* Honest score: the blend weights are fitted in a second CV over gfold_s1 (speaker-grouped, like the
  base models' folds; see 10_speaker_folds.py). Fold seeds differ, so there's a small optimism shared
  by every stacking setup.

* --batch: the 45.06 s recording batch (12% of the train clips, 40% of the test clips, scored lower) gets its own
  non-negative weights, shifted so that their mean on the training batch equals the global blend's: the batch is
  weighted differently but gets no extra offset (a batch offset made the public score worse).

Output: artifacts/submission.csv and artifacts/oof_blend.parquet (out-of-fold blend predictions for
the train clips, used by the notebook's evaluation and plots).
Usage: python src/08_blend.py [--batch] <model names: artifacts/pred_<name>.parquet> ...
       (the gate always uses pred_s3_ridge_wavlm.parquet, whether or not it's in the list)
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from common import ART, fold_of

GATE_MODEL, GATE = "s3_ridge_wavlm", 0.5   # train 0-clips: WavLM max 0.40; lowest non-zero clip sits far above
BATCH_SECONDS = 45.06   # every clip of the recording batch has this length


def fit(X, y):
    """Intercept + non-negative weights (centring lets the intercept stay unconstrained)."""
    w, _ = nnls(X - X.mean(0), y - y.mean())
    return y.mean() - X.mean(0) @ w, w


def blend(names, batch_weights=False):
    P = pd.concat([pd.read_parquet(ART / f"pred_{n}.parquet").set_index(["filename", "split"]).pred.rename(n)
                   for n in names], axis=1)
    ids = pd.read_parquet(ART / f"pred_{GATE_MODEL}.parquet").set_index(["filename", "split"])
    P = P.loc[ids.index]   # train and test reuse filenames, hence the (filename, split) key
    gated = (ids.pred < GATE).values
    is_tr = (ids.reset_index().split == "train").values
    y = ids.label.values
    folds = fold_of(ids.index.get_level_values(0)[is_tr], "gfold_s1")
    batch = np.zeros(len(ids), bool)
    if batch_weights:
        dur = pd.read_parquet(ART / "feat_text_prompt.parquet").set_index(["filename", "split"]).duration
        batch = np.abs(dur.reindex(ids.index).values - BATCH_SECONDS) < 0.05
    X = P.values
    Xtr, ytr, gtr, btr = X[is_tr], y[is_tr], gated[is_tr], batch[is_tr]

    def predict(rows, Xp, bp):
        """Blend fitted on the train rows `rows`, applied to Xp (bp: which of them are batch clips)."""
        b, w = fit(Xtr[rows], ytr[rows])
        out, extra = b + Xp @ w, None
        if batch_weights:
            m = rows & btr
            bb, wb = fit(Xtr[m], ytr[m])
            shift = (bb + Xtr[m] @ wb).mean() - (b + Xtr[m] @ w).mean()   # the batch's own offset, removed
            out, extra = np.where(bp, bb + Xp @ wb - shift, out), wb
        return out, b, w, extra

    # ---- CV of the blend itself (fold_s1) ----
    oof = np.zeros(is_tr.sum())
    for k in range(5):
        fit_m, va = (folds != k) & ~gtr, folds == k
        oof[va] = predict(fit_m, Xtr[va], btr[va])[0]
    oof = np.where(gtr, 0, np.clip(oof, 0, 5))
    print(f"blend CV RMSE {np.sqrt(np.mean((oof - ytr) ** 2)):.4f} | Pearson {np.corrcoef(oof, ytr)[0, 1]:.4f} | "
          f"MAE {np.abs(oof - ytr).mean():.4f} | gated train {gtr.sum()} "
          f"(of which label 0: {(ytr[gtr] == 0).sum()}), test {gated[~is_tr].sum()}")
    for band, m in [("0", ytr == 0), ("1.5-3", (ytr > 0) & (ytr <= 3)), ("3.5-5", ytr >= 3.5)]:
        print(f"  band {band}: RMSE {np.sqrt(np.mean((oof[m] - ytr[m]) ** 2)):.3f} (n={m.sum()})")
    pd.DataFrame({"filename": ids.index.get_level_values(0)[is_tr], "label": ytr, "oof": oof}).to_parquet(
        ART / "oof_blend.parquet", index=False)

    # ---- Final weights on all non-gated train clips -> test ----
    pred, b, w, wb = predict(~gtr, X[~is_tr], batch[~is_tr])
    print("weights:", dict(zip(names, w.round(3))), "intercept", round(b, 3))
    if batch_weights:
        print(f"batch weights ({btr.sum()} train / {batch[~is_tr].sum()} test clips):", dict(zip(names, wb.round(3))))
    te = np.where(gated[~is_tr], 0, np.clip(pred, 0, 5))
    sub = pd.DataFrame({"filename": ids.index.get_level_values(0)[~is_tr], "label": te})
    sub.to_csv(ART / "submission.csv", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    blend([a for a in args if a != "--batch"], batch_weights="--batch" in args)
