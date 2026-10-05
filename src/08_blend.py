"""Step 8: score-0 gate + non-negative blend of out-of-fold predictions -> submission.csv.

* Gate: score-0 clips are acoustically distinct (most contain ordinary speech, so it's probably the
  recording conditions, not the grammar). Frozen WavLM + Ridge predicts
  them almost perfectly, while the text models miss them by 1.5-2 points. So any clip whose
  WavLM prediction is below GATE is set to 0. The text models never get a vote on those clips.
* Blend: the remaining clips get intercept + non-negative weights over the models' predictions.
  Non-negative weights keep correlated models from cancelling each other out (overfitting).
* No rounding: train labels sit on a 0.5 grid and snapping to it cut CV MAE by 0.02, but it made
  the public leaderboard 0.02 *worse* in two paired submissions, so test labels are probably not on
  the grid (e.g. averaged raters). The leaderboard tracks CV MAE (public 0.346 vs CV MAE 0.370).
* Honest score: the blend weights are fitted in a second CV over gfold_s1 (speaker-grouped, like the
  base models' folds; see 10_speaker_folds.py). Fold seeds differ, so there's a small optimism shared
  by every stacking setup.

Usage: python src/08_blend.py s1_lgbm_prompt s2_ridge_deberta s3_ridge_wavlm s4_deberta_ft
       (the gate always uses s3_ridge_wavlm, which must be in the list)
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from common import ART, fold_of

GATE_MODEL, GATE = "s3_ridge_wavlm", 0.5   # train 0-clips: WavLM max 0.40; lowest non-zero clip sits far above


def fit(X, y):
    """Intercept + non-negative weights (centring lets the intercept stay unconstrained)."""
    w, _ = nnls(X - X.mean(0), y - y.mean())
    return y.mean() - X.mean(0) @ w, w


def blend(names):
    P = pd.concat([pd.read_parquet(ART / f"pred_{n}.parquet").set_index(["filename", "split"]).pred.rename(n)
                   for n in names], axis=1)
    ids = pd.read_parquet(ART / f"pred_{GATE_MODEL}.parquet").set_index(["filename", "split"])
    P = P.loc[ids.index]   # train and test reuse filenames, hence the (filename, split) key
    gated = (P[GATE_MODEL] < GATE).values
    is_tr = (ids.reset_index().split == "train").values
    y = ids.label.values
    folds = fold_of(ids.index.get_level_values(0)[is_tr], "gfold_s1")

    # ---- CV of the blend itself (fold_s1) ----
    X, oof = P.values, np.zeros(is_tr.sum())
    Xtr, ytr, gtr = X[is_tr], y[is_tr], gated[is_tr]
    for k in range(5):
        fit_m, va = (folds != k) & ~gtr, folds == k
        b, w = fit(Xtr[fit_m], ytr[fit_m])
        oof[va] = b + Xtr[va] @ w
    oof = np.where(gtr, 0, np.clip(oof, 0, 5))
    print(f"blend CV MAE {np.abs(oof - ytr).mean():.4f} | RMSE {np.sqrt(np.mean((oof - ytr) ** 2)):.4f} | gated train {gtr.sum()} "
          f"(of which label 0: {(ytr[gtr] == 0).sum()}), test {gated[~is_tr].sum()}")
    for band, m in [("0", ytr == 0), ("1.5-3", (ytr > 0) & (ytr <= 3)), ("3.5-5", ytr >= 3.5)]:
        print(f"  band {band}: MAE {np.abs(oof[m] - ytr[m]).mean():.3f} (n={m.sum()})")

    # ---- Final weights on all non-gated train clips -> test ----
    b, w = fit(Xtr[~gtr], ytr[~gtr])
    print("weights:", dict(zip(names, w.round(3))), "intercept", round(b, 3))
    te = np.where(gated[~is_tr], 0, np.clip(b + X[~is_tr] @ w, 0, 5))
    sub = pd.DataFrame({"filename": ids.index.get_level_values(0)[~is_tr], "label": te})
    sub.to_csv(ART / "submission.csv", index=False)


if __name__ == "__main__":
    blend(sys.argv[1:])
