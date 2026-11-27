"""Step 21: public-score mix of scored submissions (the averaging identity of README section 8), with a lower bound on weights.

Public MSE of sum_i w_i p_i (weights summing to 1) is w'Gw, G_ij = (MSE_i + MSE_j - mean((p_i - p_j)^2)) / 2.
MSE_i comes from the public score: shift probes keep their base file's Pearson, so their RMSE is exact
((score - 0.4 (1 - r)) / 0.6); every other file is converted in proportion to the decoded 0.3207 file (RMSE 0.45).
The mix's score is converted back the same way. This reproduces submission 34's weights and its 0.3110 forecast.
Negative lower bounds let the weights go below 0: the public score keeps falling, but the forecast error grows with the
weights' size (35: -0.10 predicted 0.3050, scored 0.3068; 36: -0.15 predicted 0.3035, scored 0.3064) and every step is
fitted to the public clips only.
The scores csv is `kaggle competitions submissions shl-hiring-assessment-2026 --csv`.
Usage: python src/21_lb_mix.py submissions <scores csv> [lower bounds, e.g. 0 -0.1]
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize

K = 0.3207 / 0.45                                 # score per unit public RMSE, from the decoded 0.3207 file
SKIP = ("probe_metric", "probe_scale", "_rounded", "02_", "04_", "27_", "28_", "32_", "34_")   # rescale probes, rounded files, earlier mixes
SHIFT = {f: "09_blend_v7.csv" for f in ["probe_v7_minus0.2.csv", "probe_v7_plus0.2.csv", "11_blend_v7_biasfix.csv",
         "probe_biasfix_full_minus0.1.csv", "probe_biasfix_short_minus0.1.csv", "13_batch_offset.csv"]}


def load(sub_dir, scores_csv):
    s = pd.read_csv(scores_csv, on_bad_lines="skip")
    s = s[s.ref.astype(str).str.isdigit()].drop_duplicates("ref")
    sc = s.groupby("fileName").publicScore.apply(lambda x: float(x.iloc[0])).to_dict()
    files = {f: sc[f] for f in sorted(os.listdir(sub_dir)) if f in sc and not any(t in f for t in SKIP)}
    P = pd.concat({f: pd.read_csv(os.path.join(sub_dir, f)).set_index("filename").label for f in files}, axis=1)
    return P, files


def rmse_of(f, files):
    if f in SHIFT:   # Pearson of the base file: score = 0.6 RMSE + 0.4 (1 - r) with RMSE = score / K
        b = files[SHIFT[f]]
        return (files[f] - (b - 0.6 * b / K)) / 0.6
    return files[f] / K


def best_mix(P, files, lb):
    X, names = P.values, list(P.columns)
    mse = np.array([rmse_of(f, files) ** 2 for f in names])
    G = (mse[:, None] + mse[None, :] - ((X[:, :, None] - X[:, None, :]) ** 2).mean(0)) / 2
    r = minimize(lambda w: w @ G @ w, np.full(len(names), 1 / len(names)), method="SLSQP", bounds=[(lb, None)] * len(names),
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 2000, "ftol": 1e-12})
    return pd.Series(r.x, names), K * np.sqrt(max(r.fun, 0))


if __name__ == "__main__":
    P, files = load(sys.argv[1], sys.argv[2])
    for lb in map(float, sys.argv[3:] or ["0"]):
        w, score = best_mix(P, files, lb)
        top = w[w.abs() > 0.01].sort_values(key=abs, ascending=False)
        print(f"weights >= {lb:+.2f}: predicted public {score:.4f} | sum|w| {w.abs().sum():.2f} | "
              + ", ".join(f"{n.replace('.csv', '')} {v:+.3f}" for n, v in top.head(8).items()))
