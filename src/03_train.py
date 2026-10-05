"""Step 3: cross-validated models (mean baseline, LightGBM, Ridge) on cached feature tables.

* Same folds for every model (artifacts/folds.csv: 5 folds x 3 seeds, stratified by label),
  so out-of-fold (OOF) predictions from different steps can be blended later without leakage.
* Each run appends one line to artifacts/results.csv (the results table for the report) and
  saves OOF + test predictions to artifacts/pred_<name>.parquet for the blend (08_blend.py).

Usage:
  python src/03_train.py --name s0_mean --model mean --feats feat_text_prompt
  python src/03_train.py --name s1_lgbm --model lgbm --feats feat_text_prompt
  python src/03_train.py --name s2_ridge --model ridge --feats feat_emb_deberta-v3-large_prompt
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import ART

SEEDS, N_FOLDS = [0, 1, 2], 5
ID_COLS = ["filename", "split", "label"]


def get_folds(train):
    """Create the shared fold table once, then always reuse it."""
    path = ART / "folds.csv"
    if path.exists():
        return pd.read_csv(path)
    folds = pd.DataFrame({"filename": train.filename})
    # Stratify on the label (as a string, so 2.5 and 3.0 are separate classes); the very rare
    # 1.0/1.5 labels are merged into 2.0 so every class has at least N_FOLDS members.
    strat = train.label.clip(lower=2.0).where(train.label > 0, 0).astype(str)
    for s in SEEDS:
        folds[f"fold_s{s}"] = -1
        for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(train, strat)):
            folds.loc[va, f"fold_s{s}"] = k
    folds.to_csv(path, index=False)
    return folds


def fit_predict(model, X_tr, y_tr, X_ev, seed):
    """Train one model and predict for each matrix in X_ev."""
    if model == "mean":
        return [np.full(len(X), y_tr.mean()) for X in X_ev]
    if model == "ridge":
        # For wide embedding tables (~1000 dims vs 769 rows): standardise, then pick the L2
        # strength by efficient leave-one-out CV inside the training fold only.
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 21)))
        m.fit(X_tr, y_tr)
        return [m.predict(X) for X in X_ev]
    if model == "svr":
        # Epsilon-insensitive loss is closer to the leaderboard's MAE than Ridge's squared error,
        # and the RBF kernel adds some non-linearity. Beat Ridge on every embedding table tried
        # (WavLM: MAE 0.442 -> 0.418 on non-zero clips); C >= 10 gave the same result.
        m = make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1))
        m.fit(X_tr, y_tr)
        return [m.predict(X) for X in X_ev]
    # Small data (769 rows): shallow trees, strong subsampling, many slow steps.
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=15,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                          random_state=seed, verbose=-1)
    m.fit(X_tr, y_tr)
    return [m.predict(X) for X in X_ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, used for output files and the results table")
    ap.add_argument("--model", choices=["mean", "lgbm", "ridge", "svr"], required=True)
    ap.add_argument("--feats", nargs="+", required=True, help="artifacts/<name>.parquet tables, merged on filename")
    args = ap.parse_args()

    # ---- Load and merge feature tables (ids/label taken from the first one) ----
    df = pd.read_parquet(ART / f"{args.feats[0]}.parquet")
    for f in args.feats[1:]:
        # Train and test reuse filenames, so the key must include the split.
        df = df.merge(pd.read_parquet(ART / f"{f}.parquet").drop(columns=["label"], errors="ignore"),
                      on=["filename", "split"])
    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    folds = get_folds(train).set_index("filename").loc[train.filename].reset_index()

    # ---- CV: average OOF over seeds; test prediction = mean over all fold models ----
    y = train.label.values
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for s in SEEDS:
        for k in range(N_FOLDS):
            tr_idx = folds[f"fold_s{s}"].values != k
            va_idx = ~tr_idx
            p_va, p_te = fit_predict(args.model, train.loc[tr_idx, feat_cols], y[tr_idx],
                                     [train.loc[va_idx, feat_cols], test[feat_cols]], seed=s)
            oof[va_idx] += p_va / len(SEEDS)
            test_pred += p_te / (len(SEEDS) * N_FOLDS)

    # Scores are bounded to [0, 5], so clipping can only reduce the error.
    oof, test_pred = np.clip(oof, 0, 5), np.clip(test_pred, 0, 5)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    r = float(np.corrcoef(oof, y)[0, 1]) if oof.std() > 0 else float("nan")
    print(f"{args.name}: CV RMSE {rmse:.4f} | MAE {np.abs(oof - y).mean():.4f} | Pearson {r:.4f} | {len(feat_cols)} features")

    # ---- Save predictions + append to the results table ----
    pd.concat([pd.DataFrame({"filename": train.filename, "split": "train", "label": y, "pred": oof}),
               pd.DataFrame({"filename": test.filename, "split": "test", "label": np.nan, "pred": test_pred})]
              ).to_parquet(ART / f"pred_{args.name}.parquet", index=False)
    row = pd.DataFrame([dict(name=args.name, model=args.model, feats=" ".join(args.feats),
                             n_feats=len(feat_cols), cv_rmse=round(rmse, 4), cv_pearson=round(r, 4))])
    res = ART / "results.csv"
    row.to_csv(res, mode="a", header=not res.exists(), index=False)


if __name__ == "__main__":
    main()
