"""Step 22: soft pseudo-labels from the main model for the frozen-feature ridge/SVR members, validated honestly.

Why: the test clips answer new questions and 40% come from the hard 45.06 s batch. Refitting a member on train + test
with the main model's test predictions as soft targets is the usual transductive fix. To validate it, every CV fold
does the same thing to its held-out clips: in fold k (speaker-grouped, fold seeds 0-2 as in 03_train.py) the pseudo
rows are the 4 test-length crops of the fold-k clips plus the 216 test clips (weight 4w each, so a pseudo clip weighs
4w against 5 rows for a labelled clip), and their targets come from a teacher that never saw fold k's labels:
the main model's three blends (08_blend.py --batch) applied to the members' fold-k models, with blend weights fitted
without fold k. Only fold k's real labels score the result. The ridge strength stays at the value picked without
pseudo rows; the score-0 gate's clips get target 0, and members trained without score-0 clips skip them.

Two documented gaps: the fine-tuned text model and the attention head enter the teacher through their saved
out-of-fold and test predictions (their fold models are not kept). Both use fold seed 0, so seed 0 is fully honest
for the fold-k clips; their test predictions saw every label. Removing the test pseudo rows, or scoring seed 0 only,
leaves the gain of the text ridges about the same (submissions/README.md, 19 Oct).

Result (19 Oct): pseudo-labelling all eight ridge/SVR members makes the main model worse (each member gets closer to
the teacher and the blend loses diversity); pseudo-labelling only the two DeBERTa text ridges helps a little.

The ridge solver is an exact eigen-decomposition version of 03_train.py's StandardScaler + Ridge with the grouped
alpha search (same alphas, same GroupKFold, same MAE choice; test predictions agree to 1e-5), so the 15 x 8 fold models
run in about 45 minutes on 4 CPU cores. SVRs use sklearn as 03_train.py does.
Outputs: artifacts/plfold_<member>.parquet (per seed/fold predictions without pseudo rows, cached),
         artifacts/pred_pl<w>_<member>.parquet (the pseudo-labelled member, 03_train.py format), and the main
         model's out-of-fold metric with and without them (paired bootstrap over clips).
Usage: python src/22_pseudo_label.py [--weight 1] [--members t4_ridge_deberta z4_ridge_deberta]
"""
import argparse
import importlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR

from common import ART, fold_of

T3, B8 = importlib.import_module("03_train"), importlib.import_module("08_blend")
ALPHAS = np.logspace(0, 5, 11)
AUDIO = ["feat_voxtral", "feat_wavlm-base-plus", "feat_wavlm-large", "feat_whisper-large-v3-turbo"]
SPECS = {   # the ridge/SVR members of the main model (notebook cell "SPECS"): model, tables, trained without score-0 clips
    "t4_ridge_deberta": ("ridge", ["feat_emb_deberta-v3-large_prompt"], False),
    "c4c_ridge_vox_allaudio": ("ridge", AUDIO, False),
    "c4_svr_wavlm-large": ("svr", ["feat_wavlm-large"], False),
    "c4_ridge_whisper": ("ridge", ["feat_whisper-large-v3-turbo"], False),
    "x_svr_voxtral_wavlml": ("svr", ["feat_voxtral", "feat_wavlm-large"], False),
    "z4_ridge_deberta": ("ridge", ["feat_emb_deberta-v3-large_prompt"], True),
    "z4c_ridge_vox_allaudio": ("ridge", AUDIO, True),
    "z4_svr_wavlm-large": ("svr", ["feat_wavlm-large"], True),
}
FT, VP = "s4_deberta_ft_crops_nozero", "vp_attn"
BLENDS = [["t4_ridge_deberta", FT, "c4c_ridge_vox_allaudio", VP, "c4_svr_wavlm-large"],
          ["t4_ridge_deberta", FT, "c4_ridge_whisper", "c4_svr_wavlm-large", "x_svr_voxtral_wavlml", VP],
          ["z4_ridge_deberta", FT, "z4c_ridge_vox_allaudio", VP, "z4_svr_wavlm-large"]]
KEY = ["filename", "split"]


def metric(p, y):
    return 0.6 * np.sqrt(np.mean((p - y) ** 2)) + 0.4 * (1 - np.corrcoef(p, y)[0, 1])


def ridge(X, y, alphas, Xev, sw):
    """StandardScaler + Ridge(alpha) with sample weights, for several alphas: [[pred per Xev] per alpha]."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    X, Xev = (X - mu) / sd, [(Z - mu) / sd for Z in Xev]
    xm, ym = sw @ X / sw.sum(), sw @ y / sw.sum()
    A, b = (X - xm) * np.sqrt(sw)[:, None], (y - ym) * np.sqrt(sw)
    if A.shape[1] > A.shape[0]:   # dual form for the wide tables
        ev, U = np.linalg.eigh(A @ A.T)
        E, c0 = [(Z - xm) @ A.T @ U for Z in Xev], U.T @ b
    else:
        ev, U = np.linalg.eigh(A.T @ A)
        E, c0 = [(Z - xm) @ U for Z in Xev], U.T @ (A.T @ b)
    return [[e @ (c0 / (ev + a)) + ym for e in E] for a in alphas]


def fit_predict(model, X, y, groups, Xev, alpha=None, sw=None):
    sw = np.ones(len(y)) if sw is None else sw
    if model == "svr":
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        m = SVR(C=10, epsilon=0.1).fit((X - mu) / sd, y, sample_weight=sw)
        return [m.predict((Z - mu) / sd) for Z in Xev], None
    if alpha is None:   # 03_train.py: GridSearchCV over ALPHAS, GroupKFold(5) by clip, MAE
        mae = np.zeros(len(ALPHAS))
        for itr, iva in GroupKFold(5).split(X, y, groups):
            mae += [np.abs(p[0] - y[iva]).mean() for p in ridge(X[itr], y[itr], ALPHAS, [X[iva]], sw[itr])]
        alpha = ALPHAS[int(np.argmin(mae))]
    return ridge(X, y, [alpha], Xev, sw)[0], alpha


def member(name, teacher=None, w=0.0):
    """Per (seed, fold) predictions of one member: val crops averaged per clip + test. teacher -> with pseudo rows."""
    model, feats, drop_zero = SPECS[name]
    train, test, crop, cols = T3.load_tables(feats, crops=True)
    rows = pd.concat([train, crop], ignore_index=True)
    rows = rows[rows.label > 0].reset_index(drop=True) if drop_zero else rows
    Xf, Xc, Xt = (d[cols].values.astype(np.float64) for d in (rows, crop, test))
    base = None if teacher is None else pd.read_parquet(ART / f"plfold_{name}.parquet")
    out = []
    for s in range(3):
        for k in range(5):
            fm = fold_of(rows.filename, f"gfold_s{s}") != k
            vm = fold_of(crop.filename, f"gfold_s{s}") == k
            X, y, g, sw, alpha = Xf[fm], rows.label.values[fm], rows.filename.values[fm], None, None
            if teacher is not None:
                alpha = base.query("seed == @s and fold == @k").alpha.iloc[0]
                tv, tt = teacher[s, k]
                pv, pt = tv.reindex(crop.filename[vm]).values, tt.reindex(test.filename).values
                kv, kt = ~np.isnan(pv), ~np.isnan(pt)
                X = np.vstack([X, Xc[vm][kv], Xt[kt]])
                y = np.r_[y, pv[kv], pt[kt]]
                g = np.r_[g, "v_" + crop.filename[vm][kv], "t_" + test.filename[kt]]
                sw = np.r_[np.ones(fm.sum()), np.full(kv.sum(), w), np.full(kt.sum(), 4 * w)]
            (p_va, p_te), a = fit_predict(model, X, y, g, [Xc[vm], Xt], alpha, sw)
            va = pd.Series(p_va, index=crop.filename[vm].values).groupby(level=0).mean()
            out += [pd.DataFrame({"filename": va.index, "split": "train", "seed": s, "fold": k, "pred": va.values}),
                    pd.DataFrame({"filename": test.filename, "split": "test", "seed": s, "fold": k, "pred": p_te})]
            out[-1]["alpha"] = out[-2]["alpha"] = np.nan if a is None else a
        print(name, "seed", s, "done", flush=True)
    return pd.concat(out, ignore_index=True)


def as_pred(folds, labels):
    """Per-fold table -> 03_train.py prediction format (OOF averaged over seeds, test over all 15 models)."""
    tr = folds[folds.split == "train"].groupby("filename").pred.mean()   # one fold per seed -> mean over seeds
    te = folds[folds.split == "test"].groupby("filename").pred.mean()
    p = pd.concat([tr.rename("pred").to_frame().assign(split="train"), te.rename("pred").to_frame().assign(split="test")])
    p = p.reset_index().set_index(KEY).pred.clip(0, 5)
    return p.reindex(labels.index)


def main_blend(P, rows=None, ids=None):
    """Main model on prediction table P (one column per model): gate + 3 blends with batch weights, averaged.
    rows: train rows the blend weights are fitted on (default: CV over gfold_s1 as 08_blend.py, giving OOF + test)."""
    gated = (ids.pred < B8.GATE).values
    is_tr = (ids.reset_index().split == "train").values
    y = ids.label.values
    dur = pd.read_parquet(ART / "feat_text_prompt.parquet").set_index(KEY).duration.reindex(ids.index).values
    bat = np.abs(dur - B8.BATCH_SECONDS) < 0.05

    def apply(names, fit_rows):
        X = P[names].values
        b, w = B8.fit(X[fit_rows], y[fit_rows])
        m = fit_rows & bat
        bb, wb = B8.fit(X[m], y[m])
        shift = (bb + X[m] @ wb).mean() - (b + X[m] @ w).mean()
        return np.where(gated, 0, np.clip(np.where(bat, bb + X @ wb - shift, b + X @ w), 0, 5))

    if rows is not None:
        return np.mean([apply(n, rows & is_tr & ~gated) for n in BLENDS], 0)
    folds = fold_of(ids.index.get_level_values(0), "gfold_s1")
    oof = np.zeros(len(ids))
    for k in range(5):
        fit_rows = is_tr & ~gated & (folds != k)
        oof[folds == k] = np.mean([apply(n, fit_rows) for n in BLENDS], 0)[folds == k]
    return oof[is_tr], y[is_tr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, default=1.0, help="pseudo-row weight w (test rows get 4w)")
    ap.add_argument("--members", nargs="+", default=["t4_ridge_deberta", "z4_ridge_deberta"])
    args = ap.parse_args()
    ids = pd.read_parquet(ART / f"pred_{B8.GATE_MODEL}.parquet").set_index(KEY)
    folds = {}
    for n in SPECS:   # per-fold predictions without pseudo rows (cached; ~45 min for all eight)
        path = ART / f"plfold_{n}.parquet"
        if not path.exists():
            member(n).to_parquet(path, index=False)
        folds[n] = pd.read_parquet(path)
    P = pd.DataFrame({n: as_pred(f, ids) for n, f in folds.items()})
    for n in (FT, VP):
        P[n] = pd.read_parquet(ART / f"pred_{n}.parquet").set_index(KEY).pred.reindex(ids.index)

    # Teacher for (seed, fold): members' fold models (FT/VP: saved predictions), blend weights fitted without the fold.
    teacher = {}
    for s in range(3):
        fk = fold_of(ids.index.get_level_values(0), f"gfold_s{s}")
        for k in range(5):
            Q = P.copy()
            for n, f in folds.items():
                f = f[(f.seed == s) & (f.fold == k)].set_index(KEY).pred.clip(0, 5)
                Q.loc[f.index, n] = f
            tv = main_blend(Q, (fk != k), ids)
            names = ids.index.get_level_values(0)
            val = (fk == k) & (ids.reset_index().split == "train").values
            te = (ids.reset_index().split == "test").values
            gated = (ids.pred < B8.GATE).values
            teacher[s, k] = (pd.Series(tv[val], index=names[val]), pd.Series(tv[te], index=names[te]))
            teacher[s, k] += (pd.Series(gated[val], index=names[val]), pd.Series(gated[te], index=names[te]))

    S = P.copy()
    for n in args.members:
        t = {sk: (v.where(~gv) if SPECS[n][2] else v, tt.where(~gt) if SPECS[n][2] else tt)
             for sk, (v, tt, gv, gt) in teacher.items()}   # members without score-0 clips skip gated pseudo rows
        S[n] = as_pred(member(n, t, args.weight), ids)
        S[n].rename("pred").reset_index().merge(ids.label.reset_index(), on=KEY, how="left")[KEY + ["label", "pred"]] \
            .to_parquet(ART / f"pred_pl{args.weight:g}_{n}.parquet", index=False)

    o0, y = main_blend(P, ids=ids)
    o1, _ = main_blend(S, ids=ids)
    rng = np.random.default_rng(0)
    d = np.array([metric(o1[i], y[i]) - metric(o0[i], y[i]) for i in rng.integers(0, len(y), (2000, len(y)))])
    print(f"main model OOF metric: without pseudo-labels {metric(o0, y):.4f} | with ({', '.join(args.members)}, "
          f"w {args.weight:g}) {metric(o1, y):.4f} | delta {metric(o1, y) - metric(o0, y):+.4f} | "
          f"bootstrap P(better) {np.mean(d < 0):.3f}")


if __name__ == "__main__":
    main()
