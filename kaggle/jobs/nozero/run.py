# Kaggle job (CPU, research): train the regressors without the 37 score-0 clips. Those clips are loud noise that
# the gate already sets to 0 at prediction time, and Whisper turns them into fluent hallucinated text, so the text
# model learns "fluent text -> 0" from them. Compared with the current training (all clips) under the same fold
# schemes, then applied to the test clips. Inputs (precomputed by the research scripts, uploaded with the code):
#   <table>.npy  per-table linear kernels (standardised, divided by the table width) over keys.parquet's rows
#   keys.parquet the row order: train full clips, train test-length crops, test clips
#   folds*.csv   speaker-grouped (folds.csv) and topic+speaker-grouped (folds_topic*.csv) CV folds
# Output: artifacts/nozero_results.csv (scores per fold scheme), artifacts/nozero_test.parquet
import glob, os, time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import nnls
from sklearn.svm import SVR

T0 = time.time()
IN = os.path.dirname(glob.glob("/kaggle/input/**/keys.parquet", recursive=True)[0])
OUT = "/kaggle/working/artifacts"
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)

keys = pd.read_parquet(f"{IN}/keys.parquet")
fn, y = keys.filename.values, keys.label.values.astype(float)
n_tr, n_te, N = int((keys.split == "train").sum()), int((keys.split == "test").sum()), len(keys)
fit_idx, crop_idx, te_idx = np.arange(N - n_te), np.arange(n_tr, N - n_te), np.arange(N - n_te, N)
train_fn, y_tr = fn[:n_tr], y[:n_tr]
clip_of_row = pd.Series(np.arange(n_tr), index=train_fn)   # train clip index of a filename

# ---- the three models of the 0.3207 blend, in kernel form (ridge = linear kernel of the standardised concatenation)
WIDTH = {"feat_voxtral": 6144, "feat_wavlm-base-plus": 1536, "feat_wavlm-large": 2048,
         "feat_whisper-large-v3-turbo": 2560, "feat_emb_deberta-v3-large_prompt": 1024}
Kb = {b: np.load(f"{IN}/{b}.npy") for b in WIDTH}


def lin(tabs):
    return sum(np.float32(WIDTH[b]) * Kb[b] for b in tabs)


def rbf(tabs):   # SVR's RBF kernel, gamma = 1 / (total width) on standardised features
    dist = sum(WIDTH[b] * (np.diag(Kb[b])[:, None] + np.diag(Kb[b])[None] - 2 * Kb[b]) for b in tabs)
    return np.exp(-np.maximum(dist, 0) / sum(WIDTH[b] for b in tabs)).astype(np.float32)


MODELS = {"deberta_ridge": ("ridge", lin(["feat_emb_deberta-v3-large_prompt"])),
          "vox_allaudio_ridge": ("ridge", lin(["feat_voxtral", "feat_wavlm-base-plus", "feat_wavlm-large",
                                               "feat_whisper-large-v3-turbo"])),
          "wavlml_svr": ("svr", rbf(["feat_wavlm-large"]))}
del Kb
NAMES = list(MODELS)
log("kernels ready")


def fit_predict(kind, K, rows, targets, weights, ev, alpha):
    """Train on rows (sample weights: 1 for labelled rows, w for pseudo-labelled ones), predict ev."""
    if kind == "svr":
        m = SVR(kernel="precomputed", C=10, epsilon=0.1).fit(K[np.ix_(rows, rows)], targets, sample_weight=weights)
        return m.predict(K[np.ix_(ev, rows)])
    ym = np.average(targets, weights=weights)
    A = K[np.ix_(rows, rows)].astype(np.float64)
    A[np.diag_indices_from(A)] += alpha / weights        # weighted kernel ridge: penalty alpha / w per row
    c = cho_solve(cho_factor(A, overwrite_a=True), targets - ym)
    return K[np.ix_(ev, rows)] @ c + ym


def clip_mean(pred, rows):
    """Crop-level predictions -> per-clip mean, as an n_tr vector (NaN where the clip has no rows)."""
    return pd.Series(pred, index=fn[rows]).groupby(level=0).mean().reindex(train_fn).values


def metric(p, t):
    return 0.6 * np.sqrt(np.mean((p - t) ** 2)) + 0.4 * (1 - np.corrcoef(p, t)[0, 1])


def nnls_blend(X, t):
    w, _ = nnls(np.column_stack([X, np.ones(len(X)), -np.ones(len(X))]), t)
    return w[:-2], w[-2] - w[-1]


# ---- ridge strength per model and training set: chosen on base OOF (topic folds, seed 0), eigendecomposition per fold
fd0 = pd.read_csv(f"{IN}/folds_topic.csv").set_index("filename")["tfold_s0"]
fr0, vr0 = fd0.reindex(fn[fit_idx]).values, fd0.reindex(fn[crop_idx]).values
GRID = np.logspace(0, 5, 11)


def choose_alpha(keep):
    out = {}
    for m, (kind, K) in MODELS.items():
        if kind != "ridge":
            continue
        oof = np.zeros((len(GRID), n_tr))
        for k in range(5):
            f, v = fit_idx[(fr0 != k) & keep], crop_idx[vr0 == k]
            lam, Q = eigh(K[np.ix_(f, f)].astype(np.float64))
            ym = y[f].mean(); qy = Q.T @ (y[f] - ym); Kvf = K[np.ix_(v, f)] @ Q
            for j, a in enumerate(GRID):
                cm = clip_mean(Kvf @ (qy / (lam + a)) + ym, v); oof[j][~np.isnan(cm)] = cm[~np.isnan(cm)]
        nz = y_tr > 0
        scores = [metric(np.where(nz, np.clip(o, 0, 5), 0), y_tr) for o in oof]
        out[m] = GRID[int(np.argmin(scores))]
    return out


TRAINSETS = {"all clips (current)": np.ones(len(fit_idx), bool), "without score-0 clips": y[fit_idx] > 0}
ALPHAS = {t: choose_alpha(keep) for t, keep in TRAINSETS.items()}
log("ridge alphas", ALPHAS)


def run_scheme(folds_csv, cols):
    fd = pd.read_csv(f"{IN}/{folds_csv}").set_index("filename")
    nz = y_tr > 0
    rows = []
    for tname, keep in TRAINSETS.items():
        oof = np.zeros(n_tr)
        for col in cols:
            fr, vr = fd[col].reindex(fn[fit_idx]).values, fd[col].reindex(fn[crop_idx]).values
            X = np.full((n_tr, len(NAMES)), np.nan)
            for k in range(5):
                f, v = fit_idx[(fr != k) & keep], crop_idx[vr == k]
                for j, (m, (kind, K)) in enumerate(MODELS.items()):
                    cm = clip_mean(fit_predict(kind, K, f, y[f], np.ones(len(f)), v, ALPHAS[tname].get(m)), v)
                    X[~np.isnan(cm), j] = cm[~np.isnan(cm)]
            cf = fd[col].reindex(train_fn).values
            for k in range(5):   # blend weights from the other folds (scored clips)
                msk = nz & (cf != k); w, b = nnls_blend(X[msk], y_tr[msk]); sel = cf == k
                oof[sel] += (X[sel] @ w + b) / len(cols)
        oof = np.where(nz, np.clip(oof, 0, 5), 0)   # score-0 clips: the perfect gate
        rows.append(dict(scheme=folds_csv, training=tname, metric_all=metric(oof, y_tr), metric_scored=metric(oof[nz], y_tr[nz]),
                         rmse_scored=float(np.sqrt(np.mean((oof[nz] - y_tr[nz]) ** 2))), r_scored=float(np.corrcoef(oof[nz], y_tr[nz])[0, 1])))
        log(rows[-1])
    return rows


results = []
for folds_csv, cols in [("folds_topic.csv", ["tfold_s0", "tfold_s1", "tfold_s2"]), ("folds_topic95.csv", ["tfold_s0", "tfold_s1", "tfold_s2"]),
                        ("folds.csv", ["gfold_s0", "gfold_s1", "gfold_s2"])]:
    results += run_scheme(folds_csv, cols)
    pd.DataFrame(results).to_csv(f"{OUT}/nozero_results.csv", index=False)

# ---- test predictions of each model under both training sets (the blend weights are fitted locally)
out = pd.DataFrame({"filename": fn[te_idx]})
for tname, keep in TRAINSETS.items():
    f = fit_idx[keep]
    for m, (kind, K) in MODELS.items():
        out[f"{m} | {tname}"] = fit_predict(kind, K, f, y[f], np.ones(len(f)), te_idx, ALPHAS[tname].get(m))
out.to_parquet(f"{OUT}/nozero_test.parquet", index=False)
print(pd.DataFrame(results).round(4).to_string(index=False))
