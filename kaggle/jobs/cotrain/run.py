# Kaggle job (CPU, research): co-training on unlabelled clips. The test set is a different domain (new questions,
# a quieter recording batch), so: let each model of the 3-model blend also learn from the blend's own predictions
# on the unlabelled clips, so that what one view (text, Voxtral, WavLM) concludes about them reaches the others.
# Simulated honestly inside CV first (each held-out fold plays the test set: its labels are only used for scoring),
# then applied to the real test clips. Inputs (precomputed by the research scripts, uploaded with the code):
#   <table>.npy  per-table linear kernels (standardised, divided by the table width) over keys.parquet's rows
#   keys.parquet the row order: train full clips, train test-length crops, test clips
#   folds*.csv   speaker-grouped (folds.csv) and topic+speaker-grouped (folds_topic*.csv) CV folds
# Output: artifacts/cotrain_results.csv (scores per fold scheme and variant), artifacts/cotrain_test.parquet
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


# ---- ridge strength per model: chosen once on base OOF (topic folds, seed 0), eigendecomposition per fold
fd0 = pd.read_csv(f"{IN}/folds_topic.csv").set_index("filename")["tfold_s0"]
fr0, vr0 = fd0.reindex(fn[fit_idx]).values, fd0.reindex(fn[crop_idx]).values
GRID = np.logspace(0, 5, 11)
ALPHA = {}
for m, (kind, K) in MODELS.items():
    if kind != "ridge":
        continue
    oof = np.zeros((len(GRID), n_tr))
    for k in range(5):
        f, v = fit_idx[fr0 != k], crop_idx[vr0 == k]
        lam, Q = eigh(K[np.ix_(f, f)].astype(np.float64))
        ym = y[f].mean(); qy = Q.T @ (y[f] - ym); Kvf = K[np.ix_(v, f)] @ Q
        for j, a in enumerate(GRID):
            pv = Kvf @ (qy / (lam + a)) + ym
            cm = clip_mean(pv, v); oof[j][~np.isnan(cm)] = cm[~np.isnan(cm)]
    scores = [metric(np.clip(o, 0, 5), y_tr) for o in oof]
    ALPHA[m] = GRID[int(np.argmin(scores))]
log("ridge alphas", ALPHA)


def run_scheme(folds_csv, cols, variants):
    fd = pd.read_csv(f"{IN}/{folds_csv}").set_index("filename")
    nz = y_tr > 0
    # pass 1: base crop-level predictions per seed and fold
    base = {}
    for col in cols:
        fr, vr = fd[col].reindex(fn[fit_idx]).values, fd[col].reindex(fn[crop_idx]).values
        for k in range(5):
            f, v = fit_idx[fr != k], crop_idx[vr == k]
            base[col, k] = (v, {m: fit_predict(kind, K, f, y[f], np.ones(len(f)), v, ALPHA.get(m))
                                for m, (kind, K) in MODELS.items()})
    log(f"{folds_csv}: base fits done")

    def clip_oof(preds_by_fold, col):
        X = np.full((n_tr, len(NAMES)), np.nan)
        for k in range(5):
            v, P = preds_by_fold[col, k]
            for j, m in enumerate(NAMES):
                cm = clip_mean(P[m], v); X[~np.isnan(cm), j] = cm[~np.isnan(cm)]
        return X

    clipfold = {col: fd[col].reindex(train_fn).values for col in cols}
    # blend weights for fold k: NNLS on the other folds' base clip-level OOF (scored clips), per seed
    W = {}
    for col in cols:
        X = clip_oof(base, col)
        for k in range(5):
            m = nz & (clipfold[col] != k)
            W[col, k] = nnls_blend(X[m], y_tr[m])

    def score(preds_by_fold):
        oof = np.zeros(n_tr)
        for col in cols:
            X = clip_oof(preds_by_fold, col)
            for k in range(5):
                w, b = W[col, k]; sel = clipfold[col] == k
                oof[sel] += (X[sel] @ w + b) / len(cols)
        oof = np.where(nz, np.clip(oof, 0, 5), 0)          # score-0 clips: the perfect gate
        return metric(oof, y_tr), metric(oof[nz], y_tr[nz])

    rows = [dict(scheme=folds_csv, variant="base", round=0, metric_all=score(base)[0], metric_scored=score(base)[1])]
    log(rows[-1])
    for w_pl, n_rounds in variants:
        cur = base
        for rnd in range(1, n_rounds + 1):
            new = {}
            for col in cols:
                fr = fd[col].reindex(fn[fit_idx]).values
                for k in range(5):
                    v, P = cur[col, k]
                    f = fit_idx[fr != k]
                    wk, bk = W[col, k]
                    pseudo = np.column_stack([P[m] for m in NAMES]) @ wk + bk     # the blend on the held-out crops
                    keep = y[v] > 0                                              # the gate drops score-0 clips
                    rows_ = np.concatenate([f, v[keep]])
                    tg = np.concatenate([y[f], np.clip(pseudo[keep], 0, 5)])
                    wt = np.concatenate([np.ones(len(f)), np.full(keep.sum(), w_pl)])
                    new[col, k] = (v, {m: fit_predict(kind, K, rows_, tg, wt, v, ALPHA.get(m))
                                       for m, (kind, K) in MODELS.items()})
            cur = new
            s_all, s_sc = score(cur)
            rows.append(dict(scheme=folds_csv, variant=f"cotrain w={w_pl}", round=rnd, metric_all=s_all, metric_scored=s_sc))
            log(rows[-1])
    return rows, W


results = []
r, W_topic = run_scheme("folds_topic.csv", ["tfold_s0", "tfold_s1", "tfold_s2"], [(0.5, 2), (1.0, 2)])
results += r
pd.DataFrame(results).to_csv(f"{OUT}/cotrain_results.csv", index=False)
r, _ = run_scheme("folds_topic95.csv", ["tfold_s0"], [(0.5, 1), (1.0, 1)])
results += r
r, _ = run_scheme("folds.csv", ["gfold_s0"], [(0.5, 1), (1.0, 1)])
results += r
pd.DataFrame(results).to_csv(f"{OUT}/cotrain_results.csv", index=False)

# ---- the real test clips: base blend vs co-trained blend (weights: NNLS on the topic-fold base OOF, all folds)
f = fit_idx
P0 = {m: fit_predict(kind, K, f, y[f], np.ones(len(f)), te_idx, ALPHA.get(m)) for m, (kind, K) in MODELS.items()}
w_all = np.mean([W_topic[key][0] for key in W_topic], axis=0); b_all = np.mean([W_topic[key][1] for key in W_topic])
out = pd.DataFrame({"filename": fn[te_idx], "base": np.column_stack([P0[m] for m in NAMES]) @ w_all + b_all})
for w_pl in (0.5, 1.0):
    P = P0
    for rnd in (1, 2):
        pseudo = np.clip(np.column_stack([P[m] for m in NAMES]) @ w_all + b_all, 0, 5)
        rows_ = np.concatenate([f, te_idx]); tg = np.concatenate([y[f], pseudo])
        wt = np.concatenate([np.ones(len(f)), np.full(n_te, w_pl)])
        P = {m: fit_predict(kind, K, rows_, tg, wt, te_idx, ALPHA.get(m)) for m, (kind, K) in MODELS.items()}
        out[f"cotrain_w{w_pl}_r{rnd}"] = np.column_stack([P[m] for m in NAMES]) @ w_all + b_all
out.to_parquet(f"{OUT}/cotrain_test.parquet", index=False)
log("test predictions written; blend weights", dict(zip(NAMES, np.round(w_all, 3))), "intercept", round(b_all, 3))
print(pd.DataFrame(results).round(4).to_string(index=False))
