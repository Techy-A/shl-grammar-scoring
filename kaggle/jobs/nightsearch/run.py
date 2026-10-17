# Kaggle job (CPU, research, overnight): which blend generalises best to clips like the test set's?
# Every candidate model is trained without the 37 score-0 clips (the gate sets them to 0), in kernel form, and scored
# with the decoded leaderboard metric (0.6 RMSE + 0.4 (1 - Pearson)) under three fold schemes, 3 seeds each:
# topic+speaker folds (folds_topic.csv, matched to the test set's topic and speaker distances), strict topic folds
# (folds_topic95.csv) and speaker folds (folds.csv). A greedy blend search runs on the topic folds only; the other two
# schemes and a paired bootstrap check what it picks. Test predictions of every model are saved for building blends.
# Inputs (uploaded with the code): <table>.npy per-table kernels over keys.parquet's rows, keys.parquet, folds*.csv.
# Output: artifacts/night_models.csv, night_blends.csv, night_oof.parquet, night_test.parquet
# SMOKE=1 (environment): 2 models, 1 seed, for a quick local check against artifacts/kern.
import glob, os, time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import nnls
from sklearn.svm import SVR

T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
ON_KAGGLE = os.path.exists("/kaggle/input")
IN = (os.path.dirname(glob.glob("/kaggle/input/**/keys.parquet", recursive=True)[0]) if ON_KAGGLE
      else "artifacts/kern")
FOLDS_DIR = IN if ON_KAGGLE else "artifacts"
OUT = "/kaggle/working/artifacts" if ON_KAGGLE else os.path.join(os.environ.get("TEMP", "."), "night_smoke")
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)

keys = pd.read_parquet(f"{IN}/keys.parquet")
fn, y = keys.filename.values, keys.label.values.astype(float)
n_tr, n_te, N = int((keys.split == "train").sum()), int((keys.split == "test").sum()), len(keys)
fit_idx, crop_idx, te_idx = np.arange(N - n_te), np.arange(n_tr, N - n_te), np.arange(N - n_te, N)
train_fn, y_tr = fn[:n_tr], y[:n_tr]
nz = y_tr > 0
KEEP = y[fit_idx] > 0                     # train rows (full clips and crops) of scored clips only

V, W, WL, T = "feat_voxtral", "feat_wavlm-base-plus", "feat_wavlm-large", "feat_whisper-large-v3-turbo"
DB, DBC, Q = "feat_emb_deberta-v3-large_prompt", "feat_emb_deberta-v3-large_crisper", "feat_qwen2audio"
WIDTH = {V: 6144, W: 1536, WL: 2048, T: 2560, DB: 1024, DBC: 1024, Q: 8192}
Kb = {b: np.load(f"{IN}/{b}.npy") for b in WIDTH if os.path.exists(f"{IN}/{b}.npy")}
log("tables", list(Kb))

# name: (kind, spec, C)   ridge = linear kernel of the standardised concatenation; eq = sum of per-table kernels with
# weights ("rbf:" = that table's RBF kernel); svr = RBF SVR (gamma = 1 / width on standardised features)
MODELS = {
    "deberta_ridge": ("ridge", [DB], None), "crisper_ridge": ("ridge", [DBC], None),
    "voxall_ridge": ("ridge", [V, W, WL, T], None), "vox_ridge": ("ridge", [V], None),
    "whisper_ridge": ("ridge", [T], None), "wavlm_ridge": ("ridge", [W], None), "wavlml_ridge": ("ridge", [WL], None),
    "qwen_ridge": ("ridge", [Q], None), "voxallq_ridge": ("ridge", [V, W, WL, T, Q], None),
    "eq4_kernel": ("eq", {V: 1, W: 1, WL: 1, T: 1}, None),
    "mkr_kernel": ("eq", {V: 1, W: 1, WL: 0.25, "rbf:" + WL: 2, T: 0.25, DB: 0.5}, None),
    "wavlml_svr": ("svr", [WL], 10), "wavlml_svr_c3": ("svr", [WL], 3), "wavlml_svr_c30": ("svr", [WL], 30),
    "wavlm_svr": ("svr", [W], 10), "voxwl_svr": ("svr", [V, WL], 10), "allaudio_svr": ("svr", [W, WL, T], 10),
    "qwen_svr": ("svr", [Q], 10),
}
if SMOKE:
    MODELS = {k: MODELS[k] for k in ["deberta_ridge", "wavlml_svr"]}
MODELS = {k: v for k, v in MODELS.items()
          if all((b[4:] if b.startswith("rbf:") else b) in Kb for b in (v[1] if isinstance(v[1], list) else v[1]))}
log("models", list(MODELS))


def rbf_of(tabs):
    dist = sum(WIDTH[b] * (np.diag(Kb[b])[:, None] + np.diag(Kb[b])[None] - 2 * Kb[b]) for b in tabs)
    return np.exp(-np.maximum(dist, 0) / sum(WIDTH[b] for b in tabs)).astype(np.float32)


def kernel_of(kind, spec):
    if kind == "ridge":
        return sum(np.float32(WIDTH[b]) * Kb[b] for b in spec)
    if kind == "svr":
        return rbf_of(spec)
    return sum(np.float32(w) * (rbf_of([b[4:]]) if b.startswith("rbf:") else Kb[b]) for b, w in spec.items())


def fit_predict(kind, K, rows, ev, alpha=None, C=10):
    if kind == "svr":
        return SVR(kernel="precomputed", C=C, epsilon=0.1).fit(K[np.ix_(rows, rows)], y[rows]).predict(K[np.ix_(ev, rows)])
    ym = y[rows].mean(); A = K[np.ix_(rows, rows)].astype(np.float64); A[np.diag_indices_from(A)] += alpha
    return K[np.ix_(ev, rows)] @ cho_solve(cho_factor(A, overwrite_a=True), y[rows] - ym) + ym


def clip_mean(pred, rows):
    return pd.Series(pred, index=fn[rows]).groupby(level=0).mean().reindex(train_fn).values


def metric(p, t):
    return 0.6 * np.sqrt(np.mean((p - t) ** 2)) + 0.4 * (1 - np.corrcoef(p, t)[0, 1])


SCHEMES = {"topic": ("folds_topic.csv", ["tfold_s0", "tfold_s1", "tfold_s2"]),
           "strict_topic": ("folds_topic95.csv", ["tfold_s0", "tfold_s1", "tfold_s2"]),
           "speaker": ("folds.csv", ["gfold_s0", "gfold_s1", "gfold_s2"])}
if SMOKE:
    SCHEMES = {"topic": ("folds_topic.csv", ["tfold_s0"])}
FOLDS = {s: pd.read_csv(f"{FOLDS_DIR}/{f}").set_index("filename") for s, (f, _) in SCHEMES.items()}


def choose_alpha(K):
    """Ridge strength on topic folds, seed 0 (eigendecomposition per fold, all strengths at once)."""
    fd = FOLDS["topic"][SCHEMES["topic"][1][0]]
    fr, vr = fd.reindex(fn[fit_idx]).values, fd.reindex(fn[crop_idx]).values
    grid = np.logspace(-2, 5, 15) if K is not None else None
    oof = np.zeros((len(grid), n_tr))
    for k in range(5):
        f, v = fit_idx[(fr != k) & KEEP], crop_idx[vr == k]
        lam, Qm = eigh(K[np.ix_(f, f)].astype(np.float64))
        ym = y[f].mean(); qy = Qm.T @ (y[f] - ym); Kvf = K[np.ix_(v, f)] @ Qm
        for j, a in enumerate(grid):
            cm = clip_mean(Kvf @ (qy / (lam + a)) + ym, v); oof[j][~np.isnan(cm)] = cm[~np.isnan(cm)]
    return grid[int(np.argmin([metric(np.where(nz, np.clip(o, 0, 5), 0), y_tr) for o in oof]))]


# ---- out-of-fold predictions of every model under every scheme and seed (clip level), and test predictions
OOF = {}            # (scheme, seed col, model) -> clip-level OOF
TEST = {}           # model -> test prediction (trained on all scored train rows)
rows = []
for name, (kind, spec, C) in MODELS.items():
    K = kernel_of(kind, spec)
    alpha = None if kind == "svr" else choose_alpha(K)
    for s, (_, cols) in SCHEMES.items():
        for col in cols:
            fd = FOLDS[s][col]; fr, vr = fd.reindex(fn[fit_idx]).values, fd.reindex(fn[crop_idx]).values
            o = np.full(n_tr, np.nan)
            for k in range(5):
                f, v = fit_idx[(fr != k) & KEEP], crop_idx[vr == k]
                cm = clip_mean(fit_predict(kind, K, f, v, alpha, C), v); o[~np.isnan(cm)] = cm[~np.isnan(cm)]
            OOF[s, col, name] = np.clip(o, 0, 5)
    TEST[name] = fit_predict(kind, K, fit_idx[KEEP], te_idx, alpha, C)
    row = dict(model=name, kind=kind, alpha=alpha, C=C)
    for s, (_, cols) in SCHEMES.items():
        p = np.mean([OOF[s, c, name] for c in cols], axis=0)
        row[f"{s}_scored"] = metric(p[nz], y_tr[nz])
    rows.append(row); log(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/night_models.csv", index=False)
    pd.DataFrame({f"{s}|{c}|{m}": v for (s, c, m), v in OOF.items()}, index=train_fn).to_parquet(f"{OUT}/night_oof.parquet")
    pd.DataFrame({"filename": fn[te_idx], **TEST}).to_parquet(f"{OUT}/night_test.parquet", index=False)
    del K


def blend_oof(names, s):
    """Blend OOF averaged over the scheme's seeds: NNLS + intercept per fold, fitted on the other folds' scored clips."""
    cols = SCHEMES[s][1]; out = np.zeros(n_tr)
    for col in cols:
        X = np.column_stack([OOF[s, col, m] for m in names]); cf = FOLDS[s][col].reindex(train_fn).values
        for k in range(5):
            msk = nz & (cf != k)
            w, _ = nnls(np.column_stack([X[msk], np.ones(msk.sum()), -np.ones(msk.sum())]), y_tr[msk])
            sel = cf == k; out[sel] += (X[sel] @ w[:-2] + w[-2] - w[-1]) / len(cols)
    return np.where(nz, np.clip(out, 0, 5), 0)   # score-0 clips: the perfect gate


def score_all(names):
    return {s: metric(blend_oof(names, s), y_tr) for s in SCHEMES}


blends = []
REF = [m for m in ["deberta_ridge", "voxall_ridge", "wavlml_svr"] if m in MODELS]   # the 0.3207 structure, no-zero
ref = score_all(REF); blends.append(dict(blend="reference: " + " + ".join(REF), **ref)); log(blends[-1])
# greedy forward selection on the topic folds only
chosen, best = [], np.inf
while len(chosen) < 6:
    cand = {m: score_all(chosen + [m])["topic"] for m in MODELS if m not in chosen}
    if not cand:
        break
    m, sc = min(cand.items(), key=lambda kv: kv[1])
    if sc > best - 0.0003:
        break
    chosen.append(m); best = sc
    blends.append(dict(blend="greedy: " + " + ".join(chosen), **score_all(chosen))); log(blends[-1])
    pd.DataFrame(blends).to_csv(f"{OUT}/night_blends.csv", index=False)

# paired bootstrap over clips (topic scheme): how often is the greedy blend better than the reference?
if chosen and REF:
    a, b = blend_oof(chosen, "topic"), blend_oof(REF, "topic")
    rng = np.random.default_rng(0); idx = np.arange(n_tr); d = []
    for _ in range(2000):
        s_ = rng.choice(idx, n_tr); d.append(metric(a[s_], y_tr[s_]) - metric(b[s_], y_tr[s_]))
    d = np.array(d)
    blends.append(dict(blend=f"bootstrap greedy - reference (topic): mean {d.mean():+.4f}, P(greedy better) {np.mean(d < 0):.2f}"))
    log(blends[-1])
pd.DataFrame(blends).to_csv(f"{OUT}/night_blends.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(pd.DataFrame(blends).round(4).to_string(index=False))
log("done")
