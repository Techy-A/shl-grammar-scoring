"""Step 23: topic-grouped CV folds, a second check next to the speaker-grouped folds of 10_speaker_folds.py.

Why: the test clips mostly answer questions that training never asks (median nearest-train tf-idf similarity of a test
transcript 0.22, against 0.30 train -> train). Speaker-grouped folds still put same-question answers on both sides, so a
change can look better there because it learnt the question, not the grammar. Here each question stays inside one fold.

How: there are no question ids, so they are derived from the full-clip transcripts (01_transcribe.py): tf-idf (English
stop words removed), average-linkage cosine clusters on the scored train clips. The threshold is the one at which a
validation clip's nearest transcript in the other folds is as similar as a test clip's nearest train transcript (median),
the same rule 10_speaker_folds.py uses for voices. Clusters are then joined with likely same-speaker clips (cached
WavLM-base-plus + WavLM-large features, 32 principal components, average-linkage cosine distance < 0.05), so that a
speaker who answers two questions cannot sit on both sides. The cached features are used because the speaker-verification
embeddings of step 10 are not cached; they are a proxy (91% of their pairs fall in one speaker-grouped fold in all three
fold seeds). Score-0 clips get their own group each, as in step 10. Stratified by label like step 10.

Output: artifacts/folds_topic.csv (filename, tfold_s0..tfold_s2: 5 folds x 3 seeds) and artifacts/topics.csv (filename,
split, topic: the cluster id of every train and test clip, test clips assigned to their nearest train cluster or -1).
Usage : python src/23_topic_folds.py
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from common import ART, DATA

SPEAKER_FEATS, SPEAKER_PCS, SPEAKER_DIST = ["feat_wavlm-base-plus", "feat_wavlm-large"], 32, 0.05


def speaker_proxy(names):
    """Unit vectors of the leading principal components of the standardised WavLM features (train + test rows)."""
    Xs = []
    for f in SPEAKER_FEATS:
        d = pd.read_parquet(ART / f"{f}.parquet")
        X = d[[c for c in d.columns if c.startswith("a")]].values.astype(float)
        Xs.append((X - X.mean(0)) / (X.std(0) + 1e-9) / np.sqrt(X.shape[1]))
    X = np.hstack(Xs)
    U = np.linalg.svd(X - X.mean(0), full_matrices=False)[0][:, :SPEAKER_PCS]
    Z = pd.DataFrame(U / np.linalg.norm(U, axis=1, keepdims=True), index=pd.MultiIndex.from_frame(d[["filename", "split"]]))
    return Z.xs("train", level=1).loc[names].values


def folds_for(groups, strat, seed):
    f = np.full(len(groups), -1)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(groups, strat, groups)):
        f[va] = k
    return f


def main():
    train = pd.read_csv(DATA / "train.csv")
    t = pd.read_parquet(ART / "transcripts_prompt.parquet")
    X = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit_transform(t.text).toarray()
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    is_tr = (t.split == "train").values
    pos = pd.Series(np.where(is_tr)[0], index=t.filename[is_tr]).loc[train.filename].values
    Xtr, Xte = X[pos], X[~is_tr]
    y = train.label.values
    zero = y == 0
    strat = train.label.clip(lower=2.0).where(train.label > 0, 0).astype(str)
    S = Xtr @ Xtr.T
    target = np.median((Xte @ Xtr.T).max(1))
    L = linkage(Xtr[~zero], "average", metric="cosine")

    def build(thr):
        g = np.zeros(len(y), int)
        g[~zero] = fcluster(L, thr, "distance")
        topic = g.copy()
        # join clusters that share a likely speaker (connected components over topic + speaker links)
        sp = np.zeros(len(y), int)
        sp[~zero] = fcluster(linkage(speaker_proxy(train.filename)[~zero], "average", metric="cosine"), SPEAKER_DIST,
                             "distance") + g.max()
        n = max(g.max(), sp.max()) + 1
        A = coo_matrix((np.ones(2 * (~zero).sum()), (np.r_[g[~zero], sp[~zero]], np.r_[sp[~zero], g[~zero]])), (n, n))
        comp = connected_components(A, directed=False)[1]
        groups = comp[g]
        groups[zero] = groups.max() + 1 + np.arange(zero.sum())
        f = folds_for(groups, strat, 0)
        nn = np.array([S[i, f != f[i]].max() for i in range(len(y))])
        return topic, groups, np.median(nn[~zero])

    lo, hi = 0.7, 0.99   # larger threshold -> coarser topics -> less similar across folds
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if build(mid)[2] > target else (lo, mid)
    thr = hi
    topic, groups, med = build(thr)
    out = pd.DataFrame({"filename": train.filename})
    for s in range(3):
        out[f"tfold_s{s}"] = folds_for(groups, strat, s)
    out.to_csv(ART / "folds_topic.csv", index=False)
    # test clips: the topic of their nearest scored train clip, if it is closer than the clustering threshold
    St = Xte @ Xtr[~zero].T
    te_topic = np.where(1 - St.max(1) < thr, topic[~zero][St.argmax(1)], -1)
    pd.concat([pd.DataFrame({"filename": train.filename, "split": "train", "topic": np.where(zero, -1, topic)}),
               pd.DataFrame({"filename": t.filename[~is_tr].values, "split": "test", "topic": te_topic})]
              ).to_csv(ART / "topics.csv", index=False)
    print(f"threshold {thr:.3f}: {len(set(topic[~zero]))} topics, {len(set(groups))} groups (largest "
          f"{pd.Series(groups).value_counts().max()}); val -> other-fold nearest transcript similarity median {med:.3f} "
          f"(test -> train {target:.3f}); test clips with a train topic {np.mean(te_topic >= 0):.2f}")


if __name__ == "__main__":
    main()
