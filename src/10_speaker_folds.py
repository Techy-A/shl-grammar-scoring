"""Step 10: speaker-grouped CV folds.

Why: train clips come in speaker "twins" (same voice, similar score), and with random folds ~80% of
validation clips had a twin in the training folds. Test clips are clearly farther from train (median
nearest-train speaker similarity 0.87 vs 0.92 train->train). So random-fold CV was optimistic, most
of all for audio models that can recognise voices (WavLM SVR: MAE 0.397 random vs 0.448 grouped;
text models barely move). It does NOT explain why v3 beat v2 on CV but lost on the public
leaderboard: grouped CV still ranks v3 > v2 > v1, so that gap is most likely public-score noise.

How: a speaker-verification model (WavLM-base-plus-SV) embeds the first 15 s of every clip (same
length for all, so the shorter test clips compare fairly). Train clips are clustered (average-linkage
cosine distance < 0.12); each cluster stays inside one fold. The 0.12 threshold makes validation
clips exactly as far from their training folds as test clips are from train (median 0.870 vs 0.87).

Output: adds gfold_s0..gfold_s2 (5 folds x 3 seeds, stratified by label, grouped by speaker
cluster) to artifacts/folds.csv; the original random fold_s* columns are kept for reference.
Usage : python src/10_speaker_folds.py
"""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoFeatureExtractor, WavLMForXVector

from common import ART, DATA, SR

MODEL, CROP, THRESHOLD = "microsoft/wavlm-base-plus-sv", 15 * SR, 0.12


@torch.inference_mode()
def speaker_embeddings(files):
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = WavLMForXVector.from_pretrained(MODEL).to("cuda").eval()   # small model: fp32 everywhere
    E = []
    for r in files.itertuples():
        w = sf.read(DATA / r.split / r.filename, dtype="float32")[0][:CROP]
        E.append(model(**fe(w, sampling_rate=SR, return_tensors="pt").to("cuda")).embeddings[0].cpu().numpy())
    E = np.stack(E)
    E = E - E.mean(0)   # centring spreads the raw cosines, which all sit above 0.9
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def main():
    train = pd.read_csv(DATA / "train.csv")
    E = speaker_embeddings(train.assign(split="train"))
    groups = fcluster(linkage(E, "average", metric="cosine"), THRESHOLD, "distance")
    # Score-0 clips are acoustically distinct (the same property the gate in 08_blend.py uses), so their
    # speaker embeddings all look alike and form one cluster, which put every 0 into a single fold.
    # Give each its own group instead.
    zero = (train.label == 0).values
    groups[zero] = groups.max() + 1 + np.arange(zero.sum())
    # Same stratification as the random folds (03_train.py): rare 1.0/1.5 labels merged into 2.0.
    strat = train.label.clip(lower=2.0).where(train.label > 0, 0).astype(str)

    folds = pd.read_csv(ART / "folds.csv").set_index("filename").loc[train.filename].reset_index()
    for s in range(3):
        folds[f"gfold_s{s}"] = -1
        for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(E, strat, groups)):
            folds.loc[va, f"gfold_s{s}"] = k
    folds.to_csv(ART / "folds.csv", index=False)

    S = E @ E.T
    f = folds.gfold_s0.values
    print(f"{groups.max()} speaker groups; val->train-fold nearest similarity median "
          f"{np.median([S[i, f != f[i]].max() for i in range(len(E))]):.3f} (test->train: 0.87)")


if __name__ == "__main__":
    main()
