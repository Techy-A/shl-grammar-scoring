"""Step 20: attention-pooling head on cached Voxtral frames (artifacts/voxtral_frames.npz from 19_voxtral_frames.py).

The ridge models see each clip as one mean + std vector and only the 4 fixed test-length crops per training clip.
Here a small head learns which frames matter (attention weights over time) and trains on a new random test-length
prefix of every clip in every epoch, so it sees far more length variation than the 4 crops. Crop training is the kind of
change that has carried over to the public leaderboard before.

Head: LayerNorm -> Linear 256->128 + GELU -> [attention-weighted mean, mean, std] -> Linear -> score.
Training: without the score-0 clips (the 08_blend.py gate handles them), MSE, AdamW, speaker-grouped folds gfold_s0,
3 seeds. Validation as 03_train.py --crops: the 4 standard crops of each validation clip (prefixes of its frames),
averaged per clip; test: the full clips, averaged over folds and seeds.
--check compares the prefix approximation with the true crop embeddings of 11_voxtral_embed.py (feat_voxtral_crops).
--source phi4 / wavlm-large trains on that encoder's frame cache (19_voxtral_frames.py --source) -> pred_vp_attn_<source>.
Output: artifacts/pred_vp_attn.parquet (+ a results.csv row)
Usage : python src/20_attn_pool.py [--check] [--epochs 40] [--source voxtral]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import ART, DATA, N_CROPS, SR, crop_lengths, fold_of

FPS, SEEDS, BS = 12.5, [0, 1, 2], 32
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    def __init__(self, d=256, h=128):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1))
        self.att = nn.Linear(h, 1)
        self.out = nn.Linear(3 * h, 1)

    def forward(self, x, m):   # x (B, T, d), m (B, T) 1 = real frame
        h = self.proj(x)
        a = self.att(h).squeeze(-1).masked_fill(m == 0, -1e4).softmax(-1).unsqueeze(-1)
        w = m.unsqueeze(-1) / m.sum(1, keepdim=True).unsqueeze(-1)
        mean = (h * w).sum(1)
        std = ((h - mean.unsqueeze(1)) ** 2 * w).sum(1).clamp_min(1e-6).sqrt()
        return self.out(torch.cat([(h * a).sum(1), mean, std], -1)).squeeze(-1)


def batch(F, start, n):   # pad prefixes of n[i] frames from clip offsets start[i] -> (x, mask)
    T = int(max(n))
    x = np.zeros((len(n), T, F.shape[1]), np.float32)
    m = np.zeros((len(n), T), np.float32)
    for i, (s, k) in enumerate(zip(start, n)):
        x[i, :k], m[i, :k] = F[s:s + k], 1
    return torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)


@torch.inference_mode()
def predict(model, F, start, n):
    model.eval()
    return np.concatenate([model(*batch(F, start[i:i + 64], n[i:i + 64])).cpu().numpy() for i in range(0, len(n), 64)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--source", default="voxtral")
    args = ap.parse_args()
    z = np.load(ART / f"{args.source}_frames.npz", allow_pickle=True)
    name = "vp_attn" + ("" if args.source == "voxtral" else f"_{args.source}")
    F = z["frames"].astype(np.float32)
    ids = pd.DataFrame({"filename": z["filename"], "split": z["split"], "start": z["start"], "lens": z["lens"]})
    train = pd.read_csv(DATA / "train.csv").merge(ids[ids.split == "train"], on="filename")   # train.csv order
    test = ids[ids.split == "test"].reset_index(drop=True)
    to_frames = lambda samples: np.minimum(np.ceil(samples / SR * FPS).astype(int), train.lens.values)
    crops = [to_frames(c) for c in crop_lengths(len(train))]   # the 4 standard test-length crops, in frames

    if args.check:   # prefix-of-frames mean vs the true crop embedding's mean part, both in PCA space
        fc = pd.read_parquet(ART / "feat_voxtral_crops.parquet")
        c0 = fc[fc.split == "crop0"].set_index("filename").loc[train.filename].iloc[:, 2:2 + 3072].values
        true = (c0 - z["mu"]) @ z["basis"]
        pref = np.stack([F[s:s + k].mean(0) for s, k in zip(train.start, crops[0])])
        r = [np.corrcoef(a, b)[0, 1] for a, b in zip(true, pref)]
        return print(f"prefix approximation, crop 0: per-clip correlation median {np.median(r):.4f}, min {np.min(r):.4f}")

    y = train.label.values.astype(np.float32)
    folds = fold_of(train.filename, "gfold_s0")
    test_durs = pd.read_parquet(ART / "feat_text_prompt.parquet").query("split == 'test'").duration.values
    oof, te = np.zeros(len(train)), np.zeros(len(test))
    for seed in SEEDS:
        for k in range(5):
            torch.manual_seed(seed * 10 + k)
            rng = np.random.default_rng(seed * 10 + k)
            tr = np.where((folds != k) & (y > 0))[0]
            model = Head().to(dev)
            with torch.no_grad():
                model.out.bias.fill_(float(y[tr].mean()))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.epochs * int(np.ceil(len(tr) / BS)))
            for ep in range(args.epochs):
                model.train()
                order = rng.permutation(tr)
                n = to_frames(rng.choice(test_durs, len(train)) * SR)   # a random test-length prefix per clip
                for i in range(0, len(order), BS):
                    b = order[i:i + BS]
                    loss = nn.functional.mse_loss(model(*batch(F, train.start.values[b], n[b])), torch.from_numpy(y[b]).to(dev))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            va = np.where(folds == k)[0]
            p = np.mean([predict(model, F, train.start.values[va], c[va]) for c in crops], axis=0)
            oof[va] += p / len(SEEDS)
            te += predict(model, F, test.start.values, test.lens.values) / (len(SEEDS) * 5)
            sc = y[va] > 0
            print(f"seed {seed} fold {k}: val RMSE (scored) {np.sqrt(np.mean((p[sc] - y[va][sc]) ** 2)):.4f}", flush=True)

    oof, te = np.clip(oof, 0, 5), np.clip(te, 0, 5)
    m = y > 0
    rmse, r = float(np.sqrt(np.mean((oof - y) ** 2))), float(np.corrcoef(oof, y)[0, 1])
    print(f"{name}: CV RMSE {rmse:.4f} | Pearson {r:.4f} | scored clips: RMSE {np.sqrt(np.mean((oof[m] - y[m]) ** 2)):.4f} "
          f"r {np.corrcoef(oof[m], y[m])[0, 1]:.4f}")
    pd.concat([pd.DataFrame({"filename": train.filename, "split": "train", "label": y, "pred": oof}),
               pd.DataFrame({"filename": test.filename, "split": "test", "label": np.nan, "pred": te})]
              ).to_parquet(ART / f"pred_{name}.parquet", index=False)
    pd.DataFrame([dict(name=name, model="attn-pool", feats=f"{args.source}_frames", n_feats=256, cv_rmse=round(rmse, 4),
                       cv_pearson=round(r, 4))]).to_csv(ART / "results.csv", mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
