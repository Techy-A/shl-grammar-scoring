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
--text conditions the head on the transcript: the frozen DeBERTa embedding (04_text_embed.py, PCA to 64) of the crop
whose length is closest to the frame prefix (the 4 standard crops or the full clip; the full clip at test) shifts the
attention logits and joins the pooled vector, so the head can learn where to listen given what was said -> _text.
--dropout / --ema: head dropout (default 0.1) and an exponential moving average of the weights used for prediction
(e.g. 0.99; default off) -> _do<p> / _ema<d>.
--out bins / corn: a distribution over the 9 half-point labels 1-5 instead of one number, decoded by its expected value
(never rounded). bins: a softmax, trained with MSE on the expected value; corn: CORN ordinal loss (8 conditional
"above k?" classifiers). Output biases start at the training labels' (conditional) log-frequencies -> _bins / _corn.
--oversample f: clips scored <= 2 or >= 4.5 appear f times per epoch (same crop length within an epoch) -> _os<f>.
--corr l: loss + l * (1 - Pearson of the batch), since the metric rewards correlation -> _corr<l>.
--seed0 s: seeds s.. instead of 0.. (another retrain) -> _r<s>. --folds tfold_s0: topic-grouped folds of
23_topic_folds.py instead of gfold_s0 -> _tfold_s0.
Output: artifacts/pred_vp_attn.parquet (+ a results.csv row)
Usage : python src/20_attn_pool.py [--check] [--epochs 40] [--source voxtral]
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import ART, DATA, N_CROPS, SR, crop_lengths, fold_of

FPS, SEEDS, BS = 12.5, [0, 1, 2], 32
VALUES = np.arange(1.0, 5.01, 0.5)   # the scored labels (score-0 clips are left to the gate)
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    def __init__(self, d=256, h=128, dropout=0.1, dt=0, kind="reg"):
        super().__init__()
        self.kind = kind
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout))
        self.att = nn.Linear(h, 1)
        self.out = nn.Linear(3 * h + (h if dt else 0), {"reg": 1, "bins": len(VALUES), "corn": len(VALUES) - 1}[kind])
        if dt:   # created last, so the audio-only head keeps its initialisation
            self.tproj = nn.Sequential(nn.LayerNorm(dt), nn.Linear(dt, h), nn.GELU(), nn.Dropout(dropout))

    def init_bias(self, y):
        """Start at the training labels' mean (reg) or their (conditional) log-frequencies (bins / corn)."""
        with torch.no_grad():
            if self.kind == "reg":
                return self.out.bias.fill_(float(y.mean()))
            freq = (np.bincount(rank(y), minlength=len(VALUES)) + 1) / (len(y) + len(VALUES))
            if self.kind == "bins":
                return self.out.bias.copy_(torch.tensor(np.log(freq), dtype=torch.float32))
            above = 1 - np.cumsum(freq)[:-1]                          # P(rank > k)
            cond = above / np.r_[1, above[:-1]]                       # P(rank > k | rank > k - 1)
            self.out.bias.copy_(torch.tensor(np.log(cond / (1 - cond)), dtype=torch.float32))

    def score(self, o):   # head output -> expected score
        if self.kind == "reg":
            return o
        v = torch.as_tensor(VALUES, dtype=o.dtype, device=o.device)
        if self.kind == "bins":
            return o.softmax(-1) @ v
        return v[0] + 0.5 * torch.sigmoid(o).cumprod(-1).sum(-1)   # sum of P(rank > k) over the 0.5 steps

    def forward(self, x, m, t=None):   # x (B, T, d), m (B, T) 1 = real frame, t (B, dt) transcript embedding
        h = self.proj(x)
        ht = self.tproj(t) if t is not None else None
        a = self.att(h if ht is None else h + ht.unsqueeze(1)).squeeze(-1).masked_fill(m == 0, -1e4).softmax(-1).unsqueeze(-1)
        w = m.unsqueeze(-1) / m.sum(1, keepdim=True).unsqueeze(-1)
        mean = (h * w).sum(1)
        std = ((h - mean.unsqueeze(1)) ** 2 * w).sum(1).clamp_min(1e-6).sqrt()
        return self.out(torch.cat([(h * a).sum(1), mean, std] + ([] if ht is None else [ht]), -1)).squeeze(-1)


def rank(y):   # label -> bin index 0..8
    return np.clip(np.round((np.asarray(y) - VALUES[0]) / 0.5), 0, len(VALUES) - 1).astype(int)


def corn_loss(o, r):
    """CORN: task k is "rank > k", trained only on the rows with rank > k - 1; mean over all (row, task) terms used."""
    k = torch.arange(o.shape[1], device=o.device)
    use = (r.unsqueeze(1) >= k).float()
    target = (r.unsqueeze(1) > k).float()
    return (nn.functional.binary_cross_entropy_with_logits(o, target, reduction="none") * use).sum() / use.sum()


def pearson_loss(p, y):
    p, y = p - p.mean(), y - y.mean()
    return 1 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)


def text_table(train, test, dim=64):
    """Frozen DeBERTa transcript embeddings, PCA-reduced on the train rows: (train views (5, n, dim) as crop0-3 + full,
    test (n_test, dim)). Crop transcripts are of the crops themselves (01_transcribe.py --crops), so nothing leaks."""
    full = pd.read_parquet(ART / "feat_emb_deberta-v3-large_prompt.parquet")
    crop = pd.read_parquet(ART / "feat_emb_deberta-v3-large_prompt_crops.parquet")
    cols = [c for c in full.columns if c.startswith("e")]
    get = lambda df, split, names: df[df.split == split].set_index("filename").loc[names, cols].values.astype(np.float32)
    views = [get(crop, f"crop{k}", train.filename) for k in range(N_CROPS)] + [get(full, "train", train.filename)]
    te = get(full, "test", test.filename)
    X = np.concatenate(views)
    mu = X.mean(0)
    basis = np.linalg.svd(X - mu, full_matrices=False)[2][:dim].T
    sd = ((X - mu) @ basis).std(0)
    proj = lambda a: ((a - mu) @ basis / sd).astype(np.float32)
    return np.stack([proj(v) for v in views]), proj(te)


def batch(F, start, n, t=None):   # pad prefixes of n[i] frames from clip offsets start[i] -> (x, mask[, text])
    T = int(max(n))
    x = np.zeros((len(n), T, F.shape[1]), np.float32)
    m = np.zeros((len(n), T), np.float32)
    for i, (s, k) in enumerate(zip(start, n)):
        x[i, :k], m[i, :k] = F[s:s + k], 1
    out = (torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev))
    return out if t is None else out + (torch.from_numpy(t).to(dev),)


@torch.inference_mode()
def predict(model, F, start, n, t=None):
    model.eval()
    return np.concatenate([model.score(model(*batch(F, start[i:i + 64], n[i:i + 64], None if t is None else t[i:i + 64])))
                           .cpu().numpy() for i in range(0, len(n), 64)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--source", default="voxtral")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="more seeds -> name gets _s<n> (and _e<epochs> if changed)")
    ap.add_argument("--text", action="store_true", help="condition on the transcript embedding of the matching crop")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights used for prediction (0 = off)")
    ap.add_argument("--out", default="reg", choices=["reg", "bins", "corn"], help="one score, or a distribution over the labels")
    ap.add_argument("--oversample", type=int, default=1, help="clips scored <= 2 or >= 4.5 appear this often per epoch")
    ap.add_argument("--corr", type=float, default=0.0, help="weight of a (1 - batch Pearson) loss term")
    ap.add_argument("--seed0", type=int, default=0, help="first seed (another retrain)")
    ap.add_argument("--folds", default="gfold_s0", help="fold column; tfold_* reads folds_topic.csv (23_topic_folds.py)")
    args = ap.parse_args()
    z = np.load(ART / f"{args.source}_frames.npz", allow_pickle=True)
    name = "vp_attn" + ("" if args.source == "voxtral" else f"_{args.source}")
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    name += (f"_s{args.seeds}" if args.seeds != len(SEEDS) else "") + (f"_e{args.epochs}" if args.epochs != 40 else "")
    name += ("_text" if args.text else "") + (f"_do{args.dropout:g}" if args.dropout != 0.1 else "") + \
        (f"_ema{args.ema:g}" if args.ema else "")
    name += (f"_{args.out}" if args.out != "reg" else "") + (f"_os{args.oversample}" if args.oversample != 1 else "") + \
        (f"_corr{args.corr:g}" if args.corr else "") + (f"_r{args.seed0}" if args.seed0 else "") + \
        (f"_{args.folds}" if args.folds != "gfold_s0" else "")
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

    if args.text:   # views 0-3 = the standard crops, 4 = the full clip; view lengths in frames for the nearest match
        tview, tte = text_table(train, test)
        vlen = np.stack(crops + [train.lens.values])                          # (5, n)
        # ponytail: nearest view by length, so a training prefix can get a slightly longer transcript; validation and
        # test always pair matching audio and text. Restrict to views no longer than the prefix if this ever matters.
        nearest = lambda n, rows: np.abs(vlen[:, rows] - n[rows]).argmin(0)   # view index per row
    y = train.label.values.astype(np.float32)
    folds = fold_of(train.filename, args.folds) if not args.folds.startswith("tfold") else \
        pd.read_csv(ART / "folds_topic.csv").set_index("filename")[args.folds].loc[train.filename].values
    test_durs = pd.read_parquet(ART / "feat_text_prompt.parquet").query("split == 'test'").duration.values
    oof, te = np.zeros(len(train)), np.zeros(len(test))
    for seed in seeds:
        for k in range(5):
            torch.manual_seed(seed * 10 + k)
            rng = np.random.default_rng(seed * 10 + k)
            tr = np.where((folds != k) & (y > 0))[0]
            rows = tr if args.oversample == 1 else np.r_[tr, np.repeat(tr[(y[tr] <= 2) | (y[tr] >= 4.5)], args.oversample - 1)]
            model = Head(dropout=args.dropout, dt=tview.shape[2] if args.text else 0, kind=args.out).to(dev)
            model.init_bias(y[tr])
            ema = copy.deepcopy(model) if args.ema else None
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.epochs * int(np.ceil(len(rows) / BS)))
            for ep in range(args.epochs):
                model.train()
                order = rng.permutation(rows)
                n = to_frames(rng.choice(test_durs, len(train)) * SR)   # a random test-length prefix per clip
                for i in range(0, len(order), BS):
                    b = order[i:i + BS]
                    t = tview[nearest(n, b), b] if args.text else None
                    o, yb = model(*batch(F, train.start.values[b], n[b], t)), torch.from_numpy(y[b]).to(dev)
                    loss = corn_loss(o, torch.from_numpy(rank(y[b])).to(dev)) if args.out == "corn" else \
                        nn.functional.mse_loss(model.score(o), yb)
                    if args.corr:
                        loss = loss + args.corr * pearson_loss(model.score(o), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
                    if ema is not None:
                        with torch.no_grad():
                            for pe, pm in zip(ema.parameters(), model.parameters()):
                                pe.mul_(args.ema).add_(pm, alpha=1 - args.ema)
            final = ema if ema is not None else model
            va = np.where(folds == k)[0]
            p = np.mean([predict(final, F, train.start.values[va], c[va], tview[j, va] if args.text else None)
                         for j, c in enumerate(crops)], axis=0)
            oof[va] += p / len(seeds)
            te += predict(final, F, test.start.values, test.lens.values, tte if args.text else None) / (len(seeds) * 5)
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
