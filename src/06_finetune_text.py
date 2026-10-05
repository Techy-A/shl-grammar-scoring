"""Step 6: fine-tune DeBERTa-v3 to regress the score directly from the transcript.

04_text_embed.py keeps DeBERTa frozen and fit a Ridge on its pooled states. Fine-tuning lets the whole network
adapt to "what makes a 3.5 differ from a 4.5", which is usually much stronger on text-scoring tasks.

Design choices (small data: about 615 training texts per fold):
* Mean-pooled last hidden state -> linear head, MSE loss on label / 5 (targets in [0, 1]).
* A fixed number of epochs, no early stopping on the validation fold, so the OOF predictions stay
  honest and can be blended with the other steps' OOF in 08_blend.py.
* Folds = fold_s0 from artifacts/folds.csv (the folds every other step uses).
* Separate learning rates: small for the pretrained body, larger for the freshly initialised head.

Two modes:
  train : python src/06_finetune_text.py --folds 0 1 2    -> artifacts/s4_fold{k}.parquet
          (on Kaggle, 2 processes split the folds across the two T4s)
  merge : python src/06_finetune_text.py --merge          -> artifacts/pred_<name>.parquet + results.csv row
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import ART, DTYPE, merge_folds

# Mixed precision only where fp16 works (Kaggle T4); the local GTX 1650 trains in plain fp32.
AMP = DTYPE == torch.float16


class Regressor(nn.Module):
    def __init__(self, name):
        super().__init__()
        # fp32 master weights: transformers 5 otherwise keeps the checkpoint's dtype (fp16 for
        # DeBERTa-v3), and pure-fp16 training overflows to NaN. Autocast still runs fp16 math on T4.
        self.body = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = nn.Linear(self.body.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)   # mean over real tokens


def batches(enc, idx, bs, y=None, shuffle=False, rng=None):
    idx = rng.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        out = {k: v[b].to("cuda") for k, v in enc.items()}
        yield out, (torch.tensor(y[b], dtype=torch.float32, device="cuda") if y is not None else None)


@torch.inference_mode()
def predict(model, enc, idx, bs):
    model.eval()
    preds = []
    for x, _ in batches(enc, idx, bs):
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            preds.append(model(**x).float().cpu().numpy())
    return np.concatenate(preds) * 5


def train_fold(k, df, folds, enc, args):
    torch.manual_seed(k)
    rng = np.random.default_rng(k)
    is_tr = (df.split == "train").values
    tr_idx = np.where(is_tr & (folds != k))[0]
    va_idx = np.where(is_tr & (folds == k))[0]
    te_idx = np.where(~is_tr)[0]
    y = (df.label.fillna(0).values / 5).astype(np.float32)

    model = Regressor(args.model).to("cuda")
    model.body.gradient_checkpointing_enable()   # fits large in T4 memory
    opt = torch.optim.AdamW([{"params": model.body.parameters(), "lr": args.lr},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(tr_idx) / args.bs))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=AMP)   # loss scaling for fp16; weights stay fp32

    for ep in range(args.epochs):
        model.train()
        for x, yb in batches(enc, tr_idx, args.bs, y, shuffle=True, rng=rng):
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = nn.functional.mse_loss(model(**x).float(), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        # Logged only for monitoring; not used to pick an epoch (keeps the OOF honest).
        p = np.clip(predict(model, enc, va_idx, args.bs * 2), 0, 5)
        print(f"fold {k} epoch {ep}: val RMSE {np.sqrt(np.mean((p - df.label.values[va_idx]) ** 2)):.4f}", flush=True)

    idx = np.r_[va_idx, te_idx]
    pd.DataFrame({"filename": df.filename.values[idx], "split": df.split.values[idx],
                  "pred": predict(model, enc, idx, args.bs * 2)}).to_parquet(ART / f"s4_fold{k}.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--tag", default="prompt", help="transcripts_<tag>.parquet")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--name", default="s4_deberta_ft")
    args = ap.parse_args()
    if args.merge:
        return merge_folds("s4", args.name, args.model)

    df = pd.read_parquet(ART / f"transcripts_{args.tag}.parquet")
    folds = df[["filename"]].merge(pd.read_csv(ART / "folds.csv"), how="left")["fold_s0"].fillna(-1).values
    tok = AutoTokenizer.from_pretrained(args.model)
    t = tok(df.text.tolist(), padding="max_length", truncation=True, max_length=args.max_len, return_tensors="pt")
    enc = {"input_ids": t.input_ids, "attention_mask": t.attention_mask}   # DeBERTa-v3 ignores token_type_ids
    for k in args.folds:
        train_fold(k, df, folds, enc, args)


if __name__ == "__main__":
    main()
