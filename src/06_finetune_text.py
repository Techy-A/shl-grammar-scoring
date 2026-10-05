"""Step 6: fine-tune DeBERTa-v3 to regress the score directly from the transcript.

04_text_embed.py keeps DeBERTa frozen and fit a Ridge on its pooled states. Fine-tuning lets the whole network
adapt to "what makes a 3.5 differ from a 4.5", which is usually much stronger on text-scoring tasks.

Design choices (small data: about 615 training texts per fold):
* Mean-pooled last hidden state -> linear head, MSE loss on label / 5 (targets in [0, 1]).
* A fixed number of epochs, no early stopping on the validation fold, so the OOF predictions stay
  honest and can be blended with the other steps' OOF in 08_blend.py.
* Folds = gfold_s0 from artifacts/folds.csv (speaker-grouped, like every other step).
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
from transformers import AutoModel, AutoTokenizer

from common import AMP, ART, finetune_fold, merge_folds

MODEL, TAG, NAME = "microsoft/deberta-v3-large", "prompt", "s4_deberta_ft"
EPOCHS, LR, BS, MAX_LEN = 4, 1e-5, 8, 256


class Regressor(nn.Module):
    def __init__(self, name):
        super().__init__()
        # fp32 master weights: transformers 5 otherwise keeps the checkpoint's dtype (fp16 for
        # DeBERTa-v3), and pure-fp16 training overflows to NaN. Autocast still runs fp16 math on T4.
        self.body = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = nn.Linear(self.body.config.hidden_size, 1)

    def forward(self, x):   # x: dict with input_ids / attention_mask
        h = self.body(**x).last_hidden_state
        m = x["attention_mask"].unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)   # mean over real tokens


@torch.inference_mode()
def predict(model, enc, idx):
    model.eval()
    preds = []
    for i in range(0, len(idx), BS * 2):
        x = {k: v[idx[i:i + BS * 2]].to("cuda") for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            preds.append(model(x).float().cpu().numpy())
    return np.concatenate(preds) * 5


def make_model():
    model = Regressor(MODEL).to("cuda")
    model.body.gradient_checkpointing_enable()   # fits large in T4 memory
    # Small learning rate for the pretrained body, larger for the freshly initialised head.
    return model, [{"params": model.body.parameters(), "lr": LR}, {"params": model.head.parameters(), "lr": 1e-3}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        return merge_folds("s4", NAME, MODEL)

    df = pd.read_parquet(ART / f"transcripts_{TAG}.parquet")
    folds = df[["filename"]].merge(pd.read_csv(ART / "folds.csv"), how="left")["gfold_s0"].fillna(-1).values
    t = AutoTokenizer.from_pretrained(MODEL)(df.text.tolist(), padding="max_length", truncation=True,
                                             max_length=MAX_LEN, return_tensors="pt")
    enc = {"input_ids": t.input_ids, "attention_mask": t.attention_mask}   # DeBERTa-v3 ignores token_type_ids

    def train_batches(idx, y, rng):
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            yield {k: v[b].to("cuda") for k, v in enc.items()}, torch.from_numpy(y[b]).cuda()

    for k in args.folds:
        finetune_fold(k, df, folds, make_model, train_batches, lambda m, idx: predict(m, enc, idx),
                      EPOCHS, BS, "s4")


if __name__ == "__main__":
    main()
