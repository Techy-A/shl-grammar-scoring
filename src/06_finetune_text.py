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
  --lora: a decoder LLM (e.g. --model Qwen/Qwen2.5-7B-Instruct) in 4-bit with LoRA adapters instead of full fine-tuning,
  pooled at the last real token (a decoder's summary position); an LLM's knowledge of grammar may learn more from
  ~600 texts than DeBERTa does.
  --crops (train on the test-length crop transcripts too, validate on them) and --drop-zero (train without the
  score-0 clips) add _crops / _nozero to the run name; --seed n adds _seed<n>; pass the same flags to --merge.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from common import AMP, ART, DTYPE, finetune_fold, fold_of, merge_folds

MODEL, TAG, NAME = "microsoft/deberta-v3-large", "prompt", "s4_deberta_ft"
EPOCHS, LR, BS, MAX_LEN = 4, 1e-5, 8, 256
LORA_LR = 1e-4   # LoRA adapters start at zero and need a larger step than full fine-tuning


class Regressor(nn.Module):
    def __init__(self, name, lora=False):
        super().__init__()
        self.lora = lora
        if lora:   # 4-bit frozen body + trainable LoRA adapters (fp32) on every linear layer
            from peft import LoraConfig, get_peft_model
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE)
            body = AutoModel.from_pretrained(name, dtype=DTYPE, quantization_config=q, device_map={"": 0})
            body.enable_input_require_grads()   # gradient checkpointing needs grads flowing from the frozen embeddings
            self.body = get_peft_model(body, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear"))
        else:
            # fp32 master weights: transformers 5 otherwise keeps the checkpoint's dtype (fp16 for
            # DeBERTa-v3), and pure-fp16 training overflows to NaN. Autocast still runs fp16 math on T4.
            self.body = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = nn.Linear(self.body.config.hidden_size, 1)

    def forward(self, x):   # x: dict with input_ids / attention_mask (right-padded)
        h = self.body(**x).last_hidden_state.float()
        if self.lora:   # last real token: a decoder's mean also averages its huge first-token activations
            return self.head(h[torch.arange(len(h)), x["attention_mask"].sum(1) - 1]).squeeze(-1)
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
    model = Regressor(MODEL, LORA)
    (model.head if LORA else model).to("cuda")   # a 4-bit body is placed on the GPU when loaded and can't be moved
    model.body.gradient_checkpointing_enable()   # fits large in T4 memory
    # Small learning rate for the pretrained body (or the LoRA adapters), larger for the freshly initialised head.
    body = [p for p in model.body.parameters() if p.requires_grad]
    return model, [{"params": body, "lr": LORA_LR if LORA else LR}, {"params": model.head.parameters(), "lr": 1e-3}]


def main():
    global MODEL, LORA   # --model / --lora replace them for make_model() and the tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--crops", action="store_true",
                    help="also train on the test-length crop transcripts, validate on them (as 03_train.py --crops)")
    ap.add_argument("--drop-zero", action="store_true",
                    help="train without the score-0 clips (noise recordings with invented fluent transcripts)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--model", default=MODEL, help="another Hugging Face encoder, e.g. roberta-large")
    ap.add_argument("--tag", default=TAG, help="transcripts_<tag>[_crops].parquet, e.g. phi4 (adds _<tag> to the name)")
    ap.add_argument("--lora", action="store_true", help="4-bit decoder LLM + LoRA adapters, last-token pooling")
    ap.add_argument("--seed", type=int, default=0, help="seed > 0: another run of the same recipe (adds _seed<n>)")
    args = ap.parse_args()
    other = args.model != MODEL   # a non-default encoder gets its own run name, e.g. s4_roberta-large_ft
    MODEL, LORA, short = args.model, args.lora, args.model.split("/")[-1]
    prefix = "s4" + "c" * args.crops + "z" * args.drop_zero + f"_{short}" * other
    name = (f"s4_{short}_ft" if other else NAME) + "_crops" * args.crops + "_nozero" * args.drop_zero
    if args.tag != TAG:
        prefix, name = prefix + f"_{args.tag}", name + f"_{args.tag}"
    if args.seed:   # seed runs are averaged later; the fine-tune alone is noisy (one run's OOF RMSE varies by fold luck)
        prefix, name = prefix + f"_seed{args.seed}", name + f"_seed{args.seed}"
    if args.merge:
        return merge_folds(prefix, name, MODEL)

    df = pd.read_parquet(ART / f"transcripts_{args.tag}.parquet")
    if args.crops:
        df = pd.concat([df, pd.read_parquet(ART / f"transcripts_{args.tag}_crops.parquet")], ignore_index=True)
    folds = fold_of(df.filename, "gfold_s0")
    t = AutoTokenizer.from_pretrained(MODEL, padding_side="right")(df.text.fillna("").tolist(), padding="max_length", truncation=True,
                                             max_length=MAX_LEN, return_tensors="pt")
    enc = {"input_ids": t.input_ids, "attention_mask": t.attention_mask}   # DeBERTa-v3 ignores token_type_ids

    def train_batches(idx, y, rng):
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            yield {k: v[b].to("cuda") for k, v in enc.items()}, torch.from_numpy(y[b]).cuda()

    is_test, is_crop = (df.split == "test").values, df.split.str.startswith("crop").values
    keep = (df.label.fillna(0).values > 0) if args.drop_zero else np.ones(len(df), bool)

    def rows(k):   # train: full clips (+ crops) of the other folds; val: fold k (its crops with --crops); test: full
        val = (is_crop if args.crops else ~is_test) & (folds == k)
        return np.where(~is_test & (folds != k) & keep)[0], np.where(val & ~is_test)[0], np.where(is_test)[0]

    def save(k):   # validation crops -> mean per clip, so the OOF is one row per train clip (as 03_train.py --crops)
        def out(model):
            _, va, te = rows(k)
            v = pd.DataFrame({"filename": df.filename.values[va], "pred": predict(model, enc, va)})
            v = v.groupby("filename", as_index=False).pred.mean().assign(split="train")
            t_ = pd.DataFrame({"filename": df.filename.values[te], "split": "test", "pred": predict(model, enc, te)})
            pd.concat([v, t_])[["filename", "split", "pred"]].to_parquet(ART / f"{prefix}_fold{k}.parquet", index=False)
        return out

    custom = args.crops or args.drop_zero
    for k in args.folds:
        finetune_fold(k, df, folds, make_model, train_batches, lambda m, idx: predict(m, enc, idx),
                      args.epochs, BS, prefix, rows=rows if custom else None, save=save(k) if custom else None,
                      seed=args.seed)


if __name__ == "__main__":
    main()
