"""Step 18: task-adaptive pretraining (TAPT) of DeBERTa-v3-large on our own transcripts, before 06_finetune_text.py.

The test clips answer questions the training clips rarely cover (topic overlap 34% vs 70% within train), so the text
model meets unfamiliar vocabulary at test time. Continuing masked-language-model training on the transcripts of every
clip (train and test, text only, no labels) adapts the encoder to that vocabulary and to transcribed speech before the
supervised fine-tune (Gururangan et al. 2020, "Don't Stop Pretraining").

DeBERTa-v3's released checkpoint has no MLM head, so the head starts random; it learns within the first epoch.
Output: <out>/ (the adapted encoder + tokenizer), then: 06_finetune_text.py --model <out> ...
Usage : python src/18_tapt.py --out /kaggle/working/tapt_deberta [--epochs 8]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from common import AMP, ART

MODEL, BS, MAX_LEN, LR = "microsoft/deberta-v3-large", 4, 256, 2e-5   # batch 4: the 128k-word output layer fills a T4 at 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    texts = pd.read_parquet(ART / "transcripts_prompt.parquet").query("label != 0").text.fillna("").tolist()   # not the noise clips
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(texts, truncation=True, max_length=MAX_LEN)
    rows = [{"input_ids": i, "attention_mask": m} for i, m in zip(enc.input_ids, enc.attention_mask)]
    collate = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32).to("cuda")   # fp32 master weights (fp16 NaNs)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(rows) / BS))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=AMP)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    for ep in range(args.epochs):
        model.train()
        losses = []
        order = rng.permutation(len(rows))
        for i in range(0, len(rows), BS):
            b = {k: v.to("cuda") for k, v in collate([rows[j] for j in order[i:i + BS]]).items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = model(**b).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        print(f"TAPT epoch {ep}: MLM loss {np.mean(losses):.3f}", flush=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
