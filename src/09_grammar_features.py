"""Step 9: direct grammar-error features from the Whisper transcripts.

The DeBERTa embeddings (step 4) mostly encode *meaning*, which is why the text models trail the
audio ones on a *grammar* score. These three small pretrained models measure grammar directly:
  * edit rate: a T5 grammar corrector rewrites each sentence; the share of words it changes
    is an estimate of errors per word (word-level diff, so punctuation fixes barely count)
  * acceptability: a RoBERTa classifier trained on CoLA (grammatical vs ungrammatical sentences)
    gives P(grammatical) per sentence -> mean, min and share of sentences below 0.5
  * perplexity: GPT-2's mean token loss over the transcript (fluent, well-formed text is less surprising)
All are rates or means, so they don't depend on clip length (test clips are shorter).
Silent clips (no sentences) get the column median; the score-0 gate in 08_blend.py handles them.

Input : artifacts/transcripts_<tag>.parquet
Output: artifacts/feat_grammar_<tag>.parquet (filename, split, label, g_*)
Usage : python src/09_grammar_features.py --tag prompt
"""
import argparse
import difflib
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          AutoTokenizer)

from common import ART, DTYPE

GEC, COLA, LM = "vennify/t5-base-grammar-correction", "textattack/roberta-base-CoLA", "gpt2"
BS = 32


def sentences(text):
    """Whisper's punctuation splits the transcript into sentences; very short fragments are skipped."""
    return [s for s in re.split(r"(?<=[.?!])\s+", text or "") if len(s.split()) >= 3]


def edit_rate(src, fixed):
    a, b = src.lower().split(), fixed.lower().split()
    return 1 - difflib.SequenceMatcher(None, a, b).ratio()


@torch.inference_mode()
def run_gec(sents):
    tok = AutoTokenizer.from_pretrained(GEC)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEC, dtype=DTYPE).to("cuda").eval()
    out = []
    for i in tqdm(range(0, len(sents), BS), desc="gec"):
        enc = tok(["grammar: " + s for s in sents[i:i + BS]], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to("cuda")
        gen = model.generate(**enc, max_new_tokens=128, num_beams=1)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return [edit_rate(s, f) for s, f in zip(sents, out)]


@torch.inference_mode()
def run_cola(sents):
    tok = AutoTokenizer.from_pretrained(COLA)
    model = AutoModelForSequenceClassification.from_pretrained(COLA, dtype=DTYPE).to("cuda").eval()
    out = []
    for i in tqdm(range(0, len(sents), BS), desc="cola"):
        enc = tok(sents[i:i + BS], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        out += model(**enc).logits.float().softmax(-1)[:, 1].tolist()   # label 1 = acceptable
    return out


@torch.inference_mode()
def run_lm(texts):
    tok = AutoTokenizer.from_pretrained(LM)
    model = AutoModelForCausalLM.from_pretrained(LM, dtype=DTYPE).to("cuda").eval()
    out = []
    for t in tqdm(texts, desc="gpt2"):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.to("cuda")
        out.append(model(ids, labels=ids).loss.item() if ids.shape[1] > 1 else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="prompt", help="which transcripts_<tag>.parquet to use")
    args = ap.parse_args()

    tr = pd.read_parquet(ART / f"transcripts_{args.tag}.parquet")
    per_clip = [sentences(t) for t in tr.text]
    flat = [s for ss in per_clip for s in ss]
    owner = np.repeat(np.arange(len(tr)), [len(ss) for ss in per_clip])   # clip index of each sentence

    s = pd.DataFrame({"clip": owner, "edit": run_gec(flat), "cola": run_cola(flat)})
    g = s.groupby("clip").agg(g_edit_mean=("edit", "mean"), g_edit_any=("edit", lambda x: (x > 0).mean()),
                              g_cola_mean=("cola", "mean"), g_cola_min=("cola", "min"),
                              g_cola_bad=("cola", lambda x: (x < 0.5).mean()))
    feats = g.reindex(range(len(tr)))
    feats["g_lm_loss"] = run_lm(tr.text.fillna("").tolist())
    feats = feats.fillna(feats.median())

    out = pd.concat([tr[["filename", "split", "label"]], feats.reset_index(drop=True)], axis=1)
    out.to_parquet(ART / f"feat_grammar_{args.tag}.parquet", index=False)
    print(out.drop(columns=["filename", "split"]).corr(numeric_only=True)["label"].round(3).sort_values())


if __name__ == "__main__":
    main()
