"""Step 4: frozen transformer embeddings of the transcripts.

The step-2 features count surface signals (fillers, repeats, speaking rate). A pretrained language
model also "knows" whether a sentence is well-formed, so its hidden states should carry grammar
quality that hand-crafted counts miss. No fine-tuning here (that's 06_finetune_text.py): we only pool hidden states
and fit a Ridge on top in 03_train.py, which is cheap and hard to overfit on 769 clips.

Pooling: mean over tokens (ignoring padding), averaged over the last 4 layers. Middle-to-late
layers tend to carry more syntactic information than the very last one.

Input : artifacts/transcripts_<tag>.parquet
Output: artifacts/feat_emb_<short model name>_<tag>.parquet  (filename, split, label, e0..eN)

Usage: python src/04_text_embed.py --tag prompt
"""
import argparse

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import ART, DTYPE

LAST_LAYERS = 4
MODEL = "microsoft/deberta-v3-large"
BATCH = 8


@torch.inference_mode()
def embed(texts, tok, model):
    out = []
    for i in tqdm(range(0, len(texts), BATCH)):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to("cuda")
        hidden = model(**enc, output_hidden_states=True).hidden_states[-LAST_LAYERS:]
        h = torch.stack(hidden).float().mean(0)                  # (batch, tokens, dim), avg of last layers
        mask = enc.attention_mask.unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu())     # mean over real tokens only
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="prompt", help="which transcripts_<tag>.parquet to embed")
    args = ap.parse_args()

    tr = pd.read_parquet(ART / f"transcripts_{args.tag}.parquet")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=DTYPE).to("cuda").eval()

    # Empty transcripts (silent clips) still need a row; an empty string embeds to the [CLS]/[SEP] mean.
    emb = embed(tr.text.fillna("").tolist(), tok, model)
    feats = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    out = pd.concat([tr[["filename", "split", "label"]], feats], axis=1)
    name = f"feat_emb_{MODEL.split('/')[-1]}_{args.tag}"
    out.to_parquet(ART / f"{name}.parquet", index=False)
    print(f"saved {name}: {emb.shape}, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
