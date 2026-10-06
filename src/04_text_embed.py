"""Step 4: frozen transformer embeddings of the transcripts.

The step-2 features count surface signals (fillers, repeats, speaking rate). A pretrained language
model also "knows" whether a sentence is well-formed, so its hidden states should carry grammar
quality that hand-crafted counts miss. No fine-tuning here (that's 06_finetune_text.py): we only pool hidden states
and fit a Ridge on top in 03_train.py, which is cheap and hard to overfit on 769 clips.

Pooling: mean over tokens (ignoring padding), averaged over the last 4 layers. Middle-to-late
layers tend to carry more syntactic information than the very last one.

Input : artifacts/transcripts_<tag>.parquet
Output: artifacts/feat_emb_<short model name>_<tag>.parquet  (filename, split, label, e0..eN)

Decoder LLMs (--model Qwen/Qwen3-4B --layers 12 25): their middle layers, not the last ones, carry the most
transferable features, so pass the layer slice to pool.

Usage: python src/04_text_embed.py --tag prompt
       python src/04_text_embed.py --tag prompt --model Qwen/Qwen3-4B --layers 12 25
"""
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import ART, DTYPE

BATCH = 8


@torch.inference_mode()
def embed(texts, tok, model, layers):
    out = []
    for i in tqdm(range(0, len(texts), BATCH)):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to("cuda")
        hidden = model(**enc, output_hidden_states=True).hidden_states[layers]
        h = torch.stack(hidden).float().mean(0)                  # (batch, tokens, dim), avg over the layers
        mask = enc.attention_mask.unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())   # mean over real tokens; an empty text (no tokens) -> 0
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="prompt", help="which transcripts_<tag>.parquet to embed")
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--layers", type=int, nargs=2, default=[-4, None], help="hidden-state slice [start, stop); default: last 4")
    args = ap.parse_args()

    tr = pd.read_parquet(ART / f"transcripts_{args.tag}.parquet")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=DTYPE).to("cuda").eval()

    # Empty transcripts (silent clips) still need a row; an empty string embeds to the [CLS]/[SEP] mean.
    emb = embed(tr.text.fillna("").tolist(), tok, model, slice(*args.layers))
    assert np.isfinite(emb).all(), "non-finite embeddings (fp16 overflow?)"
    feats = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    out = pd.concat([tr[["filename", "split", "label"]], feats], axis=1)
    name = f"feat_emb_{args.model.split('/')[-1]}_{args.tag}"
    out.to_parquet(ART / f"{name}.parquet", index=False)
    print(f"saved {name}: {emb.shape}, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
