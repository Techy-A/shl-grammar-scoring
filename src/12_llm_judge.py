"""Step 12: LLM judge. An instruction-tuned LLM reads each transcript with the scoring rubric.

The text models so far (02, 04, 09) describe a transcript: statistics, embeddings, error rates from a small
corrector. None of them is asked the actual question. Here an LLM is given the rubric and one transcript and
answers with a single digit 1-5. Instead of generating, we read the next-token probabilities of "1".."5"
at the answer position, which gives a continuous score (their weighted mean) plus the full distribution.
The idea of a rubric-reading LLM judge comes from another participant's public solution; this implementation
and its crop training are ours.

Input : artifacts/transcripts_prompt[_crops].parquet (01_transcribe.py)
Output: artifacts/feat_llm_judge[_crops].parquet (filename, split, label, j_mean, j_p1..j_p5)
Usage : python src/12_llm_judge.py [--crops] [--model Qwen/Qwen3-8B] [--limit 5]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ART, DTYPE

RUBRIC = """You are an examiner scoring the GRAMMAR of spontaneous spoken English.
The text is an automatic transcript of a 45-60 second spoken answer. Ignore punctuation, capitalisation and
fillers (um, uh); judge sentence structure, syntax, and grammatical accuracy and complexity.

1: Struggles with sentence structure and syntax; limited control of simple grammatical structures.
2: Simple structures with consistent basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of either grammar or sentence structure, with errors in the other.
4: Strong control of sentence structure and grammar; occasional minor errors that don't impede understanding.
5: High grammatical accuracy; complex structures handled well; self-corrects when necessary.

Reply with a single digit from 1 to 5."""


@torch.inference_mode()
def judge(text, tok, model, digits):
    msgs = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": f"Transcript:\n{text}"}]
    # enable_thinking=False: Qwen3 answers directly instead of reasoning first (ignored by other templates).
    x = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
    logits = model(**x.to(model.device)).logits[0, -1, digits].float()
    return torch.softmax(logits, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--crops", action="store_true", help="judge the transcripts of the test-length crops instead")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = ap.parse_args()

    tr = pd.read_parquet(ART / f"transcripts_prompt{'_crops' if args.crops else ''}.parquet")
    if args.limit:
        tr = tr.head(args.limit)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE, device_map="auto").eval()   # spans both T4s
    digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]

    p = np.stack([judge(t, tok, model, digits) for t in tqdm(tr.text, mininterval=60)])
    assert np.isfinite(p).all(), "non-finite probabilities (fp16 overflow?)"
    out = tr[["filename", "split", "label"]].assign(j_mean=p @ np.arange(1, 6),
                                                    **{f"j_p{d}": p[:, d - 1] for d in range(1, 6)})
    name = "feat_llm_judge" + ("_crops" if args.crops else "")
    out.to_parquet(ART / f"{name}.parquet", index=False)
    print(f"saved {name}: {len(out)} rows | j_mean {out.j_mean.mean():.2f} +- {out.j_mean.std():.2f}"
          + (f" | r with label (train) {out[out.split == 'train'][['j_mean', 'label']].corr().iloc[0, 1]:.3f}"
             if (out.split == "train").sum() > 2 else ""))


if __name__ == "__main__":
    main()
