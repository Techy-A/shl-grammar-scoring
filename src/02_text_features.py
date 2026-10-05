"""Step 2: hand-crafted features from the Whisper transcripts.

Every feature is a *rate* or *ratio* rather than a raw count, because test clips are shorter
than train clips (median 45 s vs 60 s) and raw counts would shift with clip length.

Input : artifacts/transcripts_<tag>.parquet   (from 01_transcribe.py)
Output: artifacts/feat_text_<tag>.parquet     (one row per clip; ASR confidence columns kept)

Usage: python src/02_text_features.py --tag prompt
"""
import argparse
import re

import numpy as np
import pandas as pd

from common import ART

FILLERS = {"uh", "um", "umm", "uhm", "hmm", "mm", "er", "ah", "eh"}


def mattr(words, window=50):
    """Moving-average type-token ratio: vocabulary richness that doesn't fall as texts get longer
    (plain TTR does, which would penalise the longer train clips)."""
    if len(words) <= window:
        return len(set(words)) / max(len(words), 1)
    return np.mean([len(set(words[i:i + window])) / window for i in range(len(words) - window + 1)])


def text_features(text, duration):
    words = re.findall(r"[a-z']+", text.lower())
    n = max(len(words), 1)
    # Sentences as Whisper punctuates them; a rough proxy for utterance structure.
    sents = [s for s in re.split(r"[.?!]+", text) if s.strip()]
    return dict(
        words_per_sec=len(words) / duration,                         # speaking rate
        filler_rate=sum(w in FILLERS for w in words) / n,            # hesitation
        repeat_rate=sum(a == b for a, b in zip(words, words[1:])) / n,   # "the, the" self-repairs
        mattr=mattr(words),                                          # vocabulary richness
        mean_word_len=np.mean([len(w) for w in words]) if words else 0.0,
        mean_sent_len=n / max(len(sents), 1),                        # longer sentences ~ more complex grammar
        comma_rate=text.count(",") / n,                              # Whisper inserts commas at pauses
        ellipsis_rate=text.count("...") / n,                         # trailing off / unclear speech
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="prompt", help="which transcripts_<tag>.parquet to use")
    args = ap.parse_args()

    tr = pd.read_parquet(ART / f"transcripts_{args.tag}.parquet")
    feats = pd.DataFrame([text_features(t, d) for t, d in zip(tr.text, tr.duration)])
    # Keep ids/label/duration + ASR confidence columns from step 1 alongside the new features.
    keep = ["filename", "split", "label", "duration"] + [c for c in tr.columns if c.startswith("asr_")]
    out = pd.concat([tr[keep], feats], axis=1)
    # asr_n_tok is a raw count -> turn it into a rate like everything else.
    out["asr_n_tok"] = out["asr_n_tok"] / out["duration"]
    out.to_parquet(ART / f"feat_text_{args.tag}.parquet", index=False)
    print(out.drop(columns=["filename", "split"]).corr(numeric_only=True)["label"].round(3).sort_values())


if __name__ == "__main__":
    main()
