"""Step 13: rule-based grammar-error counts (LanguageTool) on the transcripts.

09_grammar_features.py measured errors with a neural corrector (T5 edit rate): one number that mixes
grammar with punctuation and style. LanguageTool is rule-based and labels every match with a category
(GRAMMAR, TYPOS, PUNCTUATION, STYLE, ...), so grammar errors can be counted on their own. Speech isn't
punctuated like writing, so categories, not the total, are the useful signal; the regressor decides
which ones matter. Rates are per 100 words, so clip length doesn't drive them.

Needs Java 17+ (set JAVA_HOME). language_tool_python downloads LanguageTool on first use.
Input : artifacts/transcripts_prompt[_crops].parquet (01_transcribe.py)
Output: artifacts/feat_languagetool[_crops].parquet (filename, split, label, lt_<category>, lt_all, n_words)
Usage : python src/13_languagetool_features.py [--crops] [--limit 20]
"""
import argparse
from collections import Counter

import language_tool_python
import pandas as pd
from tqdm import tqdm

from common import ART

# Categories with enough matches on speech transcripts to be worth a column; the rest go to lt_other.
CATEGORIES = ["GRAMMAR", "TYPOS", "PUNCTUATION", "STYLE", "CASING", "CONFUSED_WORDS", "REDUNDANCY",
              "COLLOCATIONS", "SEMANTICS", "MISC", "TYPOGRAPHY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", action="store_true", help="the transcripts of the test-length crops instead")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    args = ap.parse_args()

    tr = pd.read_parquet(ART / f"transcripts_prompt{'_crops' if args.crops else ''}.parquet")
    if args.limit:
        tr = tr.head(args.limit)
    tool = language_tool_python.LanguageTool("en-US")   # local server, nothing leaves the machine
    rows = []
    for text in tqdm(tr.text.fillna(""), mininterval=30):
        n = max(len(text.split()), 1)
        c = Counter(m.category for m in tool.check(text))
        rows.append({**{f"lt_{k.lower()}": 100 * c.pop(k, 0) / n for k in CATEGORIES},
                     "lt_other": 100 * sum(c.values()) / n, "n_words": n})
    tool.close()
    out = pd.concat([tr[["filename", "split", "label"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["lt_all"] = out.filter(like="lt_").sum(axis=1)
    name = "feat_languagetool" + ("_crops" if args.crops else "")
    out.to_parquet(ART / f"{name}.parquet", index=False)
    t = out[out.split == "train"]
    print(f"saved {name}: {len(out)} rows | r with label (train): "
          + ", ".join(f"{c} {t[c].corr(t.label):+.2f}" for c in t.filter(like="lt_").columns if t[c].std() > 0))


if __name__ == "__main__":
    main()
