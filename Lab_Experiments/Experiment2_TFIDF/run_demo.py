#!/usr/bin/env python3
"""Run the TF-IDF pipeline (3 chained MapReduce jobs) step by step on any OS.
Writes output/job1_counts.txt, output/job2_tf.txt, output/tfidf.txt and output/top_terms.txt.

Usage:  python run_demo.py              (readable trace)
        python run_demo.py --markdown   (trace as Markdown, used by build_docx.py)
"""
import collections
import glob
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = "--markdown" in sys.argv
DOCS = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", "*.txt")))
D = len(DOCS)


def run(script, lines, env=None):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, script)], input="".join(l + "\n" for l in lines),
                          capture_output=True, text=True, check=True, env={**os.environ, **(env or {})})
    return proc.stdout.splitlines()


def show(title, lines, command, limit=None):
    shown = lines if limit is None or len(lines) <= limit + 2 else lines[:limit]
    body = [f"$ {command}"] + [l.replace("\t", "    ") for l in shown]
    if len(shown) < len(lines):
        body.append(f"... ({len(lines) - len(shown)} more lines)")
    if MD:
        print(f"**{title}** ({len(lines)} lines)\n\n```text\n" + "\n".join(body) + "\n```\n")
    else:
        print(f"\n--- {title}  [{len(lines)} lines]\n" + "\n".join(body))


def heading(text):
    print(f"\n## {text}\n" if MD else "\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def save(name, lines):
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    with open(os.path.join(ROOT, "output", name), "w") as f:
        f.writelines(l + "\n" for l in lines)


heading("Input corpus")
for path in DOCS:
    with open(path) as f:
        show(os.path.basename(path), f.read().splitlines(), f"cat data/corpus/{os.path.basename(path)}")

heading("Job 1: term count n for every (word, document)")
mapped = []
for path in DOCS:  # Hadoop runs one map task per file and sets mapreduce_map_input_file
    with open(path) as f:
        mapped += run("mapper1.py", f.read().splitlines(), {"mapreduce_map_input_file": path})
show("Mapper 1 output (word@doc TAB 1)", mapped, "python mapper1.py < each document", 6)
shuffled = sorted(mapped)
show("Shuffle and Sort", shuffled, "... | sort", 6)
counts = run("reducer1.py", shuffled)
show("Reducer 1 output (word@doc TAB n)", counts, "... | sort | python reducer1.py", 6)
save("job1_counts.txt", counts)

heading("Job 2: document length N, giving n and N for every (word, document)")
m2 = sorted(run("mapper2.py", counts))
show("Mapper 2 output after Shuffle and Sort (doc TAB word,n)", m2, "python mapper2.py < job1 | sort", 6)
tf = run("reducer2.py", m2)
show("Reducer 2 output (word@doc TAB n,N)", tf, "... | python reducer2.py", 6)
save("job2_tf.txt", tf)

heading(f"Job 3: document frequency df, IDF and TF-IDF (D = {D} documents)")
m3 = sorted(run("mapper3.py", tf))
show("Mapper 3 output after Shuffle and Sort (word TAB doc,n,N)", m3, "python mapper3.py < job2 | sort", 6)
result = run("reducer3.py", m3, {"TOTAL_DOCS": str(D)})
show("Reducer 3 output (word@doc TAB tf, idf, tfidf)", result,
     f"... | TOTAL_DOCS={D} python reducer3.py", 6)
save("tfidf.txt", result)

# How common words are pushed to zero: show 'the' and 'hadoop' in full.
for word in ("the", "hadoop"):
    show(f"Example: every line for '{word}'", [l for l in result if l.startswith(word + "@")],
         f"grep '^{word}@' output/tfidf.txt")

# Final answer: the 3 highest TF-IDF terms of each document.
rows = []
for line in result:
    key, value = line.split("\t")
    word, doc = key.rsplit("@", 1)
    fields = dict(kv.split("=") for kv in value.split(","))
    rows.append((doc, -float(fields["tfidf"]), word, fields))
rows.sort()
top = []
for doc in sorted({r[0] for r in rows}):
    top += [r for r in rows if r[0] == doc][:3]
save("top_terms.txt", [f"{d}\t{w}\ttfidf={-s:.4f}" for d, s, w, _ in top])
if MD:
    print("\n## Result: top 3 terms of each document\n")
    print("| Document | Rank | Word | tf | idf | TF-IDF |")
    print("|------|----|------|------|------|------|")
    for i, (doc, s, word, f) in enumerate(top):
        print(f"| {doc} | {i % 3 + 1} | **{word}** | {f['tf']} | {f['idf']} | {f['tfidf']} |")
    print()
else:
    heading("Result: top 3 terms of each document")
    for doc, s, word, f in top:
        print(f"{doc}  {word:<12} tf={f['tf']}  idf={f['idf']}  tfidf={f['tfidf']}")

# Verification: recompute TF-IDF directly in Python, without MapReduce.
docs = {}
for path in DOCS:
    with open(path) as f:
        docs[os.path.splitext(os.path.basename(path))[0]] = re.findall(r"[a-z0-9']+", f.read().lower())
df = collections.Counter(w for words in docs.values() for w in set(words))
direct = {f"{w}@{d}": round(c / len(words) * math.log10(D / df[w]), 4)
          for d, words in docs.items() for w, c in collections.Counter(words).items()}
mr = {l.split("\t")[0]: float(l.split("tfidf=")[1]) for l in result}
ok = mr.keys() == direct.keys() and all(abs(mr[k] - direct[k]) < 1e-4 for k in mr)
summary = [f"Documents (D)                       : {D}",
           f"Words in corpus (mapper 1 records)  : {len(mapped)}",
           f"(word, document) pairs             : {len(counts)}",
           f"Distinct words                     : {len(df)}",
           f"Matches a direct Python computation: {'YES' if ok else 'NO'}"]
show("Summary and verification", summary, "python run_demo.py")
sys.exit(0 if ok else 1)
