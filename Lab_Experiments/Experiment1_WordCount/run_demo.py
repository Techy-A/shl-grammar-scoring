#!/usr/bin/env python3
"""Run the word count step by step: input -> map -> shuffle & sort -> combine -> reduce.
Works on Windows, macOS and Linux. Writes output/wordcount.txt and output/top10.txt.

Usage:  python run_demo.py              (readable trace)
        python run_demo.py --markdown   (trace as Markdown, used by build_docx.py)
"""
import collections
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = "--markdown" in sys.argv


def run(script, lines):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, script)], input="".join(l + "\n" for l in lines),
                          capture_output=True, text=True, check=True)
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


with open(os.path.join(ROOT, "data", "input.txt")) as f:
    text = f.read().splitlines()

show("Step 1: Input file", text, "cat data/input.txt")
mapped = run("mapper.py", text)
show("Step 2: Mapper output (word TAB 1)", mapped, "python mapper.py < data/input.txt", 8)
shuffled = sorted(mapped)
show("Step 3: Shuffle and Sort (equal words become adjacent)", shuffled, "... | sort", 8)
combined = run("reducer.py", shuffled)
show("Step 4: Combiner output (local partial counts)", combined, "... | sort | python reducer.py   (as combiner)", 8)
reduced = run("reducer.py", sorted(combined))
if MD:  # the full result as a compact 3-pair table instead of one long column
    show("Step 5: Reducer output (final count for every word)", reduced, "... | sort | python reducer.py", 6)
    cols = 3
    rows = -(-len(reduced) // cols)
    cells = [l.split("\t") for l in reduced] + [["", ""]] * (rows * cols - len(reduced))
    print("Complete reducer output (" + str(len(reduced)) + " distinct words, alphabetical, read down each column):\n")
    print("|" + " Word | Count |" * cols)
    print("|" + "------|------|" * cols)
    for r in range(rows):
        print("|" + "".join(f" {cells[c * rows + r][0]} | {cells[c * rows + r][1]} |" for c in range(cols)))
    print()
else:
    show("Step 5: Reducer output (final count for every word)", reduced, "... | sort | python reducer.py")
top = sorted(reduced, key=lambda l: (-int(l.split("\t")[1]), l))[:10]
show("Step 6: Top 10 most frequent words", top, "sort -k2,2nr output/wordcount.txt | head -10")

# Check against a direct count with collections.Counter (no MapReduce).
direct = collections.Counter(w for line in text for w in re.findall(r"[a-z0-9']+", line.lower()))
ok = dict((w, int(c)) for w, c in (l.split("\t") for l in reduced)) == dict(direct)
summary = [f"Total words (mapper records) : {len(mapped)}",
           f"Distinct words (reducer keys): {len(reduced)}",
           f"Records after combiner      : {len(combined)}",
           f"Matches a direct Counter()  : {'YES' if ok else 'NO'}"]
show("Step 7: Summary and verification", summary, "python run_demo.py")

os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
for name, lines in (("wordcount.txt", reduced), ("top10.txt", top)):
    with open(os.path.join(ROOT, "output", name), "w") as f:
        f.writelines(l + "\n" for l in lines)
sys.exit(0 if ok else 1)
