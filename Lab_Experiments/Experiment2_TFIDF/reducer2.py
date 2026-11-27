#!/usr/bin/env python3
"""Job 2 reducer: N = total number of words in the document.
In : doc \t word,n   (sorted by doc)
Out: word@doc \t n,N
"""
import sys


def emit(doc, words):
    total = sum(n for _, n in words)
    for word, n in words:
        print(f"{word}@{doc}\t{n},{total}")


current, words = None, []
for line in sys.stdin:
    doc, value = line.rstrip("\n").split("\t", 1)
    word, n = value.rsplit(",", 1)
    if doc != current:
        if current is not None:
            emit(current, words)
        current, words = doc, []
    words.append((word, int(n)))
if current is not None:
    emit(current, words)
