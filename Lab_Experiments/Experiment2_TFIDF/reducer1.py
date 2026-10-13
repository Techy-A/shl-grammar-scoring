#!/usr/bin/env python3
"""Job 1 reducer: n = number of times a word occurs in a document (also the combiner).
In : word@doc \t count   (sorted)
Out: word@doc \t n
"""
import sys

current, total = None, 0
for line in sys.stdin:
    key, count = line.rstrip("\n").split("\t", 1)
    if key != current:
        if current is not None:
            print(f"{current}\t{total}")
        current, total = key, 0
    total += int(count)
if current is not None:
    print(f"{current}\t{total}")
