#!/usr/bin/env python3
"""Word count reducer (also used as the combiner).
In : word \t count   sorted by word, so equal words arrive on consecutive lines
Out: word \t total count
"""
import sys

current, total = None, 0
for line in sys.stdin:
    word, count = line.rstrip("\n").split("\t", 1)
    if word != current:
        if current is not None:
            print(f"{current}\t{total}")
        current, total = word, 0
    total += int(count)
if current is not None:
    print(f"{current}\t{total}")
