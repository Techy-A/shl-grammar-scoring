#!/usr/bin/env python3
"""Job 4 stage 1 reducer - all unique sorted item pairs of one transaction.
In : TransactionID \t Item   (sorted by TransactionID)
Out: ItemA|ItemB \t 1        (ItemA < ItemB alphabetically)
A set removes repeated items, so a pair is counted once per transaction.
"""
import sys
from itertools import combinations


def emit_pairs(items):
    for a, b in combinations(sorted(items), 2):
        print(f"{a}|{b}\t1")


current, items = None, set()
for line in sys.stdin:
    key, item = line.rstrip("\n").split("\t", 1)
    if key != current:
        if current is not None:
            emit_pairs(items)
        current, items = key, set()
    items.add(item)
if current is not None:
    emit_pairs(items)
