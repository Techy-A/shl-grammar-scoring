#!/usr/bin/env python3
"""(i) Matrix-vector multiplication, reducer (also the combiner).
In : i \t partial product   (sorted by i)
Out: i \t x_i = sum over j of m_ij * v_j
"""
import sys

current, total = None, 0.0
for line in sys.stdin:
    i, value = line.rstrip("\n").split("\t", 1)
    if i != current:
        if current is not None:
            print(f"{current}\t{total:g}")
        current, total = i, 0.0
    total += float(value)
if current is not None:
    print(f"{current}\t{total:g}")
