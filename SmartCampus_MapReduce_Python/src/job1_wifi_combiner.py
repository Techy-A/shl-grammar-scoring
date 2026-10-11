#!/usr/bin/env python3
"""Generic sum combiner (used by Jobs 1, 2, 3, 4-stage-2, 5-stage-1 and 6).
In : key \t a,b,c   (sorted by key, any number of comma-separated numbers)
Out: key \t sum(a),sum(b),sum(c)  - same format as the mapper, so the reducer is unchanged.
Safe because every value it touches is a sum (associative and commutative);
averages are only computed in the reducer from (sum, count).
"""
import sys


def emit(key, acc):
    print(key + "\t" + ",".join(f"{x:g}" for x in acc))


current, acc = None, None
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    nums = [float(x) for x in value.split(",")]
    if key != current:
        if current is not None:
            emit(current, acc)
        current, acc = key, nums
    else:
        acc = [a + b for a, b in zip(acc, nums)]
if current is not None:
    emit(current, acc)
