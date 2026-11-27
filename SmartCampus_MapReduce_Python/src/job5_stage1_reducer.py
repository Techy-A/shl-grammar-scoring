#!/usr/bin/env python3
"""Job 5 stage 1 reducer.
Out: StudentID \t TotalDurationMin,Visits   (kept as raw sum,count so later jobs can combine them)
"""
import sys

current, total, visits = None, 0.0, 0
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    d, n = value.split(",")
    if key != current:
        if current is not None:
            print(f"{current}\t{total:.2f},{visits}")
        current, total, visits = key, 0.0, 0
    total += float(d)
    visits += int(float(n))
if current is not None:
    print(f"{current}\t{total:.2f},{visits}")
