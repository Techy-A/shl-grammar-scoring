#!/usr/bin/env python3
"""Job 5 global mapper - send every student's (sum, count) to one key.
In : StudentID \t TotalDurationMin,Visits   (stage 1 output)
Out: ALL \t TotalDurationMin,Visits
Only one record per student reaches the single reducer, and a combiner can pre-sum them.
"""
import sys

for line in sys.stdin:
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 2:
        print(f"ALL\t{parts[1]}")
