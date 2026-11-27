#!/usr/bin/env python3
"""Job 5 filter mapper - keep students whose average visit > 1.5 x overall average.
In : StudentID \t TotalDurationMin,Visits   (stage 1 output)
Side input: the global job's output, shipped to every mapper with `-files global_avg.txt`
            (path can be overridden with the GLOBAL_AVG_FILE environment variable).
Out: StudentID \t StudentAvgMin,OverallAvgMin,Ratio
"""
import os
import sys

THRESHOLD = 1.5
path = os.environ.get("GLOBAL_AVG_FILE", "global_avg.txt")
overall = None
with open(path) as f:
    for line in f:
        k, v = line.rstrip("\n").split("\t", 1)
        if k == "OverallAvgMin":
            overall = float(v)
if not overall:
    sys.exit("overall average not found in " + path)

for line in sys.stdin:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        continue
    total, visits = parts[1].split(",")
    avg = float(total) / int(visits)
    if avg > THRESHOLD * overall:
        print(f"{parts[0]}\t{avg:.2f},{overall:.2f},{avg / overall:.2f}")
