#!/usr/bin/env python3
"""Job 5 filter reducer - format the anomaly report (one line per flagged student).
Out: StudentID \t AvgVisitMin=..,OverallAvgMin=..,Ratio=..,Flag=ANOMALY
"""
import sys

for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    avg, overall, ratio = value.split(",")
    print(f"{key}\tAvgVisitMin={avg},OverallAvgMin={overall},Ratio={ratio}x,Flag=ANOMALY")
