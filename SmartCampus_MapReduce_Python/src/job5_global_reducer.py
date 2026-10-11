#!/usr/bin/env python3
"""Job 5 global reducer - overall average visit duration.
Out: OverallAvgMin \t <avg>  (also TotalDurationMin and Visits for checking)
Overall average = sum of all durations / number of visits (not an average of averages).
"""
import sys

total, visits = 0.0, 0
for line in sys.stdin:
    _, value = line.rstrip("\n").split("\t", 1)
    d, n = value.split(",")
    total += float(d)
    visits += int(float(n))
avg = total / visits if visits else 0.0
print(f"OverallAvgMin\t{avg:.4f}")
print(f"TotalDurationMin\t{total:.2f}")
print(f"Visits\t{visits}")
