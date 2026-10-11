#!/usr/bin/env python3
"""Job 2 reducer.
Out: AccessPoint \t TotalDataMB,TotalDurationMin,Sessions,AvgSessionMin
"""
import sys


def emit(key, data, duration, sessions):
    avg = duration / sessions if sessions else 0.0
    print(f"{key}\tTotalDataMB={data:.2f},TotalDurationMin={duration:.2f},"
          f"Sessions={sessions},AvgSessionMin={avg:.2f}")


current = None
data = duration = 0.0
sessions = 0
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    d, t, n = value.split(",")
    if key != current:
        if current is not None:
            emit(current, data, duration, sessions)
        current, data, duration, sessions = key, 0.0, 0.0, 0
    data += float(d)
    duration += float(t)
    sessions += int(float(n))
if current is not None:
    emit(current, data, duration, sessions)
