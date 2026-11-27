#!/usr/bin/env python3
"""Job 1 reducer - totals per hour.
In : YYYY-MM-DD-HH \t DataMB,DurationMin,Sessions   (sorted by key)
Out: YYYY-MM-DD-HH \t TotalDataMB=..,TotalDurationMin=..,Sessions=..

Note: the reference reducer in the worksheet treats each input line as a whole key group
(values joined by ';'). Hadoop Streaming instead delivers one line per value with equal keys
on consecutive lines, so this version groups consecutive keys itself.
"""
import sys


def emit(key, data, duration, sessions):
    print(f"{key}\tTotalDataMB={data:.2f},TotalDurationMin={duration:.2f},Sessions={sessions}")


current = None
total_data = total_duration = 0.0
sessions = 0
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    d, t, n = value.split(",")
    if key != current:
        if current is not None:
            emit(current, total_data, total_duration, sessions)
        current, total_data, total_duration, sessions = key, 0.0, 0.0, 0
    total_data += float(d)
    total_duration += float(t)
    sessions += int(float(n))
if current is not None:
    emit(current, total_data, total_duration, sessions)
