#!/usr/bin/env python3
"""Job 6 reducer - average speed per route-hour, flag congestion (< 20 km/h).
Out: RouteID-HH \t AvgSpeedKmph=..,Readings=..,Status=CONGESTED|NORMAL
"""
import sys

LIMIT = 20.0


def emit(key, total, n):
    avg = total / n
    status = "CONGESTED" if avg < LIMIT else "NORMAL"
    print(f"{key}\tAvgSpeedKmph={avg:.2f},Readings={n},Status={status}")


current, total, n = None, 0.0, 0
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    s, c = value.split(",")
    if key != current:
        if current is not None:
            emit(current, total, n)
        current, total, n = key, 0.0, 0
    total += float(s)
    n += int(float(c))
if current is not None:
    emit(current, total, n)
