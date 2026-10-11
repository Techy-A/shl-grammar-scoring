#!/usr/bin/env python3
"""Job 3 reducer.
Out: Item \t TotalQuantity=..,TotalRevenue=..
"""
import sys


def emit(key, qty, revenue):
    print(f"{key}\tTotalQuantity={qty},TotalRevenue={revenue:.2f}")


current = None
qty, revenue = 0, 0.0
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    q, a = value.split(",")
    if key != current:
        if current is not None:
            emit(current, qty, revenue)
        current, qty, revenue = key, 0, 0.0
    qty += int(float(q))
    revenue += float(a)
if current is not None:
    emit(current, qty, revenue)
