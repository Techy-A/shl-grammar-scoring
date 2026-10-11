#!/usr/bin/env python3
"""Job 4 stage 2 reducer - number of transactions containing each pair.
Out: ItemA|ItemB \t Transactions=N
"""
import sys

current, count = None, 0
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    if key != current:
        if current is not None:
            print(f"{current}\tTransactions={count}")
        current, count = key, 0
    count += int(float(value))
if current is not None:
    print(f"{current}\tTransactions={count}")
