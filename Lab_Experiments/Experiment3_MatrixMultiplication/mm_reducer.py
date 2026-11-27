#!/usr/bin/env python3
"""(ii) Matrix-matrix multiplication, reducer.
In : i,k \t A,j,a  or  B,j,b   (sorted by i,k)
Out: i,k \t C[i][k] = sum over j of a_ij * b_jk
Values are matched on j; a j present in only one matrix is a zero and adds nothing.
"""
import sys


def emit(key, a, b):
    total = sum(a[j] * b[j] for j in a if j in b)
    print(f"{key}\t{total:g}")


current, a, b = None, {}, {}
for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    name, j, x = value.split(",")
    if key != current:
        if current is not None:
            emit(current, a, b)
        current, a, b = key, {}, {}
    (a if name == "A" else b)[int(j)] = float(x)
if current is not None:
    emit(current, a, b)
