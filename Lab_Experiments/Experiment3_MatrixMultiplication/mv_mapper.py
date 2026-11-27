#!/usr/bin/env python3
"""(i) Matrix-vector multiplication x = M v, mapper.
In : M,i,j,m_ij lines of the matrix
Side input: the vector v, shipped to every mapper with -files vector_v.txt (lines v,j,v_j).
            The path can be changed with the VECTOR_FILE environment variable.
Out: i \t m_ij * v_j
"""
import os
import sys

v = {}
with open(os.environ.get("VECTOR_FILE", "vector_v.txt")) as f:
    for line in f:
        _, j, value = line.strip().split(",")
        v[int(j)] = float(value)

for line in sys.stdin:
    parts = line.strip().split(",")
    if len(parts) != 4 or parts[0] != "M":
        continue
    _, i, j, m = parts
    print(f"{i}\t{float(m) * v.get(int(j), 0.0):g}")
