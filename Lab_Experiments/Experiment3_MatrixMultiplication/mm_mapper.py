#!/usr/bin/env python3
"""(ii) Matrix-matrix multiplication C = A B (one-pass algorithm), mapper.
A is m x n, B is n x p. The driver passes m and p: -cmdenv ROWS_A=m -cmdenv COLS_B=p
In : A,i,j,a_ij  and  B,j,k,b_jk  lines
Out: every element of A and B is sent to each result cell C[i][k] that needs it:
     A,i,j,a  ->  (i,k) \t A,j,a   for k = 0 .. p-1
     B,j,k,b  ->  (i,k) \t B,j,b   for i = 0 .. m-1
"""
import os
import sys

m = int(os.environ["ROWS_A"])
p = int(os.environ["COLS_B"])

for line in sys.stdin:
    parts = line.strip().split(",")
    if len(parts) != 4:
        continue
    name, r, c, value = parts
    if name == "A":
        for k in range(p):
            print(f"{r},{k}\tA,{c},{value}")
    elif name == "B":
        for i in range(m):
            print(f"{i},{c}\tB,{r},{value}")
