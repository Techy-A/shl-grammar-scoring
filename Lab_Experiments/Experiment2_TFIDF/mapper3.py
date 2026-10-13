#!/usr/bin/env python3
"""Job 3 mapper: re-key by word so one reducer sees every document that contains it.
In : word@doc \t n,N
Out: word \t doc,n,N
"""
import sys

for line in sys.stdin:
    key, value = line.rstrip("\n").split("\t", 1)
    word, doc = key.rsplit("@", 1)
    print(f"{word}\t{doc},{value}")
