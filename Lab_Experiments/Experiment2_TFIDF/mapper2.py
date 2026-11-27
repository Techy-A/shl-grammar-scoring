#!/usr/bin/env python3
"""Job 2 mapper: re-key by document so one reducer sees every word of a document.
In : word@doc \t n
Out: doc \t word,n
"""
import sys

for line in sys.stdin:
    key, n = line.rstrip("\n").split("\t", 1)
    word, doc = key.rsplit("@", 1)
    print(f"{doc}\t{word},{n}")
