#!/usr/bin/env python3
"""Word count mapper.
In : lines of text on stdin
Out: word \t 1   for every word (lower-cased, punctuation removed)
"""
import re
import sys

for line in sys.stdin:
    for word in re.findall(r"[a-z0-9']+", line.lower()):
        print(f"{word}\t1")
