#!/usr/bin/env python3
"""Job 1 mapper: term count per document.
In : lines of one document (Hadoop sets the file name in mapreduce_map_input_file)
Out: word@doc \t 1
"""
import os
import re
import sys

path = os.environ.get("mapreduce_map_input_file") or os.environ.get("map_input_file", "unknown")
doc = os.path.splitext(os.path.basename(path))[0]
for line in sys.stdin:
    for word in re.findall(r"[a-z0-9']+", line.lower()):
        print(f"{word}@{doc}\t1")
