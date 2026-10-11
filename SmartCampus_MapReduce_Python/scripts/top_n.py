#!/usr/bin/env python3
"""Print the top N lines of a reducer output by one numeric field.
Usage: top_n.py <file> <FieldName> [N]     e.g. top_n.py output/job1_wifi.txt TotalDataMB 3
"""
import sys

path, field = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rows = []
with open(path) as f:
    for line in f:
        key, value = line.rstrip("\n").split("\t", 1)
        fields = dict(kv.split("=", 1) for kv in value.split(","))
        rows.append((float(fields[field].rstrip("x")), line.rstrip("\n")))
for _, line in sorted(rows, key=lambda r: -r[0])[:n]:
    print(line)
