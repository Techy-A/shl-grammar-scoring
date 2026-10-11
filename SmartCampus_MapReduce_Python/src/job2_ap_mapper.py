#!/usr/bin/env python3
"""Job 2 mapper - access point utilisation.
Out: AccessPoint \t DataMB,DurationMin,1
"""
import csv
import sys

for line in sys.stdin:
    row = next(csv.reader([line]), [])
    if not row or row[0] == "StudentID":
        continue
    try:
        if len(row) < 5 or not row[2].strip():
            raise ValueError("bad record")
        print(f"{row[2].strip()}\t{float(row[3])},{float(row[4])},1")
    except ValueError:
        sys.stderr.write("reporter:counter:SmartCampus,RejectedRecords,1\n")
