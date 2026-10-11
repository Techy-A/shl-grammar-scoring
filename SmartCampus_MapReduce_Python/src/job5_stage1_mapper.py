#!/usr/bin/env python3
"""Job 5 stage 1 mapper - library visits per student.
In : StudentID,Timestamp,EntryType,DurationMin
Out: StudentID \t DurationMin,1
Only ENTRY rows carry the visit duration; EXIT rows (duration 0) would halve the averages,
so they are skipped.
"""
import csv
import sys

for line in sys.stdin:
    row = next(csv.reader([line]), [])
    if not row or row[0] == "StudentID":
        continue
    try:
        if len(row) < 4:
            raise ValueError("bad record")
        if row[2].strip().upper() != "ENTRY":
            continue
        print(f"{row[0].strip()}\t{float(row[3])},1")
    except ValueError:
        sys.stderr.write("reporter:counter:SmartCampus,RejectedRecords,1\n")
