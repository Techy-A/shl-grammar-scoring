#!/usr/bin/env python3
"""Job 1 mapper - Peak Wi-Fi usage.
In : StudentID,Timestamp,AccessPoint,DataMB,DurationMin
Out: YYYY-MM-DD-HH \t DataMB,DurationMin,1
"""
import csv
import sys

for line in sys.stdin:
    row = next(csv.reader([line]), [])
    if not row or row[0] == "StudentID":
        continue
    try:
        if len(row) < 5:
            raise ValueError("too few fields")
        hour = row[1].strip()[:13].replace(" ", "-")  # "2026-08-10 09:15" -> "2026-08-10-09"
        if len(hour) != 13:
            raise ValueError("bad timestamp")
        data_mb = float(row[3])
        duration = float(row[4])
        print(f"{hour}\t{data_mb},{duration},1")
    except ValueError:
        # Hadoop Streaming counter: shows up in the job's counters as SmartCampus:RejectedRecords.
        sys.stderr.write("reporter:counter:SmartCampus,RejectedRecords,1\n")
