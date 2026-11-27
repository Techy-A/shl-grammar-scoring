#!/usr/bin/env python3
"""Job 6 mapper - transport congestion.
In : VehicleID,Timestamp,RouteID,SpeedKmph
Out: RouteID-HH \t SpeedKmph,1   (HH = hour of day, so the same slot is pooled across days)
"""
import csv
import sys

for line in sys.stdin:
    row = next(csv.reader([line]), [])
    if not row or row[0] == "VehicleID":
        continue
    try:
        if len(row) < 4 or not row[2].strip():
            raise ValueError("bad record")
        hour = row[1].strip()[11:13]
        if not hour.isdigit():
            raise ValueError("bad timestamp")
        print(f"{row[2].strip()}-{hour}\t{float(row[3])},1")
    except ValueError:
        sys.stderr.write("reporter:counter:SmartCampus,RejectedRecords,1\n")
