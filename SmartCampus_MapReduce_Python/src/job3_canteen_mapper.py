#!/usr/bin/env python3
"""Job 3 mapper - canteen item frequency.
In : TransactionID,Timestamp,StudentID,Item,Quantity,Amount
Out: Item \t Quantity,Amount
"""
import csv
import sys

for line in sys.stdin:
    row = next(csv.reader([line]), [])
    if not row or row[0] == "TransactionID":
        continue
    try:
        if len(row) < 6 or not row[3].strip():
            raise ValueError("bad record")
        print(f"{row[3].strip()}\t{int(row[4])},{float(row[5])}")
    except ValueError:
        sys.stderr.write("reporter:counter:SmartCampus,RejectedRecords,1\n")
