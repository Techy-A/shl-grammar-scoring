#!/usr/bin/env python3
"""Job 4 stage 1 mapper - group items by transaction.
Out: TransactionID \t Item
"""
import csv
import sys

for line in sys.stdin:
    row = next(csv.reader([line]), [])
    if not row or row[0] == "TransactionID":
        continue
    if len(row) < 6 or not row[0].strip() or not row[3].strip():
        sys.stderr.write("reporter:counter:SmartCampus,RejectedRecords,1\n")
        continue
    print(f"{row[0].strip()}\t{row[3].strip()}")
