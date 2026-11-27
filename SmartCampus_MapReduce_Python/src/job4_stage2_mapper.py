#!/usr/bin/env python3
"""Job 4 stage 2 mapper - identity: ItemA|ItemB \t 1 passes straight to the shuffle."""
import sys

for line in sys.stdin:
    line = line.rstrip("\n")
    if "\t" in line:
        print(line)
