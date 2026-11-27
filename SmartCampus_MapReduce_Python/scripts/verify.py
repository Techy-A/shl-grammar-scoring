#!/usr/bin/env python3
"""Recompute every job directly from the CSVs (no MapReduce) and compare with output/.
Exit code 0 only if all six jobs match.
"""
import csv
import os
import sys
from collections import defaultdict
from itertools import combinations

ROOT = os.path.join(os.path.dirname(__file__), "..")


def rows(name, header):
    with open(os.path.join(ROOT, "data", name)) as f:
        for r in csv.reader(f):
            if r and r[0] != header:
                yield r


def load(name):
    out = {}
    with open(os.path.join(ROOT, "output", name)) as f:
        for line in f:
            k, v = line.rstrip("\n").split("\t", 1)
            out[k] = dict(kv.split("=", 1) for kv in v.split(","))
    return out


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


ok = True


def check(job, cond):
    global ok
    print(f"{job}: {'PASS' if cond else 'FAIL'}")
    ok &= cond


# Job 1 and 2
hourly, ap = defaultdict(lambda: [0.0, 0.0, 0]), defaultdict(lambda: [0.0, 0.0, 0])
for r in rows("wifi_logs.csv", "StudentID"):
    if len(r) < 5 or num(r[3]) is None or num(r[4]) is None:
        continue
    for agg, k in ((hourly, r[1][:13].replace(" ", "-")), (ap, r[2])):
        agg[k][0] += float(r[3]); agg[k][1] += float(r[4]); agg[k][2] += 1
mr = load("job1_wifi.txt")
check("Job 1", mr.keys() == hourly.keys() and all(
    abs(float(mr[k]["TotalDataMB"]) - v[0]) < 0.01 and int(mr[k]["Sessions"]) == v[2] for k, v in hourly.items()))
mr = load("job2_ap.txt")
check("Job 2", mr.keys() == ap.keys() and all(
    abs(float(mr[k]["AvgSessionMin"]) - v[1] / v[2]) < 0.01 and int(mr[k]["Sessions"]) == v[2] for k, v in ap.items()))

# Job 3 and 4
items, tx = defaultdict(lambda: [0, 0.0]), defaultdict(set)
for r in rows("canteen.csv", "TransactionID"):
    if len(r) < 6:
        continue
    tx[r[0]].add(r[3])
    if num(r[4]) is None or num(r[5]) is None:
        continue
    items[r[3]][0] += int(r[4]); items[r[3]][1] += float(r[5])
mr = load("job3_canteen.txt")
check("Job 3", mr.keys() == items.keys() and all(
    int(mr[k]["TotalQuantity"]) == v[0] and abs(float(mr[k]["TotalRevenue"]) - v[1]) < 0.01 for k, v in items.items()))
pairs = defaultdict(int)
for s in tx.values():
    for a, b in combinations(sorted(s), 2):
        pairs[f"{a}|{b}"] += 1
mr = load("job4_pairs.txt")
check("Job 4", {k: int(v["Transactions"]) for k, v in mr.items()} == dict(pairs))

# Job 5
stu = defaultdict(list)
for r in rows("library.csv", "StudentID"):
    if len(r) >= 4 and r[2] == "ENTRY" and num(r[3]) is not None:
        stu[r[0]].append(float(r[3]))
overall = sum(map(sum, stu.values())) / sum(map(len, stu.values()))
flagged = {s for s, d in stu.items() if sum(d) / len(d) > 1.5 * overall}
check("Job 5", set(load("job5_anomalies.txt")) == flagged)

# Job 6
rh = defaultdict(list)
for r in rows("transport.csv", "VehicleID"):
    if len(r) >= 4 and num(r[3]) is not None:
        rh[f"{r[2]}-{r[1][11:13]}"].append(float(r[3]))
mr = load("job6_transport.txt")
check("Job 6", mr.keys() == rh.keys() and all(
    (mr[k]["Status"] == "CONGESTED") == (sum(v) / len(v) < 20) for k, v in rh.items()))

sys.exit(0 if ok else 1)
