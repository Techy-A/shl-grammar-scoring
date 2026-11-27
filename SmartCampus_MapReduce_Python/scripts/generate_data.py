"""Generate the four Smart Campus input datasets (seeded, so results are reproducible).

The worksheet gives only the schema and one example row per file, so the data is
synthetic but shaped like a real campus week: Wi-Fi peaks late morning, the canteen
peaks at lunch, buses slow down at rush hour, and a few students camp in the library.
A handful of deliberately malformed rows are mixed in to test record rejection.
"""
import csv
import os
import random

random.seed(2026)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
DAYS = [f"2026-08-{d:02d}" for d in range(10, 15)]  # Mon 10 Aug - Fri 14 Aug
STUDENTS = [f"S{n}" for n in range(101, 161)]        # 60 students


def ts(day, hour, minute):
    return f"{day} {hour:02d}:{minute:02d}"


def wifi():
    # AP03 (library) and AP01 (main block) are busier than the rest -> realistic skew.
    aps = ["AP01"] * 4 + ["AP02"] * 2 + ["AP03"] * 5 + ["AP04"] * 2 + ["AP05"] * 2 + ["AP06", "AP07", "AP08"]
    hour_weight = {8: 3, 9: 6, 10: 8, 11: 10, 12: 7, 13: 6, 14: 7, 15: 6, 16: 5, 17: 3, 18: 2, 19: 2, 20: 1}
    hours = [h for h, w in hour_weight.items() for _ in range(w)]
    rows = []
    for day in DAYS:
        for _ in range(140):
            h = random.choice(hours)
            dur = random.randint(5, 120)
            data = round(dur * random.uniform(0.8, 4.5) * (1.6 if h in (11, 14) else 1.0), 1)
            rows.append([random.choice(STUDENTS), ts(day, h, random.randint(0, 59)), random.choice(aps), data, dur])
    rows.sort(key=lambda r: r[1])
    return ["StudentID", "Timestamp", "AccessPoint", "DataMB", "DurationMin"], rows


MENU = {"Coffee": 40, "Tea": 20, "Sandwich": 80, "Samosa": 15, "Burger": 120,
        "Pasta": 140, "Juice": 50, "Muffin": 45, "Noodles": 110, "Water": 20}
# Items that tend to be bought together.
COMBOS = [("Coffee", "Sandwich"), ("Tea", "Samosa"), ("Burger", "Juice"),
          ("Coffee", "Muffin"), ("Pasta", "Juice"), ("Noodles", "Water")]


def canteen():
    rows, tid = [], 1001
    for day in DAYS:
        for _ in range(60):
            h = random.choices([8, 9, 10, 11, 12, 13, 14, 15, 16, 17], weights=[2, 3, 2, 3, 8, 10, 6, 3, 4, 3])[0]
            t = ts(day, h, random.randint(0, 59))
            sid = random.choice(STUDENTS)
            items = set(random.choice(COMBOS)) if random.random() < 0.55 else {random.choice(list(MENU))}
            while random.random() < 0.3:
                items.add(random.choice(list(MENU)))
            for item in sorted(items, key=lambda _: random.random()):
                q = random.choices([1, 2, 3], weights=[7, 2, 1])[0]
                rows.append([f"T{tid}", t, sid, item, q, q * MENU[item]])
            tid += 1
    return ["TransactionID", "Timestamp", "StudentID", "Item", "Quantity", "Amount"], rows


def library():
    heavy = {"S107", "S123", "S131", "S148"}  # students who study for very long stretches
    rows = []
    for day in DAYS:
        for sid in random.sample(STUDENTS, 28) + sorted(heavy):
            h, m = random.randint(8, 18), random.randint(0, 59)
            dur = random.randint(150, 260) if sid in heavy else random.randint(15, 110)
            rows.append([sid, ts(day, h, m), "ENTRY", dur])
            end = h * 60 + m + dur
            rows.append([sid, ts(day, min(end // 60, 23), end % 60), "EXIT", 0])
    rows.sort(key=lambda r: r[1])
    return ["StudentID", "Timestamp", "EntryType", "DurationMin"], rows


def transport():
    routes = {"R01": 32, "R02": 28, "R03": 35, "R04": 24, "R05": 30}  # free-flow speed
    rush = {8: 0.55, 9: 0.65, 17: 0.6, 18: 0.7, 13: 0.85}
    rows = []
    for day in DAYS:
        for r, base in routes.items():
            buses = [f"BUS{n:02d}" for n in range(int(r[1:]) * 2 - 1, int(r[1:]) * 2 + 1)]
            for h in range(7, 21):
                for _ in range(3):
                    s = base * rush.get(h, 1.0) * random.uniform(0.85, 1.15)
                    rows.append([random.choice(buses), ts(day, h, random.randint(0, 59)), r, round(s, 1)])
    rows.sort(key=lambda r: r[1])
    return ["VehicleID", "Timestamp", "RouteID", "SpeedKmph"], rows


# Deliberately bad lines appended to each file (blank, wrong field count, non-numeric).
MALFORMED = {
    "wifi_logs.csv": ["", "S999,2026-08-11 10:05,AP02,abc,30", "S998,2026-08-12 11:00,AP01"],
    "canteen.csv": ["T9999,2026-08-12 13:00,S101,Coffee,two,80", "T9998,,S102"],
    "library.csv": ["S150,2026-08-13 10:00,ENTRY,", "garbage line without commas"],
    "transport.csv": ["BUS01,2026-08-10 08:15,R01,fast", ""],
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("wifi_logs.csv", wifi), ("canteen.csv", canteen),
                     ("library.csv", library), ("transport.csv", transport)]:
        header, rows = fn()
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
            for bad in MALFORMED[name]:
                f.write(bad + "\n")
        print(f"{name}: {len(rows)} valid rows + {len(MALFORMED[name])} malformed")
