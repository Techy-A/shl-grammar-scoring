#!/usr/bin/env python3
"""Step-by-step demonstration of every MapReduce job (works on Windows, macOS and Linux).

For each job it shows what goes in and what comes out of every phase:
    input CSV -> Mapper -> Shuffle & Sort -> Combiner -> Reducer -> answer
with record counts at each step and a few sample lines, then writes the full
results to output/ (the same files run_local.sh produces).

Usage:
    python run_demo.py              # all jobs
    python run_demo.py 4            # only Job 4
    python run_demo.py --markdown   # print the trace as Markdown (used for the report appendix)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC, DATA, OUT = (os.path.join(ROOT, d) for d in ("src", "data", "output"))
COMBINER = "job1_wifi_combiner.py"
SAMPLE = 3 if "--markdown" in sys.argv else 4
MD = "--markdown" in sys.argv
ONLY = [a for a in sys.argv[1:] if a.isdigit()]


def out(text=""):
    print(text)


def show(title, lines, command=None, sample=SAMPLE, everything=False):
    """Print one phase: its title, the equivalent command, the record count and sample lines."""
    count = f"{len(lines)} line{'s' if len(lines) != 1 else ''}"
    if MD:
        out(f"**{title}** ({count})")
        out()
        out("```text")
        if command:
            out(f"$ {command}")
    else:
        out(f"\n--- {title}  [{count}]")
        if command:
            out(f"$ {command}")
    shown = lines if everything or len(lines) <= sample + 2 else lines[:sample]
    for line in shown:
        out(line.replace("\t", "    "))
    if len(shown) < len(lines):
        out(f"... ({len(lines) - len(shown)} more lines)")
    if MD:
        out("```")
        out()


def run(script, lines, env=None):
    """Run a mapper/combiner/reducer on a list of lines; return (output lines, rejected count)."""
    proc = subprocess.run([sys.executable, os.path.join(SRC, script)],
                          input="".join(l + "\n" for l in lines), capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, check=True)
    rejected = proc.stderr.count("RejectedRecords")
    return proc.stdout.splitlines(), rejected


def read(path):
    with open(path) as f:
        return f.read().splitlines()


def save(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.writelines(l + "\n" for l in lines)


def mapreduce(label, input_lines, input_name, mapper, reducer, combine=True, env=None, full=False):
    """One MapReduce stage, shown phase by phase. Returns the reducer output."""
    show(f"{label}: input ({input_name})", input_lines, f"head {input_name}")
    mapped, rejected = run(mapper, input_lines, env)
    show(f"{label}: Mapper output (key TAB value)", mapped,
         f"python src/{mapper} < {input_name}")
    if rejected:
        if MD:
            out(f"Malformed records rejected by the mapper: **{rejected}**\n")
        else:
            out(f"(malformed records rejected by the mapper: {rejected})")
    shuffled = sorted(mapped)  # the framework's shuffle & sort: equal keys become adjacent
    show(f"{label}: after Shuffle & Sort (grouped by key)", shuffled, "... | sort")
    if combine:
        combined, _ = run(COMBINER, shuffled)
        show(f"{label}: after Combiner (one partial sum per key)", combined,
             f"... | sort | python src/{COMBINER}")
        shuffled = sorted(combined)
    reduced, _ = run(reducer, shuffled, env)
    show(f"{label}: Reducer output", reduced, f"... | sort | python src/{reducer}", everything=full)
    return reduced


def top(lines, field, n):
    def value(line):
        fields = dict(kv.split("=", 1) for kv in line.split("\t", 1)[1].split(","))
        return float(fields[field].rstrip("x"))
    return sorted(lines, key=value, reverse=True)[:n]


def heading(text):
    out(f"\n## {text}\n" if MD else "\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def answer(text, lines):
    show(f"Answer: {text}", lines, everything=True)


def job1():
    heading("Job 1: Peak Wi-Fi Usage")
    res = mapreduce("Job 1", read(os.path.join(DATA, "wifi_logs.csv")), "data/wifi_logs.csv",
                    "job1_wifi_mapper.py", "job1_wifi_reducer.py")
    save("job1_wifi.txt", res)
    answer("top 3 hours by TotalDataMB (first line = peak hour)", top(res, "TotalDataMB", 3))


def job2():
    heading("Job 2: Access Point Utilisation")
    res = mapreduce("Job 2", read(os.path.join(DATA, "wifi_logs.csv")), "data/wifi_logs.csv",
                    "job2_ap_mapper.py", "job2_ap_reducer.py", full=True)
    save("job2_ap.txt", res)
    answer("busiest access point by TotalDataMB", top(res, "TotalDataMB", 1))


def job3():
    heading("Job 3: Canteen Item Frequency")
    res = mapreduce("Job 3", read(os.path.join(DATA, "canteen.csv")), "data/canteen.csv",
                    "job3_canteen_mapper.py", "job3_canteen_reducer.py", full=True)
    save("job3_canteen.txt", res)
    answer("highest revenue item", top(res, "TotalRevenue", 1))
    answer("highest quantity item", top(res, "TotalQuantity", 1))


def job4():
    heading("Job 4: Top Co-Purchased Items (2 stages)")
    pairs = mapreduce("Job 4 Stage 1", read(os.path.join(DATA, "canteen.csv")), "data/canteen.csv",
                      "job4_stage1_mapper.py", "job4_stage1_reducer.py", combine=False)
    save("job4_stage1_pairs.txt", pairs)
    res = mapreduce("Job 4 Stage 2", pairs, "output/job4_stage1_pairs.txt",
                    "job4_stage2_mapper.py", "job4_stage2_reducer.py")
    save("job4_pairs.txt", res)
    answer("top 5 pairs by number of transactions", top(res, "Transactions", 5))


def job5():
    heading("Job 5: Library Usage Anomaly (3 stages)")
    students = mapreduce("Job 5 Stage 1", read(os.path.join(DATA, "library.csv")), "data/library.csv",
                         "job5_stage1_mapper.py", "job5_stage1_reducer.py")
    save("job5_stage1_students.txt", students)
    overall = mapreduce("Job 5 Stage 2 (global average)", students, "output/job5_stage1_students.txt",
                        "job5_global_mapper.py", "job5_global_reducer.py", full=True)
    save("job5_global_avg.txt", overall)
    avg = float(overall[0].split("\t")[1])
    if MD:
        out(f"Threshold = 1.5 x {avg:.2f} = **{1.5 * avg:.2f} min**. "
            "The global average file is passed to every filter mapper (`-files global_avg.txt`).\n")
    else:
        out(f"\nThreshold = 1.5 x {avg:.2f} = {1.5 * avg:.2f} min (global file shipped to every mapper)")
    env = {"GLOBAL_AVG_FILE": os.path.join(OUT, "job5_global_avg.txt")}
    res = mapreduce("Job 5 Stage 3 (filter)", students, "output/job5_stage1_students.txt",
                    "job5_filter_mapper.py", "job5_filter_reducer.py", combine=False, env=env, full=True)
    save("job5_anomalies.txt", res)
    answer("students above 1.5 x the overall average", res)


def job6():
    heading("Job 6: Transport Congestion")
    res = mapreduce("Job 6", read(os.path.join(DATA, "transport.csv")), "data/transport.csv",
                    "job6_transport_mapper.py", "job6_transport_reducer.py")
    save("job6_transport.txt", res)
    congested = [l for l in res if l.endswith("CONGESTED")]
    save("job6_congested.txt", congested)
    answer(f"{len(congested)} of {len(res)} route-hours are congested (avg speed < 20 km/h)", congested)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = {"1": job1, "2": job2, "3": job3, "4": job4, "5": job5, "6": job6}
    for key, fn in jobs.items():
        if not ONLY or key in ONLY:
            fn()
    if not ONLY and not MD:
        out("\nAll results written to output/. Check them with:  python scripts/verify.py")
