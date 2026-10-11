# SmartCampusAnalytics — Python Hadoop Streaming

Six MapReduce jobs (ten stages) over campus Wi-Fi, canteen, library and transport logs.
The full write-up, with results, worksheet answers and viva answers, is in [REPORT.md](REPORT.md) (also `.docx` / `.pdf`).

```bash
python3 scripts/generate_data.py   # (re)create data/*.csv, seeded
./run_local.sh                     # run all jobs: map | sort | combine | reduce  -> output/
python3 scripts/verify.py          # recompute every job without MapReduce and compare (all PASS)
scripts/compare_combiner.sh        # combiner shuffle saving + 3-reducer equivalence
./run_hadoop.sh                    # same jobs on a Hadoop cluster (hadoop-streaming.jar)
```
