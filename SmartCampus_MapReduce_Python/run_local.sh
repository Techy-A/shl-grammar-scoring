#!/usr/bin/env bash
# Run all six jobs locally, simulating Hadoop Streaming as:  map | sort (shuffle) | [combine] | reduce
# Usage: ./run_local.sh            (with combiners)
#        NO_COMBINER=1 ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")"
export LC_ALL=C              # byte-order sort, same as Hadoop's default key comparator
S=src; D=data; O=output
mkdir -p "$O"

combine() { if [ -n "${NO_COMBINER:-}" ]; then cat; else python3 $S/job1_wifi_combiner.py; fi; }
# map-side sort+combine, then the shuffle sort, then reduce (mapper stderr = Hadoop counters, tallied at the end) (mapper stderr = Hadoop counters, counted at the end)
run() { # run <input> <mapper> <reducer> [combine]
  if [ "${4:-}" = combine ]; then
    python3 "$2" < "$1" 2>/dev/null | sort | combine | sort | python3 "$3"
  else
    python3 "$2" < "$1" 2>/dev/null | sort | python3 "$3"
  fi
}

echo "== Job 1: Peak Wi-Fi usage per hour"
run $D/wifi_logs.csv $S/job1_wifi_mapper.py $S/job1_wifi_reducer.py combine > $O/job1_wifi.txt
echo "Peak hour:"; python3 scripts/top_n.py $O/job1_wifi.txt TotalDataMB 1 | tee $O/job1_peak_hour.txt
echo "Top 3 hours:"; python3 scripts/top_n.py $O/job1_wifi.txt TotalDataMB 3 | tee $O/job1_top3_hours.txt

echo; echo "== Job 2: Access point utilisation"
run $D/wifi_logs.csv $S/job2_ap_mapper.py $S/job2_ap_reducer.py combine | tee $O/job2_ap.txt

echo; echo "== Job 3: Canteen item frequency"
run $D/canteen.csv $S/job3_canteen_mapper.py $S/job3_canteen_reducer.py combine | tee $O/job3_canteen.txt

echo; echo "== Job 4: Co-purchased item pairs (2 stages)"
run $D/canteen.csv $S/job4_stage1_mapper.py $S/job4_stage1_reducer.py > $O/job4_stage1_pairs.txt
run $O/job4_stage1_pairs.txt $S/job4_stage2_mapper.py $S/job4_stage2_reducer.py combine > $O/job4_pairs.txt
echo "Top 5 pairs:"; python3 scripts/top_n.py $O/job4_pairs.txt Transactions 5 | tee $O/job4_top5_pairs.txt

echo; echo "== Job 5: Library usage anomaly (3 stages)"
run $D/library.csv $S/job5_stage1_mapper.py $S/job5_stage1_reducer.py combine > $O/job5_stage1_students.txt
run $O/job5_stage1_students.txt $S/job5_global_mapper.py $S/job5_global_reducer.py combine | tee $O/job5_global_avg.txt
GLOBAL_AVG_FILE=$O/job5_global_avg.txt \
  run $O/job5_stage1_students.txt $S/job5_filter_mapper.py $S/job5_filter_reducer.py | tee $O/job5_anomalies.txt

echo; echo "== Job 6: Transport congestion per route-hour"
run $D/transport.csv $S/job6_transport_mapper.py $S/job6_transport_reducer.py combine > $O/job6_transport.txt
grep CONGESTED $O/job6_transport.txt | tee $O/job6_congested.txt

echo; echo "== Rejected (malformed) records per input file"
for f in wifi_logs canteen library transport; do
  case $f in wifi_logs) m=job1_wifi_mapper;; canteen) m=job3_canteen_mapper;;
             library) m=job5_stage1_mapper;; transport) m=job6_transport_mapper;; esac
  n=$(python3 $S/$m.py < $D/$f.csv 2>&1 >/dev/null | grep -c RejectedRecords || true)
  echo "$f.csv: $n"
done | tee $O/rejected_records.txt
