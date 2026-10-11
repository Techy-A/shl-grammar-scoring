#!/usr/bin/env bash
# Extension challenges: (1) shuffle volume with vs without the combiner for Job 1,
# (2) Job 1 with 3 reducers (hash partitioning) gives the same result as 1 reducer.
set -euo pipefail
cd "$(dirname "$0")/.."
export LC_ALL=C
S=src; IN=data/wifi_logs.csv
SPLITS=4   # pretend the input is split across 4 map tasks; the combiner runs per map task

lines=$(wc -l < $IN); per=$(( (lines + SPLITS - 1) / SPLITS ))
tmp=$(mktemp -d); split -l $per $IN $tmp/split_
no_comb=0; with_comb=0
for f in $tmp/split_*; do
  python3 $S/job1_wifi_mapper.py < $f 2>/dev/null | sort > $f.map
  no_comb=$(( no_comb + $(wc -l < $f.map) ))
  python3 $S/job1_wifi_combiner.py < $f.map > $f.comb
  with_comb=$(( with_comb + $(wc -l < $f.comb) ))
done
echo "Job 1 records shuffled without combiner: $no_comb"
echo "Job 1 records shuffled with combiner   : $with_comb  ($SPLITS map tasks)"
cat $tmp/split_*.comb | sort | python3 $S/job1_wifi_reducer.py > $tmp/combined.txt
cat $tmp/split_*.map  | sort | python3 $S/job1_wifi_reducer.py > $tmp/plain.txt
cmp -s $tmp/combined.txt $tmp/plain.txt && echo "Combiner output identical to no-combiner output: YES"

# 3 reducers: partition by hash(key) % 3, reduce each partition separately, then merge.
cat $tmp/split_*.map | python3 -c '
import sys, zlib
outs = [open(sys.argv[1] + f"/part{i}", "w") for i in range(3)]
for line in sys.stdin:
    outs[zlib.crc32(line.split("\t")[0].encode()) % 3].write(line)
' $tmp
for i in 0 1 2; do sort $tmp/part$i | python3 $S/job1_wifi_reducer.py > $tmp/out$i; echo "reducer $i: $(wc -l < $tmp/out$i) keys"; done
cat $tmp/out0 $tmp/out1 $tmp/out2 | sort | cmp -s - $tmp/plain.txt && echo "3-reducer output identical to 1-reducer output: YES"
rm -rf $tmp
