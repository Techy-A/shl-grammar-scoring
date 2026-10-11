#!/usr/bin/env bash
# Run the same jobs on a Hadoop cluster with Hadoop Streaming.
# Requires HADOOP_HOME (or `hadoop` on PATH). Input is uploaded to $BASE/input.
set -euo pipefail
cd "$(dirname "$0")"
JAR=${STREAMING_JAR:-$(ls $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -1)}
BASE=${BASE:-/user/$USER/smartcampus}
R=${REDUCERS:-2}

hdfs dfs -mkdir -p $BASE/input
hdfs dfs -put -f data/*.csv $BASE/input/

stream() { # stream <name> <input> <mapper> <reducer> [extra args...]
  local name=$1 in=$2 map=$3 red=$4; shift 4
  hdfs dfs -rm -r -f $BASE/out/$name
  hadoop jar "$JAR" -D mapreduce.job.name="SmartCampus-$name" "$@" \
    -files src/$map,src/$red,src/job1_wifi_combiner.py \
    -mapper "python3 $map" -reducer "python3 $red" \
    -input "$in" -output $BASE/out/$name
}
C=(-combiner "python3 job1_wifi_combiner.py")

stream job1 $BASE/input/wifi_logs.csv job1_wifi_mapper.py job1_wifi_reducer.py -D mapreduce.job.reduces=$R "${C[@]}"
stream job2 $BASE/input/wifi_logs.csv job2_ap_mapper.py job2_ap_reducer.py -D mapreduce.job.reduces=$R "${C[@]}"
stream job3 $BASE/input/canteen.csv job3_canteen_mapper.py job3_canteen_reducer.py -D mapreduce.job.reduces=$R "${C[@]}"

stream job4_s1 $BASE/input/canteen.csv job4_stage1_mapper.py job4_stage1_reducer.py -D mapreduce.job.reduces=$R
stream job4_s2 $BASE/out/job4_s1 job4_stage2_mapper.py job4_stage2_reducer.py -D mapreduce.job.reduces=$R "${C[@]}"

stream job5_s1 $BASE/input/library.csv job5_stage1_mapper.py job5_stage1_reducer.py -D mapreduce.job.reduces=$R "${C[@]}"
stream job5_global $BASE/out/job5_s1 job5_global_mapper.py job5_global_reducer.py -D mapreduce.job.reduces=1 "${C[@]}"
hdfs dfs -cat $BASE/out/job5_global/part-* > global_avg.txt
stream job5_filter $BASE/out/job5_s1 job5_filter_mapper.py job5_filter_reducer.py \
  -D mapreduce.job.reduces=1 -files global_avg.txt

# Job 6 with a custom partitioner: partition on RouteID only (first '-' separated field of the key),
# so every hour of one route goes to the same reducer and that reducer's output is one route's day.
stream job6 $BASE/input/transport.csv job6_transport_mapper.py job6_transport_reducer.py \
  -D mapreduce.job.reduces=$R \
  -D mapreduce.map.output.key.field.separator=- \
  -D mapreduce.partition.keypartitioner.options=-k1,1 \
  -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner "${C[@]}"

for j in job1 job2 job3 job4_s2 job5_global job5_filter job6; do
  echo "== $j"; hdfs dfs -cat $BASE/out/$j/part-* | sort | head -20
done
