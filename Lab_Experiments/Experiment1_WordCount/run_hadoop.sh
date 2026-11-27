#!/usr/bin/env bash
# Run the word count on a Hadoop cluster with Hadoop Streaming.
set -euo pipefail
cd "$(dirname "$0")"
JAR=${STREAMING_JAR:-$(ls $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -1)}
BASE=${BASE:-/user/$USER/wordcount}
hdfs dfs -mkdir -p $BASE/input
hdfs dfs -put -f data/input.txt $BASE/input/
hdfs dfs -rm -r -f $BASE/output
hadoop jar "$JAR" -files mapper.py,reducer.py \
  -mapper "python3 mapper.py" -combiner "python3 reducer.py" -reducer "python3 reducer.py" \
  -input $BASE/input -output $BASE/output
hdfs dfs -cat $BASE/output/part-* | sort -k2,2nr | head -10
