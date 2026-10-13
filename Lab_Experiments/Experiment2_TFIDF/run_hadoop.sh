#!/usr/bin/env bash
# TF-IDF on a Hadoop cluster: three chained Hadoop Streaming jobs.
set -euo pipefail
cd "$(dirname "$0")"
JAR=${STREAMING_JAR:-$(ls $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -1)}
BASE=${BASE:-/user/$USER/tfidf}
D=$(ls data/corpus/*.txt | wc -l)
hdfs dfs -mkdir -p $BASE/input
hdfs dfs -put -f data/corpus/*.txt $BASE/input/
hdfs dfs -rm -r -f $BASE/job1 $BASE/job2 $BASE/job3
S="hadoop jar $JAR -files mapper1.py,reducer1.py,mapper2.py,reducer2.py,mapper3.py,reducer3.py"
$S -mapper "python3 mapper1.py" -combiner "python3 reducer1.py" -reducer "python3 reducer1.py" \
   -input $BASE/input -output $BASE/job1
$S -mapper "python3 mapper2.py" -reducer "python3 reducer2.py" -input $BASE/job1 -output $BASE/job2
$S -cmdenv TOTAL_DOCS=$D -mapper "python3 mapper3.py" -reducer "python3 reducer3.py" \
   -input $BASE/job2 -output $BASE/job3
hdfs dfs -cat $BASE/job3/part-* | sort -t= -k4,4gr | head -15
