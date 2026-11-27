#!/usr/bin/env bash
# Both multiplications on a Hadoop cluster with Hadoop Streaming.
set -euo pipefail
cd "$(dirname "$0")"
JAR=${STREAMING_JAR:-$(ls $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -1)}
BASE=${BASE:-/user/$USER/matrix}
hdfs dfs -mkdir -p $BASE/mv_in $BASE/mm_in
hdfs dfs -put -f data/matrix_M.txt $BASE/mv_in/
hdfs dfs -put -f data/matrix_A.txt data/matrix_B.txt $BASE/mm_in/
hdfs dfs -rm -r -f $BASE/mv_out $BASE/mm_out

# (i) x = M v : the vector travels to every mapper through the distributed cache.
hadoop jar "$JAR" -files mv_mapper.py,mv_reducer.py,data/vector_v.txt \
  -mapper "python3 mv_mapper.py" -combiner "python3 mv_reducer.py" -reducer "python3 mv_reducer.py" \
  -input $BASE/mv_in -output $BASE/mv_out

# (ii) C = A B : A is 3 x 3, B is 3 x 2.
hadoop jar "$JAR" -files mm_mapper.py,mm_reducer.py -cmdenv ROWS_A=3 -cmdenv COLS_B=2 \
  -mapper "python3 mm_mapper.py" -reducer "python3 mm_reducer.py" \
  -input $BASE/mm_in -output $BASE/mm_out

hdfs dfs -cat $BASE/mv_out/part-* | sort -n
hdfs dfs -cat $BASE/mm_out/part-* | sort -t, -k1,1n -k2,2n
