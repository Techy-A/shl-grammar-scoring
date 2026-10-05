#!/usr/bin/env bash
# Run one Kaggle GPU job from the local machine:
#   1. upload src/ as a new version of the private "shl-src" dataset
#   2. push kaggle/jobs/<job>/ as a private kernel, which runs on Kaggle's GPUs
#   3. wait for it to finish, then download its outputs into artifacts/
# Usage: bash kaggle/run_job.sh transcribe
set -euo pipefail
cd "$(dirname "$0")/.."
JOB=$1
K=.venv/Scripts/kaggle
export PYTHONIOENCODING=utf-8   # Kaggle logs contain non-ASCII; the Windows console codec would crash the download
DS=kaggle/src_dataset
KID=$(.venv/Scripts/python -c "import json;print(json.load(open('kaggle/jobs/$JOB/kernel-metadata.json'))['id'])")

# 1. Code upload: the dataset folder gets a fresh copy of src/*.py.
# Plus any artifacts the job needs (listed in kaggle/jobs/<job>/inputs.txt).
find $DS -type f ! -name dataset-metadata.json -delete
cp src/*.py $DS/
[ -f kaggle/jobs/$JOB/inputs.txt ] && while read -r f; do cp "artifacts/$f" $DS/; done < kaggle/jobs/$JOB/inputs.txt
# Run from inside the folder: the Kaggle CLI on Windows breaks on relative -p paths.
(cd $DS && ../../$K datasets version -p . -m "$JOB $(date +%F_%T)" -q)
# Wait until the NEW version is live: "status: ready" can still refer to the previous version,
# and a kernel pushed too early mounts old code. So wait until Kaggle lists every local file
# with its local size.
until $K datasets files aryanawasthiteykik/shl-src 2>/dev/null | .venv/Scripts/python -c "
import os, sys
listed = {l.split()[0]: l.split()[1] for l in sys.stdin if l.strip() and not l.startswith(('name', '-'))}
local = {f: str(os.path.getsize(f'$DS/{f}')) for f in os.listdir('$DS') if f != 'dataset-metadata.json'}
sys.exit(any(listed.get(f) != size for f, size in local.items()))"; do sleep 15; done

# 2. Start the job.
# A refused push (e.g. "Maximum batch GPU session count of 2 reached") still exits 0, so check the text.
out=$($K kernels push -p kaggle/jobs/$JOB 2>&1); echo "$out"
echo "$out" | grep -q "successfully pushed" || exit 1

# 3. Wait, then fetch outputs (artifacts/* land in artifacts/, log in kaggle/jobs/<job>/out/).
while :; do
  s=$($K kernels status $KID 2>&1 || true)   # a transient API error must not kill the wait
  echo "$(date +%T) $s"
  echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL" && break
  sleep 60
done
rm -rf kaggle/jobs/$JOB/out
# The log download is flaky on Windows (comes back empty / non-zero exit); the outputs still arrive.
$K kernels output $KID -p kaggle/jobs/$JOB/out || echo "warning: kernels output exited non-zero (usually just the log)"
[ -d kaggle/jobs/$JOB/out/artifacts ] && cp kaggle/jobs/$JOB/out/artifacts/* artifacts/
echo "$s" | grep -q COMPLETE
