# Kaggle job: step 1 with CrisperWhisper (01_transcribe.py --model nyrahealth/CrisperWhisper), full train + test clips.
# A Whisper-large-v3 fine-tuned for verbatim transcripts; greedy decoding, one shard per T4, merged at the end.
#   -> artifacts/transcripts_crisper.parquet (the train crops run separately)
import glob, os, subprocess

import pandas as pd

SRC = os.path.dirname(glob.glob("/kaggle/input/**/01_transcribe.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)
cmd = f"python {SRC}/01_transcribe.py --model nyrahealth/CrisperWhisper --no-prompt --beams 1 --tag crisper"
procs = [subprocess.Popen(f"{cmd} --shard {k} 2", shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)})
         for k in range(2)]
assert all(p.wait() == 0 for p in procs), "a shard failed"
shards = [f"{ART}/transcripts_crisper_shard{k}.parquet" for k in range(2)]
pd.concat(map(pd.read_parquet, shards)).to_parquet(f"{ART}/transcripts_crisper.parquet", index=False)
for f in shards:
    os.remove(f)
