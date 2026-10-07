# Kaggle job: step 11 (11_voxtral_embed.py --upper), Voxtral-Mini-3B upper layers (60-90% of depth), one model per T4:
#   GPU 0: full train + test clips, then crops shard 0 of 3;  GPU 1: crops shards 1 and 2
#   -> artifacts/feat_voxtral_hi.parquet, artifacts/feat_voxtral_hi_crops.parquet
import glob, os, subprocess

import pandas as pd

SRC = os.path.dirname(glob.glob("/kaggle/input/**/11_voxtral_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 'mistral-common[audio]'", shell=True, check=True)

run = lambda gpu, *argsets: subprocess.Popen(" && ".join(f"python {SRC}/11_voxtral_embed.py --upper {a}" for a in argsets),
                                             shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
procs = [run("0", "", "--crops --shard 0 3"), run("1", "--crops --shard 1 3", "--crops --shard 2 3")]
assert all(p.wait() == 0 for p in procs), "a Voxtral run failed"

art = "/kaggle/working/artifacts"
shards = [f"{art}/feat_voxtral_hi_crops_shard{k}.parquet" for k in range(3)]
pd.concat(map(pd.read_parquet, shards), ignore_index=True).to_parquet(f"{art}/feat_voxtral_hi_crops.parquet", index=False)
for s in shards:
    os.remove(s)
print({f: pd.read_parquet(f"{art}/{f}.parquet").shape for f in ("feat_voxtral_hi", "feat_voxtral_hi_crops")})
