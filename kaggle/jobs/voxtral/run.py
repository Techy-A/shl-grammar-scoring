# Kaggle job: step 11 (11_voxtral_embed.py), Voxtral-Mini-3B audio-LLM embeddings, one model per T4:
#   GPU 0: full train + test clips, then crops shard 0 of 3;  GPU 1: crops shards 1 and 2
#   -> artifacts/feat_voxtral[_judge].parquet, artifacts/feat_voxtral[_judge]_crops.parquet
# SMOKE: 3 rows per run first, to catch install or API errors in minutes instead of hours.
import glob, os, subprocess

import pandas as pd

SMOKE = False
SRC = os.path.dirname(glob.glob("/kaggle/input/**/11_voxtral_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 'mistral-common[audio]'", shell=True, check=True)

lim = " --limit 3" if SMOKE else ""
run = lambda gpu, *argsets: subprocess.Popen(" && ".join(f"python {SRC}/11_voxtral_embed.py {a}{lim}" for a in argsets),
                                             shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
procs = [run("0", "", "--crops --shard 0 3"), run("1", "--crops --shard 1 3", "--crops --shard 2 3")]
assert all(p.wait() == 0 for p in procs), "a Voxtral run failed"

art = "/kaggle/working/artifacts"
for t in ("feat_voxtral", "feat_voxtral_judge"):   # embeddings and the direct judgement, sharded the same way
    shards = [f"{art}/{t}_crops_shard{k}.parquet" for k in range(3)]
    pd.concat(map(pd.read_parquet, shards), ignore_index=True).to_parquet(f"{art}/{t}_crops.parquet", index=False)
    for s in shards:
        os.remove(s)
print({f: pd.read_parquet(f"{art}/{f}.parquet").shape
       for f in ("feat_voxtral", "feat_voxtral_crops", "feat_voxtral_judge", "feat_voxtral_judge_crops")})
