# Kaggle job: step 14 (14_qwen_audio_embed.py), Qwen2-Audio-7B-Instruct embeddings. 7B in fp16 needs both T4s,
# so the runs go one after another: full train + test clips first (enough for a quick CV check without crops),
# then the crops in 2 shards (each shard saved on its own, so a timeout keeps what finished).
#   -> artifacts/feat_qwenaudio[_judge].parquet, artifacts/feat_qwenaudio[_judge]_crops.parquet
# SMOKE: 3 rows per run first, to catch install or API errors in minutes instead of hours.
import glob, os, subprocess

import pandas as pd

SMOKE = False
SRC = os.path.dirname(glob.glob("/kaggle/input/**/14_qwen_audio_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 accelerate", shell=True, check=True)

lim = " --limit 3" if SMOKE else ""
for a in ["", "--crops --shard 0 2", "--crops --shard 1 2"]:
    subprocess.run(f"python {SRC}/14_qwen_audio_embed.py {a}{lim}", shell=True, check=True)

art = "/kaggle/working/artifacts"
for t in ("feat_qwenaudio", "feat_qwenaudio_judge"):
    shards = [f"{art}/{t}_crops_shard{k}.parquet" for k in range(2)]
    pd.concat(map(pd.read_parquet, shards), ignore_index=True).to_parquet(f"{art}/{t}_crops.parquet", index=False)
    for s in shards:
        os.remove(s)
print({f: pd.read_parquet(f"{art}/{f}.parquet").shape
       for f in ("feat_qwenaudio", "feat_qwenaudio_crops", "feat_qwenaudio_judge", "feat_qwenaudio_judge_crops")})
