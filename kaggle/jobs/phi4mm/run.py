# Kaggle job: step 17 (17_phi4mm_embed.py), Phi-4-multimodal audio-LLM embeddings, one fp16 model per T4:
# each GPU embeds every second row of the full clips, then of the crops; the shards are merged at the end.
#   -> artifacts/feat_phi4mm.parquet, artifacts/feat_phi4mm_crops.parquet
# SMOKE: 3 rows + a test transcription first, to catch load/adapter/fp16 problems in minutes.
import glob, os, subprocess

import pandas as pd

SMOKE = False
SRC = os.path.dirname(glob.glob("/kaggle/input/**/17_phi4mm_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 peft 'mistral-common[audio]' && pip uninstall -y -q torchao", shell=True, check=True)   # Kaggle's old torchao breaks peft's adapter loading

lim = " --limit 3 --check" if SMOKE else ""
run = lambda gpu: subprocess.Popen(" && ".join(f"python {SRC}/17_phi4mm_embed.py {a} --shard {gpu} 2{lim}" for a in ["", "--crops"]),
                                   shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
procs = [run(0), run(1)]
assert all(p.wait() == 0 for p in procs), "a Phi-4 run failed"

art = "/kaggle/working/artifacts"
for t in ("feat_phi4mm", "feat_phi4mm_crops"):
    shards = [f"{art}/{t}_shard{k}.parquet" for k in range(2)]
    pd.concat(map(pd.read_parquet, shards), ignore_index=True).to_parquet(f"{art}/{t}.parquet", index=False)
    for s in shards:
        os.remove(s)
    print(t, pd.read_parquet(f"{art}/{t}.parquet").shape)
