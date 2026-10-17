# Kaggle job: Phi-4-multimodal transcripts (17_phi4mm_embed.py --transcribe) of the full clips and the test-length crops,
# one fp16 model per T4 (every second row each), shards merged at the end. For 06_finetune_text.py --tag phi4.
#   -> artifacts/transcripts_phi4.parquet, artifacts/transcripts_phi4_crops.parquet
import glob, os, subprocess

import pandas as pd

SRC = os.path.dirname(glob.glob("/kaggle/input/**/17_phi4mm_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 peft 'mistral-common[audio]' && pip uninstall -y -q torchao",
               shell=True, check=True)   # Kaggle's old torchao breaks peft's adapter loading
run = lambda gpu: subprocess.Popen(" && ".join(f"python {SRC}/17_phi4mm_embed.py --transcribe {a} --shard {gpu} 2"
                                               for a in ["", "--crops"]),
                                   shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
procs = [run(0), run(1)]
assert all(p.wait() == 0 for p in procs), "a Phi-4 transcription run failed"

art = "/kaggle/working/artifacts"
for t in ("transcripts_phi4", "transcripts_phi4_crops"):
    shards = [f"{art}/{t}_shard{k}.parquet" for k in range(2)]
    pd.concat(map(pd.read_parquet, shards), ignore_index=True).to_parquet(f"{art}/{t}.parquet", index=False)
    for s in shards:
        os.remove(s)
    print(t, pd.read_parquet(f"{art}/{t}.parquet").shape)
