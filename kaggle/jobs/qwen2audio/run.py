# Kaggle job: step 14 (14_qwen2audio_embed.py), Qwen2-Audio-7B-Instruct hidden states. The fp16 model spans both T4s,
# so full clips and crops run one after the other in one process at a time.
#   -> artifacts/feat_qwen2audio[_q].parquet, artifacts/feat_qwen2audio[_q]_crops.parquet
# SMOKE: 3 rows per run first, to catch install or API errors in minutes instead of an hour.
import glob, os, subprocess

SMOKE = False
SRC = os.path.dirname(glob.glob("/kaggle/input/**/14_qwen2audio_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 accelerate", shell=True, check=True)
lim = " --limit 3" if SMOKE else ""
for extra in ("", "--crops"):
    subprocess.run(f"python {SRC}/14_qwen2audio_embed.py {extra}{lim}", shell=True, check=True)
