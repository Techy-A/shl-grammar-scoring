# Kaggle job: step 11 with Voxtral-Small-24B (11_voxtral_embed.py --small): one 4-bit model split over both T4s,
# full train + test clips, then the test-length crops.
#   -> artifacts/feat_voxtral_small[_judge].parquet, artifacts/feat_voxtral_small[_judge]_crops.parquet
# SMOKE: 3 rows first (download fits on disk: 1 TB free; ~4.2 s per clip, ~4.7 h for all 4061 rows).
import glob, os, subprocess, time

SMOKE = False
SRC = os.path.dirname(glob.glob("/kaggle/input/**/11_voxtral_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 'mistral-common[audio]' bitsandbytes", shell=True, check=True)
subprocess.run("df -h / /kaggle/working /tmp; free -g; nvidia-smi --query-gpu=memory.total --format=csv", shell=True)

lim = " --limit 3" if SMOKE else ""
for a in ["", "--crops"]:
    t = time.time()
    subprocess.run(f"python {SRC}/11_voxtral_embed.py --small {a}{lim}", shell=True, check=True)
    print(f"--small {a}: {time.time() - t:.0f} s", flush=True)
