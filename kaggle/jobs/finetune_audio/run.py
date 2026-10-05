# Kaggle job: step 7 (07_finetune_audio.py), WavLM fine-tune, 5 folds split across the two T4s (folds 0-2 / 3-4).
# Each process writes its own log into the outputs (Kaggle's log download is unreliable on Windows).
# The per-fold outputs (artifacts/s4b_fold*.parquet) are merged locally with --merge.
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/07_finetune_audio.py", recursive=True)[0])
os.makedirs("/kaggle/working/artifacts", exist_ok=True)
shutil.copy(f"{SRC}/folds.csv", "/kaggle/working/artifacts/")   # input uploaded with the code
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)

jobs = []
for gpu, folds in [("0", ["0", "1", "2"]), ("1", ["3", "4"])]:
    log = open(f"/kaggle/working/s4b_gpu{gpu}.log", "w")
    jobs.append(subprocess.Popen(["python", f"{SRC}/07_finetune_audio.py", "--folds", *folds],
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT))
codes = [j.wait() for j in jobs]
os.remove("/kaggle/working/artifacts/folds.csv")   # don't download the input back
assert codes == [0, 0], f"a fine-tuning run failed: {codes}"
