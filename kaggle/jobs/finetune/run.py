# Kaggle job: step 6 (06_finetune_text.py), DeBERTa-v3-large fine-tune, 5 folds split across the two T4s (folds 0-2 / 3-4).
# The per-fold outputs (artifacts/s4_fold*.parquet) are merged locally with --merge.
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/06_finetune_text.py", recursive=True)[0])
os.makedirs("/kaggle/working/artifacts", exist_ok=True)
for f in ["transcripts_prompt.parquet", "folds.csv"]:   # inputs uploaded with the code (see inputs.txt)
    shutil.copy(f"{SRC}/{f}", "/kaggle/working/artifacts/")
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)

jobs = [subprocess.Popen(["python", f"{SRC}/06_finetune_text.py", "--folds", *folds],
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        for gpu, folds in [("0", ["0", "1", "2"]), ("1", ["3", "4"])]]
assert all(j.wait() == 0 for j in jobs), "a fine-tuning run failed"
for f in ["transcripts_prompt.parquet", "folds.csv"]:    # don't download the inputs back
    os.remove(f"/kaggle/working/artifacts/{f}")
