# Kaggle job: step 18 (TAPT: DeBERTa-v3-large MLM on all transcripts, no labels), then four seeds of the fine-tune from
# the adapted encoder (06_finetune_text.py --model <tapt> --crops --drop-zero --epochs 2 --seed n), two per T4, merges.
#   -> artifacts/pred_s4_tapt_deberta_ft_crops_nozero_seed<n>.parquet
import glob, os, shutil, subprocess

PAIRS = [(1, 3), (2, 4)]   # (seed, seed) per GPU
INPUTS = ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]
TAPT = "/kaggle/working/tapt_deberta"
SRC = os.path.dirname(glob.glob("/kaggle/input/**/18_tapt.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
for f in INPUTS:
    shutil.copy(f"{SRC}/{f}", ART)
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)
subprocess.run(f"python {SRC}/18_tapt.py --out {TAPT}", shell=True, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
ft = f"python {SRC}/06_finetune_text.py --model {TAPT} --crops --drop-zero --epochs 2"
procs = [subprocess.Popen(f"{ft} --seed {a} && {ft} --seed {b}", shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
         for gpu, (a, b) in enumerate(PAIRS)]
assert all(p.wait() == 0 for p in procs), "a run failed"
for s in sum(PAIRS, ()):
    subprocess.run(f"{ft} --seed {s} --merge", shell=True, check=True)
for f in INPUTS:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
shutil.rmtree(TAPT)   # 1.7 GB; the predictions are what's needed
