# Kaggle job: step 6, four more seeds of the fine-tuned DeBERTa-v3-large in the final model
# (06_finetune_text.py --crops --drop-zero --epochs 2 --seed n); two seeds per T4 (PAIRS), then the merges.
# Version 1 ran seeds 1-4, version 2 seeds 5-8. A single fine-tune is noisy, so the seed average should be a steadier text model.
#   -> artifacts/pred_s4_deberta_ft_crops_nozero_seed<n>.parquet (+ the per-fold files)
import glob, os, shutil, subprocess

PAIRS = [(5, 7), (6, 8)]   # (seed, seed) per GPU

SRC = os.path.dirname(glob.glob("/kaggle/input/**/06_finetune_text.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:
    shutil.copy(f"{SRC}/{f}", ART)
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)
cmd = f"python {SRC}/06_finetune_text.py --crops --drop-zero --epochs 2"
procs = [subprocess.Popen(f"{cmd} --seed {a} && {cmd} --seed {b}", shell=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
         for gpu, (a, b) in enumerate(PAIRS)]
assert all(p.wait() == 0 for p in procs), "a seed run failed"
for s in sum(PAIRS, ()):
    subprocess.run(f"{cmd} --seed {s} --merge", shell=True, check=True)
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
