# Kaggle job: step 6, four more seeds of the fine-tuned DeBERTa-v3-large in the final model
# (06_finetune_text.py --crops --drop-zero --epochs 2 --seed 1..4); seeds 1, 3 on one T4 and 2, 4 on the other,
# then the merges. A single fine-tune is noisy, so the seed average should be a steadier text model.
#   -> artifacts/pred_s4_deberta_ft_crops_nozero_seed{1..4}.parquet (+ the per-fold files)
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/06_finetune_text.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:
    shutil.copy(f"{SRC}/{f}", ART)
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)
cmd = f"python {SRC}/06_finetune_text.py --crops --drop-zero --epochs 2"
procs = [subprocess.Popen(f"{cmd} --seed {a} && {cmd} --seed {b}", shell=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
         for gpu, (a, b) in enumerate([(1, 3), (2, 4)])]
assert all(p.wait() == 0 for p in procs), "a seed run failed"
for s in range(1, 5):
    subprocess.run(f"{cmd} --seed {s} --merge", shell=True, check=True)
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
