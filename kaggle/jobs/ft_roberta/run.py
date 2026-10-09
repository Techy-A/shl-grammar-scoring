# Kaggle job: step 6 with RoBERTa-large (06_finetune_text.py --model roberta-large --crops --drop-zero), a second fine-tuned
# text model; folds 0, 2, 4 on one T4 and 1, 3 on the other, then the merge.
#   -> artifacts/pred_s4_roberta-large_ft_crops_nozero.parquet (+ the per-fold files)
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/06_finetune_text.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:
    shutil.copy(f"{SRC}/{f}", ART)
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)
cmd = f"python {SRC}/06_finetune_text.py --crops --drop-zero --epochs 2 --model roberta-large"
procs = [subprocess.Popen(f"{cmd} --folds {folds}", shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
         for gpu, folds in enumerate(["0 2 4", "1 3"])]
assert all(p.wait() == 0 for p in procs), "a fold failed"
subprocess.run(f"{cmd} --merge", shell=True, check=True)
