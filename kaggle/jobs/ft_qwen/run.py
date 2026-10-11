# Kaggle job: step 6 with a decoder LLM, Qwen2.5-7B-Instruct (Apache-2.0) in 4-bit with LoRA adapters
# (06_finetune_text.py --lora --crops --drop-zero --epochs 2). FOLDS per T4; the pilot runs folds 0 and 1 only and is
# kept only if its validation RMSE clearly beats the DeBERTa fine-tune's on the same folds.
#   -> artifacts/s4cz_Qwen2.5-7B-Instruct_fold{k}.parquet (merged locally with --merge once all 5 exist)
import glob, os, shutil, subprocess

FOLDS = ["0", "1"]   # one entry per GPU, e.g. ["2 4", "3"] for the rest
SRC = os.path.dirname(glob.glob("/kaggle/input/**/06_finetune_text.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:
    shutil.copy(f"{SRC}/{f}", ART)
subprocess.run("pip install -q transformers==5.18.0 peft bitsandbytes", shell=True, check=True)
cmd = f"python {SRC}/06_finetune_text.py --lora --model Qwen/Qwen2.5-7B-Instruct --crops --drop-zero --epochs 2"
procs = [subprocess.Popen(f"{cmd} --folds {folds}", shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
         for gpu, folds in enumerate(FOLDS)]
assert all(p.wait() == 0 for p in procs), "a fold failed"
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
