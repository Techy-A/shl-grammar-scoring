# Kaggle job: step 15 (15_voxtral_qlora.py), QLoRA fine-tuning of Voxtral-Mini-3B on one T4.
# SMOKE: 16 train clips, 1 epoch, fold 0, to catch install or API errors in minutes.
#   -> artifacts/s5v_fold{k}.parquet
import glob, os, shutil, subprocess

SMOKE = False
SRC = os.path.dirname(glob.glob("/kaggle/input/**/15_voxtral_qlora.py", recursive=True)[0])
os.makedirs("/kaggle/working/artifacts", exist_ok=True)
shutil.copy(f"{SRC}/folds.csv", "/kaggle/working/artifacts/")
subprocess.run("pip install -q transformers==5.18.0 accelerate peft bitsandbytes 'mistral-common[audio]'", shell=True, check=True)
args = "--folds 0 --limit 16 --epochs 1" if SMOKE else "--folds 0 1"   # folds 2-4 run in the shl-qlora-b job; merged locally
subprocess.run(f"python {SRC}/15_voxtral_qlora.py {args}", shell=True, check=True)
os.remove("/kaggle/working/artifacts/folds.csv")
