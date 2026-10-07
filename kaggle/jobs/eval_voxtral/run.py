# Kaggle job (CPU, no GPU session): step 3 (03_train.py) on the Voxtral features. Too big for the local 8 GB RAM
# (SVR on ~3800 rows x 6144-8192 features). Only the prediction files come back; the blends run locally.
#   -> artifacts/pred_x_*.parquet, artifacts/results_eval.csv
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/03_train.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
inputs = [f for f in os.listdir(SRC) if f.endswith((".parquet", ".csv"))]   # every artifact uploaded with the code
for f in inputs:
    shutil.copy(f"{SRC}/{f}", ART)
for spec in ["x_ridge_all ridge feat_voxtral feat_wavlm-base-plus feat_wavlm-large feat_whisper-large-v3-turbo feat_emb_deberta-v3-large_prompt",
             "x_ridge_vox_whisper ridge feat_voxtral feat_whisper-large-v3-turbo"]:
    name, model, *feats = spec.split()
    subprocess.run(f"python {SRC}/03_train.py --name {name} --model {model} --feats {' '.join(feats)} --crops",
                   shell=True, check=True)
for f in inputs:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
os.rename(f"{ART}/results.csv", f"{ART}/results_eval.csv")   # never overwrite the local results table
