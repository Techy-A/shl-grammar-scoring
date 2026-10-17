# Kaggle job (CPU, no GPU session): step 3 (03_train.py) for the 3 blend models, trained without the score-0 clips
# (--drop-zero; the gate model keeps them). Too big for the local 8 GB RAM (Ridge on 3800 rows x 12288 features).
#   -> artifacts/pred_z*.parquet, artifacts/results_eval.csv
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/03_train.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
inputs = [f for f in os.listdir(SRC) if f.endswith((".parquet", ".csv"))]   # every artifact uploaded with the code
for f in inputs:
    shutil.copy(f"{SRC}/{f}", ART)
for spec in ["z4_ridge_deberta ridge feat_emb_deberta-v3-large_prompt",
             "z4c_ridge_vox_allaudio ridge feat_voxtral feat_wavlm-base-plus feat_wavlm-large feat_whisper-large-v3-turbo",
             "z4_svr_wavlm-large svr feat_wavlm-large"]:
    name, model, *feats = spec.split()
    subprocess.run(f"python {SRC}/03_train.py --name {name} --model {model} --feats {' '.join(feats)} --crops --drop-zero",
                   shell=True, check=True)
for f in inputs:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
os.rename(f"{ART}/results.csv", f"{ART}/results_eval.csv")   # never overwrite the local results table
