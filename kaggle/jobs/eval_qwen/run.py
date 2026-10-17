# Kaggle job (CPU, no GPU session): step 3 (03_train.py) on the Qwen2-Audio features. Too big for the local 8 GB RAM
# (SVR on ~3800 rows x 6144-8192 features). Only the prediction files come back; the blends run locally.
#   -> artifacts/pred_x_*.parquet, artifacts/results_eval.csv
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/03_train.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
inputs = [f for f in os.listdir(SRC) if f.endswith((".parquet", ".csv"))]   # every artifact uploaded with the code
for f in inputs:
    shutil.copy(f"{SRC}/{f}", ART)
for spec in ["x_ridge_qwen ridge feat_qwen2audio",
             "x_ridge_qwen_vox_allaudio ridge feat_qwen2audio feat_voxtral feat_wavlm-base-plus feat_wavlm-large feat_whisper-large-v3-turbo",
             "x_ridge_qwen_q ridge feat_qwen2audio_q",
             "x_ridge_qwen_both ridge feat_qwen2audio feat_qwen2audio_q",
             "x_svr_qwen_wavlml svr feat_qwen2audio feat_wavlm-large"]:
    name, model, *feats = spec.split()
    subprocess.run(f"python {SRC}/03_train.py --name {name} --model {model} --feats {' '.join(feats)} --crops",
                   shell=True, check=True)
for f in inputs:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
os.rename(f"{ART}/results.csv", f"{ART}/results_eval.csv")   # never overwrite the local results table
