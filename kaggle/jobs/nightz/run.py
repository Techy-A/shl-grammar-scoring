# Kaggle job (CPU, overnight): step 3 (03_train.py --drop-zero) for extra blend candidates, from the saved GPU features
# of the verified run (dataset shl-features), so that a winning blend from the overnight search can be built with the
# real pipeline. -> artifacts/pred_z*.parquet, artifacts/results_eval.csv
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/03_train.py", recursive=True)[0])
FEAT = os.path.dirname(glob.glob("/kaggle/input/**/feat_voxtral_crops.parquet", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
inputs = [f"{FEAT}/{f}" for f in os.listdir(FEAT) if f.startswith("feat_")] + [f"{SRC}/folds.csv"]
for f in inputs:
    shutil.copy(f, ART)
for spec in ["z4_ridge_whisper ridge feat_whisper-large-v3-turbo",
             "z4_ridge_wavlm ridge feat_wavlm-base-plus",
             "z4_svr_wavlm svr feat_wavlm-base-plus",
             "zx_svr_voxtral_wavlml svr feat_voxtral feat_wavlm-large",
             "z4c_svr_allaudio svr feat_wavlm-base-plus feat_wavlm-large feat_whisper-large-v3-turbo"]:
    name, model, *feats = spec.split()
    subprocess.run(f"python {SRC}/03_train.py --name {name} --model {model} --feats {' '.join(feats)} --crops --drop-zero",
                   shell=True, check=True)
for f in inputs:   # don't download the inputs back
    os.remove(f"{ART}/{os.path.basename(f)}")
os.rename(f"{ART}/results.csv", f"{ART}/results_eval.csv")   # never overwrite the local results table
