# Kaggle job: step 12 (12_llm_judge.py), Qwen3-8B reads every transcript with the rubric. fp16 weights (16 GB)
# span both T4s (device_map="auto"), so the full clips and the crops run one after the other.
#   -> artifacts/feat_llm_judge.parquet, artifacts/feat_llm_judge_crops.parquet
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/12_llm_judge.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0 accelerate", shell=True, check=True)
# The transcripts travel in the code dataset (kaggle/jobs/llm_judge/inputs.txt); the script reads them from ART.
os.makedirs("/kaggle/working/artifacts", exist_ok=True)
for f in ("transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"):
    shutil.copy(f"{SRC}/{f}", "/kaggle/working/artifacts/")
for extra in ("", "--crops"):
    subprocess.run(f"python {SRC}/12_llm_judge.py {extra}", shell=True, check=True)
for f in ("transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"):   # don't download them back
    os.remove(f"/kaggle/working/artifacts/{f}")
