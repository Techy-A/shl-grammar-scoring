# Kaggle job: transcripts of the test-length crops of the train clips (01_transcribe.py --crops), so the
# text models can also train/validate at test-like lengths. Crops 0-1 on GPU 0, crops 2-3 on GPU 1, then
# the text features (02) and DeBERTa embeddings (04) of those transcripts.
#   -> artifacts/transcripts_prompt_crops.parquet, feat_text_prompt_crops.parquet,
#      feat_emb_deberta-v3-large_prompt_crops.parquet
import glob, os, subprocess
import pandas as pd

SRC = os.path.dirname(glob.glob("/kaggle/input/**/01_transcribe.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)

jobs = [subprocess.Popen(["python", f"{SRC}/01_transcribe.py", "--tag", "prompt", "--crops", *ids],
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        for gpu, ids in [("0", ["0", "1"]), ("1", ["2", "3"])]]
assert all(j.wait() == 0 for j in jobs), "a transcription run failed"
parts = [f"{ART}/transcripts_prompt_crops01.parquet", f"{ART}/transcripts_prompt_crops23.parquet"]
pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(f"{ART}/transcripts_prompt_crops.parquet", index=False)
for p in parts:
    os.remove(p)
for step in ["02_text_features.py", "04_text_embed.py"]:
    subprocess.run(["python", f"{SRC}/{step}", "--tag", "prompt_crops"], check=True)
