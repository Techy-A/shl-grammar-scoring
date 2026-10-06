# Kaggle job: two newer representations to strengthen the blend.
#   GPU 0: Qwen3-4B hidden states of the transcripts (full clips and crop transcripts), 04_text_embed.py
#   GPU 1: w2v-BERT 2.0 audio embeddings (full clips and test-length crops), 05_audio_embed.py
#   -> artifacts/feat_emb_Qwen3-4B_prompt[_crops].parquet, artifacts/feat_w2v-bert-2.0[_crops].parquet
import glob, os, shutil, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/04_text_embed.py", recursive=True)[0])
INPUTS = ["transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]   # uploaded with the code (inputs.txt)
os.makedirs("/kaggle/working/artifacts", exist_ok=True)
for f in INPUTS:
    shutil.copy(f"{SRC}/{f}", "/kaggle/working/artifacts/")
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)

def run(gpu, *cmds):   # commands on one GPU run one after another
    return subprocess.Popen(" && ".join(f"python {SRC}/{c}" for c in cmds), shell=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})

procs = [run("0", *[f"04_text_embed.py --tag {t} --model Qwen/Qwen3-4B --layers 12 25" for t in ["prompt", "prompt_crops"]]),
         run("1", *[f"05_audio_embed.py --model facebook/w2v-bert-2.0 --layers 6 18 {c}".strip() for c in ["", "--crops"]])]
ok = [p.wait() == 0 for p in procs]
for f in INPUTS:   # don't download the inputs back
    os.remove(f"/kaggle/working/artifacts/{f}")
assert all(ok), f"a run failed: {ok}"
