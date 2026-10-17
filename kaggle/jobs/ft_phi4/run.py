# Kaggle job: the text models on Phi-4 transcripts (which keep more of the speaker's grammar than Whisper's):
# frozen DeBERTa embeddings of the full clips and crops (04_text_embed.py --tag phi4[_crops]), then four seeds of the
# fine-tuned DeBERTa (06_finetune_text.py --tag phi4 --crops --drop-zero --epochs 2 --seed n), two per T4, then the merges.
#   -> artifacts/feat_emb_deberta-v3-large_phi4[_crops].parquet, pred_s4_deberta_ft_crops_nozero_phi4_seed<n>.parquet
import glob, os, shutil, subprocess

PAIRS = [(1, 3), (2, 4)]   # (seed, seed) per GPU
INPUTS = ["folds.csv", "transcripts_phi4.parquet", "transcripts_phi4_crops.parquet"]
SRC = os.path.dirname(glob.glob("/kaggle/input/**/06_finetune_text.py", recursive=True)[0])
ART = "/kaggle/working/artifacts"
os.makedirs(ART, exist_ok=True)
for f in INPUTS:
    shutil.copy(f"{SRC}/{f}", ART)
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)
ft = f"python {SRC}/06_finetune_text.py --tag phi4 --crops --drop-zero --epochs 2"
emb = f"python {SRC}/04_text_embed.py --tag phi4 && python {SRC}/04_text_embed.py --tag phi4_crops && "
procs = [subprocess.Popen((emb if gpu == 0 else "") + f"{ft} --seed {a} && {ft} --seed {b}", shell=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
         for gpu, (a, b) in enumerate(PAIRS)]
assert all(p.wait() == 0 for p in procs), "a run failed"
for s in sum(PAIRS, ()):
    subprocess.run(f"{ft} --seed {s} --merge", shell=True, check=True)
for f in INPUTS:   # don't download the inputs back
    os.remove(f"{ART}/{f}")
