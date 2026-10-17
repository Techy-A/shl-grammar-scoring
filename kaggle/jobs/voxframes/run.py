# Kaggle job: step 19 (19_voxtral_frames.py), Voxtral-Mini frame-level states of every full clip, one model per T4
# (every second clip each), then the PCA merge.  -> artifacts/voxtral_frames.npz (for 20_attn_pool.py)
import glob, os, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/19_voxtral_frames.py", recursive=True)[0])
RAW = "/kaggle/tmp/voxframes"   # outside /kaggle/working, so the raw shards are not downloaded
subprocess.run("pip install -q transformers==5.18.0 'mistral-common[audio]'", shell=True, check=True)
procs = [subprocess.Popen(f"python {SRC}/19_voxtral_frames.py --shard {g} 2 --raw {RAW}", shell=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g in (0, 1)]
assert all(p.wait() == 0 for p in procs), "a shard failed"
subprocess.run(f"python {SRC}/19_voxtral_frames.py --merge --raw {RAW}", shell=True, check=True)
