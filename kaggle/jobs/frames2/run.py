# Kaggle job: step 19 for two more encoders, Phi-4-multimodal on GPU 0 and WavLM-large on GPU 1 (all clips each), then
# each source's PCA merge.  -> artifacts/phi4_frames.npz, artifacts/wavlm-large_frames.npz (for 20_attn_pool.py --source)
import glob, os, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/19_voxtral_frames.py", recursive=True)[0])
RAW = "/kaggle/tmp/frames"   # outside /kaggle/working, so the raw frames are not downloaded
subprocess.run("pip install -q transformers==5.18.0 peft 'mistral-common[audio]' && pip uninstall -y -q torchao",
               shell=True, check=True)   # Kaggle's old torchao breaks peft's adapter loading (Phi-4's speech adapter)
procs = [subprocess.Popen(f"python {SRC}/19_voxtral_frames.py --source {s} --raw {RAW}", shell=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g, s in enumerate(["phi4", "wavlm-large"])]
assert all(p.wait() == 0 for p in procs), "a frame run failed"
for s in ["phi4", "wavlm-large"]:
    subprocess.run(f"python {SRC}/19_voxtral_frames.py --source {s} --merge --raw {RAW}", shell=True, check=True)
