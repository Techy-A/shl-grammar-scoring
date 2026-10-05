# Kaggle job: test-length crops of the train clips for every audio embedding (05_audio_embed.py --crops),
# the length-bias fix. GPU 0: WavLM-base then WavLM-large; GPU 1: Whisper-large-v3-turbo encoder.
#   -> artifacts/feat_<model>_crops.parquet
import glob, os, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/05_audio_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)

def run(gpu, *jobs):   # jobs on one GPU run one after another
    cmd = " && ".join(f"python {SRC}/05_audio_embed.py --crops --model {m} --layers {a} {b}" for m, a, b in jobs)
    return subprocess.Popen(cmd, shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})

procs = [run("0", ("microsoft/wavlm-base-plus", 4, 10), ("microsoft/wavlm-large", 8, 16)),
         run("1", ("openai/whisper-large-v3-turbo", 16, 33))]
assert all(p.wait() == 0 for p in procs), "a crop-embedding run failed"
