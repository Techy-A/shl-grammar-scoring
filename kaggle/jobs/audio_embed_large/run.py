# Kaggle job: step 5 (05_audio_embed.py) with two bigger frozen encoders, one per T4:
#   GPU 0: WavLM-large, layers 8-15                  -> artifacts/feat_wavlm-large.parquet
#   GPU 1: Whisper-large-v3-turbo encoder, layers 16-32 -> artifacts/feat_whisper-large-v3-turbo.parquet
# Too slow (~1.5 h each) and crash-prone on the local 4 GB GTX 1650.
import glob, os, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/05_audio_embed.py", recursive=True)[0])
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)

jobs = [subprocess.Popen(["python", f"{SRC}/05_audio_embed.py", "--model", model, "--layers", *layers],
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        for gpu, model, layers in [("0", "microsoft/wavlm-large", ["8", "16"]),
                                   ("1", "openai/whisper-large-v3-turbo", ["16", "33"])]]
assert all(j.wait() == 0 for j in jobs), "an embedding run failed"
