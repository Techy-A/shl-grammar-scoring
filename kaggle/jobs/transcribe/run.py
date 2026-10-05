# Kaggle job: full Whisper transcription of train + test, run with and without the verbatim
# prompt in parallel, one run per T4. The two outputs are compared later by downstream CV.
import glob, os, subprocess

SRC = os.path.dirname(glob.glob("/kaggle/input/**/01_transcribe.py", recursive=True)[0])
# Pin transformers to the local version so Kaggle and local runs behave the same.
subprocess.run("pip install -q transformers==5.18.0", shell=True, check=True)

jobs = [subprocess.Popen(["python", f"{SRC}/01_transcribe.py", *args],
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        for gpu, args in [("0", ["--tag", "prompt"]), ("1", ["--tag", "noprompt", "--no-prompt"])]]
assert all(j.wait() == 0 for j in jobs), "a transcription run failed"
