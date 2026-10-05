"""Step 5: self-supervised audio embeddings (WavLM).

Transcripts lose *how* something was said: hesitation, rhythm, pronunciation. Raters hear those,
and they correlate with grammar scores, so WavLM's speech representations add a second view.

Length robustness: each clip is cut into 20 s windows, every window is embedded on its own, and
the windows are averaged (weighted by length). So a 45 s test clip and a 60 s train clip produce
comparable vectors.

Pooling: hidden states of a middle-layer range (default 4-9 for base; middle layers are where WavLM keeps phonetic and prosodic
information) are averaged, then mean and std over time -> 2 x hidden size features (1536 for base, 2048 for large).

Whisper (--model openai/whisper-*): only its encoder is used. Whisper always pads input to 30 s,
so the padding frames (50 per second of audio) are cut off before pooling.

Output: artifacts/feat_<model short name>.parquet (filename, split, label, a0..)
Usage : python src/05_audio_embed.py                                   # wavlm-base-plus, layers 4-9
        python src/05_audio_embed.py --model microsoft/wavlm-large --layers 8 16
        python src/05_audio_embed.py --model openai/whisper-large-v3-turbo --layers 16 33
"""
import argparse

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel, WhisperModel

from common import ART, DATA, DTYPE, SR

WINDOW = 20 * SR


@torch.inference_mode()
def embed_clip(wav, fe, model, layers):
    vecs, weights = [], []
    for start in range(0, len(wav), WINDOW):
        w = wav[start:start + WINDOW]
        if len(w) < SR:   # skip a trailing piece under 1 s: too short to describe anything
            continue
        x = fe(w, sampling_rate=SR, return_tensors="pt")
        x = (x.input_features if "input_features" in x else x.input_values).to("cuda", DTYPE)
        hs = model(x, output_hidden_states=True).hidden_states[layers]
        h = torch.stack(hs).float().mean(0)[0]                   # (frames, hidden), averaged over layers
        h = h[:len(w) // 320 + 1]                                # Whisper: drop the 30 s padding (no-op for WavLM)
        vecs.append(torch.cat([h.mean(0), h.std(0)]).cpu().numpy())
        weights.append(len(w))
    return np.average(vecs, axis=0, weights=weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--layers", type=int, nargs=2, default=[4, 10], help="hidden-state slice [start, stop)")
    args = ap.parse_args()
    layers = slice(*args.layers)

    files = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                       pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    model = (WhisperModel.from_pretrained(args.model, dtype=DTYPE).encoder if "whisper" in args.model
             else WavLMModel.from_pretrained(args.model, dtype=DTYPE)).to("cuda").eval()

    emb = np.stack([embed_clip(sf.read(DATA / r.split / r.filename, dtype="float32")[0], fe, model, layers)
                    for r in tqdm(files.itertuples(), total=len(files))])
    feats = pd.DataFrame(emb, columns=[f"a{i}" for i in range(emb.shape[1])])
    name = f"feat_{args.model.split('/')[-1]}"
    pd.concat([files[["filename", "split", "label"]], feats], axis=1).to_parquet(ART / f"{name}.parquet", index=False)
    print(f"saved {name}: {emb.shape}, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
