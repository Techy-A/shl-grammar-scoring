"""Step 17: audio-LLM embeddings from Phi-4-multimodal-instruct (MIT), the same recipe as 11_voxtral_embed.py.

Voxtral's audio encoder is Whisper-derived and its language model is Mistral; Qwen2-Audio (also Whisper-derived) added
nothing next to it. Phi-4-multimodal differs on both sides: a conformer speech encoder and the Phi-4-mini language model,
so its hidden states are the most likely to carry information Voxtral misses.

One forward pass per clip: the audio, then the scoring rubric (11_voxtral_embed.PROMPT). Hidden states of the
lower-middle layers (20-60% of depth) are averaged, then mean and std over the audio tokens -> 2 x 3072 features.
The audio comes first and attention is causal, so these depend on the audio only. The speech LoRA adapter that ships
with the model is switched on (the model's speech skills live in it).

--transcribe writes transcripts instead (greedy decoding) -> transcripts_phi4[_crops][_shard<k>].parquet, same columns
as transcripts_prompt (filename, split, label, text) for 06_finetune_text.py --tag phi4. Unlike Whisper, Phi-4 keeps
much of the speaker's grammar as spoken ("having a more more greenery trees"), which is what a grammar scorer needs.

--crops / --shard / --limit as in 11_voxtral_embed.py. --check transcribes the first clip and prints it (smoke test:
proves the weights and the speech adapter loaded, rather than silently random-initialised layers).
Output: artifacts/feat_phi4mm[_crops][_shard<k>].parquet (filename, split, label, a0..)
Usage : python src/17_phi4mm_embed.py [--crops] [--shard 0 2] [--limit 3] [--check] [--transcribe]
"""
import argparse
import importlib

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from common import ART, DATA, DTYPE, N_CROPS, SR, crop_lengths

ASR = "Transcribe the audio to text."
MODEL = "microsoft/Phi-4-multimodal-instruct"
REV = "refs/pr/70"   # the weights converted to the transformers-native format (main still needs custom code)
PROMPT = importlib.import_module("11_voxtral_embed").PROMPT


@torch.inference_mode()
def embed_clip(wav, proc, model, layers):
    x = proc(text=f"<|user|><|audio|>{PROMPT}<|end|><|assistant|>", audio=[wav], sampling_rate=SR,
             return_tensors="pt").to("cuda")
    x["audio_input_features"] = x["audio_input_features"].to(DTYPE)
    hs = model(**x, output_hidden_states=True).hidden_states
    h = torch.stack([hs[l][0] for l in layers]).float().mean(0)   # (tokens, hidden), averaged over layers
    h = h[x["input_ids"][0] == proc.audio_token_id]                 # the audio tokens (no padding: one per 80 ms)
    return torch.cat([h.mean(0), h.std(0)]).cpu().numpy()


@torch.inference_mode()
def transcribe(wav, proc, model):
    x = proc(text=f"<|user|><|audio|>{ASR}<|end|><|assistant|>", audio=[wav], sampling_rate=SR, return_tensors="pt").to("cuda")
    x["audio_input_features"] = x["audio_input_features"].to(DTYPE)
    out = model.generate(**x, max_new_tokens=640, do_sample=False)   # 60 s of speech: at most ~450 words
    return proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", action="store_true", help="embed test-length crops of the train clips instead")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("K", "N"), help="rows K, K+N, K+2N, ...")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--check", action="store_true", help="transcribe the first clip and print it")
    ap.add_argument("--transcribe", action="store_true", help="write transcripts_phi4 instead of embeddings")
    args = ap.parse_args()

    # Same row list as 05_audio_embed.py / 11_voxtral_embed.py, so crops line up with the other audio tables.
    files = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                       pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    files["n"] = 10**9
    if args.crops:
        tr = files[files.split == "train"]
        lens = crop_lengths(len(tr))
        files = pd.concat([tr.assign(split=f"crop{i}", n=lens[i]) for i in range(N_CROPS)], ignore_index=True)
    k, n = args.shard
    files = files.iloc[k::n].reset_index(drop=True)
    if args.limit:
        files = files.head(args.limit)

    proc = AutoProcessor.from_pretrained(MODEL, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=DTYPE).to("cuda").eval()
    model.load_adapter(MODEL, adapter_name="speech", adapter_kwargs={"subfolder": "speech-lora", "revision": REV})
    model.set_adapter("speech")
    L = model.config.num_hidden_layers
    layers = range(int(0.2 * L), int(0.6 * L) + 1)   # hidden_states[0] is the embedding layer

    folder = lambda split: "test" if split == "test" else "train"   # crops are cut from train clips
    read = lambda r: sf.read(DATA / folder(r.split) / r.filename, dtype="float32")[0][:r.n]
    sfx = ("_crops" if args.crops else "") + (f"_shard{k}" if n > 1 else "")
    if args.check:
        print("check transcript:", transcribe(read(next(files.itertuples())), proc, model))
    if args.transcribe:
        text = [transcribe(read(r), proc, model) for r in tqdm(files.itertuples(), total=len(files), mininterval=60)]
        files[["filename", "split", "label"]].assign(text=text).to_parquet(ART / f"transcripts_phi4{sfx}.parquet", index=False)
        return print(f"saved transcripts_phi4{sfx}: {len(text)} rows, mean {np.mean([len(t.split()) for t in text]):.0f} words")

    emb = np.stack([embed_clip(read(r), proc, model, layers)
                    for r in tqdm(files.itertuples(), total=len(files), mininterval=60)])
    # ponytail: an fp16 overflow fails the whole run (Phi-4 was trained in bf16); add an fp32 fallback if it happens
    assert np.isfinite(emb).all(), "non-finite outputs (fp16 overflow?)"
    pd.concat([files[["filename", "split", "label"]], pd.DataFrame(emb, columns=[f"a{i}" for i in range(emb.shape[1])])],
              axis=1).to_parquet(ART / f"feat_phi4mm{sfx}.parquet", index=False)
    print(f"saved feat_phi4mm{sfx} {emb.shape}, layers {layers.start}-{layers.stop - 1} of {L}, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
