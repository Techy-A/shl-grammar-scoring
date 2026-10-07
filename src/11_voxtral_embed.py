"""Step 11: audio-LLM embeddings (Voxtral-Mini-3B).

The other audio encoders (05_audio_embed.py) are generic speech models. Voxtral is an audio language model,
and one forward pass gives two things:
* embeddings: hidden states of its language model over the audio tokens. The audio comes before the text in
  the prompt and attention is causal, so these depend on the audio only.
* a direct judgement: after the audio, the scoring rubric asks for one digit 1-5, and the next-token
  probabilities of "1".."5" give a score (their weighted mean), as in 12_llm_judge.py but from the audio itself.
The idea of using an audio LLM this way comes from another participant's public solution; this
implementation, the direct judgement and the crop training are ours.

Pooling: hidden states of the lower-middle layers (20-60% of depth) are averaged, then mean and std over
the audio tokens -> 2 x 3072 features. --upper pools the upper layers (60-90%) instead, saved as feat_voxtral_hi
(the judgement is the same either way, so it isn't saved again). Voxtral pads audio to 30 s chunks of 375 tokens (12.5 per second),
so tokens that only cover padding are dropped before pooling (as the Whisper padding in 05_audio_embed.py).

--crops embeds the same test-length crops of the train clips as 05_audio_embed.py --crops, so
03_train.py --crops can train and validate at test-like lengths. --shard k n embeds every n-th row from
row k (two T4s, one model each); the Kaggle job merges the shards.

Output: artifacts/feat_voxtral[_crops][_shard<k>].parquet (filename, split, label, a0..)
        artifacts/feat_voxtral_judge[_crops][_shard<k>].parquet (filename, split, label, v_mean, v_p1..v_p5)
Usage : python src/11_voxtral_embed.py [--crops] [--shard 0 3] [--limit 5]
"""
import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor, VoxtralForConditionalGeneration

from common import ART, DATA, DTYPE, N_CROPS, SR, crop_lengths

MODEL = "mistralai/Voxtral-Mini-3B-2507"
PROMPT = """You heard a 45-60 second spoken answer from an English test. Score the speaker's GRAMMAR: sentence
structure, syntax, and grammatical accuracy and complexity. Fillers and self-corrections are normal in speech.

1: Struggles with sentence structure and syntax; limited control of simple grammatical structures.
2: Simple structures with consistent basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of either grammar or sentence structure, with errors in the other.
4: Strong control of sentence structure and grammar; occasional minor errors that don't impede understanding.
5: High grammatical accuracy; complex structures handled well; self-corrects when necessary.

Reply with a single digit from 1 to 5."""
CHUNK_TOKENS, TOKENS_PER_SEC = 375, 12.5   # Voxtral: one 30 s chunk = 375 audio tokens


@torch.inference_mode()
def embed_clip(wav, proc, model, layers, digits, tmp):
    sf.write(tmp, wav, SR)   # the chat template takes audio by path
    conv = [{"role": "user", "content": [{"type": "audio", "path": str(tmp)}, {"type": "text", "text": PROMPT}]}]
    x = proc.apply_chat_template(conv).to("cuda", dtype=DTYPE)
    out = model(**x, output_hidden_states=True)
    probs = torch.softmax(out.logits[0, -1, digits].float(), 0).cpu().numpy()   # the answer digit
    hs = out.hidden_states
    h = torch.stack([hs[l][0] for l in layers]).float().mean(0)   # (tokens, hidden), averaged over layers
    # Audio tokens that cover real audio: in chunk c, the first ceil(seconds of audio in c * 12.5).
    pos = (x["input_ids"][0] == model.config.audio_token_id).nonzero().squeeze(-1)
    dur = len(wav) / SR
    keep = [p for i, p in enumerate(pos.tolist())
            if i % CHUNK_TOKENS < math.ceil(min(max(dur - 30 * (i // CHUNK_TOKENS), 0), 30) * TOKENS_PER_SEC)]
    h = h[keep]
    return torch.cat([h.mean(0), h.std(0)]).cpu().numpy(), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", action="store_true", help="embed test-length crops of the train clips instead")
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1], metavar=("K", "N"), help="rows K, K+N, K+2N, ...")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--upper", action="store_true", help="pool layers 60-90%% of depth -> feat_voxtral_hi")
    args = ap.parse_args()

    # Same row list as 05_audio_embed.py, so crops line up with the other audio tables.
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

    proc = AutoProcessor.from_pretrained(MODEL)
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL, dtype=DTYPE).to("cuda").eval()
    L = model.config.text_config.num_hidden_layers
    lo, hi = (0.6, 0.9) if args.upper else (0.2, 0.6)
    layers = range(int(lo * L), int(hi * L) + 1)   # hidden_states[0] is the embedding layer
    digits = [proc.tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]
    assert len(set(digits)) == 5, f"digit token ids not distinct: {digits}"

    folder = lambda split: "test" if split == "test" else "train"   # crops are cut from train clips
    tmp = Path(tempfile.gettempdir()) / f"voxtral_{k}.wav"
    emb, probs = map(np.stack, zip(*[embed_clip(sf.read(DATA / folder(r.split) / r.filename, dtype="float32")[0][:r.n],
                                                proc, model, layers, digits, tmp)
                                     for r in tqdm(files.itertuples(), total=len(files), mininterval=60)]))
    # ponytail: an fp16 overflow fails the whole run; add a per-clip fp32 retry if it ever happens
    assert np.isfinite(emb).all() and np.isfinite(probs).all(), "non-finite outputs (fp16 overflow?)"
    ids = files[["filename", "split", "label"]]
    sfx = ("_crops" if args.crops else "") + (f"_shard{k}" if n > 1 else "")
    pd.concat([ids, pd.DataFrame(emb, columns=[f"a{i}" for i in range(emb.shape[1])])], axis=1).to_parquet(
        ART / f"feat_voxtral{'_hi' if args.upper else ''}{sfx}.parquet", index=False)
    if not args.upper:
        ids.assign(v_mean=probs @ np.arange(1, 6), **{f"v_p{d}": probs[:, d - 1] for d in range(1, 6)}).to_parquet(
            ART / f"feat_voxtral_judge{sfx}.parquet", index=False)
    print(f"saved feat_voxtral{'_hi' if args.upper else ''}{sfx} {emb.shape}, layers {layers.start}-{layers.stop - 1} of {L}, "
          f"judge mean {(probs @ np.arange(1, 6)).mean():.2f}, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
