"""Step 15: QLoRA fine-tuning of Voxtral-Mini-3B to regress the score from the audio.

Frozen Voxtral hidden states (11_voxtral_embed.py) were the biggest single gain, so letting the audio LLM adapt
to the task is the next step (the Voxtral counterpart of 07_finetune_audio.py). Fits an 8 GB GPU:
* The model is loaded in 4-bit (NF4, QLoRA); only LoRA adapters on the language model's attention projections
  (r=16) and a linear head train. The audio encoder stays frozen.
* Input = the clip, then the scoring rubric (the same prompt as 11_voxtral_embed.py).
* Head on the middle layer (where the frozen features were best): mean over the audio tokens that cover real
  audio, concatenated with the last position (which has read the audio and the rubric) -> linear -> score / 5.
* Training on random 30 s crops (one Voxtral audio chunk: augmentation, test-like lengths, bounded memory).
  Prediction = average over 30 s windows weighted by length; validation clips are scored on their test-length
  crops (common.crop_lengths), like 07, so the OOF blends with the other models.
* Fixed epochs and gfold_s1 folds (common.finetune_fold), so the OOF stays honest.

  train : python src/15_voxtral_qlora.py --folds 0 1 2 3 4   -> artifacts/s5v_fold{k}.parquet
  merge : python src/15_voxtral_qlora.py --merge             -> artifacts/pred_s5v_voxtral_qlora.parquet
  smoke : python src/15_voxtral_qlora.py --folds 0 --limit 16 --epochs 1
"""
import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration

from common import AMP, DATA, DTYPE, SR, crop_lengths, finetune_fold, fold_of, merge_folds

MODEL, NAME, PREFIX = "mistralai/Voxtral-Mini-3B-2507", "s5v_voxtral_qlora", "s5v"
CROP = 30 * SR   # one Voxtral audio chunk = 375 tokens
EPOCHS, LR, BS = 3, 1e-4, 2   # batch 2: peak VRAM 7.5 GB at batch 4 on a T4, too close to 8 GB
PROMPT = """You heard a 45-60 second spoken answer from an English test. Score the speaker's GRAMMAR: sentence
structure, syntax, and grammatical accuracy and complexity. Fillers and self-corrections are normal in speech.

1: Struggles with sentence structure and syntax; limited control of simple grammatical structures.
2: Simple structures with consistent basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of either grammar or sentence structure, with errors in the other.
4: Strong control of sentence structure and grammar; occasional minor errors that don't impede understanding.
5: High grammatical accuracy; complex structures handled well; self-corrects when necessary.

Reply with a single digit from 1 to 5."""
TMP = Path(tempfile.gettempdir()) / "voxtral_qlora.wav"   # the chat template takes audio by path


class VoxRegressor(nn.Module):
    def __init__(self, proc):
        super().__init__()
        self.proc = proc
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=DTYPE)
        body = VoxtralForConditionalGeneration.from_pretrained(MODEL, quantization_config=q, device_map={"": 0})
        body = prepare_model_for_kbit_training(body, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
        self.body = get_peft_model(body, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"))
        cfg = body.config
        self.layer, self.audio_id = cfg.text_config.num_hidden_layers // 2, cfg.audio_token_id
        self.head = nn.Linear(2 * cfg.text_config.hidden_size, 1).cuda()

    def forward(self, wavs):
        out = []
        for w in wavs:   # one clip per forward pass: no padding logic, memory bounded by one 30 s chunk
            sf.write(TMP, w, SR)
            conv = [{"role": "user", "content": [{"type": "audio", "path": str(TMP)}, {"type": "text", "text": PROMPT}]}]
            x = self.proc.apply_chat_template(conv).to("cuda")
            x["input_features"] = x["input_features"].to(DTYPE)
            h = self.body(**x, output_hidden_states=True).hidden_states[self.layer][0].float()
            pos = (x["input_ids"][0] == self.audio_id).nonzero().squeeze(-1)
            pos = pos[:math.ceil(len(w) / SR * 12.5)]   # tokens that cover real audio (<= 30 s, one chunk)
            out.append(self.head(torch.cat([h[pos].mean(0), h[-1]])))
        return torch.cat(out)


def random_crop(wav, rng):
    if len(wav) <= CROP:
        return wav
    s = rng.integers(0, len(wav) - CROP)
    return wav[s:s + CROP]


@torch.inference_mode()
def predict(model, wavs):
    model.eval()
    preds = []
    for wav in wavs:
        windows = [wav[s:s + CROP] for s in range(0, len(wav), CROP)]
        windows = [w for w in windows if len(w) >= SR] or [wav]   # drop <1 s tails
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            ps = model(windows).float().cpu().numpy()
        preds.append(np.average(ps, weights=[len(w) for w in windows]) * 5)
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="first N train + 4 test clips only (smoke test)")
    args = ap.parse_args()
    if args.merge:
        return merge_folds(PREFIX, NAME, MODEL + " QLoRA")

    tr, te = pd.read_csv(DATA / "train.csv").assign(split="train"), pd.read_csv(DATA / "test.csv").assign(split="test")
    if args.limit:
        tr, te = tr.head(args.limit), te.head(4)
    df = pd.concat([tr, te], ignore_index=True)
    folds = fold_of(df.filename, "gfold_s1")
    folds[df.split.values == "test"] = -1
    wavs = [sf.read(DATA / r.split / r.filename, dtype="float32")[0] for r in df.itertuples()]
    proc = AutoProcessor.from_pretrained(MODEL)

    def make_model():
        model = VoxRegressor(proc)
        return model, [{"params": [p for p in model.body.parameters() if p.requires_grad], "lr": LR},
                       {"params": model.head.parameters(), "lr": 1e-3}]

    def train_batches(idx, y, rng):   # a fresh random 30 s crop of every clip, every epoch
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            yield [random_crop(wavs[j], rng) for j in b], torch.from_numpy(y[b]).cuda()

    n_tr = len(tr)   # train rows come first
    lens = crop_lengths(len(pd.read_csv(DATA / "train.csv")))   # the shared crops, indexed by train row

    def predict_idx(m, idx):   # train (validation) clips: mean over their crops; test clips: whole clip
        out, is_tr = np.empty(len(idx)), idx < n_tr
        if is_tr.any():
            out[is_tr] = np.mean([predict(m, [wavs[j][:L[j]] for j in idx[is_tr]]) for L in lens], axis=0)
        if (~is_tr).any():
            out[~is_tr] = predict(m, [wavs[j] for j in idx[~is_tr]])
        return out

    for k in args.folds:
        finetune_fold(k, df, folds, make_model, train_batches, predict_idx, args.epochs, BS, PREFIX)
        print(f"fold {k} done, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
