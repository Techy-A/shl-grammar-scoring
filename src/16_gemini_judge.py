"""Step 16: a frontier audio LLM (Gemini, through its public API) listens to each clip and grades its grammar.

The competition rules allow external models and APIs (section 6b). Every other model here is small and open; this one
hears the audio itself, so the speaker's own errors reach the grader without a transcript in between (Whisper tends to
write what the speaker meant), and it reads the official rubric the way a human rater does.
* Zero-shot: no training labels go into the prompt, so its grades on the training clips are already out-of-fold and
  can be blended (08_blend.py) like any other model's predictions.
* Output per clip (JSON schema, temperature 0): a score 1.0-5.0, the number of grammatical errors heard, and the
  complexity of the structures used (1-5).
* Every answer is cached in artifacts/gemini_cache.jsonl as soon as it arrives, so the run can be stopped and resumed.
* The API key is read from the environment variable GEMINI_API_KEY and sent in a request header, never printed.

Usage : python src/16_gemini_judge.py --list-models
        python src/16_gemini_judge.py --model gemini-2.5-flash --limit 8      # smoke test on 8 training clips
        python src/16_gemini_judge.py --model gemini-2.5-flash --rpm 10       # all clips -> pred_gemini_judge.parquet
"""
import argparse
import base64
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from common import ART, DATA

API = "https://generativelanguage.googleapis.com/v1beta"
RUBRIC = """You are an examiner scoring the GRAMMAR of spontaneous spoken English. You will hear a 45-60 second spoken
answer to an open question from an English test. Listen to the whole answer. Judge sentence structure, syntax, and
grammatical accuracy and complexity of what the speaker actually says (not what they meant). Fillers, hesitations and
self-corrections are normal in speech; accent and pronunciation are not part of this score.

1: Struggles with sentence structure and syntax; limited control of simple grammatical structures.
2: Simple structures with consistent basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of either grammar or sentence structure, with errors in the other.
4: Strong control of sentence structure and grammar; occasional minor errors that don't impede understanding.
5: High grammatical accuracy; complex structures handled well; self-corrects when necessary.

Use the full scale with one decimal (for example 3.5 or 4.2)."""
SCHEMA = {"type": "OBJECT", "properties": {
    "score": {"type": "NUMBER", "description": "grammar score, 1.0 to 5.0"},
    "errors": {"type": "INTEGER", "description": "number of grammatical errors heard"},
    "complexity": {"type": "NUMBER", "description": "complexity of the structures used, 1 to 5"}},
    "required": ["score", "errors", "complexity"]}


def headers():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set the environment variable GEMINI_API_KEY first (Google AI Studio API key).")
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def grade(path, model, thinking):
    body = {"system_instruction": {"parts": [{"text": RUBRIC}]},
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(path.read_bytes()).decode()}},
                {"text": "Grade the grammar of this answer."}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA}}
    if thinking >= 0:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking}
    for attempt in range(8):   # back off on rate limits and transient server errors
        r = requests.post(f"{API}/models/{model}:generateContent", headers=headers(), json=body, timeout=180)
        if r.status_code == 200:
            out = r.json()
            ans = json.loads(out["candidates"][0]["content"]["parts"][-1]["text"])
            return {**ans, "tokens": out.get("usageMetadata", {}).get("totalTokenCount", 0)}
        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == 0:   # show which limit was hit (the message names the quota and its value)
                print(f"HTTP {r.status_code}: {r.text[:400]}", flush=True)
            time.sleep(min(120, 10 * 2 ** attempt))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("gave up after repeated rate-limit / server errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--rpm", type=float, default=10, help="requests per minute (free tier limits are low)")
    ap.add_argument("--thinking", type=int, default=-1, help="thinking token budget (-1: the model's default)")
    ap.add_argument("--limit", type=int, default=0, help="first N training clips only (smoke test)")
    ap.add_argument("--test-first", action="store_true", help="grade the test clips before the training clips")
    ap.add_argument("--workers", type=int, default=3, help="requests in flight at once (replies take ~10 s)")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    if args.list_models:
        ms = requests.get(f"{API}/models?pageSize=200", headers=headers(), timeout=60).json().get("models", [])
        for m in ms:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].split("/")[-1], "|", m.get("displayName", ""))
        return

    clips = pd.concat([pd.read_csv(DATA / "train.csv").assign(split="train"),
                       pd.read_csv(DATA / "test.csv").assign(split="test")], ignore_index=True)
    if args.limit:
        clips = clips.head(args.limit)
    order = clips.sort_values("split", key=lambda s: s != "test", kind="stable") if args.test_first else clips
    cache_path = ART / "gemini_cache.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            done[(c["filename"], c["split"], c["model"])] = c
    todo = [r for r in order.itertuples() if (r.filename, r.split, args.model) not in done]
    lock, slot = threading.Lock(), [time.time()]

    def work(r):   # request starts are spaced 60/rpm s apart across the workers; replies arrive in parallel
        with lock:
            wait = slot[0] - time.time()
            slot[0] = max(slot[0], time.time()) + 60 / args.rpm
        time.sleep(max(0.0, wait))
        return r, grade(DATA / r.split / r.filename, args.model, args.thinking)

    tokens, t0, n = 0, time.time(), 0
    with cache_path.open("a", encoding="utf-8") as cache, ThreadPoolExecutor(args.workers) as pool:
        try:
            for r, ans in pool.map(work, todo):
                c = {"filename": r.filename, "split": r.split, "model": args.model, **ans}
                cache.write(json.dumps(c) + "\n"); cache.flush()
                done[(r.filename, r.split, args.model)] = c
                tokens += ans["tokens"]; n += 1
                if n % 25 == 0 or args.limit:
                    print(f"{n}/{len(todo)}: {r.filename} ({r.split}) score {ans['score']} errors {ans['errors']} "
                          f"| {tokens} tokens so far, {(time.time() - t0) / 60:.1f} min", flush=True)
        except RuntimeError as e:   # e.g. the free tier's daily quota: keep what we have, resume later
            print(f"stopped after {n} new clips: {e}", flush=True)
            pool.shutdown(wait=False, cancel_futures=True)

    rows = [done[(r.filename, r.split, args.model)] for r in clips.itertuples() if (r.filename, r.split, args.model) in done]
    g = pd.DataFrame(rows).merge(clips[["filename", "split", "label"]], on=["filename", "split"], how="left")
    tr = g[(g.split == "train") & (g.label > 0)]
    if len(tr) > 2:
        print(f"training clips scored 1-5 (n={len(tr)}): Pearson(score, label) {np.corrcoef(tr.score, tr.label)[0, 1]:.3f}, "
              f"Pearson(errors, label) {np.corrcoef(tr.errors, tr.label)[0, 1]:.3f}")
    if not args.limit:
        g[["filename", "split", "label", "score", "errors", "complexity"]].to_parquet(ART / "feat_gemini.parquet", index=False)
        if len(g) == len(clips):   # the blend needs every clip
            g.assign(pred=g.score.clip(0, 5))[["filename", "split", "label", "pred"]].to_parquet(
                ART / "pred_gemini_judge.parquet", index=False)
        print(f"saved artifacts/feat_gemini.parquet ({len(g)} of {len(clips)} clips)"
              + (" and pred_gemini_judge.parquet" if len(g) == len(clips) else "; run again to finish"))


if __name__ == "__main__":
    main()
