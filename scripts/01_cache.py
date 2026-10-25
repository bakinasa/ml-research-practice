#!/usr/bin/env python3
"""Run a model once over a WikiMIA split and cache the per-token quantities.

This is the only step that needs a model. Everything else in the study is a
function of what this writes, which is why the baseline and every variant are
compared at exactly matched compute.

For each item it stores, per scored token:
    logp   log p(x_t | x_<t)
    mu     E_{v ~ p(.|x_<t)} [ log p(v | x_<t) ]   = -H(p)
    sigma  sqrt( E_{v ~ p} [ (log p(v) - mu)^2 ] )

Position 0 is excluded: it has no preceding context. This matches the released
`run.py` of the paper, where the scored tokens are `input_ids[1:]`.

Checkpoints every 25 items, so an interrupted run resumes instead of restarting.

Usage
-----
    python 01_cache.py --self-check
    python 01_cache.py --model EleutherAI/pythia-160m --split WikiMIA_length32
    python 01_cache.py --model EleutherAI/pythia-160m --all-splits
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = ROOT / "cache"
SPLITS = ["WikiMIA_length32", "WikiMIA_length64", "WikiMIA_length128",
          "WikiMIA_length256"]


def cache_path(model: str, split: str, dtype: str) -> Path:
    return CACHE / f"{model.split('/')[-1]}_{split}_{dtype}.npz"


def per_token_stats(model, tok, text: str, device: str):
    """Return (logp, mu, sigma) arrays for the scored positions of one item."""
    import torch

    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = ids["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        logits = model(input_ids).logits.float()

    # predict position t from positions < t, so drop the last logit row and the
    # first token id
    lp = torch.log_softmax(logits[0, :-1], dim=-1)      # (T-1, V)
    target = input_ids[0, 1:]                            # (T-1,)

    logp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    p = lp.exp()
    mu = (p * lp).sum(-1)                                # = -H(p)
    var = (p * (lp - mu.unsqueeze(-1)) ** 2).sum(-1)
    sigma = var.clamp_min(0).sqrt()

    return (logp.cpu().numpy().astype(np.float32),
            mu.cpu().numpy().astype(np.float32),
            sigma.cpu().numpy().astype(np.float32))


def run(model_name: str, split: str, dtype: str, device: str, limit: int | None):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = cache_path(model_name, split, dtype)
    CACHE.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.npz")

    ds = load_dataset("swj0419/WikiMIA", split=split)
    texts = list(ds["input"])
    labels = [int(x) for x in ds["label"]]
    if limit:
        texts, labels = texts[:limit], labels[:limit]

    done = 0
    store: dict[str, np.ndarray] = {}
    if part.exists():
        z = np.load(part, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        done = int(store.get("_done", np.array(0)))
        print(f"resuming at item {done}")

    if out.exists():
        print(f"{out.name} exists, nothing to do")
        return out

    print(f"loading {model_name} ({dtype}) on {device}")
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.to(device).eval()

    t0 = time.time()
    for i in range(done, len(texts)):
        st = per_token_stats(model, tok, texts[i], device)
        if st is None:
            store[f"logp_{i}"] = np.zeros(0, np.float32)
            store[f"mu_{i}"] = np.zeros(0, np.float32)
            store[f"sigma_{i}"] = np.zeros(0, np.float32)
        else:
            store[f"logp_{i}"], store[f"mu_{i}"], store[f"sigma_{i}"] = st
        if (i + 1) % 25 == 0 or i == len(texts) - 1:
            store["_done"] = np.array(i + 1)
            np.savez_compressed(part, **store)
            el = time.time() - t0
            rate = el / max(i + 1 - done, 1)
            eta = rate * (len(texts) - i - 1)
            print(f"  {i+1}/{len(texts)}  {rate:.2f}s/item  eta {eta/60:.1f} min",
                  flush=True)

    store["labels"] = np.array(labels, np.int64)
    store["n_items"] = np.array(len(texts))
    store["_meta"] = np.array(json.dumps(
        {"model": model_name, "split": split, "dtype": dtype,
         "device": device, "n_items": len(texts)}))
    store.pop("_done", None)
    np.savez_compressed(out, **store)
    part.unlink(missing_ok=True)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB, "
          f"{(time.time()-t0)/60:.1f} min)")
    return out


def load_cache(path: Path):
    """-> (list of (logp, mu, sigma), labels)"""
    z = np.load(path, allow_pickle=True)
    n = int(z["n_items"])
    items = [(z[f"logp_{i}"], z[f"mu_{i}"], z[f"sigma_{i}"]) for i in range(n)]
    return items, z["labels"]


def self_check() -> bool:
    """Verify the per-token maths against a hand-computed two-token example."""
    import torch
    ok = True

    # a tiny fake distribution: logits over a 4-token vocabulary
    logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0],
                            [3.0, 2.0, 1.0, 0.0]]])
    ids = torch.tensor([[1, 2, 0]])
    lp = torch.log_softmax(logits[0], dim=-1)
    p = lp.exp()
    mu = (p * lp).sum(-1)
    var = (p * (lp - mu.unsqueeze(-1)) ** 2).sum(-1)

    # mu must equal the negative entropy
    ent = -(p * lp).sum(-1)
    c1 = torch.allclose(mu, -ent)
    print(f"  [{'PASS' if c1 else 'FAIL'}] mu == -entropy")
    ok &= bool(c1)

    # sigma must be non-negative and finite
    c2 = bool((var >= 0).all() and torch.isfinite(var).all())
    print(f"  [{'PASS' if c2 else 'FAIL'}] sigma finite and non-negative")
    ok &= c2

    # the gather must pick the observed token's log-probability
    got = lp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    want = torch.tensor([lp[0, 2], lp[1, 0]])
    c3 = torch.allclose(got, want)
    print(f"  [{'PASS' if c3 else 'FAIL'}] gather picks the observed token")
    ok &= bool(c3)

    sys.path.insert(0, str(HERE))
    import minkpp
    ok &= minkpp.self_check(verbose=True)
    print(f"\ncache self-check: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--split", default="WikiMIA_length32")
    ap.add_argument("--all-splits", action="store_true")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        sys.exit(0 if self_check() else 1)

    if not self_check():
        sys.exit("self-check failed; refusing to run")

    splits = SPLITS[:3] if a.all_splits else [a.split]
    for s in splits:
        run(a.model, s, a.dtype, a.device, a.limit)


if __name__ == "__main__":
    main()
