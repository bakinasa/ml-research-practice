#!/usr/bin/env python3
"""Follow-up D — WikiMIA_concat: where H1 windows *should* help.

Dataset: zjysteven/WikiMIA_concat (387 passages). Each item is a concatenation
of 32-word chunks with a per-chunk membership label (0/1). The original study's
mechanism (§5.1 / §11) predicts global Min-K%++ dilutes a localised member
span, while sliding-window aggregation can isolate it.

Protocol (chunk-level AUROC, matching the paper's concat evaluation spirit)
--------------------------------------------------------------------------
1. Run the model once on the full concatenated text; cache per-token
   logp / mu / sigma (same as 01_cache.py).
2. Partition scored tokens into n_chunks contiguous segments (one per label).
3. Score every chunk with:
     - chunk_minkpp     Min-K%++ on that segment only
     - global_minkpp    Min-K%++ on the whole passage (broadcast to chunks)
     - h1_w{W}_overlap  max Min-K%++ over sliding windows that overlap the chunk
4. Pool all chunks across passages → AUROC / TPR@5% / paired bootstrap vs
   chunk_minkpp and vs global_minkpp.

Needs GPU (or patience) the first time to build the concat cache. Default model
is pythia-160m so an RTX 3060 6GB can run it; pass Mamba if you have VRAM.

Usage
-----
    python scripts/07_concat_windows.py --model EleutherAI/pythia-160m --device cuda
    python scripts/07_concat_windows.py --model EleutherAI/pythia-160m --score-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import followup_common as F  # noqa: E402
import minkpp as M  # noqa: E402

SPLIT = "WikiMIA_concat"
HF_DS = "zjysteven/WikiMIA_concat"
WINDOWS = (16, 32, 64)


def short_name(model: str) -> str:
    return model.split("/")[-1]


def cache_path(model: str, dtype: str) -> Path:
    return F.CACHE / f"{short_name(model)}_{SPLIT}_{dtype}.npz"


def build_cache(model_name: str, dtype: str, device: str, limit: int | None) -> Path:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # reuse per-token maths from 01_cache
    import importlib.util
    _spec = importlib.util.spec_from_file_location("cachemod", HERE / "01_cache.py")
    cachemod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cachemod)

    out = cache_path(model_name, dtype)
    F.CACHE.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.npz")
    if out.exists():
        print(f"{out.name} exists, nothing to do")
        return out

    ds = load_dataset(HF_DS, split=SPLIT)
    texts = list(ds["input"])
    chunk_labels = [list(map(int, x)) for x in ds["label"]]
    if limit:
        texts, chunk_labels = texts[:limit], chunk_labels[:limit]

    done = 0
    store: dict = {}
    if part.exists():
        z = np.load(part, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        done = int(store.get("_done", np.array(0)))
        print(f"resuming at item {done}")

    print(f"loading {model_name} ({dtype}) on {device}")
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.to(device).eval()

    t0 = time.time()
    for i in range(done, len(texts)):
        st = cachemod.per_token_stats(model, tok, texts[i], device)
        labs = np.array(chunk_labels[i], dtype=np.int64)
        if st is None:
            store[f"logp_{i}"] = np.zeros(0, np.float32)
            store[f"mu_{i}"] = np.zeros(0, np.float32)
            store[f"sigma_{i}"] = np.zeros(0, np.float32)
        else:
            store[f"logp_{i}"], store[f"mu_{i}"], store[f"sigma_{i}"] = st
        store[f"chunk_labels_{i}"] = labs
        store[f"n_chunks_{i}"] = np.array(len(labs))
        if (i + 1) % 10 == 0 or i == len(texts) - 1:
            store["_done"] = np.array(i + 1)
            np.savez_compressed(part, **store)
            el = time.time() - t0
            rate = el / max(i + 1 - done, 1)
            eta = rate * (len(texts) - i - 1)
            print(f"  {i+1}/{len(texts)}  {rate:.2f}s/item  eta {eta/60:.1f} min",
                  flush=True)

    store["n_items"] = np.array(len(texts))
    store["_meta"] = np.array(json.dumps({
        "model": model_name, "split": SPLIT, "dataset": HF_DS,
        "dtype": dtype, "device": device, "n_items": len(texts),
    }))
    store.pop("_done", None)
    np.savez_compressed(out, **store)
    part.unlink(missing_ok=True)
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB, "
          f"{(time.time()-t0)/60:.1f} min)")
    return out


def load_concat_cache(path: Path):
    z = np.load(path, allow_pickle=True)
    n = int(z["n_items"])
    items, chunk_labs = [], []
    for i in range(n):
        items.append((z[f"logp_{i}"], z[f"mu_{i}"], z[f"sigma_{i}"]))
        chunk_labs.append(np.asarray(z[f"chunk_labels_{i}"]))
    return items, chunk_labs


def segment_bounds(n_tokens: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split scored tokens into n_chunks contiguous segments (as equal as possible)."""
    if n_chunks <= 0 or n_tokens <= 0:
        return []
    edges = np.linspace(0, n_tokens, n_chunks + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_chunks)
            if edges[i + 1] > edges[i]]


def window_starts(n: int, w: int, stride: int) -> list[int]:
    if n <= w:
        return [0]
    starts = list(range(0, n - w + 1, stride))
    if starts[-1] != n - w:
        starts.append(n - w)
    return starts


def minkpp_slice(logp, mu, sigma, lo: int, hi: int, k: float = 0.2) -> float:
    if hi - lo < 2:
        return np.nan
    return M.minkpp(logp[lo:hi], mu[lo:hi], sigma[lo:hi], k=k)


def score_passage(logp, mu, sigma, n_chunks: int, k: float = 0.2) -> dict[str, list[float]]:
    n = len(logp)
    bounds = segment_bounds(n, n_chunks)
    # if tokenisation dropped a chunk, pad/truncate scores later by caller
    out: dict[str, list[float]] = {
        "chunk_minkpp": [],
        "global_minkpp": [],
    }
    for w in WINDOWS:
        out[f"h1_w{w}_overlap"] = []

    g = M.minkpp(logp, mu, sigma, k=k)
    for lo, hi in bounds:
        out["chunk_minkpp"].append(minkpp_slice(logp, mu, sigma, lo, hi, k))
        out["global_minkpp"].append(g)
        for w in WINDOWS:
            stride = max(1, w // 2)
            best = -np.inf
            for s0 in window_starts(n, min(w, n), stride):
                s1 = min(n, s0 + w)
                # overlap with chunk?
                if s1 <= lo or s0 >= hi:
                    continue
                val = minkpp_slice(logp, mu, sigma, s0, s1, k)
                if np.isfinite(val) and val > best:
                    best = val
            out[f"h1_w{w}_overlap"].append(best if np.isfinite(best) else np.nan)
    return out


def evaluate(items, chunk_labs, n_boot: int) -> pd.DataFrame:
    scores: dict[str, list[float]] = {}
    labels: list[int] = []

    for (logp, mu, sigma), labs in zip(items, chunk_labs):
        labs = np.asarray(labs)
        if len(logp) < 4 or labs.size == 0:
            continue
        sc = score_passage(logp, mu, sigma, int(labs.size))
        # align lengths
        m = min(labs.size, min(len(v) for v in sc.values()))
        labels.extend(labs[:m].tolist())
        for name, vals in sc.items():
            scores.setdefault(name, []).extend(vals[:m])

    y = np.array(labels, dtype=int)
    rows = []
    base_name = "chunk_minkpp"
    base = np.array(scores[base_name], dtype=float)
    print(f"chunks scored: n={len(y)}  members={int((y==1).sum())}  "
          f"non={int((y==0).sum())}")

    for name, vals in scores.items():
        s = np.array(vals, dtype=float)
        au = 100 * M.auroc(y, s)
        tpr = 100 * M.tpr_at_fpr(y, s, 0.05)
        lo, hi, pgt = M.paired_bootstrap(y, s, base, n_boot, seed=0)
        d = au - 100 * M.auroc(y, base)
        rows.append({
            "variant": name,
            "auroc": au, "tpr5": tpr,
            "delta_vs_chunk_minkpp": d,
            "lo": 100 * lo, "hi": 100 * hi, "p_gt0": pgt,
            "significant": bool(lo > 0 or hi < 0),
            "n_chunks": int(len(y)),
        })
        tag = "SIG" if rows[-1]["significant"] else "n.s."
        print(f"  {name:20s} AUROC {au:5.1f}  TPR5 {tpr:5.1f}  "
              f"d_vs_chunk {d:+5.2f} [{100*lo:+.2f},{100*hi:+.2f}]  {tag}")
    return pd.DataFrame(rows)


def fig_concat(tab: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt

    order = ["global_minkpp", "chunk_minkpp"] + [f"h1_w{w}_overlap" for w in WINDOWS]
    order = [v for v in order if v in set(tab.variant)]
    sub = tab.set_index("variant").loc[order].reset_index()
    labels = {
        "global_minkpp": "global Min-K%++\n(whole passage)",
        "chunk_minkpp": "chunk Min-K%++\n(oracle segments)",
        "h1_w16_overlap": "H1 w=16\n(overlap max)",
        "h1_w32_overlap": "H1 w=32\n(overlap max)",
        "h1_w64_overlap": "H1 w=64\n(overlap max)",
    }

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(sub))
    colors = [F.NEG if "global" in v else (F.S1 if "chunk" in v else F.S3)
              for v in sub.variant]
    ax.bar(x, sub.auroc, color=colors, edgecolor=F.SURFACE, width=0.72)
    ax.axhline(sub.loc[sub.variant == "chunk_minkpp", "auroc"].iloc[0],
               color=F.NEUTRAL, ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(v, v) for v in sub.variant],
                       color=F.INK, fontsize=8)
    F.style(ax, f"WikiMIA_concat chunk AUROC — {model}",
            ylabel="AUROC (%)")
    F.save(fig, f"figD_concat_auroc_{model}.png")

    # deltas vs chunk minkpp
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    y = np.arange(len(sub))
    for i, r in sub.iterrows():
        color = F.POS if r.delta_vs_chunk_minkpp > 0 else F.NEG
        ax.plot([r.lo, r.hi], [i, i], color=color, lw=2)
        ax.scatter([r.delta_vs_chunk_minkpp], [i], color=color, s=55, zorder=3,
                   edgecolor=F.SURFACE)
    ax.axvline(0, color=F.NEUTRAL, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(v, v).replace("\n", " ") for v in sub.variant],
                       color=F.INK, fontsize=8)
    F.style(ax, f"Δ AUROC vs chunk Min-K%++ — {model}",
            xlabel="Δ AUROC (pp), paired 95% CI")
    F.save(fig, f"figD_concat_delta_{model}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--dtype", default="float16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cache only first N items (smoke test)")
    ap.add_argument("--score-only", action="store_true",
                    help="do not build cache; fail if missing")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    F.ensure_dirs()
    short = short_name(a.model)
    print(f"=== follow-up D: WikiMIA_concat + H1 ({short}) ===")

    path = cache_path(a.model, a.dtype)
    if not path.exists():
        if a.score_only:
            raise SystemExit(
                f"missing cache {path.name}. Re-run without --score-only "
                f"(needs GPU roughly once)."
            )
        # prefer cuda; fall back
        device = a.device
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                print("WARNING: cuda unavailable, using cpu")
                device = "cpu"
        except ImportError:
            device = "cpu"
        build_cache(a.model, a.dtype, device, a.limit)
    else:
        print(f"using cache {path.name}")

    items, chunk_labs = load_concat_cache(path)
    tab = evaluate(items, chunk_labs, a.n_boot)
    tab.insert(0, "model", short)
    out = F.FOLLOW / f"concat_windows_{short}.csv"
    tab.to_csv(out, index=False)
    print(f"wrote {out.relative_to(F.ROOT)}")
    fig_concat(tab, short)

    h1 = tab[tab.variant.str.startswith("h1_")]
    wins = h1[h1.significant & (h1.delta_vs_chunk_minkpp > 0)]
    print(f"\nH1 significant gains vs chunk minkpp: {len(wins)} of {len(h1)}")
    glob = tab[tab.variant == "global_minkpp"].iloc[0]
    print(f"global minkpp vs chunk: d={glob.delta_vs_chunk_minkpp:+.2f} "
          f"[{glob.lo:+.2f},{glob.hi:+.2f}] "
          f"(dilution check; expect <= 0)")


if __name__ == "__main__":
    main()
