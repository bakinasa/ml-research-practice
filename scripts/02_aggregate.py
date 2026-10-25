#!/usr/bin/env python3
"""Score every variant from the caches and write the result tables.

No model and no GPU. Every number in the report comes from here, so the tables
are regenerated rather than transcribed.

Outputs, under results/:
    scores_<model>_<split>.csv    per-item score for every variant
    auroc_<model>.csv             AUROC and TPR@5%FPR per variant per split
    paired_<model>.csv            paired bootstrap of (variant - baseline)
    resolution_<model>.csv        bootstrap half-width per split
    per_item_<model>.csv          per-item diffs vs token count (mechanism)

Usage
-----
    python 02_aggregate.py --self-check
    python 02_aggregate.py --model pythia-160m
    python 02_aggregate.py --model pythia-160m --n-boot 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import minkpp as M  # noqa: E402

_spec = importlib.util.spec_from_file_location("cachemod", HERE / "01_cache.py")
_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache)

SPLITS = ["WikiMIA_length32", "WikiMIA_length64", "WikiMIA_length128"]


def texts_for(split: str) -> list[str]:
    from datasets import load_dataset
    return list(load_dataset("swj0419/WikiMIA", split=split)["input"])


def score_all(items, texts, registry) -> pd.DataFrame:
    cols = {}
    for name, (fn, kw) in registry.items():
        cols[name] = [fn(a, b, c, text=t, **kw)
                      for (a, b, c), t in zip(items, texts)]
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        sys.exit(0 if M.self_check() else 1)
    if not M.self_check(verbose=False):
        sys.exit("minkpp self-check failed; refusing to run")

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    registry = M.build_registry()

    auroc_rows, paired_rows, res_rows, item_rows = [], [], [], []

    for split in SPLITS:
        p = ROOT / "cache" / f"{a.model}_{split}_{a.dtype}.npz"
        if not p.exists():
            print(f"skip {split}: no cache at {p.name}")
            continue
        items, y = _cache.load_cache(p)
        texts = texts_for(split)[:len(items)]
        ntok = np.array([len(x[0]) for x in items])
        print(f"{split}: n={len(items)} scoring {len(registry)} variants")

        df = score_all(items, texts, registry)
        df.insert(0, "label", y)
        df.insert(1, "n_tokens", ntok)
        df.to_csv(out / f"scores_{a.model}_{split}.csv", index=False)

        base = df[M.BASELINE].to_numpy()

        for name in registry:
            s = df[name].to_numpy()
            auroc_rows.append({
                "split": split, "variant": name,
                "family": M.family_of(name),
                "auroc": 100 * M.auroc(y, s),
                "tpr5": 100 * M.tpr_at_fpr(y, s, 0.05),
                "n_nan": int((~np.isfinite(s)).sum()),
            })

        # paired bootstrap of every variant against the baseline
        for name in registry:
            if name == M.BASELINE:
                continue
            s = df[name].to_numpy()
            lo, hi, pgt = M.paired_bootstrap(y, s, base, a.n_boot, seed=0)
            d = 100 * (M.auroc(y, s) - M.auroc(y, base))
            paired_rows.append({
                "split": split, "variant": name, "family": M.family_of(name),
                "delta": d, "lo": 100 * lo, "hi": 100 * hi, "p_gt0": pgt,
                "significant": bool(lo > 0 or hi < 0),
            })

            # mechanism: does the per-item gain depend on length?
            diff = s - base
            ok = np.isfinite(diff)
            if ok.sum() > 10:
                r = np.corrcoef(diff[ok], ntok[ok])[0, 1]
                item_rows.append({"split": split, "variant": name,
                                  "r_diff_vs_ntokens": r})

        # resolution of the benchmark itself
        lo, hi, _ = M.paired_bootstrap(y, base, base, a.n_boot, seed=0)
        rng = np.random.default_rng(0)
        single = []
        for _ in range(a.n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            single.append(M.auroc(y[idx], base[idx]))
        single = np.array(single)
        res_rows.append({
            "split": split, "n": len(y),
            "single_auroc_halfwidth": 100 * (np.percentile(single, 97.5) -
                                             np.percentile(single, 2.5)) / 2,
        })

    pd.DataFrame(auroc_rows).to_csv(out / f"auroc_{a.model}.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / f"paired_{a.model}.csv", index=False)
    pd.DataFrame(res_rows).to_csv(out / f"resolution_{a.model}.csv", index=False)
    pd.DataFrame(item_rows).to_csv(out / f"per_item_{a.model}.csv", index=False)

    pr = pd.DataFrame(paired_rows)
    if not pr.empty:
        print("\nbest variants by mean delta across splits:")
        agg = (pr.groupby(["variant", "family"])["delta"].mean()
                 .sort_values(ascending=False).head(12))
        print(agg.to_string())
        wins = pr[(pr.delta > 0) & pr.significant]
        print(f"\nsignificant improvements: {len(wins)} of {len(pr)} comparisons")
        if len(wins):
            print(wins[["split", "variant", "delta", "lo", "hi"]].to_string(index=False))
    print(f"\nwrote tables to {out}")


if __name__ == "__main__":
    main()
