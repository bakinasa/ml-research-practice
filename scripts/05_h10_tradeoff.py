#!/usr/bin/env python3
"""Follow-up B — H10 as an accuracy–compute trade-off (not an AUROC win).

Re-scores prefix fractions from committed caches and plots AUROC / TPR@5%
retained vs fraction of tokens scored. Old paired tables stay untouched.

Usage
-----
    python scripts/05_h10_tradeoff.py --model mamba-1.4b-hf
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import followup_common as F  # noqa: E402
import minkpp as M  # noqa: E402

_spec = importlib.util.spec_from_file_location("cachemod", HERE / "01_cache.py")
_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache)

FRACS = [0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]


def score_prefix(items, frac: float, k: float = 0.2) -> np.ndarray:
    return np.array([M.h10_prefix(a, b, c, frac=frac, k=k) for a, b, c in items],
                    dtype=float)


def run_model(model: str, dtype: str, n_boot: int) -> pd.DataFrame:
    rows = []
    for split in F.SPLITS:
        p = F.CACHE / f"{model}_{split}_{dtype}.npz"
        if not p.exists():
            print(f"skip {split}: no cache")
            continue
        items, y = _cache.load_cache(p)
        y = np.asarray(y)
        ntok = np.array([len(x[0]) for x in items])
        base = np.array([M.minkpp(a, b, c, k=0.2) for a, b, c in items])
        base_au = 100 * M.auroc(y, base)
        base_tpr = 100 * M.tpr_at_fpr(y, base, 0.05)

        for frac in FRACS:
            s = score_prefix(items, frac)
            au = 100 * M.auroc(y, s)
            tpr = 100 * M.tpr_at_fpr(y, s, 0.05)
            lo, hi, pgt = M.paired_bootstrap(y, s, base, n_boot, seed=0)
            # expected compute: mean tokens scored / mean tokens
            scored = np.maximum(2, np.ceil(frac * ntok))
            compute = float(scored.mean() / ntok.mean())
            rows.append({
                "model": model, "split": split, "frac": frac,
                "compute_frac": compute,
                "auroc": au, "tpr5": tpr,
                "base_auroc": base_au, "base_tpr5": base_tpr,
                "delta_auroc": au - base_au,
                "lo": 100 * lo, "hi": 100 * hi, "p_gt0": pgt,
                "significant": bool(lo > 0 or hi < 0),
                "auroc_retained": au / base_au if base_au else np.nan,
            })
            tag = "SIG" if rows[-1]["significant"] else "n.s."
            print(f"{split} frac={frac:.2f}: AUROC {au:.1f} "
                  f"(d={au - base_au:+.1f})  TPR5 {tpr:.1f}  "
                  f"compute={compute:.0%}  {tag}")
    return pd.DataFrame(rows)


def fig_tradeoff(tab: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for sp in F.SPLITS:
        sub = tab[tab.split == sp].sort_values("frac")
        if sub.empty:
            continue
        c = F.SPLIT_COLOR[sp]
        axes[0].plot(sub.frac, sub.auroc, "-o", color=c, ms=5, lw=1.6,
                     label=F.SPLIT_LABEL[sp])
        axes[0].axhline(sub.base_auroc.iloc[0], color=c, ls="--", lw=1, alpha=0.55)
        axes[1].plot(sub.frac, sub.tpr5, "-o", color=c, ms=5, lw=1.6,
                     label=F.SPLIT_LABEL[sp])
        axes[1].axhline(sub.base_tpr5.iloc[0], color=c, ls="--", lw=1, alpha=0.55)

    F.style(axes[0], "AUROC vs prefix fraction",
            xlabel="fraction of tokens scored", ylabel="AUROC (%)")
    F.style(axes[1], "TPR@5%FPR vs prefix fraction",
            xlabel="fraction of tokens scored", ylabel="TPR@5%FPR (%)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=F.INK_2)
    axes[1].text(0.98, 0.05, "dashed = full-sequence baseline",
                 transform=axes[1].transAxes, ha="right", color=F.INK_2, fontsize=8)
    fig.suptitle(f"H10 accuracy–compute trade-off — {model}",
                 color=F.INK, fontsize=12, fontweight="600", x=0.02, ha="left")
    fig.tight_layout()
    F.save(fig, f"figB_h10_tradeoff_{model}.png")

    # retained AUROC vs compute
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for sp in F.SPLITS:
        sub = tab[tab.split == sp].sort_values("compute_frac")
        if sub.empty:
            continue
        ax.plot(100 * sub.compute_frac, 100 * sub.auroc_retained, "-o",
                color=F.SPLIT_COLOR[sp], ms=5, lw=1.6, label=F.SPLIT_LABEL[sp])
    ax.axhline(100, color=F.NEUTRAL, ls="--", lw=1.2)
    ax.axvline(50, color=F.NEUTRAL, ls=":", lw=1)
    F.style(ax, f"AUROC retained vs compute — {model}",
            xlabel="tokens scored (% of full)", ylabel="AUROC retained (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=F.INK_2)
    F.save(fig, f"figB_h10_pareto_{model}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba-1.4b-hf")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    F.ensure_dirs()
    print(f"=== follow-up B: H10 trade-off ({a.model}) ===")
    tab = run_model(a.model, a.dtype, a.n_boot)
    if tab.empty:
        raise SystemExit("no rows; caches missing?")
    out = F.FOLLOW / f"h10_tradeoff_{a.model}.csv"
    tab.to_csv(out, index=False)
    print(f"wrote {out.relative_to(F.ROOT)}")
    fig_tradeoff(tab, a.model)

    half = tab[np.isclose(tab.frac, 0.5)]
    print("\nat frac=0.5 (≈ half compute):")
    print(half[["split", "auroc", "delta_auroc", "tpr5", "compute_frac"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
