#!/usr/bin/env python3
"""Follow-up C — H7 length-conditional thresholds (TPR@5%FPR only).

AUROC is invariant to monotone per-item recalibration, so this experiment
never claims an AUROC gain. It fits separate decision thresholds for short vs
long passages (by token count) at a fixed 5% FPR on training folds, then
measures TPR@5% on held-out folds vs a single global threshold.

Uses committed caches — no GPU.

Usage
-----
    python scripts/06_h7_calibration.py --model mamba-1.4b-hf
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import followup_common as F  # noqa: E402
import minkpp as M  # noqa: E402

_spec = importlib.util.spec_from_file_location("cachemod", HERE / "01_cache.py")
_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache)


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.05) -> float:
    """Score threshold so that empirical FPR on negatives ≈ fpr (higher=member)."""
    neg = np.sort(s[y == 0])
    if neg.size == 0:
        return np.nan
    # same convention as minkpp.tpr_at_fpr
    return float(neg[max(0, int(np.ceil((1 - fpr) * neg.size)) - 1)])


def tpr_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    if not np.isfinite(thr) or (y == 1).sum() == 0:
        return np.nan
    return float((s[y == 1] > thr).mean())


def eval_global(y, s, fpr=0.05):
    thr = threshold_at_fpr(y, s, fpr)
    return tpr_at_threshold(y, s, thr), thr


def eval_length_bins(y, s, ntok, train_idx, test_idx, fpr=0.05, n_bins=2):
    """Fit per-bin thresholds on train; apply on test. Bins by train median / quantiles."""
    y_tr, s_tr, n_tr = y[train_idx], s[train_idx], ntok[train_idx]
    y_te, s_te, n_te = y[test_idx], s[test_idx], ntok[test_idx]

    if n_bins == 2:
        edges = [-np.inf, np.median(n_tr), np.inf]
    else:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(n_tr, qs)
        edges[0], edges[-1] = -np.inf, np.inf

    pred_hit = np.zeros(len(test_idx), dtype=bool)
    used = np.zeros(len(test_idx), dtype=bool)
    for lo, hi in zip(edges[:-1], edges[1:]):
        tr_m = (n_tr > lo) & (n_tr <= hi) if np.isfinite(lo) else (n_tr <= hi)
        te_m = (n_te > lo) & (n_te <= hi) if np.isfinite(lo) else (n_te <= hi)
        if tr_m.sum() < 10 or (y_tr[tr_m] == 0).sum() < 5:
            continue
        thr = threshold_at_fpr(y_tr[tr_m], s_tr[tr_m], fpr)
        if not np.isfinite(thr):
            continue
        pred_hit[te_m] = s_te[te_m] > thr
        used[te_m] = True

    # fallback: global train threshold for unused test items
    thr_g = threshold_at_fpr(y_tr, s_tr, fpr)
    pred_hit[~used] = s_te[~used] > thr_g

    # empirical FPR / TPR on test
    fpr_hat = float((pred_hit[y_te == 0]).mean()) if (y_te == 0).sum() else np.nan
    tpr_hat = float((pred_hit[y_te == 1]).mean()) if (y_te == 1).sum() else np.nan
    return tpr_hat, fpr_hat


def run_split(items, y, n_folds: int, n_bins: int, seed: int = 0) -> dict:
    y = np.asarray(y)
    ntok = np.array([len(x[0]) for x in items])
    s = np.array([M.minkpp(a, b, c, k=0.2) for a, b, c in items], dtype=float)
    ok = np.isfinite(s)
    y, s, ntok = y[ok], s[ok], ntok[ok]

    # oracle single-split metrics (fit+eval on all — upper reference only)
    tpr_all, _ = eval_global(y, s)
    auroc = 100 * M.auroc(y, s)

    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    glob_tpr, bin_tpr, glob_fpr, bin_fpr = [], [], [], []
    for tr, te in skf.split(s.reshape(-1, 1), y):
        # global threshold from train
        thr = threshold_at_fpr(y[tr], s[tr])
        glob_tpr.append(tpr_at_threshold(y[te], s[te], thr))
        glob_fpr.append(float((s[te][y[te] == 0] > thr).mean())
                        if (y[te] == 0).sum() else np.nan)
        t, f = eval_length_bins(y, s, ntok, tr, te, n_bins=n_bins)
        bin_tpr.append(t)
        bin_fpr.append(f)

    return {
        "n": int(len(y)),
        "auroc": auroc,  # unchanged by H7; reported for context
        "tpr5_insample": 100 * tpr_all,
        "tpr5_cv_global": 100 * np.nanmean(glob_tpr),
        "tpr5_cv_length": 100 * np.nanmean(bin_tpr),
        "fpr_cv_global": 100 * np.nanmean(glob_fpr),
        "fpr_cv_length": 100 * np.nanmean(bin_fpr),
        "delta_tpr5": 100 * (np.nanmean(bin_tpr) - np.nanmean(glob_tpr)),
        "ntok_median": float(np.median(ntok)),
        "ntok_min": int(ntok.min()),
        "ntok_max": int(ntok.max()),
        "corr_score_ntok": float(np.corrcoef(s, ntok)[0, 1]),
    }


def fig_h7(tab: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt

    splits = [s for s in F.SPLITS if s in set(tab.split)]
    x = np.arange(len(splits))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    g = [tab.loc[tab.split == s, "tpr5_cv_global"].iloc[0] for s in splits]
    b = [tab.loc[tab.split == s, "tpr5_cv_length"].iloc[0] for s in splits]
    ax.bar(x - w / 2, g, w, color=F.S1, label="global threshold", edgecolor=F.SURFACE)
    ax.bar(x + w / 2, b, w, color=F.S3, label="length-binned thresholds",
           edgecolor=F.SURFACE)
    ax.set_xticks(x)
    ax.set_xticklabels([F.SPLIT_LABEL[s] for s in splits], color=F.INK)
    F.style(ax, f"H7: TPR@5%FPR under CV — {model}",
            ylabel="TPR@5%FPR (%)")
    ax.legend(frameon=False, fontsize=9, labelcolor=F.INK_2)
    F.save(fig, f"figC_h7_tpr_{model}.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for sp in splits:
        sub = tab[tab.split == sp].iloc[0]
        color = F.POS if sub.delta_tpr5 > 0 else F.NEG
        ax.scatter([sub.corr_score_ntok], [sub.delta_tpr5], s=90, color=color,
                   edgecolor=F.SURFACE, linewidth=1.3, zorder=3)
        ax.text(sub.corr_score_ntok, sub.delta_tpr5 + 0.15,
                F.SPLIT_LABEL[sp], color=F.INK_2, fontsize=8, ha="center")
    ax.axhline(0, color=F.NEUTRAL, lw=1.2)
    ax.axvline(0, color=F.NEUTRAL, lw=1, ls="--")
    F.style(ax, f"When does length calibration help? — {model}",
            xlabel="corr(minkpp score, n_tokens)",
            ylabel="Δ TPR@5% (length − global), pp")
    F.save(fig, f"figC_h7_corr_{model}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba-1.4b-hf")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-bins", type=int, default=2, choices=[2, 3])
    a = ap.parse_args()

    F.ensure_dirs()
    print(f"=== follow-up C: H7 length calibration ({a.model}) ===")
    print("Note: AUROC cannot improve under monotone per-item recalibration;")
    print("      this experiment only claims TPR@5%FPR under a fixed FPR budget.\n")

    rows = []
    for split in F.SPLITS:
        p = F.CACHE / f"{a.model}_{split}_{a.dtype}.npz"
        if not p.exists():
            print(f"skip {split}: no cache")
            continue
        items, y = _cache.load_cache(p)
        stats = run_split(items, y, a.n_folds, a.n_bins)
        stats.update({"model": a.model, "split": split, "n_bins": a.n_bins})
        rows.append(stats)
        print(f"{split}: AUROC {stats['auroc']:.1f} (unchanged)  "
              f"TPR5 global {stats['tpr5_cv_global']:.1f} -> "
              f"length {stats['tpr5_cv_length']:.1f}  "
              f"d={stats['delta_tpr5']:+.1f}pp  "
              f"corr(score,n)={stats['corr_score_ntok']:+.3f}")

    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit("no rows")
    out = F.FOLLOW / f"h7_calibration_{a.model}.csv"
    tab.to_csv(out, index=False)
    print(f"wrote {out.relative_to(F.ROOT)}")
    fig_h7(tab, a.model)


if __name__ == "__main__":
    main()
