#!/usr/bin/env python3
"""Follow-up A — stacking baselines (NOT an aggregation fix).

Train logistic regression on (loss, zlib, mink_0.2, minkpp_0.2) with
stratified CV. Compare stacked AUROC to each component and to minkpp alone.

Uses committed scores_*.csv — no GPU.

Usage
-----
    python scripts/04_stacking.py --model mamba-1.4b-hf
    python scripts/04_stacking.py --model pythia-160m --n-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import followup_common as F  # noqa: E402
import minkpp as M  # noqa: E402

FEATURES = ["loss", "zlib", "mink_0.2", "minkpp_0.2"]


def stack_cv(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_aurocs = []
    coefs = []
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=4000, random_state=seed)),
        ])
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        oof[te] = proba
        fold_aurocs.append(M.auroc(y[te], proba))
        coefs.append(pipe.named_steps["lr"].coef_.ravel())
    return oof, np.array(fold_aurocs), np.mean(coefs, axis=0)


def paired_delta(y, score_a, score_b, n_boot: int):
    lo, hi, p = M.paired_bootstrap(y, score_a, score_b, n_boot=n_boot, seed=0)
    d = 100 * (M.auroc(y, score_a) - M.auroc(y, score_b))
    return d, 100 * lo, 100 * hi, p


def run_model(model: str, n_folds: int, n_boot: int) -> pd.DataFrame:
    rows = []
    for split in F.SPLITS:
        df = F.load_scores(model, split)
        y = df["label"].to_numpy()
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise SystemExit(f"{split}: missing columns {missing}")
        X = df[FEATURES].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[ok], y[ok]

        singles = {name: 100 * M.auroc(y, df.loc[ok, name].to_numpy())
                   for name in FEATURES}
        oof, fold_a, coef = stack_cv(X, y, n_folds, seed=0)
        stack_au = 100 * M.auroc(y, oof)
        d, lo, hi, p = paired_delta(y, oof, df.loc[ok, "minkpp_0.2"].to_numpy(),
                                    n_boot)

        row = {
            "model": model, "split": split, "n": int(len(y)),
            "stack_auroc": stack_au,
            "stack_cv_mean": 100 * fold_a.mean(),
            "stack_cv_std": 100 * fold_a.std(ddof=1) if len(fold_a) > 1 else 0.0,
            "minkpp_auroc": singles["minkpp_0.2"],
            "best_single": max(singles.values()),
            "best_single_name": max(singles, key=singles.get),
            "delta_vs_minkpp": d, "lo": lo, "hi": hi, "p_gt0": p,
            "significant": bool(lo > 0 or hi < 0),
        }
        for name, v in singles.items():
            row[f"auroc_{name}"] = v
        for name, c in zip(FEATURES, coef):
            row[f"coef_{name}"] = c
        rows.append(row)

        print(f"{split}: stack {stack_au:.2f}  minkpp {singles['minkpp_0.2']:.2f}  "
              f"d={d:+.2f} [{lo:+.2f},{hi:+.2f}]  "
              f"{'SIG' if row['significant'] else 'n.s.'}")
    return pd.DataFrame(rows)


def fig_stacking(tab: pd.DataFrame, model: str) -> None:
    import matplotlib.pyplot as plt

    splits = [s for s in F.SPLITS if s in set(tab.split)]
    methods = FEATURES + ["stack"]
    labels = ["Loss", "Zlib", "Min-K%", "Min-K%++", "Stack (CV)"]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ypos = np.arange(len(methods))
    for i, sp in enumerate(splits):
        sub = tab[tab.split == sp].iloc[0]
        vals = [sub[f"auroc_{m}"] if m != "stack" else sub["stack_auroc"]
                for m in methods]
        off = (i - (len(splits) - 1) / 2) * 0.22
        ax.scatter(vals, ypos + off, s=58, color=F.SPLIT_COLOR[sp],
                   label=F.SPLIT_LABEL[sp], zorder=3,
                   edgecolor=F.SURFACE, linewidth=1.4)
    ax.axvline(50, color=F.NEUTRAL, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=F.INK, fontsize=10)
    ax.invert_yaxis()
    F.style(ax, f"Stacking vs single attacks — {model}",
            xlabel="AUROC (%)  (stack = out-of-fold)")
    ax.legend(frameon=False, fontsize=9, labelcolor=F.INK_2, loc="lower right",
              ncol=3)
    F.save(fig, f"figA_stacking_{model}.png")

    # delta lollipop
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(splits))
    for i, sp in enumerate(splits):
        sub = tab[tab.split == sp].iloc[0]
        color = F.POS if sub.delta_vs_minkpp > 0 else F.NEG
        ax.plot([sub.lo, sub.hi], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.scatter([sub.delta_vs_minkpp], [i], color=color, s=55, zorder=3,
                   edgecolor=F.SURFACE, linewidth=1.2)
    ax.axvline(0, color=F.NEUTRAL, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([F.SPLIT_LABEL[s] for s in splits], color=F.INK)
    F.style(ax, f"Stack − Min-K%++ (paired 95% CI) — {model}",
            xlabel="Δ AUROC (pp)")
    F.save(fig, f"figA_stacking_delta_{model}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mamba-1.4b-hf")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    F.ensure_dirs()
    print(f"=== follow-up A: stacking ({a.model}) ===")
    tab = run_model(a.model, a.n_folds, a.n_boot)
    out = F.FOLLOW / f"stacking_{a.model}.csv"
    tab.to_csv(out, index=False)
    print(f"wrote {out.relative_to(F.ROOT)}")
    fig_stacking(tab, a.model)

    wins = tab[tab.significant & (tab.delta_vs_minkpp > 0)]
    print(f"\nsignificant stack gains vs minkpp: {len(wins)} of {len(tab)}")
    if len(wins):
        print(wins[["split", "delta_vs_minkpp", "lo", "hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
