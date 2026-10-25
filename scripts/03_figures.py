#!/usr/bin/env python3
"""Every figure in the report, built from the tables written by 02_aggregate.py.

Colour follows the reference palette of the data-visualisation method, used
unchanged: categorical slots 1-3 (blue / orange / aqua), which are documented as
clearing the all-pairs CVD and normal-vision floors in both modes, and the
documented diverging pair blue <-> red for "better / worse than baseline".
Identity is never carried by colour alone: every series is also direct-labelled
or legended, and the sign of an effect is readable from its position relative to
the zero line.

Usage
-----
    python 03_figures.py --model pythia-160m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

# ---- reference palette, light mode -------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
POS, NEG = "#2a78d6", "#e34948"                # diverging poles
NEUTRAL = "#b9b8b3"

SPLIT_COLOR = {"WikiMIA_length32": S1, "WikiMIA_length64": S2,
               "WikiMIA_length128": S3}
SPLIT_LABEL = {"WikiMIA_length32": "length 32", "WikiMIA_length64": "length 64",
               "WikiMIA_length128": "length 128"}


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="600",
                     loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote figures/{name}")


# --------------------------------------------------------------------------- #


def fig_baseline(auroc: pd.DataFrame, model: str):
    """Where the methods stand, per split. A grouped dot plot: the job is
    comparing a handful of magnitudes across three groups, not showing a trend."""
    methods = ["loss", "zlib", "mink_0.2", "minkpp_0.2"]
    labels = ["Loss", "Zlib", "Min-K%", "Min-K%++"]
    splits = [s for s in SPLIT_LABEL if s in set(auroc.split)]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ypos = np.arange(len(methods))
    for i, sp in enumerate(splits):
        sub = auroc[auroc.split == sp].set_index("variant")
        vals = [sub.loc[m, "auroc"] if m in sub.index else np.nan for m in methods]
        off = (i - (len(splits) - 1) / 2) * 0.22
        ax.scatter(vals, ypos + off, s=58, color=SPLIT_COLOR[sp],
                   label=SPLIT_LABEL[sp], zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
    ax.axvline(50, color=NEUTRAL, lw=1.2, ls="--", zorder=1)
    ax.text(50, len(methods) - 0.35, " chance", color=INK_2, fontsize=8, va="top")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.invert_yaxis()
    style(ax, f"Where each method stands — {model}",
          xlabel="AUROC (%)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right",
              ncol=3, handletextpad=0.3, columnspacing=1.2)
    save(fig, f"fig1_baseline_{model}.png")


def fig_main_deltas(paired: pd.DataFrame, model: str):
    """THE result figure. Every variant's change against the baseline, with its
    95% paired interval. Horizontal lollipop: magnitude plus polarity, so the
    zero line does the work and colour only reinforces the sign."""
    # The bar is the RANGE of the per-split 95% intervals, not an average of
    # interval bounds -- averaging CI endpoints does not produce a valid
    # interval. Significance is judged per split, in the tables.
    agg = (paired.groupby(["variant", "family"])
           .agg(delta=("delta", "mean"), lo=("lo", "min"), hi=("hi", "max"),
                sig=("significant", "any"))
           .reset_index())
    agg = agg[~agg.family.eq("baseline")]
    agg = agg.sort_values("delta")

    fig, ax = plt.subplots(figsize=(7.6, max(5.5, 0.21 * len(agg))))
    y = np.arange(len(agg))
    for yi, (_, r) in zip(y, agg.iterrows()):
        col = POS if r.delta > 0 else NEG
        ax.plot([r.lo, r.hi], [yi, yi], color=col, lw=1.6, alpha=0.30,
                solid_capstyle="round", zorder=2)
        ax.scatter([r.delta], [yi], s=34, color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(0, color=INK_2, lw=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(agg.variant, fontsize=7.5, color=INK_2)
    ax.set_ylim(-1, len(agg))
    style(ax,
          f"No aggregation variant beats Min-K%++ — {model}",
          xlabel="change in AUROC vs Min-K%++ baseline (points)\n"
                 "dot = mean over splits · bar = range of the per-split 95% paired intervals")
    # direct-label only the extremes, never every point
    for yi, (_, r) in zip(y, agg.iterrows()):
        if r.delta == agg.delta.min() or r.delta == agg.delta.max():
            ax.annotate(f"{r.delta:+.1f}", (r.delta, yi), fontsize=8,
                        color=INK, xytext=(6 if r.delta > 0 else -6, 0),
                        textcoords="offset points",
                        va="center", ha="left" if r.delta > 0 else "right")
    save(fig, f"fig2_all_variants_{model}.png")


def fig_fragmentation(paired: pd.DataFrame, model: str):
    """H1 and H6: the more finely the sequence is cut, the worse it gets.
    A line chart, because the x axis is an ordered amount of fragmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # H1: window width (wider = less fragmented, so plot against width)
    ax = axes[0]
    for sp in [s for s in SPLIT_LABEL if s in set(paired.split)]:
        sub = paired[(paired.split == sp) & paired.variant.str.match(r"h1_w\d+_max")]
        if sub.empty:
            continue
        w = sub.variant.str.extract(r"w(\d+)")[0].astype(int)
        o = np.argsort(w.values)
        ax.plot(w.values[o], sub.delta.values[o], "-o", ms=5, lw=2,
                color=SPLIT_COLOR[sp], label=SPLIT_LABEL[sp])
    ax.axhline(0, color=INK_2, lw=1.2)
    style(ax, "H1 · sliding window", xlabel="window width (tokens)",
          ylabel="Δ AUROC vs baseline")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)

    # H6: number of chunks
    ax = axes[1]
    for sp in [s for s in SPLIT_LABEL if s in set(paired.split)]:
        sub = paired[(paired.split == sp) & paired.variant.str.match(r"h6_K\d+_max")]
        if sub.empty:
            continue
        K = sub.variant.str.extract(r"K(\d+)")[0].astype(int)
        o = np.argsort(K.values)
        ax.plot(K.values[o], sub.delta.values[o], "-o", ms=5, lw=2,
                color=SPLIT_COLOR[sp], label=SPLIT_LABEL[sp])
    ax.axhline(0, color=INK_2, lw=1.2)
    style(ax, "H6 · fixed chunk count", xlabel="number of chunks",
          ylabel=None)
    save(fig, f"fig3_fragmentation_{model}.png")


def fig_k_sweep(auroc: pd.DataFrame, model: str):
    """k moves the score more than any aggregation variant does."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for sp in [s for s in SPLIT_LABEL if s in set(auroc.split)]:
        sub = auroc[(auroc.split == sp) & auroc.variant.str.match(r"minkpp_0\.\d")]
        if sub.empty:
            continue
        k = sub.variant.str.extract(r"minkpp_(0\.\d)")[0].astype(float)
        o = np.argsort(k.values)
        ax.plot(k.values[o], sub.auroc.values[o], "-o", ms=5, lw=2,
                color=SPLIT_COLOR[sp], label=SPLIT_LABEL[sp])
    style(ax, f"The k hyper-parameter moves AUROC more than aggregation does — {model}",
          xlabel="k (fraction of lowest z-scores averaged)", ylabel="AUROC (%)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    save(fig, f"fig4_k_sweep_{model}.png")


def fig_length_mechanism(model: str):
    """Per-item mechanism: for which items does a variant help?
    Scatter with a fitted line - a relationship, so a scatter is the form."""
    files = sorted(RES.glob(f"scores_{model}_WikiMIA_length64.csv"))
    if not files:
        return
    df = pd.read_csv(files[0])
    if "h2_m12" not in df.columns:
        return
    d = df["h2_m12"] - df["minkpp_0.2"]
    n = df["n_tokens"]
    ok = np.isfinite(d) & np.isfinite(n)
    r = np.corrcoef(d[ok], n[ok])[0, 1]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.scatter(n[ok], d[ok], s=14, color=S1, alpha=0.45,
               edgecolor="none")
    z = np.polyfit(n[ok], d[ok], 1)
    xs = np.linspace(n[ok].min(), n[ok].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=NEG, lw=2)
    ax.axhline(0, color=INK_2, lw=1.0)
    style(ax, "Fixed-count selection helps short passages and hurts long ones",
          xlabel="scored tokens in the item", ylabel="score difference vs baseline")
    ax.annotate(f"r = {r:.2f}", xy=(0.97, 0.94), xycoords="axes fraction",
                ha="right", fontsize=11, color=INK, fontweight="600")
    save(fig, f"fig5_length_mechanism_{model}.png")


def fig_resolution(res: pd.DataFrame, paired: pd.DataFrame, model: str):
    """What the benchmark can resolve, against the size of the effects claimed."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    sp = [s for s in SPLIT_LABEL if s in set(res.split)]
    hw = [res[res.split == s]["single_auroc_halfwidth"].iloc[0] for s in sp]
    ns = [res[res.split == s]["n"].iloc[0] for s in sp]
    bars = ax.barh(range(len(sp)), hw, color=NEUTRAL, height=0.5)
    best = (paired.groupby("split")["delta"].max()
            .reindex(sp).fillna(0).to_numpy())
    ax.scatter(np.abs(best), range(len(sp)), s=60, color=POS, zorder=3,
               edgecolor=SURFACE, linewidth=1.2,
               label="best variant gain (absolute)")
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels([f"{SPLIT_LABEL[s]}  (n={n})" for s, n in zip(sp, ns)],
                       color=INK, fontsize=9.5)
    ax.invert_yaxis()
    style(ax, "Every measured gain sits inside the benchmark's own noise",
          xlabel="AUROC points")
    for b, v in zip(bars, hw):
        ax.annotate(f"±{v:.1f}", (v, b.get_y() + b.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    save(fig, f"fig6_resolution_{model}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    a = ap.parse_args()

    auroc = pd.read_csv(RES / f"auroc_{a.model}.csv")
    paired = pd.read_csv(RES / f"paired_{a.model}.csv")
    res = pd.read_csv(RES / f"resolution_{a.model}.csv")

    print(f"figures for {a.model}:")
    fig_baseline(auroc, a.model)
    fig_main_deltas(paired, a.model)
    fig_fragmentation(paired, a.model)
    fig_k_sweep(auroc, a.model)
    fig_length_mechanism(a.model)
    fig_resolution(res, paired, a.model)


if __name__ == "__main__":
    main()
