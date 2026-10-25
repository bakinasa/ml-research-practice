"""Shared paths and plotting style for follow-up experiments (04–07).

These are *additional* to the original H1–H10 study: results land under
results/followups/ so the old tables/figures stay untouched for comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FOLLOW = RES / "followups"
FIG = FOLLOW / "figures"
CACHE = ROOT / "cache"

SPLITS = ["WikiMIA_length32", "WikiMIA_length64", "WikiMIA_length128"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
POS, NEG = "#2a78d6", "#e34948"
NEUTRAL = "#b9b8b3"

SPLIT_COLOR = {
    "WikiMIA_length32": S1,
    "WikiMIA_length64": S2,
    "WikiMIA_length128": S3,
}
SPLIT_LABEL = {
    "WikiMIA_length32": "length 32",
    "WikiMIA_length64": "length 64",
    "WikiMIA_length128": "length 128",
}


def ensure_dirs() -> None:
    FOLLOW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


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


def save(fig, name: str) -> Path:
    ensure_dirs()
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote followups/figures/{name}")
    return path


def load_scores(model: str, split: str):
    import pandas as pd
    p = RES / f"scores_{model}_{split}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p.name}; run 02_aggregate.py --model {model} first"
        )
    return pd.read_csv(p)
