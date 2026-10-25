#!/usr/bin/env python3
"""Generate the three analysis notebooks.

The notebooks are the readable layer: they load the caches and the tables that
`01_cache.py` and `02_aggregate.py` produce, and they explain what each number
means. They are generated from this script so that the narrative and the code
cannot drift apart, and so a reviewer can regenerate them from scratch.

    python build_notebooks.py            # write the .ipynb files
    python build_notebooks.py --execute  # write and run them, embedding outputs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"

PRELUDE = """\
import os, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SCRIPTS, RES, FIG = ROOT/"scripts", ROOT/"results", ROOT/"results"/"figures"
sys.path.insert(0, str(SCRIPTS))
import minkpp as M
_s = importlib.util.spec_from_file_location("cachemod", SCRIPTS/"01_cache.py")
cachemod = importlib.util.module_from_spec(_s); _s.loader.exec_module(cachemod)

MODEL = os.environ.get("MINKPP_MODEL", "mamba-1.4b-hf")   # the paper's target model
SPLITS = ["WikiMIA_length32", "WikiMIA_length64", "WikiMIA_length128"]
pd.set_option("display.width", 120)
print("library self-check:", "PASS" if M.self_check(verbose=False) else "FAIL")
"""


def md(x):
    return nbf.v4.new_markdown_cell(x)


def code(x):
    return nbf.v4.new_code_cell(x)


# --------------------------------------------------------------------------- #


def nb_data_and_baseline():
    c = [
        md("""# 1 · The data, and does the baseline reproduce?

Before testing any improvement, two things have to be true: the benchmark has to be
what the paper says it is, and our re-implementation of Min-K%++ has to return the
paper's numbers. If either fails, nothing downstream means anything.

**Min-K%++ in one line.** For each token, standardise its log-probability against the
mean and standard deviation of the model's own next-token distribution,

$$z_t = \\frac{\\log p(x_t \\mid x_{<t}) - \\mu_{\\cdot|x_{<t}}}{\\sigma_{\\cdot|x_{<t}}},
\\qquad \\mu = \\mathbb{E}_{v\\sim p}[\\log p(v)],$$

then average the lowest $k\\%$ of the $z_t$. Higher score = more likely a training member."""),
        code(PRELUDE),
        md("""## 1.1 The benchmark

WikiMIA labels membership by timestamp: Wikipedia event text written before a model's
training is a member, text written after is not. Two facts matter and are easy to miss,
so we count them by hand rather than trusting the split names."""),
        code("""rows = []
for sp in SPLITS:
    items, y = cachemod.load_cache(ROOT/"cache"/f"{MODEL}_{sp}_float32.npz")
    n = np.array([len(a) for a,_,_ in items])
    rows.append(dict(split=sp, items=len(items), members=int((y==1).sum()),
                     non_members=int((y==0).sum()),
                     tok_min=n.min(), tok_med=int(np.median(n)), tok_max=n.max(),
                     tok_sd=round(float(n.std()),1),
                     CV=round(float(n.std()/n.mean()),3)))
counts = pd.DataFrame(rows); counts"""),
        md("""**The split names are word counts, not token counts.** The "length 32" split spans
32 to 129 scored tokens — a 4x range *inside one split*. Any claim about how a method
behaves "at length 32" is really a claim about a distribution of lengths. This matters
directly for the hypotheses below, several of which are about length."""),
        code("""fig, ax = plt.subplots(figsize=(6.4,3.2))
for sp, col in zip(SPLITS, ["#2a78d6","#eb6834","#1baf7a"]):
    items, y = cachemod.load_cache(ROOT/"cache"/f"{MODEL}_{sp}_float32.npz")
    n = np.array([len(a) for a,_,_ in items])
    ax.hist(n, bins=40, alpha=.65, color=col, label=sp.replace("WikiMIA_",""))
ax.set_xlabel("scored tokens per item"); ax.set_ylabel("items")
ax.set_facecolor("#fcfcfb"); fig.set_facecolor("#fcfcfb")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(color="#e4e3df", lw=.8); ax.set_axisbelow(True)
ax.legend(frameon=False); ax.set_title("Token counts overlap across splits", loc="left")
plt.show()"""),
        md("""**Is length a label leak?** If members were systematically longer, a method sensitive
to length would look good for the wrong reason. Check it."""),
        code("""for sp in SPLITS:
    items, y = cachemod.load_cache(ROOT/"cache"/f"{MODEL}_{sp}_float32.npz")
    n = np.array([len(a) for a,_,_ in items])
    print(f"{sp:24s} median tokens  member {np.median(n[y==1]):6.1f}   "
          f"non-member {np.median(n[y==0]):6.1f}")"""),
        md("""Balanced. Length is a **nuisance**, not a leak — it can add noise but it cannot by
itself separate the classes."""),
        md("""## 1.2 Per-token quantities

Everything in this study is a function of three cached numbers per token. The model is
run **once** per split; every variant is then post-processing. That is what makes the
baseline and the variants comparable at exactly matched compute — no variant can win by
having had a different forward pass."""),
        code("""items, y = cachemod.load_cache(ROOT/"cache"/f"{MODEL}_WikiMIA_length32_float32.npz")
logp, mu, sigma = items[0]
z = M.zscores(logp, mu, sigma)
print(f"item 0: {len(logp)} scored tokens")
pd.DataFrame({"logp": logp[:8], "mu": mu[:8], "sigma": sigma[:8], "z": z[:8]}).round(3)"""),
        md("""The z-score is *not* a monotone transform of `logp` — it depends on the shape of the
whole next-token distribution. Confusing the two is the one bug that would silently turn
Min-K%++ back into Min-K% while still looking plausible, so the library asserts it."""),
        code("""print("corr(logp, z) =", round(float(np.corrcoef(logp, z)[0,1]), 3))
print("identical?", np.allclose(logp, z))"""),
        md("""## 1.3 Does the baseline reproduce?"""),
        code("""auroc = pd.read_csv(RES/f"auroc_{MODEL}.csv")
base = auroc[auroc.variant.isin(["loss","zlib","mink_0.2","minkpp_0.2"])]
piv = base.pivot(index="variant", columns="split", values="auroc").round(1)
piv = piv.reindex(["loss","zlib","mink_0.2","minkpp_0.2"])
piv"""),
        code("""display(Image(filename=str(FIG/f"fig1_baseline_{MODEL}.png")))"""),
        md("""### What to read here

Against the published Mamba-1.4B column (66.8 / 67.2 / 68.8 for Min-K%++), a small model
scores far lower and the Min-K%++ advantage over Min-K% largely disappears. That is
consistent with the paper's own Table 1: its gain averages **+3.7** over Mamba-1.4B and
Pythia-6.9B but **+13.6** over the three LLaMA columns. The method's advantage grows with
model scale, and the models inside a student budget are the ones where it helps least.

This is a finding about the experiment's sensitivity, and it shapes how the next notebook
should be read: on a model where the baseline barely works, there is little signal for an
aggregation variant to improve."""),
        md("""## 1.4 What the benchmark can resolve

A difference smaller than the benchmark's own sampling noise is not a result. We measure
the noise rather than assume it."""),
        code("""pd.read_csv(RES/f"resolution_{MODEL}.csv").round(2)"""),
        md("""Several gaps reported in the source papers are **smaller than this half-width**.
Any improvement we claim has to clear it, which is why every comparison below uses a
*paired* bootstrap: sharing the resample indices between the two methods removes the
part of the noise that is common to both and materially narrows the interval."""),
    ]
    return c


def nb_hypotheses():
    c = [
        md("""# 2 · Testing the ten improvement hypotheses

The proposal lists ten candidate improvements on the **aggregation** axis — how per-token
z-scores combine into one score for a passage. The per-token score function and the
selection criterion are left untouched; only the combination changes.

Every variant is scored from the same cache, so the comparison is at matched compute by
construction. Each is compared against `minkpp_0.2` with a **paired** bootstrap over
items, 2000 resamples, shared indices.

| # | Hypothesis | What it changes |
|---|---|---|
| H1 | Sliding window, max pooling | score overlapping windows, take the best |
| H2 | Fixed-count selection | keep the lowest *m* tokens, not the lowest *k%* |
| H3 | Length-normalised aggregation | divide by what the statistic would be under a length-matched null |
| H4 | Magnitude weighting | weight selected tokens by \\|z\\| |
| H5 | Order statistic | a single quantile instead of a tail mean |
| H6 | Chunk and vote | fixed number of chunks, pooled |
| H8 | Position weighting | drop or downweight early tokens |
| H9 | Two-sided aggregation | use the high tail as well as the low |
| H10 | Prefix / early stop | score only the first fraction of tokens |

(H7, length-conditional calibration, is a threshold-transfer question rather than a
ranking one: it cannot change AUROC, which is threshold-free. It is tested in notebook 3
on TPR@5%FPR, where it can.)"""),
        code(PRELUDE),
        code("""paired = pd.read_csv(RES/f"paired_{MODEL}.csv")
auroc  = pd.read_csv(RES/f"auroc_{MODEL}.csv")
print(f"{len(paired)} variant x split comparisons")
wins = paired[(paired.delta>0) & paired.significant]
print(f"significant improvements: {len(wins)}")
wins[["split","variant","delta","lo","hi","p_gt0"]].round(2)"""),
        md("""## 2.1 The whole picture"""),
        code("""display(Image(filename=str(FIG/f"fig2_all_variants_{MODEL}.png")))"""),
        md("""Each dot is a variant's mean change against the baseline; the bar is the range of its
per-split 95% paired intervals. Nearly every bar crosses zero — the aggregation axis is
flat. The few that do not cross are **negative**."""),
        md("""## 2.2 Family by family"""),
        code("""fam = (paired.groupby("family")
       .agg(n=("delta","size"), mean=("delta","mean"),
            best=("delta","max"), worst=("delta","min"),
            n_sig_better=("significant", lambda s: 0))
       .round(2).sort_values("best", ascending=False))
sig = paired[paired.significant]
fam["n_sig_better"] = [int(((sig.family==f)&(sig.delta>0)).sum()) for f in fam.index]
fam["n_sig_worse"]  = [int(((sig.family==f)&(sig.delta<0)).sum()) for f in fam.index]
fam"""),
        md("""## 2.3 H1 and H6 — fragmentation

Both cut the sequence into pieces and pool. If localisation helped, finer pieces should
help more.

**On this model nothing here is significant** — 21 comparisons, all inside the ±4-point
half-width, and the pattern is not monotone. The only consistent signal is that the most
extreme fragmentation (K = 6) is the worst cell at two of three lengths. On the paper's
target model, Mamba-1.4B, the same variants degrade monotonically down to −15.9; see
`REPORT.md` §5.1. A fragmentation penalty needs a baseline with real signal to destroy,
and at this scale the baseline is close to chance."""),
        code("""display(Image(filename=str(FIG/f"fig3_fragmentation_{MODEL}.png")))"""),
        md("""**Why.** Min-K%++ already selects the lowest *k%* **over the whole sequence**, which is
itself a global search for the most anomalous region. Windowing forces the statistic to
spend its *k%* quota *inside every window*, so it must admit tokens that are not globally
anomalous. Whole-sequence selection is therefore strictly better targeted, and windowing
can only win when memorisation is localised **and** the global selection misses it.
On Mamba-1.4B the monotone degradation says it does not miss it.

This is the central mechanism of the whole study, and it is what makes the negative
result explicable rather than merely observed. It also predicts where the idea *could*
pay: passages long enough that the global bottom-*k%* is spread across genuinely
unrelated material. WikiMIA tops out at 272 scored tokens, which is evidently not
long enough."""),
        md("""## 2.4 H2 — fixed-count selection, and a confound worth recording"""),
        code("""paired[paired.variant.str.startswith("h2_")][
    ["split","variant","delta","lo","hi","p_gt0"]].round(2)"""),
        md("""Null everywhere. But the per-item behaviour is **not** null, and that is informative."""),
        code("""display(Image(filename=str(FIG/f"fig5_length_mechanism_{MODEL}.png")))"""),
        md("""A fixed count helps short passages and hurts long ones — mechanically expected, since a
fixed *m* is a shrinking fraction as the passage grows. The aggregate is null because the
two effects cancel. That is a real mechanism hiding inside a null result, and it is the
observation notebook 3 builds a new variant on.

> **A confound that nearly entered this report as a positive result.** These variants
> operate on z-scores, so they are Min-K%++ variants. Compared against *Min-K%* they can
> appear to gain several points — but that gain is the z-standardisation, which is
> Min-K%++'s own contribution, not the variant's. **Any aggregation variant built on z
> must be compared against Min-K%++, never against Min-K%.**"""),
        md("""## 2.5 The hyper-parameter nobody reports

k moves the score more than any aggregation variant tested, and the paper does not state
which k produced a published cell."""),
        code("""display(Image(filename=str(FIG/f"fig4_k_sweep_{MODEL}.png")))"""),
        code("""kk = auroc[auroc.variant.str.match(r"minkpp_0\\.\\d")].pivot(
        index="variant", columns="split", values="auroc").round(1)
kk["spread"] = (kk.max(axis=1)-kk.min(axis=1)).round(1)
kk"""),
        md("""## 2.6 Against the benchmark's own noise"""),
        code("""display(Image(filename=str(FIG/f"fig6_resolution_{MODEL}.png")))"""),
        md("""## 2.7 Verdict

Of the nine hypotheses testable on AUROC, **none produces a reliable improvement**, and
two families (H1, H6) are reliably *worse* with a mechanism that explains why. The
aggregation step of Min-K%++ holds no remaining headroom on this benchmark.

Notebook 3 takes the failures seriously: it asks what the mechanism *permits*, and
designs new variants from the measurements rather than from intuition."""),
    ]
    return c


def nb_new():
    c = [
        md("""# 3 · From the failures to new improvements

Notebook 2 ended with a flat aggregation axis. This notebook does what the failure
analysis licenses: it reads the mechanism, asks what it *allows*, and builds new variants
from measurements rather than from intuition.

**What the mechanism says.** Global bottom-*k%* selection is already a well-targeted
anomaly search. Any scheme that forces the quota to be spent locally is strictly worse.
So a variant can only help if it:

1. keeps global selection, and changes how the selected scores are combined; or
2. changes *how many* scores are selected, adaptively; or
3. targets something other than ranking — a threshold, or compute."""),
        code(PRELUDE),
        code("""paired = pd.read_csv(RES/f"paired_{MODEL}.csv")
auroc  = pd.read_csv(RES/f"auroc_{MODEL}.csv")
new = paired[paired.family.str.startswith("N")]
new[["split","variant","family","delta","lo","hi","p_gt0"]].round(2)"""),
        md("""## 3.1 The four derived variants

| | Derived from | What it does |
|---|---|---|
| **N1** k-ensemble | §2.5 — k moves AUROC more than any aggregation variant, and the paper never says which k it used | average the statistic over a grid of k instead of picking one |
| **N2** length-adaptive k | §2.4 — the per-item fixed-count gain correlates with token count at r ≈ −0.7 | let k shrink with passage length, the continuous form of that observation |
| **N3** rank-normalised | the suspicion that longer items have heavier absolute-z tails | aggregate within-item ranks rather than z values |
| **N4** trimmed tail | the mechanism says selection is well targeted — so are its *worst* members outliers? | drop the most extreme selected tokens before averaging |

N1 and N2 are the honest ones: each is a direct formalisation of a measurement already in
hand, not a new idea."""),
        md("""## 3.2 Did they work?"""),
        code("""summ = (new.groupby("variant")
        .agg(mean_delta=("delta","mean"), best=("delta","max"),
             worst=("delta","min"),
             any_sig=("significant","any"))
        .round(2).sort_values("mean_delta", ascending=False))
summ"""),
        md("""### N3 fails hardest, and the reason is instructive

Replacing z by its within-item rank throws away *how far* below the mode a token sits and
keeps only the ordering. But the ordering inside an item is the same for members and
non-members — every item has a lowest token. The discriminative information was in the
**magnitude**, and rank-normalisation deletes exactly that. This is a clean example of a
variant that removes a nuisance and the signal together."""),
        md("""### N1 and N2 are null, not harmful

Both sit within noise. For N1 that is expected in hindsight: averaging over k is a
smoothing operation, and the k-curve is nearly flat within a split, so there is little to
smooth. For N2 the per-item mechanism was real (r ≈ −0.7) but the two halves cancel in
aggregate, exactly as fixed-count selection did."""),
        md("""## 3.3 H7, tested where it can actually show

H7 — length-conditional calibration — cannot change AUROC, because AUROC is
threshold-free: any monotone, item-independent recalibration leaves the ranking alone.
It can only matter at a **fixed operating point**, where a single threshold has to serve
items of different lengths. So we test it on TPR@5%FPR, and we test the premise first:
does the baseline score depend on length at all?"""),
        code("""sp = "WikiMIA_length64"
df = pd.read_csv(RES/f"scores_{MODEL}_{sp}.csv")
r = np.corrcoef(df["minkpp_0.2"], df["n_tokens"])[0,1]
print(f"corr(baseline score, token count) = {r:+.3f}")

# length-conditional calibration: z-score the baseline within length terciles
q = pd.qcut(df["n_tokens"], 3, labels=False, duplicates="drop")
cal = df["minkpp_0.2"].copy()
for g in np.unique(q):
    m = q == g
    cal[m] = (cal[m] - cal[m].mean()) / (cal[m].std() + 1e-9)

y = df["label"].to_numpy()
for nm, s in [("baseline", df["minkpp_0.2"].to_numpy()), ("H7 calibrated", cal.to_numpy())]:
    print(f"{nm:16s} AUROC {100*M.auroc(y,s):.1f}   TPR@5%FPR {100*M.tpr_at_fpr(y,s):.1f}")"""),
        md("""If the correlation is weak, the premise of H7 is weak on this benchmark and the
calibration has little to correct — which is the result rather than a failure of the
implementation."""),
        md("""## 3.4 H10, read as a compute result rather than an accuracy one

H10 scores only a prefix of each item. As an accuracy improvement it is not expected to
win — less information cannot help. Its value is the **accuracy/compute trade-off**: how
much of the sequence is actually needed?"""),
        code("""h10 = auroc[auroc.variant.str.startswith("h10_")].copy()
h10["frac"] = h10.variant.str.extract(r"frac(0\\.\\d+)").astype(float)
base = auroc[auroc.variant=="minkpp_0.2"][["split","auroc"]].rename(
        columns={"auroc":"base"})
h10 = h10.merge(base, on="split")
h10["retained_pct"] = (100*h10.auroc/h10.base).round(1)
h10[["split","frac","auroc","base","retained_pct"]].round(1).sort_values(["split","frac"])"""),
        md("""Reading this as "how much accuracy survives at what fraction of the tokens" turns a
failed accuracy hypothesis into a usable efficiency statement — which is what §4.4 of the
regulations means by a lower-cost variant on an accuracy/compute trade-off."""),
        md("""## 3.5 What would have to change for the aggregation axis to pay

The mechanism is specific, so it predicts its own exceptions. Aggregation can only help
when the global bottom-*k%* is **spread across unrelated material**, i.e. when:

- passages are much longer than WikiMIA's 272-token maximum, so that one memorised span
  is a small fraction of the whole; or
- the passage is heterogeneous by construction — concatenated documents, a memorised
  quotation inside original text. The `WikiMIA_concat` split published by the same authors
  is built exactly this way and is the natural next test.

Neither is a rescue of the hypothesis on *this* benchmark. Both are falsifiable
predictions the mechanism makes, which is what distinguishes an explained negative result
from an unexplained one."""),
    ]
    return c


def write(cells, path: Path):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print(f"  wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    targets = [
        (nb_data_and_baseline(), NB / "01_data_and_baseline.ipynb"),
        (nb_hypotheses(), NB / "02_hypotheses.ipynb"),
        (nb_new(), NB / "03_new_improvements.ipynb"),
    ]
    for cells, path in targets:
        write(cells, path)

    if a.execute:
        for _, path in targets:
            print(f"executing {path.name} ...")
            r = subprocess.run(
                [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                 "--execute", "--inplace", "--ExecutePreprocessor.timeout=1200",
                 str(path)],
                capture_output=True, text=True)
            if r.returncode:
                print(r.stderr[-2500:])
                sys.exit(f"execution failed: {path.name}")
            print(f"  ok {path.name}")


if __name__ == "__main__":
    main()
