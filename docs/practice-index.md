# Practical records

One file per project. Each is the hand-over document for writing the report: what was run, on what
hardware, with which commands, what came out, and what is still outstanding.

| Who | Paper | Record | State of the practical part |
|---|---|---|---|
| Yakup Allakov · S06 · Group B | Min-K% Prob (ICLR 2024) | [yakup-s06-practice.md](yakup-s06-practice.md) | Baseline reproduced at length 128; Table 1's split identified; **the improvement gains ≈ +3.8 AUROC and replicates on two models**, target model pending |
| Sofiia Bakina · S08 · Group B | Min-K%++ (ICLR 2025) | [sofiia-s08-practice.md](sofiia-s08-practice.md) · [full experiment](sofiia-s08/) | Baseline reproduced at **all three lengths**; **all ten hypotheses tested plus four derived variants — 0 of 120 improve, 53 significantly worse**, with one mechanism explaining them |

Yehia Sobeh (S23, ACR) is on a different method and has no record here yet.

## Runnable experiment

[`sofiia-s08/`](sofiia-s08/) is the full reproducible experiment behind S08: scripts that
download the data and the model, cache the per-token quantities once, score **all ten
proposal hypotheses plus four new variants derived from their failures**, and three
executed notebooks with the figures and commentary. Start at
[`sofiia-s08/REPORT.md`](sofiia-s08/REPORT.md) for the findings and
[`sofiia-s08/README.md`](sofiia-s08/README.md) for how to rerun them.

Unlike the records above, it keeps its infrastructure **inside the repository** rather than
in a private working directory, so the caches and every table and figure regenerate from a
clean clone.

## Common infrastructure

Both projects run through the same three scripts, kept outside this repository in the working
directory `E:\work-mink`:

| Script | Purpose |
|---|---|
| `cache.py` | Runs a model once per (model, split, setting) and stores the per-token quantities — `log p(x_t\|x_<t)`, and the mean and standard deviation of `log p` over the vocabulary. Resumes after an interrupt |
| `analyze.py` | Every baseline, every proposed variant, every ablation and the paired bootstrap, computed from those caches. No GPU |
| `unigram.py` | Token frequency table for S06's frequency-corrected selection |
| `server/run_all.sh` | Tiered batch driver for the university GPU server |

The point of the split is that **the model is never run twice for the same data**. Every variant is
a different function of one cached forward pass, so baseline and extension are compared at exactly
matched compute by construction, and all the analysis is a few seconds of numpy on a laptop.

A cache is about 0.5 MB per split, so the expensive artefact is small enough to move around freely.

## Reproducing anything here

```bash
cd /e/work-mink
export HF_HOME=/e/hf-cache HF_HUB_DISABLE_SYMLINKS_WARNING=1
./.venv/Scripts/python.exe cache.py   --self-check
./.venv/Scripts/python.exe analyze.py --self-check
```

Both scripts refuse to do real work until their self-checks pass. The checks that matter: a window
at least as wide as the sequence must reproduce the whole-sequence statistic exactly, λ = 0 must
reproduce the uncorrected baseline exactly, and the z-scores must not collapse onto the raw
log-probabilities — that last one is the single bug that would silently turn Min-K%++ back into
Min-K% while still looking plausible.

## Shared findings about the benchmark

These came out of the practical work and apply to both projects, and to anyone else using WikiMIA:

- Split names (`length32`, `length64`, …) are **word** counts. Token counts vary up to 4× inside a
  single split.
- The Neighbor baseline needs **no perturbation step**: ten neighbours per item are already
  published. Perturbed splits exist at lengths 32 / 64 / 128 only, so Neighbor is impossible at 256.
- Labels in the paraphrased and perturbed splits are stored as **strings**.
- Resolution: a single AUROC carries a 95% bootstrap half-width of ±3.6 at n = 776 and ±13.4 at
  n = 82. Several gaps reported in both source papers are smaller than that.
- `EleutherAI/pile` contains **no data files** — only a loading script pointing at a dead mirror.
