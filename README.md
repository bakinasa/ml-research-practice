# Min-K%++ — practical experiment (S08)

Reproducible study of aggregation improvements for [Min-K%++](https://arxiv.org/abs/2404.02936) (Zhang et al., ICLR 2025) on WikiMIA.

**Findings:** see [`REPORT.md`](REPORT.md).

Runs **natively on Windows** (or any OS) with Python + optional NVIDIA GPU. No Docker.

---

## Quick start (Windows + GPU)

Prerequisites:

1. Python 3.10+
2. NVIDIA driver (`nvidia-smi` works). An RTX 3060 **6 GB** is enough to reproduce the report and re-cache small models; **not** enough for Pythia-6.9B (~14 GB).

```powershell
git clone <this-repo>
cd ml-research-practice

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
$env:HF_HOME = "$PWD\hf-cache"

.\run.ps1
```

That runs self-checks and **rebuilds all report tables/figures from the committed caches** (no big model download).

### Useful modes

| Command | What it does |
|---|---|
| `.\run.ps1` | Aggregate + figures for `mamba-1.4b-hf` and `pythia-160m` from `cache/*.npz` |
| `.\run.ps1 check` | Self-checks only |
| `.\run.ps1 full -Model EleutherAI/pythia-160m` | GPU-cache that model if missing, then score |
| `.\run.ps1 reproduce -Notebooks` | Also rebuild/execute notebooks |
| `.\run.ps1 pythia69` | Pythia-6.9B float16 — needs ~**14 GB** VRAM (refuses on smaller GPUs) |

Same thing without PowerShell:

```powershell
python scripts\run_pipeline.py reproduce
python scripts\run_pipeline.py check
```

More detail: [`docs/LOCAL_RUN.md`](docs/LOCAL_RUN.md).

---

## Layout

```
scripts/
  minkpp.py            z-scores, all variants, AUROC, paired bootstrap, self-check
  01_cache.py          only GPU/CPU model step → cache/*.npz
  02_aggregate.py      score all variants → results/*.csv
  03_figures.py        → results/figures/*.png
  run_pipeline.py      one-command orchestrator (no Docker)
  build_notebooks.py   generate/execute notebooks
run.ps1                Windows entrypoint → run_pipeline.py
cache/                 committed per-token caches (skip re-running the model)
results/               tables + figures
notebooks/             narrative notebooks
docs/                  practice record + local instructions
REPORT.md              findings
```

Design: the model runs **once** per (model, split); every baseline and variant is a function of the same cached `logp` / `mu` / `sigma`. Matched compute by construction.

---

## VRAM guide

| Target | Approx. | Notes |
|---|---|---|
| Reproduce from caches | 0 (analysis is CPU) | Default `.\run.ps1` |
| Mamba-1.4B / pythia-160m cache | fits in 6 GB | Prefer `float16` if VRAM is tight |
| **Pythia-6.9B** cache | **~14 GB fp16** | `.\run.ps1 pythia69` only on large cards |

---

## What is / isn’t in this repo yet

**Included:** WikiMIA original, Min-K% / Min-K%++ baselines, H1–H10 + N1–N4, paired bootstrap, figures, Mamba-1.4B + pythia-160m caches, Windows one-shot.

**Still outstanding** (not wired as pipeline modes): Neighbor / Lowercase / Smaller-Ref baselines, paraphrased splits, truncation split, LLaMA columns. See `docs/sofiia-s08-practice.md` §9.
