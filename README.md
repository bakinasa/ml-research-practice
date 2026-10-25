# Min-K%++ — practical experiment (S08)

Reproducible study of aggregation improvements for [Min-K%++](https://arxiv.org/abs/2404.02936) (Zhang et al., ICLR 2025) on WikiMIA.

**Findings:** see [`REPORT.md`](REPORT.md).

Runs **natively on Windows** (or any OS) with Python + optional NVIDIA GPU. No Docker.

---

## Quick start (Windows + GPU laptop)

Prerequisites: Python 3.10+, NVIDIA driver (`nvidia-smi`). RTX 3060 **6 GB** is enough for the default report; **not** for Pythia-6.9B (~14 GB).

```powershell
git clone <this-repo>
cd ml-research-practice
Set-ExecutionPolicy -Scope Process Bypass
.\setup_and_run.ps1
```

That installs `.venv` + CUDA torch + deps, then runs self-checks and **rebuilds all report tables/figures from the committed caches**.

Every run writes `logs\run_YYYYMMDD_HHMMSS_<mode>.log` with per-step `STEP OK` / `STEP FAIL` and a final `SUMMARY` (`SUCCESS` / `FAILED`). Exit code `0` = success.

### Later runs (env already set up)

| Command | What it does |
|---|---|
| `.\run.ps1` | Aggregate + figures for `mamba-1.4b-hf` and `pythia-160m` |
| `.\run.ps1 check` | Self-checks only |
| `.\run.ps1 full -Model EleutherAI/pythia-160m` | GPU-cache that model if missing, then score |
| `.\run.ps1 reproduce -Notebooks` | Also rebuild/execute notebooks |
| `.\run.ps1 pythia69` | Pythia-6.9B float16 — needs ~**14 GB** VRAM |

```powershell
python scripts\run_pipeline.py reproduce   # same as .\run.ps1
Get-Content (Get-ChildItem logs\run_*.log | Sort LastWriteTime -Desc | Select -First 1).FullName -Tail 40
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
setup_and_run.ps1      first-time: venv + deps + reproduce
cache/                 committed per-token caches (skip re-running the model)
results/               tables + figures
logs/                  run transcripts (STEP OK/FAIL + SUMMARY; gitignored)
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
