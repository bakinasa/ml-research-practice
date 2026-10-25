# Local run (Windows + NVIDIA GPU, no Docker)

Python **3.10+**. GPU is optional for reproducing the committed report (analysis is CPU-only). GPU is needed only if you re-run `01_cache.py` for a model.

## 1. One-time setup

In PowerShell, from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CUDA wheel for your driver (pick cu124 / cu126 / cu128 on https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

$env:HF_HOME = "$PWD\hf-cache"
```

Check GPU visibility:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`nvidia-smi` showing `CUDA Version: 13.0` means the **driver** can run that CUDA; you do **not** need Toolkit 13. A matching PyTorch CUDA wheel is enough.

## 2. Run everything after clone

```powershell
.\run.ps1              # self-check + tables + figures for mamba + pythia-160m
.\run.ps1 check        # self-checks only
```

Equivalent without the helper:

```powershell
python scripts\run_pipeline.py reproduce
```

Committed `cache/*.npz` already contain the paper-matched forwards, so the default mode does **not** download big models.

## 3. Modes

| Command | What it does |
|---|---|
| `.\run.ps1` / `reproduce` | Aggregate + figures from committed caches |
| `.\run.ps1 check` | Library + cache maths self-checks |
| `.\run.ps1 full` | Cache `Model` on GPU if missing, then score |
| `.\run.ps1 cache` | Only `01_cache.py` |
| `.\run.ps1 pythia69` | Pythia-6.9B float16 (~**14 GB** VRAM) — refuses on smaller GPUs |

Examples:

```powershell
# Re-cache the small Pythia on GPU, then score
.\run.ps1 full -Model EleutherAI/pythia-160m -Device cuda

# Mamba (already cached as float32; skip unless you delete cache/)
.\run.ps1 full -Model state-spaces/mamba-1.4b-hf -Dtype float16 -Device cuda

# Also rebuild/execute notebooks
.\run.ps1 reproduce -Notebooks
```

## 4. VRAM guide (RTX 3060 Laptop 6 GB)

| Target | Fits 6 GB? | Notes |
|---|---|---|
| Reproduce from caches | yes (0 VRAM) | Default after clone |
| `pythia-160m` cache | yes | Fast smoke / re-cache |
| `mamba-1.4b-hf` cache | yes in float16; tight in float32 | Committed caches are float32 already |
| `pythia-6.9b` (`pythia69`) | **no** (~14 GB fp16) | Need a larger card |

`mamba-ssm` / `causal-conv1d` are **not** required: `transformers` uses a reference PyTorch scan.

### Runtimes (reference)

| Model | Split | Wall clock |
|---|---|---:|
| pythia-160m | 32/64/128 | ~6 min total (CPU) / faster on GPU |
| mamba-1.4b-hf | 32/64/128 | ~1h45 total on CPU |
| pythia-6.9b | GPU + float16 | ~10 min on a ≥14 GB card |
