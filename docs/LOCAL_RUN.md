# Local run (Windows + NVIDIA GPU, no Docker)

Python **3.10+**. GPU is optional for reproducing the committed report (analysis is CPU-only). GPU is needed only if you re-run `01_cache.py` for a model.

## 1. One-time setup + first run

Easiest — one script (install + reproduce + log):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_and_run.ps1
```

Manual equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
$env:HF_HOME = "$PWD\hf-cache"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
.\run.ps1
```

`nvidia-smi` showing `CUDA Version: 13.0` is the **driver** ceiling — you do **not** need Toolkit 13.

## 2. Logs — how to see what finished

Every `run_pipeline.py` / `run.ps1` / `setup_and_run.ps1` run writes:

```
logs\run_YYYYMMDD_HHMMSS_<mode>.log
```

Inside you get:

- torch / CUDA / GPU line at the start
- each child command echoed with `+ ...`
- full stdout of every step
- `STEP OK [name]  12.3s` or `STEP FAIL [name] ...`
- final block `=== SUMMARY ===` with `result: SUCCESS` or `FAILED`

```powershell
# last log, last 40 lines (includes SUMMARY)
Get-Content (Get-ChildItem logs\run_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 40
```

Exit code `0` = success. Non-zero = open that log and look for `STEP FAIL`.

Committed `cache/*.npz` already contain the paper-matched forwards, so default `reproduce` does **not** download big models.

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
