# Min-K%++ — practical experiment (S08)

Reproducible study of aggregation improvements for [Min-K%++](https://arxiv.org/abs/2404.02936) (Zhang et al., ICLR 2025) on WikiMIA.

**Findings:** see [`REPORT.md`](REPORT.md).  
**Branch:** `sofiia-s08-minkpp` (work in progress — not committed yet if you cloned empty).

---

## One command on Linux + NVIDIA GPU

Prerequisites on the laptop:

1. [Docker](https://docs.docker.com/engine/install/) + Compose v2  
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  
3. Check: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

Then from this repo:

```bash
chmod +x run.sh docker/entrypoint.sh
./run.sh
```

That builds the image and **reproduces all report tables/figures from the committed caches** (no big model download). Equivalent:

```bash
docker compose up --build
# or:
docker compose run --rm minkpp reproduce
```

### Useful modes

| Command | What it does |
|---|---|
| `./run.sh` | Aggregate + figures for `mamba-1.4b-hf` and `pythia-160m` from `cache/*.npz` |
| `./run.sh full` | GPU-cache `HF_MODEL` if missing, then aggregate + figures |
| `./run.sh pythia69` | Cache **Pythia-6.9B** in `float16` on GPU (needs ~14–16 GB VRAM), then score |
| `./run.sh shell` | Interactive shell in the container |
| `NOTEBOOKS=1 ./run.sh` | Also rebuild/execute notebooks |

Examples:

```bash
# New model on GPU (default HF_MODEL is Mamba; override as needed)
HF_MODEL=EleutherAI/pythia-6.9b DTYPE=float16 ./run.sh full

# Explicit short name for aggregate (usually auto from HF id)
HF_MODEL=EleutherAI/pythia-6.9b SHORT_MODEL=pythia-6.9b DTYPE=float16 ./run.sh full
```

Outputs land on the host under `./cache`, `./results` (and `./notebooks` if enabled). Hugging Face downloads persist in the Docker volume `hf-cache`.

### Without the `./run.sh` helper

```bash
docker compose build
docker compose run --rm minkpp reproduce
MODE=full HF_MODEL=EleutherAI/pythia-6.9b DTYPE=float16 docker compose run --rm minkpp full
```

---

## Layout

```
scripts/
  minkpp.py            z-scores, all variants, AUROC, paired bootstrap, self-check
  01_cache.py          only GPU/CPU model step → cache/*.npz
  02_aggregate.py      score all variants → results/*.csv
  03_figures.py        → results/figures/*.png
  build_notebooks.py   generate/execute notebooks
cache/                 committed per-token caches (small; skip re-running the model)
results/               tables + figures
notebooks/             narrative notebooks
docker/entrypoint.sh   container orchestrator
docs/                  practice record + local (non-Docker) instructions
REPORT.md              findings
```

Design: the model runs **once** per (model, split); every baseline and variant is a function of the same cached `logp` / `mu` / `sigma`. Matched compute by construction.

---

## VRAM guide

| Target | Approx. | Notes |
|---|---|---|
| Reproduce from caches | 0 (analysis is CPU) | Default `./run.sh` |
| Mamba-1.4B cache | ~3–6 GB | Already committed in `float32` |
| **Pythia-6.9B** cache | **~14 GB fp16** | Main reason for GPU laptop; `./run.sh pythia69` |
| LLaMA-7B+ | usually more / gated | Not wired as a default mode |

If `DEVICE=cuda` but CUDA is invisible in the container, the entrypoint falls back to CPU and prints a warning.

---

## Local run (no Docker)

See [`docs/LOCAL_RUN.md`](docs/LOCAL_RUN.md).

---

## What is / isn’t in this repo yet

**Included:** WikiMIA original, Min-K% / Min-K%++ baselines, H1–H10 + N1–N4, paired bootstrap, figures, Mamba-1.4B + pythia-160m caches, Docker one-shot.

**Still outstanding** (not implemented as compose modes): Neighbor / Lowercase / Smaller-Ref baselines, paraphrased splits, truncation split, LLaMA columns. See `docs/sofiia-s08-practice.md` §9.
