#!/usr/bin/env python3
"""Orchestrate the Min-K%++ pipeline without Docker.

Modes
-----
    reproduce   self-check + aggregate + figures from committed caches (default)
    cache       run 01_cache.py for HF_MODEL if caches are missing
    full        cache (if needed) + aggregate + figures for one model
    pythia69    cache EleutherAI/pythia-6.9b in float16 (~14 GB VRAM), then score
    check       self-checks only

Examples
--------
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py full --model EleutherAI/pythia-160m --device cuda
    python scripts/run_pipeline.py reproduce --notebooks
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CACHE = ROOT / "cache"
SPLITS = ["WikiMIA_length32", "WikiMIA_length64", "WikiMIA_length128"]


def short_name(hf_or_short: str) -> str:
    return hf_or_short.split("/")[-1]


def run(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.check_call(cmd, cwd=ROOT, env=merged)


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def print_torch() -> str:
    import torch

    line = f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        line += f"  gpu={torch.cuda.get_device_name(0)}  vram={vram_gb:.1f} GB"
        print(line, flush=True)
        return "cuda"
    print(line, flush=True)
    return "cpu"


def resolve_device(requested: str) -> str:
    import torch

    if requested != "cuda":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    print("WARNING: --device cuda but torch.cuda.is_available() is False; using cpu",
          flush=True)
    return "cpu"


def vram_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def self_check() -> None:
    print("--- self-check ---", flush=True)
    run(py(str(SCRIPTS / "minkpp.py")))
    run(py(str(SCRIPTS / "01_cache.py"), "--self-check"))


def caches_present(short: str, dtype: str) -> bool:
    return all(
        (CACHE / f"{short}_{split}_{dtype}.npz").exists() for split in SPLITS
    )


def cache_one(hf_model: str, device: str, dtype: str) -> None:
    short = short_name(hf_model)
    if caches_present(short, dtype):
        print(f"caches for {short} ({dtype}) already present — skip 01_cache",
              flush=True)
        return
    print(f"--- cache {hf_model} on {device}/{dtype} ---", flush=True)
    run(py(
        str(SCRIPTS / "01_cache.py"),
        "--model", hf_model,
        "--all-splits",
        "--device", device,
        "--dtype", dtype,
    ))


def aggregate_and_figures(short: str, dtype: str, n_boot: int) -> None:
    print(f"--- aggregate {short} ---", flush=True)
    run(py(
        str(SCRIPTS / "02_aggregate.py"),
        "--model", short,
        "--dtype", dtype,
        "--n-boot", str(n_boot),
    ))
    print(f"--- figures {short} ---", flush=True)
    run(py(str(SCRIPTS / "03_figures.py"), "--model", short))


def run_notebooks(short: str) -> None:
    print(f"--- notebooks (MINKPP_MODEL={short}) ---", flush=True)
    run(py(str(SCRIPTS / "build_notebooks.py"), "--execute"),
        env={"MINKPP_MODEL": short})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="reproduce",
                    choices=["reproduce", "cache", "full", "pythia69", "check"])
    ap.add_argument("--model", default="state-spaces/mamba-1.4b-hf",
                    help="HF id or short name (default: Mamba-1.4B)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--notebooks", action="store_true")
    a = ap.parse_args()

    os.chdir(ROOT)
    print("=== Min-K%++ pipeline ===", flush=True)
    print(f"mode={a.mode}  model={a.model}  device={a.device}  dtype={a.dtype}",
          flush=True)
    print_torch()
    device = resolve_device(a.device)

    if a.mode == "check":
        self_check()
        print("=== done (check) ===", flush=True)
        return 0

    self_check()

    if a.mode == "reproduce":
        dtype = "float32"
        for short in ("mamba-1.4b-hf", "pythia-160m"):
            if caches_present(short, dtype):
                aggregate_and_figures(short, dtype, a.n_boot)
            else:
                print(f"skip {short}: no {dtype} caches in cache/", flush=True)
        if a.notebooks:
            run_notebooks("mamba-1.4b-hf")

    elif a.mode == "cache":
        cache_one(a.model, device, a.dtype)

    elif a.mode == "full":
        cache_one(a.model, device, a.dtype)
        aggregate_and_figures(short_name(a.model), a.dtype, a.n_boot)
        if a.notebooks:
            run_notebooks(short_name(a.model))

    elif a.mode == "pythia69":
        gb = vram_gb()
        if gb is not None and gb < 13.0:
            print(
                f"ERROR: pythia69 needs ~14 GB VRAM in float16; this GPU has "
                f"{gb:.1f} GB. Use `reproduce` (committed caches) or "
                f"`full --model EleutherAI/pythia-160m`.",
                flush=True,
            )
            return 1
        dtype = "float16"
        hf = "EleutherAI/pythia-6.9b"
        cache_one(hf, device, dtype)
        aggregate_and_figures(short_name(hf), dtype, a.n_boot)

    print(f"=== done ({a.mode}) ===", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e
