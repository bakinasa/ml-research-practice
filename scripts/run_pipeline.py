#!/usr/bin/env python3
"""Orchestrate the Min-K%++ pipeline without Docker.

Modes
-----
    reproduce   self-check + aggregate + figures from committed caches (default)
    cache       run 01_cache.py for HF_MODEL if caches are missing
    full        cache (if needed) + aggregate + figures for one model
    pythia69    cache EleutherAI/pythia-6.9b in float16 (~14 GB VRAM), then score
    check       self-checks only

All runs write a transcript to logs/run_YYYYMMDD_HHMMSS.log (and print the path).
Exit code 0 = success; non-zero = a step failed (see the log summary).

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
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CACHE = ROOT / "cache"
LOGS = ROOT / "logs"
SPLITS = ["WikiMIA_length32", "WikiMIA_length64", "WikiMIA_length128"]


class RunLog:
    """Tee stdout/stderr-style messages to console + a timestamped log file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8", newline="\n")
        self.steps: list[tuple[str, str, float]] = []  # name, status, seconds
        self._write(f"log file: {path}")
        self._write(f"started:  {datetime.now().isoformat(timespec='seconds')}")
        self._write(f"cwd:      {ROOT}")
        self._write(f"python:   {sys.executable}")
        self._write("")

    def _write(self, msg: str = "") -> None:
        print(msg, flush=True)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def info(self, msg: str = "") -> None:
        self._write(msg)

    def section(self, title: str) -> None:
        self._write(f"--- {title} ---")

    def step(self, name: str, fn) -> None:
        self.section(name)
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:
            dt = time.perf_counter() - t0
            self.steps.append((name, "FAIL", dt))
            self._write(f"STEP FAIL [{name}] after {dt:.1f}s: {e}")
            self._write(traceback.format_exc())
            raise
        dt = time.perf_counter() - t0
        self.steps.append((name, "OK", dt))
        self._write(f"STEP OK   [{name}]  {dt:.1f}s")
        self._write("")

    def summary(self, ok: bool, mode: str) -> None:
        self._write("")
        self._write("=== SUMMARY ===")
        self._write(f"mode:     {mode}")
        self._write(f"result:   {'SUCCESS' if ok else 'FAILED'}")
        self._write(f"finished: {datetime.now().isoformat(timespec='seconds')}")
        if self.steps:
            self._write("steps:")
            for name, status, dt in self.steps:
                self._write(f"  [{status:4s}] {name:40s} {dt:8.1f}s")
        self._write(f"log:      {self.path}")
        self._write("=== " + ("done" if ok else "failed") + f" ({mode}) ===")

    def close(self) -> None:
        self._fh.close()


LOG: RunLog | None = None


def short_name(hf_or_short: str) -> str:
    return hf_or_short.split("/")[-1]


def run(cmd: list[str], env: dict | None = None) -> None:
    assert LOG is not None
    LOG.info("+ " + " ".join(str(c) for c in cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    # Capture child output into the same log (and echo to console).
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        LOG.info(line.rstrip("\n"))
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def print_torch() -> str:
    assert LOG is not None
    import torch

    line = f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        line += f"  gpu={torch.cuda.get_device_name(0)}  vram={vram_gb:.1f} GB"
        LOG.info(line)
        return "cuda"
    LOG.info(line)
    return "cpu"


def resolve_device(requested: str) -> str:
    assert LOG is not None
    import torch

    if requested != "cuda":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    LOG.info("WARNING: --device cuda but torch.cuda.is_available() is False; using cpu")
    return "cpu"


def vram_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def self_check() -> None:
    assert LOG is not None
    LOG.step("minkpp self-check",
             lambda: run(py(str(SCRIPTS / "minkpp.py"))))
    LOG.step("01_cache self-check",
             lambda: run(py(str(SCRIPTS / "01_cache.py"), "--self-check")))


def caches_present(short: str, dtype: str) -> bool:
    return all(
        (CACHE / f"{short}_{split}_{dtype}.npz").exists() for split in SPLITS
    )


def cache_one(hf_model: str, device: str, dtype: str) -> None:
    assert LOG is not None
    short = short_name(hf_model)
    if caches_present(short, dtype):
        LOG.info(f"caches for {short} ({dtype}) already present — skip 01_cache")
        LOG.steps.append((f"cache {short}", "SKIP", 0.0))
        return

    def _go():
        run(py(
            str(SCRIPTS / "01_cache.py"),
            "--model", hf_model,
            "--all-splits",
            "--device", device,
            "--dtype", dtype,
        ))

    LOG.step(f"cache {short} ({device}/{dtype})", _go)


def aggregate_and_figures(short: str, dtype: str, n_boot: int) -> None:
    assert LOG is not None

    def _agg():
        run(py(
            str(SCRIPTS / "02_aggregate.py"),
            "--model", short,
            "--dtype", dtype,
            "--n-boot", str(n_boot),
        ))

    def _fig():
        run(py(str(SCRIPTS / "03_figures.py"), "--model", short))

    LOG.step(f"aggregate {short}", _agg)
    LOG.step(f"figures {short}", _fig)


def run_notebooks(short: str) -> None:
    assert LOG is not None
    LOG.step(
        f"notebooks {short}",
        lambda: run(py(str(SCRIPTS / "build_notebooks.py"), "--execute"),
                    env={"MINKPP_MODEL": short}),
    )


def main() -> int:
    global LOG

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="reproduce",
                    choices=["reproduce", "cache", "full", "pythia69", "check",
                             "followups"])
    ap.add_argument("--model", default="state-spaces/mamba-1.4b-hf",
                    help="HF id or short name (default: Mamba-1.4B)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--notebooks", action="store_true")
    a = ap.parse_args()

    os.chdir(ROOT)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"run_{stamp}_{a.mode}.log"
    LOG = RunLog(log_path)

    ok = False
    try:
        LOG.info("=== Min-K%++ pipeline ===")
        LOG.info(f"mode={a.mode}  model={a.model}  device={a.device}  dtype={a.dtype}")

        if a.mode == "followups":
            # A+B+C for both committed models; leaves original results/ intact.
            for short in ("mamba-1.4b-hf", "pythia-160m"):
                LOG.step(
                    f"followups abc {short}",
                    lambda s=short: run(py(
                        str(SCRIPTS / "run_followups.py"), "abc", "--model", s,
                    )),
                )
            ok = True
            return 0

        print_torch()
        device = resolve_device(a.device)
        LOG.info("")

        if a.mode == "check":
            self_check()
        elif a.mode == "reproduce":
            self_check()
            dtype = "float32"
            for short in ("mamba-1.4b-hf", "pythia-160m"):
                if caches_present(short, dtype):
                    aggregate_and_figures(short, dtype, a.n_boot)
                else:
                    LOG.info(f"skip {short}: no {dtype} caches in cache/")
                    LOG.steps.append((f"aggregate {short}", "SKIP", 0.0))
            if a.notebooks:
                run_notebooks("mamba-1.4b-hf")
        elif a.mode == "cache":
            self_check()
            cache_one(a.model, device, a.dtype)
        elif a.mode == "full":
            self_check()
            cache_one(a.model, device, a.dtype)
            aggregate_and_figures(short_name(a.model), a.dtype, a.n_boot)
            if a.notebooks:
                run_notebooks(short_name(a.model))
        elif a.mode == "pythia69":
            self_check()
            gb = vram_gb()
            if gb is not None and gb < 13.0:
                LOG.info(
                    f"ERROR: pythia69 needs ~14 GB VRAM in float16; this GPU has "
                    f"{gb:.1f} GB. Use `reproduce` (committed caches) or "
                    f"`full --model EleutherAI/pythia-160m`."
                )
                return 1
            dtype = "float16"
            hf = "EleutherAI/pythia-6.9b"
            cache_one(hf, device, dtype)
            aggregate_and_figures(short_name(hf), dtype, a.n_boot)

        ok = True
        return 0
    except subprocess.CalledProcessError as e:
        LOG.info(f"command failed with exit code {e.returncode}")
        return e.returncode or 1
    except Exception:
        LOG.info("pipeline aborted with an exception (see traceback above)")
        return 1
    finally:
        LOG.summary(ok, a.mode)
        LOG.close()
        # Always remind where the log is, even if console scrolled away.
        print(f"\nFull log: {log_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
