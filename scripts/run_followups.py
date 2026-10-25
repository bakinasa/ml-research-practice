#!/usr/bin/env python3
"""Run all follow-up experiments (A–D) without touching the original pipeline.

Modes
-----
    all       A+B+C, and D if concat cache exists or --with-concat
    abc       stacking + H10 + H7 only (no GPU; uses existing caches/scores)
    stacking / h10 / h7 / concat   single experiment

Examples
--------
    python scripts/run_followups.py abc --model mamba-1.4b-hf
    python scripts/run_followups.py all --model mamba-1.4b-hf --with-concat \\
        --concat-model EleutherAI/pythia-160m --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
LOGS = ROOT / "logs"


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def _safe_print(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding or "utf-8", errors="replace")
              .decode(sys.stdout.encoding or "utf-8", errors="replace"),
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="abc",
                    choices=["all", "abc", "stacking", "h10", "h7", "concat"])
    ap.add_argument("--model", default="mamba-1.4b-hf",
                    help="short name for A/B/C (needs scores/caches)")
    ap.add_argument("--concat-model", default="EleutherAI/pythia-160m",
                    help="HF id for WikiMIA_concat cache (D)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--concat-dtype", default="float16")
    ap.add_argument("--with-concat", action="store_true",
                    help="with mode=all, also run D (may download + GPU cache)")
    ap.add_argument("--concat-limit", type=int, default=None)
    a = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"followups_{stamp}_{a.mode}.log"
    print(f"=== follow-ups === mode={a.mode}  log={log_path}")

    # tee-ish: also append a short summary file pointer
    jobs: list[tuple[str, list[str]]] = []

    def add_stacking():
        jobs.append(("A stacking",
                     py(str(SCRIPTS / "04_stacking.py"), "--model", a.model)))

    def add_h10():
        jobs.append(("B H10 trade-off",
                     py(str(SCRIPTS / "05_h10_tradeoff.py"),
                        "--model", a.model, "--dtype", a.dtype)))

    def add_h7():
        jobs.append(("C H7 calibration",
                     py(str(SCRIPTS / "06_h7_calibration.py"),
                        "--model", a.model, "--dtype", a.dtype)))

    def add_concat():
        jobs.append(("D concat windows",
                     py(str(SCRIPTS / "07_concat_windows.py"),
                        "--model", a.concat_model,
                        "--device", a.device,
                        "--dtype", a.concat_dtype,
                        *(["--limit", str(a.concat_limit)]
                          if a.concat_limit else []))))

    if a.mode in ("all", "abc", "stacking"):
        add_stacking()
    if a.mode in ("all", "abc", "h10"):
        add_h10()
    if a.mode in ("all", "abc", "h7"):
        add_h7()
    if a.mode == "concat" or (a.mode == "all" and a.with_concat):
        add_concat()
    elif a.mode == "all" and not a.with_concat:
        print("note: skipping D (concat). Pass --with-concat to include it.")

    results = []
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    with log_path.open("w", encoding="utf-8", newline="\n") as fh:
        def log(msg: str = ""):
            _safe_print(msg)
            fh.write(msg + "\n")
            fh.flush()

        log(f"started {datetime.now().isoformat(timespec='seconds')}")
        log(f"python {sys.executable}")
        for name, cmd in jobs:
            log(f"--- {name} ---")
            log("+ " + " ".join(cmd))
            try:
                proc = subprocess.run(
                    cmd, cwd=ROOT, env=child_env,
                    text=True, encoding="utf-8", errors="replace",
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                out = proc.stdout or ""
                fh.write(out)
                if not out.endswith("\n"):
                    fh.write("\n")
                fh.flush()
                _safe_print(out.rstrip("\n"))
                if proc.returncode != 0:
                    log(f"STEP FAIL [{name}] exit={proc.returncode}")
                    results.append((name, "FAIL"))
                    return proc.returncode
                log(f"STEP OK   [{name}]")
                results.append((name, "OK"))
            except Exception as e:
                log(f"STEP FAIL [{name}] {e}")
                results.append((name, "FAIL"))
                return 1

        log("")
        log("=== SUMMARY ===")
        for name, st in results:
            log(f"  [{st:4s}] {name}")
        log("outputs: results/followups/  and  results/followups/figures/")
        log(f"finished {datetime.now().isoformat(timespec='seconds')}")
        log("=== done ===")

    _safe_print(f"\nFull log: {log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e
