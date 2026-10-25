# Follow-up experiments (A–D)

Separate from the original H1–H10 aggregation study. Old tables in `results/` and
figures in `results/figures/` are **not** overwritten. New outputs go to:

```
results/followups/*.csv
results/followups/figures/figA_*.png … figD_*.png
logs/followups_*.log
```

## What each experiment claims

| ID | Script | Claim | Needs GPU? |
|---|---|---|---|
| **A** | `04_stacking.py` | Logistic stack of Loss/Zlib/Min-K%/Min-K%++ beats Min-K%++ alone (AUROC) | no |
| **B** | `05_h10_tradeoff.py` | Prefix scoring keeps most AUROC at ~50% tokens (efficiency) | no |
| **C** | `06_h7_calibration.py` | Length-binned thresholds raise **TPR@5%FPR** (AUROC unchanged by design) | no |
| **D** | `07_concat_windows.py` | On `WikiMIA_concat`, H1 windows beat diluted global Min-K%++ | **yes** (first cache) |

## Run (preferred: one command)

```powershell
# first time on the GPU laptop (install + everything)
.\setup_and_run.ps1

# later (venv already exists)
.\run.ps1 everything
```

That runs the original report **and** A+B+C for both models **and** D (`WikiMIA_concat`).

```powershell
.\run.ps1 everything -SkipConcat   # no GPU concat step
.\run.ps1 followups                # only A+B+C
.\run_followups.ps1 concat         # only D
```

## How to read success

- **A:** `delta_vs_minkpp > 0` with `significant=True` in `stacking_*.csv`
- **B:** at `frac≈0.5`, `|delta_auroc|` small while `compute_frac≈0.5`
- **C:** `delta_tpr5 > 0` in `h7_calibration_*.csv` (ignore AUROC — it must stay flat)
- **D:** H1 rows with positive significant `delta_vs_chunk_minkpp`; `global_minkpp` should be ≤ chunk (dilution)

## Diff vs original study

Original `.\run.ps1` / `02_aggregate.py` answered: *does changing Min-K%++ aggregation help on WikiMIA?* → **no**.

Follow-ups ask: *can the detector system around it still improve under different metrics / data?*
