# One-shot setup + full report reproduce for the Windows GPU laptop.
# Run from the repo root in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_and_run.ps1
#
# What it does:
#   1) creates .venv if missing
#   2) installs CUDA torch + requirements
#   3) runs self-checks + rebuilds tables/figures from committed caches
#   4) writes a transcript under logs\run_*.log with STEP OK / FAIL + SUMMARY
#
# Exit code 0 = SUCCESS. Open the newest file in logs\ if anything looks wrong.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== setup_and_run.ps1 ===" -ForegroundColor Cyan
Write-Host "repo: $PSScriptRoot"

# --- Python / venv ---
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python not found. Install Python 3.10+ and re-open the terminal."
}
python --version

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "creating .venv ..."
    python -m venv .venv
}
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
Write-Host "using: $py"

# --- deps ---
Write-Host "installing torch (CUDA 12.6 wheel) ..."
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "installing requirements.txt ..."
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:HF_HOME = "$PWD\hf-cache"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

Write-Host "checking CUDA ..."
& $py -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# --- pipeline (logged under logs\) ---
Write-Host "running pipeline: reproduce (self-check + tables + figures) ..."
& $py scripts\run_pipeline.py reproduce
$code = $LASTEXITCODE

$latest = Get-ChildItem logs\run_*.log -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Write-Host ""
if ($code -eq 0) {
    Write-Host "RESULT: SUCCESS" -ForegroundColor Green
} else {
    Write-Host "RESULT: FAILED (exit $code)" -ForegroundColor Red
}
if ($latest) {
    Write-Host "LOG:    $($latest.FullName)"
    Write-Host "----- last 30 lines of log -----"
    Get-Content $latest.FullName -Tail 30
}
exit $code
