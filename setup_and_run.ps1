# One-shot: install deps + run EVERYTHING (original report + follow-ups A-D).
# From the repo root in PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_and_run.ps1
#
# Optional:
#   .\setup_and_run.ps1 -SkipConcat    # skip WikiMIA_concat (no GPU cache for D)
#
# Exit code 0 = SUCCESS. Logs under logs\.

param(
    [switch]$SkipConcat
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== setup_and_run.ps1 (everything) ===" -ForegroundColor Cyan
Write-Host "repo: $PSScriptRoot"

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

Write-Host "installing torch (CUDA 12.6 wheel) ..."
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "installing requirements.txt ..."
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$env:HF_HOME = "$PWD\hf-cache"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "checking CUDA ..."
& $py -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "running pipeline: everything (reproduce + follow-ups A-D) ..."
$extra = @()
if ($SkipConcat) { $extra += "-SkipConcat" }
& "$PSScriptRoot\run.ps1" everything @extra
exit $LASTEXITCODE
