# Min-K%++ pipeline on Windows (no Docker).
# Prefer setup_and_run.ps1 the first time on a machine.
#
# Usage:
#   .\run.ps1                  # reproduce tables/figures from committed caches
#   .\run.ps1 check            # self-checks only
#   .\run.ps1 full -Model EleutherAI/pythia-160m -Device cuda
#   .\run.ps1 reproduce -Notebooks
#
# Every run writes logs\run_YYYYMMDD_HHMMSS_<mode>.log with STEP OK/FAIL + SUMMARY.
param(
    [Parameter(Position = 0)]
    [ValidateSet("reproduce", "cache", "full", "pythia69", "check", "followups")]
    [string]$Mode = "reproduce",

    [string]$Model = "state-spaces/mamba-1.4b-hf",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("float32", "float16", "bfloat16")]
    [string]$Dtype = "float32",
    [int]$NBoot = 2000,
    [switch]$Notebooks
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
} else {
    Write-Host "WARNING: .venv not found — using system python. Run .\setup_and_run.ps1 first." -ForegroundColor Yellow
    $py = (Get-Command python -ErrorAction Stop).Source
}

if (-not $env:HF_HOME) {
    $env:HF_HOME = "$PWD\hf-cache"
}

$argsList = @(
    "scripts\run_pipeline.py", $Mode,
    "--model", $Model,
    "--device", $Device,
    "--dtype", $Dtype,
    "--n-boot", "$NBoot"
)
if ($Notebooks) { $argsList += "--notebooks" }

Write-Host "Using Python: $py"
& $py @argsList
$code = $LASTEXITCODE

$latest = Get-ChildItem logs\run_*.log -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latest) {
    Write-Host ""
    Write-Host "LOG: $($latest.FullName)"
    if ($code -ne 0) {
        Write-Host "----- SUMMARY from log -----" -ForegroundColor Red
        Get-Content $latest.FullName | Select-String -Pattern "SUMMARY|STEP |result:|ERROR" |
            ForEach-Object { $_.Line }
    }
}
exit $code
