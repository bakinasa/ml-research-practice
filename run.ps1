# Min-K%++ pipeline on Windows (no Docker).
# Prefer setup_and_run.ps1 the first time on a machine.
#
# Usage:
#   .\run.ps1 everything       # EVERYTHING: reproduce + follow-ups A–D
#   .\run.ps1                  # reproduce tables/figures from committed caches
#   .\run.ps1 followups        # only A+B+C
#   .\run.ps1 check            # self-checks only
#   .\run.ps1 everything -SkipConcat   # no WikiMIA_concat GPU step
#
# Every run writes logs\run_YYYYMMDD_HHMMSS_<mode>.log with STEP OK/FAIL + SUMMARY.
param(
    [Parameter(Position = 0)]
    [ValidateSet("reproduce", "everything", "cache", "full", "pythia69", "check", "followups")]
    [string]$Mode = "reproduce",

    [string]$Model = "state-spaces/mamba-1.4b-hf",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("float32", "float16", "bfloat16")]
    [string]$Dtype = "float32",
    [int]$NBoot = 2000,
    [string]$ConcatModel = "EleutherAI/pythia-160m",
    [switch]$Notebooks,
    [switch]$SkipConcat
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
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$argsList = @(
    "scripts\run_pipeline.py", $Mode,
    "--model", $Model,
    "--device", $Device,
    "--dtype", $Dtype,
    "--n-boot", "$NBoot",
    "--concat-model", $ConcatModel
)
if ($Notebooks) { $argsList += "--notebooks" }
if ($SkipConcat) { $argsList += "--skip-concat" }

Write-Host "Using Python: $py"
Write-Host "Mode: $Mode"
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
    } else {
        Write-Host "----- last 20 lines -----"
        Get-Content $latest.FullName -Tail 20
    }
}
exit $code
