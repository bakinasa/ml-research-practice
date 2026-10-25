# Min-K%++ pipeline on Windows (no Docker).
# Usage:
#   .\run.ps1                  # reproduce tables/figures from committed caches
#   .\run.ps1 check            # self-checks only
#   .\run.ps1 full             # GPU-cache HF model if missing, then aggregate
#   .\run.ps1 full -Model EleutherAI/pythia-160m -Device cuda
#   .\run.ps1 reproduce -Notebooks
param(
    [Parameter(Position = 0)]
    [ValidateSet("reproduce", "cache", "full", "pythia69", "check")]
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

$py = $null
if (Test-Path ".\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
} else {
    $py = (Get-Command python -ErrorAction Stop).Source
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
exit $LASTEXITCODE
