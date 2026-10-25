# Follow-up experiments A–D (separate from the original H1–H10 pipeline).
#
# Usage (after .\setup_and_run.ps1 or with .venv ready):
#   .\run_followups.ps1                  # A+B+C on mamba (no GPU)
#   .\run_followups.ps1 -Model pythia-160m
#   .\run_followups.ps1 all -WithConcat  # also WikiMIA_concat (GPU)
#   .\run_followups.ps1 concat           # only D
param(
    [Parameter(Position = 0)]
    [ValidateSet("all", "abc", "stacking", "h10", "h7", "concat")]
    [string]$Mode = "abc",

    [string]$Model = "mamba-1.4b-hf",
    [string]$ConcatModel = "EleutherAI/pythia-160m",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$WithConcat
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Test-Path ".\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
} else {
    $py = (Get-Command python -ErrorAction Stop).Source
}

$argsList = @(
    "scripts\run_followups.py", $Mode,
    "--model", $Model,
    "--concat-model", $ConcatModel,
    "--device", $Device
)
if ($WithConcat) { $argsList += "--with-concat" }

Write-Host "Using Python: $py"
& $py @argsList
$code = $LASTEXITCODE

$latest = Get-ChildItem logs\followups_*.log -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($latest) {
    Write-Host "LOG: $($latest.FullName)"
    Get-Content $latest.FullName -Tail 25
}
exit $code
