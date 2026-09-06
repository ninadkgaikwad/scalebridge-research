param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("laptop","home-pc","lab-pc")]
    [string]$MachineId,

    [string]$OutputDir = (Join-Path $env:USERPROFILE "Downloads"),

    [switch]$DoNotRequireCuda
)

$ErrorActionPreference = "Stop"

$ExpectedEnvironment = switch ($MachineId) {
    "laptop"  { "scalebridge-dev-gpu-laptop" }
    "home-pc" { "scalebridge-dev-gpu-homepc" }
    "lab-pc"  { "scalebridge-dev-gpu-labpc" }
}

if ($env:CONDA_DEFAULT_ENV -ne $ExpectedEnvironment) {
    throw "Wrong active Conda environment. Machine '$MachineId' requires '$ExpectedEnvironment', but active environment is '$env:CONDA_DEFAULT_ENV'."
}

$RepoRoot = (Get-Location).Path
$AuditScript = Join-Path $RepoRoot "scripts\thermal_modeling\qualify_phase_e0_environment.py"

if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
    throw "Run this command from the scalebridge-research repository root."
}
if (-not (Test-Path $AuditScript)) {
    throw "Missing Phase E0 qualifier: $AuditScript"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$JsonReport = Join-Path $OutputDir "ScaleBridge_PhaseE0_Environment_${MachineId}_${Stamp}.json"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ScaleBridge Phase E0 Environment Qualification" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Machine        : $MachineId"
Write-Host "Repository     : $RepoRoot"
Write-Host "Python         : $(python --version 2>&1)"
Write-Host "Conda env      : $env:CONDA_DEFAULT_ENV"
Write-Host "Conda prefix   : $env:CONDA_PREFIX"
Write-Host "JSON report    : $JsonReport"
Write-Host ""

$ArgsList = @(
    $AuditScript,
    "--machine-id", $MachineId,
    "--repo-root", $RepoRoot,
    "--output", $JsonReport
)

if (-not $DoNotRequireCuda) {
    $ArgsList += "--require-cuda"
}

& python @ArgsList
$ExitCode = $LASTEXITCODE

Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "[PASS] Phase E0 environment qualification passed." -ForegroundColor Green
}
else {
    Write-Host "[FAIL] Phase E0 environment qualification found required failures." -ForegroundColor Red
}

Write-Host "JSON: $JsonReport"
Write-Host "TEXT: $([System.IO.Path]::ChangeExtension($JsonReport, '.txt'))"
Write-Host ""

exit $ExitCode
