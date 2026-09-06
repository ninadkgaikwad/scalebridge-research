<#
.SYNOPSIS
Starts a local ScaleBridge MLflow tracking server.

.DESCRIPTION
Each machine runs its own local MLflow server.

Backend DB:
  machine-local only, under LOCALAPPDATA.
  never synced by Dropbox/Git.

Artifacts:
  stored under SCALEBRIDGE_GENERATED_DATA_ROOT\mlflow_artifacts.
  no top-level machine folders.
  semantic experiment/campaign organization is handled by experiment artifact locations.
#>

param(
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

if (-not $env:SCALEBRIDGE_GENERATED_DATA_ROOT) {
    throw "SCALEBRIDGE_GENERATED_DATA_ROOT is not configured."
}

if ($env:SCALEBRIDGE_MACHINE_ID) {
    $MachineId = $env:SCALEBRIDGE_MACHINE_ID
} else {
    $MachineId = $env:COMPUTERNAME
}

$LocalStateRoot = Join-Path $env:LOCALAPPDATA "ScaleBridge\mlflow"
$ArtifactRoot = Join-Path $env:SCALEBRIDGE_GENERATED_DATA_ROOT "mlflow_artifacts"
$DatabasePath = Join-Path $LocalStateRoot "mlflow.db"

New-Item -ItemType Directory -Path $LocalStateRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null

$DatabaseUriPath = $DatabasePath.Replace("\", "/")

if ($env:CONDA_PREFIX) {
    $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
    $MlflowExe = Join-Path $env:CONDA_PREFIX "Scripts\mlflow.exe"
} else {
    $PythonExe = (Get-Command python).Source
    $MlflowExe = (Get-Command mlflow).Source
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $MlflowExe)) {
    throw "MLflow executable not found: $MlflowExe"
}

Write-Host "ScaleBridge machine id: $MachineId"
Write-Host "MLflow backend: $DatabasePath"
Write-Host "MLflow artifacts: $ArtifactRoot"
Write-Host "MLflow endpoint: http://127.0.0.1:$Port"
Write-Host "MLflow Python: $PythonExe"
Write-Host "MLflow executable: $MlflowExe"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SslSafeLauncher = Join-Path $ScriptRoot "ssl_safe_mlflow.py"

if (-not (Test-Path $SslSafeLauncher)) {
    throw "SSL-safe MLflow launcher not found: $SslSafeLauncher"
}

Write-Host "MLflow SSL-safe launcher: $SslSafeLauncher"

$env:SCALEBRIDGE_PATCH_SSL_CERTIFI = "1"

if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$ScriptRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = "$ScriptRoot"
}

Write-Host "ScaleBridge SSL certifi patch: $env:SCALEBRIDGE_PATCH_SSL_CERTIFI"
Write-Host "ScaleBridge server PYTHONPATH prefix: $ScriptRoot"

& $PythonExe $SslSafeLauncher server `
    --backend-store-uri "sqlite:///$DatabaseUriPath" `
    --default-artifact-root $ArtifactRoot `
    --artifacts-destination $ArtifactRoot `
    --host 127.0.0.1 `
    --port $Port