<#
.SYNOPSIS
Starts the laptop-hosted ScaleBridge MLflow tracking server.

.DESCRIPTION
Stores the SQLite metadata database in laptop-local application data so
Dropbox never synchronizes an open database file. Compact MLflow artifacts
are stored under the shared ScaleBridge generated-data directory.
#>

$ErrorActionPreference = "Stop"

if (-not $env:SCALEBRIDGE_GENERATED_DATA_ROOT) {
    throw "SCALEBRIDGE_GENERATED_DATA_ROOT is not configured."
}

$LocalStateRoot = Join-Path $env:LOCALAPPDATA "ScaleBridge\mlflow"
$ArtifactRoot = Join-Path $env:SCALEBRIDGE_GENERATED_DATA_ROOT "mlflow_artifacts"
$DatabasePath = Join-Path $LocalStateRoot "mlflow.db"

New-Item -ItemType Directory -Path $LocalStateRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null

$DatabaseUriPath = $DatabasePath.Replace("\", "/")

Write-Host "MLflow backend: $DatabasePath"
Write-Host "MLflow artifacts: $ArtifactRoot"
Write-Host "MLflow endpoint: http://0.0.0.0:5000"

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

Write-Host "MLflow Python: $PythonExe"
Write-Host "MLflow executable: $MlflowExe"

& $MlflowExe server `
    --backend-store-uri "sqlite:///$DatabaseUriPath" `
    --default-artifact-root $ArtifactRoot `
    --artifacts-destination $ArtifactRoot `
    --host 0.0.0.0 `
    --port 5000