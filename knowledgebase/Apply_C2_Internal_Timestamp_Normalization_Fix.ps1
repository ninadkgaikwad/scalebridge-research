param(
    [string]$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$AlignmentPath = Join-Path `
    $RepoRoot `
    "src\scalebridge\data\heat_input_regression\alignment.py"

$BackupPath = "$AlignmentPath.pre_internal_normalization_fix_backup"

if (-not (Test-Path $AlignmentPath)) {
    throw "Alignment module not found: $AlignmentPath"
}

if (-not (Test-Path $BackupPath)) {
    Copy-Item `
        -Path $AlignmentPath `
        -Destination $BackupPath `
        -Force
}

Write-Host "C2 internal timestamp-normalization fix installed:"
Write-Host $AlignmentPath
Write-Host ""
Write-Host "Backup:"
Write-Host $BackupPath
Write-Host ""
Write-Host "Behavior:"
Write-Host "- timestamp normalization remains local to parsing"
Write-Host "- build_timestamp_frame emits only timestamp_raw and timestamp"
Write-Host "- timestamp_normalized cannot enter derived-feature validation"
