param(
    [string]$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$AlignmentPath = Join-Path `
    $RepoRoot `
    "src\scalebridge\data\heat_input_regression\alignment.py"

$BackupPath = "$AlignmentPath.pre_helper_column_fix_backup"

if (-not (Test-Path $AlignmentPath)) {
    throw "Alignment module not found: $AlignmentPath"
}

if (-not (Test-Path $BackupPath)) {
    Copy-Item `
        -Path $AlignmentPath `
        -Destination $BackupPath `
        -Force
}

Write-Host "C2 helper-column fix installed:"
Write-Host $AlignmentPath
Write-Host ""
Write-Host "Backup:"
Write-Host $BackupPath
Write-Host ""
Write-Host "Fix:"
Write-Host "- timestamp_normalized remains internal only"
Write-Host "- it is excluded from source-feature scoring"
Write-Host "- it is removed before derived-feature validation"
