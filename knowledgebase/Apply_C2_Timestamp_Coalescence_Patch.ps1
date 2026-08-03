param(
    [string]$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$AlignmentPath = Join-Path `
    $RepoRoot `
    "src\scalebridge\data\heat_input_regression\alignment.py"

$BackupPath = "$AlignmentPath.pre_timestamp_coalescence_backup"

if (-not (Test-Path $AlignmentPath)) {
    throw "Alignment module not found: $AlignmentPath"
}

if (-not (Test-Path $BackupPath)) {
    Copy-Item `
        -Path $AlignmentPath `
        -Destination $BackupPath `
        -Force
}

Write-Host "Timestamp coalescence alignment module installed:"
Write-Host $AlignmentPath
Write-Host ""
Write-Host "Backup:"
Write-Host $BackupPath
Write-Host ""
Write-Host "The C2 builder will now:"
Write-Host "- normalize timestamp strings before parsing"
Write-Host "- coalesce complementary same-time rows column by column"
Write-Host "- record conflicting non-null source values"
