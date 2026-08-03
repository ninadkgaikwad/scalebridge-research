param(
    [string]$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$DatasetsPath = Join-Path `
    $RepoRoot `
    "src\scalebridge\data\heat_input_regression\datasets.py"

$AlignmentPath = Join-Path `
    $RepoRoot `
    "src\scalebridge\data\heat_input_regression\alignment.py"

if (-not (Test-Path $DatasetsPath)) {
    throw "C4 datasets module not found: $DatasetsPath"
}
if (-not (Test-Path $AlignmentPath)) {
    throw "Shared alignment module not found: $AlignmentPath"
}

$BackupPath = "$DatasetsPath.pre_shared_timestamp_refactor_backup"
if (-not (Test-Path $BackupPath)) {
    Copy-Item `
        -Path $DatasetsPath `
        -Destination $BackupPath `
        -Force
}

$DatasetsText = Get-Content -Raw -Path $DatasetsPath
if ($DatasetsText -notmatch "canonicalize_wide_frame") {
    throw "The replacement datasets.py does not import the shared canonicalization utility."
}

Write-Host "Shared C2/C4 timestamp canonicalization refactor installed:"
Write-Host $DatasetsPath
Write-Host ""
Write-Host "Backup:"
Write-Host $BackupPath
Write-Host ""
Write-Host "Architecture:"
Write-Host "- alignment.py remains the single timestamp-policy owner"
Write-Host "- C2 uses canonicalize_wide_frame for feature timelines"
Write-Host "- C4 reuses canonicalize_wide_frame for Stage B target lookup"
Write-Host "- C4 joins targets to C2 on parsed timestamp"
Write-Host "- no duplicate timestamp parser was added to datasets.py"
