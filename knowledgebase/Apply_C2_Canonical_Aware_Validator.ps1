param(
    [string]$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"
)

$ErrorActionPreference = "Stop"

$RunnerPath = Join-Path `
    $RepoRoot `
    "knowledgebase\C1_to_C8_Full_Rebuild_Validation.ps1"

$CanonicalValidatorRelative = `
    "scripts\heat_input_regression\validate_heat_input_regression_features_canonical_aware.py"

if (-not (Test-Path $RunnerPath)) {
    throw "Runner not found: $RunnerPath"
}

$BackupPath = "$RunnerPath.pre_canonical_aware_backup"

if (-not (Test-Path $BackupPath)) {
    Copy-Item `
        -Path $RunnerPath `
        -Destination $BackupPath `
        -Force
}

$Content = Get-Content `
    -Path $RunnerPath `
    -Raw

$Old = "scripts\heat_input_regression\validate_heat_input_regression_features.py"

if ($Content -notmatch [regex]::Escape($Old)) {
    if ($Content -match [regex]::Escape($CanonicalValidatorRelative)) {
        Write-Host "Runner is already canonical-aware."
        exit 0
    }

    throw "Could not find the legacy C2 validator command in: $RunnerPath"
}

$Updated = $Content.Replace(
    $Old,
    $CanonicalValidatorRelative
)

Set-Content `
    -Path $RunnerPath `
    -Value $Updated `
    -Encoding UTF8

Write-Host ""
Write-Host "Updated runner:"
Write-Host $RunnerPath
Write-Host ""
Write-Host "Backup:"
Write-Host $BackupPath
Write-Host ""
Write-Host "C2 standard validation now uses:"
Write-Host $CanonicalValidatorRelative
