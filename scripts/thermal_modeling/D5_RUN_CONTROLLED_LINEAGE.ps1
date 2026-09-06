$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path ".").Path
$CampaignId = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
$MatrixRunId = "aggregation_matrix_20260715_114242"
$PhaseCRunId = "phase_c_full_updated_test_laptop_20260802_172455"

$CampaignRoot = (
    Resolve-Path (
        Join-Path $RepoRoot "..\..\Data\ScaleBridge\campaigns\$CampaignId"
    )
).Path

$OutputRoot = Join-Path $RepoRoot "_phase_d_d5_validation"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$Runs = @(
    "aggr_20260715_114247_0001_a8695a44_smoke_l01_all_to_one_equal",
    "aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal"
)

foreach ($RunId in $Runs) {
    $Output = Join-Path $OutputRoot "$RunId`_lineage.json"

    python ".\scripts\thermal_modeling\validate_phase_d_lineage.py" `
        --campaign-root $CampaignRoot `
        --matrix-run-id $MatrixRunId `
        --phase-c-campaign-run-id $PhaseCRunId `
        --aggregation-run-id $RunId `
        --output $Output

    if ($LASTEXITCODE -ne 0) {
        throw "D5 controlled validation failed for $RunId"
    }
}

$Forbidden = @(
    Get-ChildItem `
        -LiteralPath $OutputRoot `
        -Recurse `
        -File |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in @(".parquet", ".csv")
    }
)

if ($Forbidden.Count -gt 0) {
    throw "D5 storage policy failure: time-series/CSV output found in validation root."
}

Write-Host ""
Write-Host "D5 controlled lineage validation completed." -ForegroundColor Green
Write-Host "Metadata only: $OutputRoot"
