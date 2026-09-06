$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path ".").Path

$CampaignId = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
$CaseId = "epcase_827ca4812c0199221d031e59"
$MatrixRunId = "aggregation_matrix_20260715_114242"
$AggregationRunId = "aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal"
$PhaseCRunId = "phase_c_full_updated_test_laptop_20260802_172455"

$CampaignRoot = (
    Resolve-Path (
        Join-Path $RepoRoot "..\..\Data\ScaleBridge\campaigns\$CampaignId"
    )
).Path

$OutputRoot = Join-Path $RepoRoot "_phase_d_d7_validation"

Remove-Item `
    -LiteralPath $OutputRoot `
    -Recurse `
    -Force `
    -ErrorAction SilentlyContinue

python ".\scripts\thermal_modeling\build_phase_d_final_datasets.py" `
    --campaign-root $CampaignRoot `
    --matrix-run-id $MatrixRunId `
    --aggregation-run-id $AggregationRunId `
    --phase-c-campaign-run-id $PhaseCRunId `
    --output-root $OutputRoot `
    --phase-d-calendar-year 2001 `
    --heat-representation grouped `
    --ml-input-lag 12 `
    --ml-target-horizon 6 `
    --ml-train-fraction 0.70 `
    --ml-test-fraction 0.15 `
    --ml-validation-fraction 0.15 `
    --sd-season-offset-days 0 `
    --sd-train-days 21 `
    --sd-test-days 7 `
    --parquet-compression zstd

if ($LASTEXITCODE -ne 0) {
    throw "D7 controlled build failed."
}

$RunRoot = Join-Path `
    $OutputRoot `
    "phase_d\cases\$CaseId\aggregation_runs\$AggregationRunId"

if (-not (Test-Path -LiteralPath $RunRoot -PathType Container)) {
    throw "D7 controlled run root was not created: $RunRoot"
}

$Parquets = @(
    Get-ChildItem -LiteralPath $RunRoot -Recurse -File -Filter "*.parquet"
)
$Jsons = @(
    Get-ChildItem -LiteralPath $RunRoot -Recurse -File -Filter "*.json"
)
$Temps = @(
    Get-ChildItem -LiteralPath $RunRoot -Recurse -File |
    Where-Object { $_.Name -like "*.tmp" }
)

# Identity smoke run has two current zones.
# Per silo: 2 Independent + Dep1 + Dep2 = 4.
# Two silos => 8 final Parquets.
if ($Parquets.Count -ne 8) {
    throw "Expected exactly 8 final D7 Parquets; found $($Parquets.Count)."
}

if ($Temps.Count -ne 0) {
    throw "Temporary files remain after D7 build: $($Temps.Count)"
}

$ForbiddenNames = @(
    $Parquets |
    Where-Object {
        $_.Name -ne "data.parquet"
    }
)
if ($ForbiddenNames.Count -ne 0) {
    throw "D7 storage policy failure: unexpected Parquet filename."
}

$ForbiddenPathTokens = @(
    "aligned",
    "assembly",
    "assembled",
    "canonical",
    "normalized",
    "preview",
    "train.parquet",
    "test.parquet",
    "validation.parquet"
)
foreach ($Item in $Parquets) {
    $Lower = $Item.FullName.ToLowerInvariant()
    foreach ($Token in $ForbiddenPathTokens) {
        if ($Lower.Contains($Token)) {
            throw "D7 storage policy failure: forbidden path token '$Token' in $($Item.FullName)"
        }
    }
}

Write-Host ""
Write-Host "D7 controlled builders completed." -ForegroundColor Green
Write-Host "Final Parquets: $($Parquets.Count)"
Write-Host "JSON metadata files: $($Jsons.Count)"
Write-Host "Intermediate Parquets: 0"
Write-Host "Output root: $RunRoot"
