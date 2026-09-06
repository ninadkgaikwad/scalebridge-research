$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path ".").Path
$CampaignRoot = (
    Resolve-Path (
        Join-Path $RepoRoot `
        "..\..\Data\ScaleBridge\campaigns\p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
    )
).Path

$Common = @(
    "--campaign-root", $CampaignRoot,
    "--matrix-run-id", "aggregation_matrix_20260715_114242",
    "--phase-c-campaign-run-id", "phase_c_full_updated_test_laptop_20260802_172455"
)

$Cases = @(
    @{
        aggregation_run_id = "aggr_20260715_114247_0001_a8695a44_smoke_l01_all_to_one_equal"
        zone = "RestaurantFastFood_All"
    },
    @{
        aggregation_run_id = "aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal"
        zone = "Dining"
    },
    @{
        aggregation_run_id = "aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal"
        zone = "Kitchen"
    }
)

$OutputRoot = Join-Path $RepoRoot "_phase_d_d2_validation"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

foreach ($Case in $Cases) {
    $OutputJson = Join-Path $OutputRoot "$($Case.zone)_discovery.json"

    python scripts\thermal_modeling\discover_phase_d_sources.py `
        @Common `
        --aggregation-run-id $Case.aggregation_run_id `
        --aggregate-zone-id $Case.zone `
        --output-json $OutputJson

    if ($LASTEXITCODE -ne 0) {
        throw "D2 discovery failed for zone $($Case.zone)"
    }
}

Write-Host ""
Write-Host "D2 controlled discovery completed." -ForegroundColor Green
Write-Host "Outputs: $OutputRoot"
