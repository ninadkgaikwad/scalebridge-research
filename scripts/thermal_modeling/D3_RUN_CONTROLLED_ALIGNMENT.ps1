$ErrorActionPreference="Stop"
$RepoRoot=(Resolve-Path ".").Path
$CampaignRoot=(Resolve-Path (Join-Path $RepoRoot "..\..\Data\ScaleBridge\campaigns\p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3")).Path
$Out=Join-Path $RepoRoot "_phase_d_d3_validation"; New-Item -ItemType Directory -Force -Path $Out|Out-Null
$Cases=@(
@{r="aggr_20260715_114247_0001_a8695a44_smoke_l01_all_to_one_equal";z="RestaurantFastFood_All"},
@{r="aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal";z="Dining"},
@{r="aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal";z="Kitchen"})
foreach($c in $Cases){
 python scripts\thermal_modeling\validate_phase_d_alignment.py --campaign-root $CampaignRoot --matrix-run-id aggregation_matrix_20260715_114242 --aggregation-run-id $c.r --phase-c-campaign-run-id phase_c_full_updated_test_laptop_20260802_172455 --aggregate-zone-id $c.z --phase-d-calendar-year 2001 --output-json (Join-Path $Out "$($c.z)_alignment.json")
 if($LASTEXITCODE-ne 0){throw "D3 failed for $($c.z)"}
}
Write-Host "D3 controlled alignment completed: $Out" -ForegroundColor Green
