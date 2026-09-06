$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$CampaignId = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"

if (-not $env:SCALEBRIDGE_GENERATED_DATA_ROOT) {
    throw "SCALEBRIDGE_GENERATED_DATA_ROOT is not configured."
}

$CampaignRoot = Join-Path $env:SCALEBRIDGE_GENERATED_DATA_ROOT "campaigns\$CampaignId"
if (-not (Test-Path $CampaignRoot)) {
    throw "Campaign root not found: $CampaignRoot"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunId = "phase_d_testing_$Stamp"

Write-Host "Phase D D8 complete testing-campaign run"
Write-Host "campaign_root: $CampaignRoot"
Write-Host "phase_d_run_id: $RunId"

python scripts\thermal_modeling\run_phase_d_campaign.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $RunId `
    --continue-on-error

if ($LASTEXITCODE -ne 0) {
    throw "D8 testing campaign run failed. Inspect phase_d\campaign_runs\$RunId\logs."
}

python scripts\thermal_modeling\validate_phase_d_campaign.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $RunId

if ($LASTEXITCODE -ne 0) {
    throw "D8 testing campaign validation failed."
}

$ResumeRunId = "${RunId}_resume"
python scripts\thermal_modeling\run_phase_d_campaign.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $ResumeRunId `
    --resume `
    --continue-on-error

if ($LASTEXITCODE -ne 0) {
    throw "D8 resume validation run failed."
}

python scripts\thermal_modeling\validate_phase_d_campaign.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $ResumeRunId

if ($LASTEXITCODE -ne 0) {
    throw "D8 resume storage validation failed."
}

Write-Host ""
Write-Host "D8 COMPLETE TESTING CAMPAIGN VALIDATED" -ForegroundColor Green
Write-Host "Primary run: $RunId"
Write-Host "Resume run:  $ResumeRunId"
Write-Host "Phase D root: $(Join-Path $CampaignRoot 'phase_d')"
