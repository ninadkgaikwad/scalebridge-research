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

Write-Host "Running complete thermal-modeling suite with PerformanceWarning as error..."
pytest tests\thermal_modeling -v -W error::pandas.errors.PerformanceWarning
if ($LASTEXITCODE -ne 0) { throw "Thermal-modeling tests failed." }

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunId = "phase_d_all_policies_test_$Stamp"

$CommonArgs = @(
    "scripts\thermal_modeling\run_phase_d_campaign.py",
    "--campaign-root", $CampaignRoot,
    "--phase-d-run-id", $RunId,
    "--phase-d-calendar-year", "2001",
    "--heat-representation", "grouped",
    "--ml-policy", "mdh",
    "--ml-policy", "ch",
    "--ml-policy", "sh",
    "--ml-input-lag", "12",
    "--ml-target-horizon", "6",
    "--ml-train-fraction", "0.70",
    "--ml-test-fraction", "0.15",
    "--ml-validation-fraction", "0.15",
    "--ml-sh-train-seasons", "winter,spring",
    "--ml-sh-test-seasons", "summer",
    "--ml-sh-validation-seasons", "fall",
    "--ob-policy", "sd",
    "--ob-policy", "sbh",
    "--ob-policy", "ci",
    "--ob-policy", "cdr",
    "--sd-season-offset-days", "0",
    "--sd-train-days", "21",
    "--sd-test-days", "7",
    "--sbh-train-seasons", "winter,spring,fall",
    "--sbh-test-seasons", "summer",
    "--ci-start-datetime", "2001-04-01T00:05:00",
    "--ci-train-days", "21",
    "--ci-test-days", "7",
    "--cdr-train-range", "2001-01-01T00:05:00/2001-01-22T00:05:00",
    "--cdr-test-range",  "2001-01-22T00:05:00/2001-01-29T00:05:00",
    "--cdr-train-range", "2001-07-01T00:05:00/2001-07-22T00:05:00",
    "--cdr-test-range",  "2001-07-22T00:05:00/2001-07-29T00:05:00",
    "--parquet-compression", "zstd",
    "--overwrite-existing",
    "--continue-on-error"
)

Write-Host "Running complete testing campaign with all 7 Phase D policies..."
& python @CommonArgs
if ($LASTEXITCODE -ne 0) {
    throw "All-policy Phase D testing campaign failed. Inspect phase_d\campaign_runs\$RunId\logs."
}

python scripts\thermal_modeling\validate_phase_d_campaign.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $RunId
if ($LASTEXITCODE -ne 0) { throw "Generic D8 validation failed." }

python scripts\thermal_modeling\validate_phase_d_all_policies.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $RunId
if ($LASTEXITCODE -ne 0) { throw "All-policy catalog validation failed." }

$ResumeRunId = "${RunId}_resume"
$ResumeArgs = @(
    "scripts\thermal_modeling\run_phase_d_campaign.py",
    "--campaign-root", $CampaignRoot,
    "--phase-d-run-id", $ResumeRunId,
    "--phase-d-calendar-year", "2001",
    "--heat-representation", "grouped",
    "--ml-policy", "mdh",
    "--ml-policy", "ch",
    "--ml-policy", "sh",
    "--ml-input-lag", "12",
    "--ml-target-horizon", "6",
    "--ml-train-fraction", "0.70",
    "--ml-test-fraction", "0.15",
    "--ml-validation-fraction", "0.15",
    "--ml-sh-train-seasons", "winter,spring",
    "--ml-sh-test-seasons", "summer",
    "--ml-sh-validation-seasons", "fall",
    "--ob-policy", "sd",
    "--ob-policy", "sbh",
    "--ob-policy", "ci",
    "--ob-policy", "cdr",
    "--sd-season-offset-days", "0",
    "--sd-train-days", "21",
    "--sd-test-days", "7",
    "--sbh-train-seasons", "winter,spring,fall",
    "--sbh-test-seasons", "summer",
    "--ci-start-datetime", "2001-04-01T00:05:00",
    "--ci-train-days", "21",
    "--ci-test-days", "7",
    "--cdr-train-range", "2001-01-01T00:05:00/2001-01-22T00:05:00",
    "--cdr-test-range",  "2001-01-22T00:05:00/2001-01-29T00:05:00",
    "--cdr-train-range", "2001-07-01T00:05:00/2001-07-22T00:05:00",
    "--cdr-test-range",  "2001-07-22T00:05:00/2001-07-29T00:05:00",
    "--parquet-compression", "zstd",
    "--resume",
    "--continue-on-error"
)

Write-Host "Running resume validation..."
& python @ResumeArgs
if ($LASTEXITCODE -ne 0) { throw "All-policy resume run failed." }

python scripts\thermal_modeling\validate_phase_d_campaign.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $ResumeRunId
if ($LASTEXITCODE -ne 0) { throw "Resume storage validation failed." }

python scripts\thermal_modeling\validate_phase_d_all_policies.py `
    --campaign-root $CampaignRoot `
    --phase-d-run-id $ResumeRunId
if ($LASTEXITCODE -ne 0) { throw "Resume policy validation failed." }

Write-Host ""
Write-Host "ALL PHASE D POLICIES VALIDATED ON TESTING CAMPAIGN" -ForegroundColor Green
Write-Host "Primary run: $RunId"
Write-Host "Resume run:  $ResumeRunId"
Write-Host "Phase D root: $(Join-Path $CampaignRoot 'phase_d')"
