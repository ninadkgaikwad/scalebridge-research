$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$CampaignId = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
$CampaignRoot = Join-Path $env:SCALEBRIDGE_GENERATED_DATA_ROOT "campaigns\$CampaignId"
$MatrixRunId = "aggregation_matrix_20260715_114242"
$PhaseCRunId = "phase_c_full_updated_test_laptop_20260802_172455"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PhaseDRunId = "phase_d_mlflow_test_$Stamp"
$ExperimentName = "ScaleBridge_PhaseD_Test"
$RunName = "d8_4_mlflow_mdh_l1_l3_l6_h1_sd_o30_tr21_te7_$Stamp"

Write-Host "Running complete thermal-modeling test suite..." -ForegroundColor Cyan
pytest tests\thermal_modeling -v -W error::pandas.errors.PerformanceWarning
if ($LASTEXITCODE -ne 0) { throw "Thermal-modeling tests failed." }

Write-Host "Running controlled D8.4 Phase D campaign with MLflow..." -ForegroundColor Cyan
python scripts\thermal_modeling\run_phase_d_campaign.py `
    --campaign-root "$CampaignRoot" `
    --matrix-run-id "$MatrixRunId" `
    --phase-c-campaign-run-id "$PhaseCRunId" `
    --phase-d-run-id "$PhaseDRunId" `
    --phase-d-calendar-year 2001 `
    --heat-representation grouped `
    --ml-policy mdh `
    --ml-input-lag 1 `
    --ml-input-lag 3 `
    --ml-input-lag 6 `
    --ml-target-horizon 1 `
    --ml-train-fraction 0.70 `
    --ml-test-fraction 0.15 `
    --ml-validation-fraction 0.15 `
    --ob-policy sd `
    --sd-season-offset-days 30 `
    --sd-train-days 21 `
    --sd-test-days 7 `
    --parquet-compression zstd `
    --continue-on-error `
    --overwrite-existing `
    --mlflow `
    --mlflow-experiment-name "$ExperimentName" `
    --mlflow-run-name "$RunName" `
    --mlflow-strict
if ($LASTEXITCODE -ne 0) { throw "D8.4 controlled campaign failed." }

python scripts\thermal_modeling\validate_phase_d_campaign.py `
    --campaign-root "$CampaignRoot" `
    --phase-d-run-id "$PhaseDRunId"
if ($LASTEXITCODE -ne 0) { throw "Phase D campaign validation failed." }

python scripts\thermal_modeling\validate_phase_d_mlflow_tracking.py `
    --campaign-root "$CampaignRoot" `
    --phase-d-run-id "$PhaseDRunId"
if ($LASTEXITCODE -ne 0) { throw "Phase D MLflow validation failed." }

Write-Host "PHASE D D8.4 MLFLOW TESTING CAMPAIGN VALIDATED" -ForegroundColor Green
Write-Host "Phase D run: $PhaseDRunId"
Write-Host "Experiment:  $ExperimentName"
