param(
    [string]$CampaignId = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3",
    [string]$PhaseCRunId = "phase_c_c2fix_20260722_205232"
)
$ErrorActionPreference = "Stop"
if (-not $env:SCALEBRIDGE_GENERATED_DATA_ROOT) { throw "SCALEBRIDGE_GENERATED_DATA_ROOT is not configured." }
if (-not $env:SCALEBRIDGE_MACHINE_ID) { throw "SCALEBRIDGE_MACHINE_ID is not configured." }
if (-not $env:MLFLOW_TRACKING_URI) { throw "MLFLOW_TRACKING_URI is not configured." }
python `
  ".\scripts\heat_input_regression\register_phase_c_run_with_mlflow.py" `
  --campaign-id $CampaignId `
  --phase-c-run-id $PhaseCRunId
if ($LASTEXITCODE -ne 0) { throw "C9 MLflow registration failed with exit code $LASTEXITCODE" }
$RegistrationManifest = Join-Path `
    $env:SCALEBRIDGE_GENERATED_DATA_ROOT `
    "campaigns\$CampaignId\heat_input_regression\mlflow_registration_runs\$PhaseCRunId\phase_c_mlflow_registration_manifest.json"
python `
  ".\scripts\heat_input_regression\validate_phase_c_mlflow_tracking.py" `
  --registration-manifest $RegistrationManifest `
  --expected-stage-runs 8 `
  --expected-training-task-runs 30 `
  --expected-evaluation-task-runs 30 `
  --expected-inference-task-runs 3
if ($LASTEXITCODE -ne 0) { throw "C9 MLflow validation failed with exit code $LASTEXITCODE" }
Write-Host ""
Write-Host "C9 PHASE C MLFLOW SMOKE PASSED"
Write-Host "Registration manifest: $RegistrationManifest"
