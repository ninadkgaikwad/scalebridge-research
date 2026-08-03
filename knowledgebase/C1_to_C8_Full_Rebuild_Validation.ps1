$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"
$CampaignRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge\campaigns\p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
$MatrixRunId = "aggregation_matrix_20260715_114242"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$AuditRunId = "heat_input_audit_c2fix_$Stamp"
$FeatureRunId = "heat_input_features_c2fix_$Stamp"
$SplitRunId = "heat_input_splits_c2fix_$Stamp"
$DatasetRunId = "heat_input_datasets_c2fix_$Stamp"
$C5RunId = "c5_c2fix_$Stamp"
$TrainingRunId = "c6_pytorch_cuda_c2fix_$Stamp"
$EvaluationRunId = "c7_pytorch_cuda_c2fix_$Stamp"
$InferenceRunId = "c8_pytorch_cuda_c2fix_$Stamp"
$ReportPath = Join-Path $RepoRoot "C1_to_C8_C2Fix_Full_Run_$Stamp.txt"
$SummaryPath = Join-Path $RepoRoot "C1_to_C8_C2Fix_Summary_$Stamp.txt"

Set-Location $RepoRoot
Start-Transcript -Path $ReportPath -Force

function Invoke-Stage {
    param([string]$Name,[scriptblock]$Command)
    Write-Host ""
    Write-Host ("="*100)
    Write-Host $Name
    Write-Host ("="*100)

    $global:LASTEXITCODE = 0
    & $Command 2>&1 | ForEach-Object { $_ | Out-Host }
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode"
    }
}

try {
    Invoke-Stage "SOURCE SYNTAX CHECK (NO BYTECODE)" { python scripts\heat_input_regression\validate_python_source_syntax.py --paths src\scalebridge\data\heat_input_regression src\scalebridge\models\heat_input_regression src\scalebridge\training src\scalebridge\evaluation src\scalebridge\inference scripts\heat_input_regression }

    Invoke-Stage "C1 AUDIT" { python scripts\heat_input_regression\audit_aggregation_for_heat_input_regression.py --campaign-root "$CampaignRoot" --matrix-run-id "$MatrixRunId" --audit-run-id "$AuditRunId" --minimum-sample-count 1000 --internal-gain-predictor-method aggregate_average --hvac-target-method signed_zone_sensible --continue-on-error }

    Invoke-Stage "C2 BUILD WITH CANONICAL TIMESTAMPS" { python scripts\heat_input_regression\build_heat_input_regression_features.py --campaign-root "$CampaignRoot" --matrix-run-id "$MatrixRunId" --audit-run-id "$AuditRunId" --feature-run-id "$FeatureRunId" --minimum-sample-count 1000 --internal-gain-predictor-method aggregate_average --hvac-target-method signed_zone_sensible --preview-rows 100 --continue-on-error }
    $FeatureRoot = Join-Path $CampaignRoot "heat_input_regression\feature_runs\$FeatureRunId"
    Invoke-Stage "C2 STANDARD + CANONICAL-AWARE VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_features_canonical_aware.py --feature-root "$FeatureRoot" --campaign-root "$CampaignRoot" --matrix-run-id "$MatrixRunId" --audit-run-id "$AuditRunId" --feature-run-id "$FeatureRunId" --minimum-sample-count 1000 --absolute-tolerance 1e-9 --relative-tolerance 1e-9 }
    Invoke-Stage "C2 CANONICAL TIMESTAMP VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_canonical_timestamps.py --feature-root "$FeatureRoot" --expected-row-count 105120 --expected-cadence-seconds 300 }
    Invoke-Stage "C2 TIMESTAMP COALESCENCE VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_timestamp_coalescence.py --feature-root "$FeatureRoot" --expected-row-count 105120 --fail-on-conflicting-source-values }

    Invoke-Stage "C3 BUILD" { python scripts\heat_input_regression\build_heat_input_regression_splits.py --campaign-root "$CampaignRoot" --matrix-run-id "$MatrixRunId" --audit-run-id "$AuditRunId" --feature-run-id "$FeatureRunId" --split-run-id "$SplitRunId" --split-strategy monthly_distributed_holdout --train-fraction 0.70 --validation-fraction 0.15 --test-fraction 0.15 --minimum-split-samples 1000 --random-seed 42 --preview-rows 100 --continue-on-error }
    Invoke-Stage "C3 VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_splits.py --campaign-root "$CampaignRoot" --matrix-run-id "$MatrixRunId" --feature-run-id "$FeatureRunId" --split-run-id "$SplitRunId" --minimum-split-samples 1000 --fraction-tolerance 0.01 }

    Invoke-Stage "C4 BUILD" { python scripts\heat_input_regression\build_heat_input_regression_datasets.py --campaign-root "$CampaignRoot" --matrix-run-id "$MatrixRunId" --audit-run-id "$AuditRunId" --feature-run-id "$FeatureRunId" --split-run-id "$SplitRunId" --dataset-run-id "$DatasetRunId" --minimum-split-samples 1000 --preview-rows 100 --continue-on-error }
    Invoke-Stage "C4 VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_datasets.py --campaign-root "$CampaignRoot" --matrix-run-id "$MatrixRunId" --audit-run-id "$AuditRunId" --feature-run-id "$FeatureRunId" --split-run-id "$SplitRunId" --dataset-run-id "$DatasetRunId" --minimum-split-samples 1000 --absolute-tolerance 1e-9 --relative-tolerance 1e-9 }
    $DatasetRoot = Join-Path $CampaignRoot "heat_input_regression\dataset_runs\$DatasetRunId"

    $C5Root = Join-Path $CampaignRoot "heat_input_regression\model_api_validation\$C5RunId"
    Invoke-Stage "C5 MODEL API VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_model_api.py --dataset-root "$DatasetRoot" --output-root "$C5Root" --max-c4-models 3 }

    $TrainingParent = Join-Path $CampaignRoot "heat_input_regression\training_runs"
    Invoke-Stage "C6 PYTORCH CUDA TRAINING" { python scripts\heat_input_regression\train_heat_input_regression_models.py --dataset-root "$DatasetRoot" --output-root "$TrainingParent" --training-run-id "$TrainingRunId" --estimator-type pytorch_linear --pytorch-device cuda --learning-rate 0.03 --max-epochs 3000 --tolerance 1e-10 --patience 200 --seed 42 --reload-atol 1e-12 --reload-rtol 1e-12 --prediction-preview-rows 100 --continue-on-error }
    $TrainingRoot = Join-Path $TrainingParent $TrainingRunId
    Invoke-Stage "C6 VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_training.py --training-root "$TrainingRoot" --coefficient-atol 0 --prediction-atol 1e-12 --prediction-rtol 1e-12 }

    $EvaluationParent = Join-Path $CampaignRoot "heat_input_regression\evaluation_runs"
    Invoke-Stage "C7 EVALUATION" { python scripts\heat_input_regression\evaluate_heat_input_regression_models.py --training-root "$TrainingRoot" --output-root "$EvaluationParent" --evaluation-run-id "$EvaluationRunId" --estimator-type pytorch_linear --requested-device cuda --prediction-preview-rows 100 --continue-on-error }
    $EvaluationRoot = Join-Path $EvaluationParent $EvaluationRunId
    Invoke-Stage "C7 VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_evaluation.py --evaluation-root "$EvaluationRoot" --metric-atol 1e-12 --metric-rtol 1e-12 }

    $InferenceParent = Join-Path $CampaignRoot "heat_input_regression\inference_runs"
    Invoke-Stage "C8 FULL-YEAR INFERENCE" { python scripts\heat_input_regression\run_heat_input_regression_full_year_inference.py --evaluation-root "$EvaluationRoot" --output-root "$InferenceParent" --inference-run-id "$InferenceRunId" --estimator-type pytorch_linear --requested-device cuda --preview-rows 100 --continue-on-error }
    $InferenceRoot = Join-Path $InferenceParent $InferenceRunId
    Invoke-Stage "C8 VALIDATION" { python scripts\heat_input_regression\validate_heat_input_regression_full_year_inference.py --inference-root "$InferenceRoot" --prediction-atol 1e-12 --prediction-rtol 1e-12 }
    Invoke-Stage "C8 MISSING-VALUE AUDIT" { python scripts\heat_input_regression\audit_heat_input_regression_inference_missing_values.py --inference-root "$InferenceRoot" --inspect-source-files }

    $Summary = @(
        "status=passed",
        "matrix_run_id=$MatrixRunId",
        "audit_run_id=$AuditRunId",
        "feature_run_id=$FeatureRunId",
        "split_run_id=$SplitRunId",
        "dataset_run_id=$DatasetRunId",
        "training_run_id=$TrainingRunId",
        "evaluation_run_id=$EvaluationRunId",
        "inference_run_id=$InferenceRunId",
        "feature_root=$FeatureRoot",
        "dataset_root=$DatasetRoot",
        "training_root=$TrainingRoot",
        "evaluation_root=$EvaluationRoot",
        "inference_root=$InferenceRoot",
        "transcript=$ReportPath"
    )
    $Summary | Set-Content -Path $SummaryPath -Encoding UTF8
    Write-Host "FULL C1-C8 RUN PASSED"
    Write-Host "Transcript: $ReportPath"
    Write-Host "Summary: $SummaryPath"
}
catch {
    "status=failed`nerror=$($_.Exception.Message)`ntranscript=$ReportPath" | Set-Content -Path $SummaryPath -Encoding UTF8
    Write-Error $_
    throw
}
finally { Stop-Transcript }

