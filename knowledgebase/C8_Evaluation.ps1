$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"

$C8Root = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge\campaigns\p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3\heat_input_regression\inference_runs\c8_full_pytorch_cuda_smoke_002"

$ReportPath = Join-Path `
  $RepoRoot `
  "C8_Full_PyTorch_CUDA_Smoke_002_Missing_Value_Root_Cause_Report.txt"

Set-Location $RepoRoot

# --------------------------------------------------------------------------------------
# Run the deeper source-file inspection.
# This updates the C8 audit CSVs with source-file and source-value information.
# --------------------------------------------------------------------------------------

python scripts\heat_input_regression\audit_heat_input_regression_inference_missing_values.py `
  --inference-root "$C8Root" `
  --inspect-source-files

if ($LASTEXITCODE -ne 0) {
    throw "C8 missing-value source-file audit failed with exit code $LASTEXITCODE."
}

# --------------------------------------------------------------------------------------
# Load audit tables.
# --------------------------------------------------------------------------------------

$ComponentAuditPath = Join-Path `
  $C8Root `
  "missing_value_root_cause_by_component.csv"

$TimestampAuditPath = Join-Path `
  $C8Root `
  "missing_value_root_cause_by_timestamp.csv"

$TimestampOverlapPath = Join-Path `
  $C8Root `
  "missing_value_overlap_by_timestamp.csv"

$SourceInventoryPath = Join-Path `
  $C8Root `
  "missing_value_source_file_inventory.csv"

$SourceValuesPath = Join-Path `
  $C8Root `
  "missing_value_source_values.csv"

$AuditManifestPath = Join-Path `
  $C8Root `
  "missing_value_root_cause_manifest.json"

$InferenceResultsPath = Join-Path `
  $C8Root `
  "inference_results.csv"

$ValidationResultsPath = Join-Path `
  $C8Root `
  "inference_validation_results.csv"

$ValidationDiagnosticsPath = Join-Path `
  $C8Root `
  "inference_validation_diagnostics.csv"

$ComponentAudit = Import-Csv $ComponentAuditPath
$TimestampAudit = Import-Csv $TimestampAuditPath
$TimestampOverlap = Import-Csv $TimestampOverlapPath
$SourceInventory = Import-Csv $SourceInventoryPath
$SourceValues = Import-Csv $SourceValuesPath
$AuditManifest = Get-Content $AuditManifestPath -Raw | ConvertFrom-Json
$InferenceResults = Import-Csv $InferenceResultsPath
$ValidationResults = Import-Csv $ValidationResultsPath
$ValidationDiagnostics = Import-Csv $ValidationDiagnosticsPath

# --------------------------------------------------------------------------------------
# Derive summary tables.
# --------------------------------------------------------------------------------------

$ComponentsWithMissingValues = $ComponentAudit |
  Where-Object {
    [int]$_.predictor_missing_count -gt 0
  } |
  Sort-Object aggregate_zone_id, model_id

$ComponentClassificationSummary = $ComponentsWithMissingValues |
  Group-Object root_cause_classification |
  ForEach-Object {
    [PSCustomObject]@{
      root_cause_classification = $_.Name
      component_count = $_.Count
      total_missing_values = (
        $_.Group |
        Measure-Object `
          -Property predictor_missing_count `
          -Sum
      ).Sum
    }
  } |
  Sort-Object root_cause_classification

$ZoneMissingSummary = $ComponentsWithMissingValues |
  Group-Object aggregate_zone_id |
  ForEach-Object {
    [PSCustomObject]@{
      aggregate_zone_id = $_.Name
      affected_component_count = $_.Count
      total_missing_component_values = (
        $_.Group |
        Measure-Object `
          -Property predictor_missing_count `
          -Sum
      ).Sum
      maximum_missing_count_for_one_component = (
        $_.Group |
        Measure-Object `
          -Property predictor_missing_count `
          -Maximum
      ).Maximum
    }
  } |
  Sort-Object aggregate_zone_id

$ModelFamilyMissingSummary = $ComponentsWithMissingValues |
  Group-Object model_id |
  ForEach-Object {
    [PSCustomObject]@{
      model_id = $_.Name
      affected_zone_count = (
        $_.Group.aggregate_zone_id |
        Sort-Object -Unique
      ).Count
      total_missing_values = (
        $_.Group |
        Measure-Object `
          -Property predictor_missing_count `
          -Sum
      ).Sum
      predictor_columns = (
        $_.Group.predictor_column |
        Sort-Object -Unique
      ) -join "; "
    }
  } |
  Sort-Object model_id

$AffectedTimestampSummary = $TimestampOverlap |
  Sort-Object aggregate_zone_id, timestamp_raw

$CommonAllModelGaps = $TimestampOverlap |
  Where-Object {
    $_.all_selected_models_missing -eq "True"
  } |
  Sort-Object aggregate_zone_id, timestamp_raw

$PredictorSpecificGaps = $TimestampAudit |
  Where-Object {
    $_.root_cause_classification -eq "predictor_specific_gap" -or
    $_.root_cause_classification -eq "common_gap_plus_predictor_specific_gap"
  } |
  Sort-Object aggregate_zone_id, model_id, timestamp_raw

$QACMissingRows = $TimestampAudit |
  Where-Object {
    $_.model_id -eq "QAC"
  } |
  Sort-Object aggregate_zone_id, timestamp_raw

$NonFiniteSourceValues = $SourceValues |
  Where-Object {
    $_.source_value_is_finite -ne "True"
  } |
  Sort-Object aggregate_zone_id, model_id, timestamp_raw, source_column

$FailedValidationDiagnostics = $ValidationDiagnostics |
  Where-Object {
    $_.status -eq "failed"
  }

# --------------------------------------------------------------------------------------
# Write one consolidated text report.
# --------------------------------------------------------------------------------------

$ReportLines = New-Object System.Collections.Generic.List[string]

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "SCALEBRIDGE C8 FULL PYTORCH-CUDA SMOKE 002"
)

$ReportLines.Add(
  "MISSING-VALUE ROOT-CAUSE REPORT"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  "GeneratedAt: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
)

$ReportLines.Add(
  "RepositoryRoot: $RepoRoot"
)

$ReportLines.Add(
  "InferenceRoot: $C8Root"
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "1. C8 INFERENCE STATUS"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $InferenceResults |
    Format-Table `
      aggregate_zone_id,
      row_count,
      component_count,
      status `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "2. C8 VALIDATION STATUS"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $ValidationResults |
    Format-Table `
      aggregate_zone_id,
      component_count,
      status,
      check_count,
      failed_check_count `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "Failed validation diagnostic count: $($FailedValidationDiagnostics.Count)"
)

if ($FailedValidationDiagnostics.Count -gt 0) {
  $ReportLines.Add("")

  $ReportLines.Add(
    (
      $FailedValidationDiagnostics |
      Format-Table `
        aggregate_zone_id,
        model_id,
        check_name,
        status,
        message `
        -AutoSize |
      Out-String
    ).TrimEnd()
  )
}

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "3. ROOT-CAUSE AUDIT SUMMARY"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  "Zone artifact count: $($AuditManifest.zone_artifact_count)"
)

$ReportLines.Add(
  "Components with missing values: $($AuditManifest.components_with_missing_values)"
)

$ReportLines.Add(
  "Total missing component values: $($AuditManifest.total_missing_component_values)"
)

$ReportLines.Add(
  "Unique affected timestamps: $($AuditManifest.unique_affected_timestamps)"
)

$ReportLines.Add(
  "All prediction masks match predictors: $($AuditManifest.all_prediction_masks_match_predictors)"
)

$ReportLines.Add("")

$ReportLines.Add(
  "Interpretation: A prediction is unavailable only where its corresponding predictor is unavailable."
)

$ReportLines.Add(
  "No evidence of a finite predictor producing a NaN prediction was found when all masks match."
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "4. MISSING VALUES BY AGGREGATE ZONE"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $ZoneMissingSummary |
    Format-Table `
      aggregate_zone_id,
      affected_component_count,
      total_missing_component_values,
      maximum_missing_count_for_one_component `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "5. ROOT-CAUSE CLASSIFICATION SUMMARY"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $ComponentClassificationSummary |
    Format-Table `
      root_cause_classification,
      component_count,
      total_missing_values `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "6. MISSING VALUES BY MODEL FAMILY"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $ModelFamilyMissingSummary |
    Format-Table `
      model_id,
      affected_zone_count,
      total_missing_values,
      predictor_columns `
      -Wrap `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "7. COMPONENT-LEVEL ROOT-CAUSE DETAILS"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $ComponentsWithMissingValues |
    Select-Object `
      aggregate_zone_id,
      model_id,
      predictor_column,
      predictor_missing_count,
      prediction_missing_count,
      missing_masks_match,
      common_all_feature_gap_count,
      predictor_specific_gap_count,
      root_cause_classification,
      source_columns |
    Format-Table `
      -Wrap `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "8. ALL AFFECTED TIMESTAMPS"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $AffectedTimestampSummary |
    Select-Object `
      aggregate_zone_id,
      timestamp_raw,
      missing_model_count,
      all_selected_models_missing,
      missing_models |
    Format-Table `
      -Wrap `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "9. TIMESTAMPS WHERE ALL SELECTED MODELS ARE MISSING"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

if ($CommonAllModelGaps.Count -eq 0) {
  $ReportLines.Add(
    "No timestamp was missing for every selected component in the same aggregate zone."
  )
}
else {
  $ReportLines.Add(
    (
      $CommonAllModelGaps |
      Select-Object `
        aggregate_zone_id,
        timestamp_raw,
        missing_model_count,
        missing_models |
      Format-Table `
        -Wrap `
        -AutoSize |
      Out-String
    ).TrimEnd()
  )
}

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "10. PREDICTOR-SPECIFIC OR MIXED GAPS"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

if ($PredictorSpecificGaps.Count -eq 0) {
  $ReportLines.Add(
    "No predictor-specific gaps were identified."
  )
}
else {
  $ReportLines.Add(
    (
      $PredictorSpecificGaps |
      Select-Object `
        aggregate_zone_id,
        model_id,
        timestamp_raw,
        missing_derived_feature_count,
        missing_derived_features,
        is_common_all_feature_gap,
        root_cause_classification |
      Format-Table `
        -Wrap `
        -AutoSize |
      Out-String
    ).TrimEnd()
  )
}

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "11. QAC-SPECIFIC MISSING TIMESTAMPS"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  (
    $QACMissingRows |
    Select-Object `
      aggregate_zone_id,
      row_index,
      timestamp_raw,
      missing_derived_feature_count,
      missing_derived_features,
      is_common_all_feature_gap,
      root_cause_classification |
    Format-Table `
      -Wrap `
      -AutoSize |
    Out-String
  ).TrimEnd()
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "12. SOURCE-FILE INVENTORY"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

if ($SourceInventory.Count -eq 0) {
  $ReportLines.Add(
    "No matching source files were identified."
  )
}
else {
  $ReportLines.Add(
    (
      $SourceInventory |
      Select-Object `
        aggregate_zone_id,
        model_id,
        predictor_column,
        matching_source_columns,
        timestamp_columns,
        row_count,
        source_file |
      Format-Table `
        -Wrap `
        -AutoSize |
      Out-String
    ).TrimEnd()
  )
}

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "13. NON-FINITE SOURCE VALUES AT AFFECTED TIMESTAMPS"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

if ($NonFiniteSourceValues.Count -eq 0) {
  $ReportLines.Add(
    "No non-finite raw source values were identified in the discovered source files."
  )

  $ReportLines.Add(
    "This would suggest that the missingness was introduced during alignment or feature construction."
  )
}
else {
  $ReportLines.Add(
    (
      $NonFiniteSourceValues |
      Select-Object `
        aggregate_zone_id,
        model_id,
        timestamp_raw,
        source_column,
        source_value,
        source_value_is_finite,
        source_file |
      Format-Table `
        -Wrap `
        -AutoSize |
      Out-String
    ).TrimEnd()
  )
}

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "14. PRELIMINARY INTERPRETATION RULES"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add("")

$ReportLines.Add(
  "1. If all or most component predictors are missing at the same timestamps, the likely cause is a common timestamp alignment or simulation-boundary gap."
)

$ReportLines.Add(
  "2. If only one predictor family is missing, the likely cause is a source-signal or feature-family-specific gap."
)

$ReportLines.Add(
  "3. If raw source values are non-finite at the same timestamps, the missingness originates in the source data."
)

$ReportLines.Add(
  "4. If raw source values are finite but derived predictors are missing, the missingness was introduced during feature construction or timestamp alignment."
)

$ReportLines.Add(
  "5. Since prediction masks match predictor masks, the trained linear model is not generating unexplained NaNs from valid inputs."
)

$ReportLines.Add("")

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines.Add(
  "END OF REPORT"
)

$ReportLines.Add(
  "===================================================================================================="
)

$ReportLines |
  Set-Content `
    -Path $ReportPath `
    -Encoding UTF8

Write-Host ""
Write-Host "C8 missing-value root-cause report written to:"
Write-Host $ReportPath
Write-Host ""

Get-Item $ReportPath |
  Select-Object `
    FullName,
    Length,
    LastWriteTime |
  Format-List