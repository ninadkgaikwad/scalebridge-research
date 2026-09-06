###########################################################################
# ScaleBridge Phase D - D5 FINAL Tree-Aware Development Bundle
#
# Purpose:
#   Collect only the exact Phase B and Phase C metadata needed to implement
#   D5 aggregation lineage and all-to-one counterpart resolution.
#
# HARD SCOPE:
#   - Testing campaign only
#   - Selected aggregation matrix only
#   - Selected Phase C controlled campaign only
#   - Metadata/code only
#   - No Parquet, pickle, model binaries, NPY/NPZ, or previews
#
# Run from:
#   scalebridge-research repository root
#
# Output:
#   _phase_d_d5_inventory\
#       PhaseD_D5_Final_TreeAware_Bundle\
#       PhaseD_D5_Final_TreeAware_Bundle.zip
###########################################################################

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

###########################################################################
# Locked testing identities
###########################################################################

$RepoRoot = (Resolve-Path ".").Path

$CampaignId = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
$CaseId = "epcase_827ca4812c0199221d031e59"
$AggregationMatrixRunId = "aggregation_matrix_20260715_114242"

$AllToOneRunId = `
    "aggr_20260715_114247_0001_a8695a44_smoke_l01_all_to_one_equal"

$IdentityRunId = `
    "aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal"

$PhaseCCampaignRunId = `
    "phase_c_full_updated_test_laptop_20260802_172455"

$PhaseCAuditRunId = "heat_input_audit_20260802_172455"
$PhaseCFeatureRunId = "heat_input_features_20260802_172455"
$PhaseCSplitRunId = "heat_input_splits_20260802_172455"
$PhaseCDatasetRunId = "heat_input_datasets_20260802_172455"
$PhaseCTrainingRunId = "c6_pytorch_20260802_172455"
$PhaseCEvaluationRunId = "c7_pytorch_20260802_172455"
$PhaseCInferenceRunId = "c8_pytorch_20260802_172455"

###########################################################################
# Roots
###########################################################################

$CampaignRoot = (
    Resolve-Path (
        Join-Path `
            $RepoRoot `
            "..\..\Data\ScaleBridge\campaigns\$CampaignId"
    )
).Path

$AggregationRoot = Join-Path $CampaignRoot "aggregation"
$HeatInputRegressionRoot = Join-Path $CampaignRoot "heat_input_regression"

$MatrixRoot = Join-Path `
    (Join-Path $AggregationRoot "matrix_runs") `
    $AggregationMatrixRunId

$CaseRunsRoot = Join-Path `
    (Join-Path `
        (Join-Path $AggregationRoot "cases") `
        $CaseId
    ) `
    "runs"

$PhaseCCampaignRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "campaign_runs") `
    $PhaseCCampaignRunId

$PhaseCAuditRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "audit_runs") `
    $PhaseCAuditRunId

$PhaseCFeatureRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "feature_runs") `
    $PhaseCFeatureRunId

$PhaseCSplitRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "split_runs") `
    $PhaseCSplitRunId

$PhaseCDatasetRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "dataset_runs") `
    $PhaseCDatasetRunId

$PhaseCTrainingRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "training_runs") `
    $PhaseCTrainingRunId

$PhaseCEvaluationRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "evaluation_runs") `
    $PhaseCEvaluationRunId

$PhaseCInferenceRoot = Join-Path `
    (Join-Path $HeatInputRegressionRoot "inference_runs") `
    $PhaseCInferenceRunId

$OutputRoot = Join-Path $RepoRoot "_phase_d_d5_inventory"
$BundleName = "PhaseD_D5_Final_TreeAware_Bundle"
$BundleRoot = Join-Path $OutputRoot $BundleName
$ZipPath = Join-Path $OutputRoot "$BundleName.zip"

###########################################################################
# Helpers
###########################################################################

function Ensure-Directory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Required directory not found: $Path"
    }
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$DestinationRelative
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required file not found: $Source"
    }

    $Destination = Join-Path $BundleRoot $DestinationRelative
    $DestinationDirectory = Split-Path -Parent $Destination

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $DestinationDirectory |
        Out-Null

    Copy-Item `
        -LiteralPath $Source `
        -Destination $Destination `
        -Force

    Write-Host "Collected: $DestinationRelative"
}

function Copy-OptionalFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$DestinationRelative
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        Write-Host `
            "Optional metadata not found: $Source" `
            -ForegroundColor Yellow
        return
    }

    Copy-RequiredFile `
        -Source $Source `
        -DestinationRelative $DestinationRelative
}

function Copy-RootMetadataFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,

        [Parameter(Mandatory = $true)]
        [string]$DestinationPrefix
    )

    Ensure-Directory -Path $SourceRoot

    $Files = @(
        Get-ChildItem `
            -LiteralPath $SourceRoot `
            -File `
            -ErrorAction Stop |
        Where-Object {
            $_.Extension.ToLowerInvariant() -in @(".json", ".csv", ".txt")
        }
    )

    foreach ($File in $Files) {
        Copy-RequiredFile `
            -Source $File.FullName `
            -DestinationRelative (
                Join-Path $DestinationPrefix $File.Name
            )
    }
}

function Copy-MatchingFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,

        [Parameter(Mandatory = $true)]
        [string]$DestinationPrefix,

        [Parameter(Mandatory = $true)]
        [string[]]$Names
    )

    Ensure-Directory -Path $SourceRoot

    $Files = @(
        Get-ChildItem `
            -LiteralPath $SourceRoot `
            -Recurse `
            -File `
            -ErrorAction Stop |
        Where-Object {
            $_.Name -in $Names
        }
    )

    foreach ($File in $Files) {
        $Relative = $File.FullName.Substring(
            $SourceRoot.TrimEnd("\").Length
        ).TrimStart("\")

        Copy-RequiredFile `
            -Source $File.FullName `
            -DestinationRelative (
                Join-Path `
                    $DestinationPrefix `
                    $Relative
            )
    }
}

###########################################################################
# Validate exact tree roots
###########################################################################

Ensure-Directory -Path $CampaignRoot
Ensure-Directory -Path $AggregationRoot
Ensure-Directory -Path $HeatInputRegressionRoot
Ensure-Directory -Path $MatrixRoot
Ensure-Directory -Path $CaseRunsRoot

Ensure-Directory -Path $PhaseCCampaignRoot
Ensure-Directory -Path $PhaseCAuditRoot
Ensure-Directory -Path $PhaseCFeatureRoot
Ensure-Directory -Path $PhaseCSplitRoot
Ensure-Directory -Path $PhaseCDatasetRoot
Ensure-Directory -Path $PhaseCTrainingRoot
Ensure-Directory -Path $PhaseCEvaluationRoot
Ensure-Directory -Path $PhaseCInferenceRoot

###########################################################################
# Recreate bundle root
###########################################################################

New-Item `
    -ItemType Directory `
    -Force `
    -Path $OutputRoot |
    Out-Null

Remove-Item `
    -LiteralPath $BundleRoot `
    -Recurse `
    -Force `
    -ErrorAction SilentlyContinue

Remove-Item `
    -LiteralPath $ZipPath `
    -Force `
    -ErrorAction SilentlyContinue

New-Item `
    -ItemType Directory `
    -Force `
    -Path $BundleRoot |
    Out-Null

Write-Host ""
Write-Host "============================================================" `
    -ForegroundColor Cyan
Write-Host "D5 final tree-aware bundle" -ForegroundColor Green
Write-Host "============================================================" `
    -ForegroundColor Cyan
Write-Host "Campaign: $CampaignId"
Write-Host "Case:     $CaseId"
Write-Host "Matrix:   $AggregationMatrixRunId"

###########################################################################
# 1. Current Phase D implementation and tests
###########################################################################

$RepoFiles = @(
    "src\scalebridge\data\thermal_modeling\__init__.py",
    "src\scalebridge\data\thermal_modeling\constants.py",
    "src\scalebridge\data\thermal_modeling\identities.py",
    "src\scalebridge\data\thermal_modeling\models.py",
    "src\scalebridge\data\thermal_modeling\signals.py",
    "src\scalebridge\data\thermal_modeling\manifests.py",
    "src\scalebridge\data\thermal_modeling\source_refs.py",
    "src\scalebridge\data\thermal_modeling\discovery.py",
    "src\scalebridge\data\thermal_modeling\alignment.py",
    "src\scalebridge\data\thermal_modeling\assembly.py",
    "scripts\thermal_modeling\discover_phase_d_sources.py",
    "scripts\thermal_modeling\validate_phase_d_alignment.py",
    "scripts\thermal_modeling\validate_phase_d_assembly.py",
    "scripts\thermal_modeling\D4_RUN_CONTROLLED_ASSEMBLY.ps1",
    "tests\thermal_modeling\test_identities.py",
    "tests\thermal_modeling\test_signals.py",
    "tests\thermal_modeling\test_manifests.py",
    "tests\thermal_modeling\test_discovery.py",
    "tests\thermal_modeling\test_alignment.py",
    "tests\thermal_modeling\test_assembly.py",
    "tests\thermal_modeling\test_d4_storage_policy.py"
)

foreach ($Relative in $RepoFiles) {
    Copy-RequiredFile `
        -Source (Join-Path $RepoRoot $Relative) `
        -DestinationRelative (
            Join-Path "repository" $Relative
        )
}

###########################################################################
# 2. Existing D2-D4 controlled manifests
###########################################################################

foreach ($ValidationName in @(
    "_phase_d_d2_validation",
    "_phase_d_d3_validation",
    "_phase_d_d4_validation"
)) {
    $ValidationRoot = Join-Path $RepoRoot $ValidationName

    if (-not (Test-Path -LiteralPath $ValidationRoot -PathType Container)) {
        continue
    }

    $Files = @(
        Get-ChildItem `
            -LiteralPath $ValidationRoot `
            -File `
            -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Extension.ToLowerInvariant() -eq ".json"
        }
    )

    foreach ($File in $Files) {
        Copy-RequiredFile `
            -Source $File.FullName `
            -DestinationRelative (
                Join-Path `
                    "phase_d_existing\$ValidationName" `
                    $File.Name
            )
    }
}

###########################################################################
# 3. Phase B selected matrix - exact five files from real tree
###########################################################################

$MatrixFiles = @(
    "aggregation_matrix_case_runs.csv",
    "aggregation_matrix_manifest.json",
    "aggregation_matrix_outputs.csv",
    "missing_generation_rows.csv",
    "selected_aggregation_plans.csv"
)

foreach ($Name in $MatrixFiles) {
    Copy-RequiredFile `
        -Source (Join-Path $MatrixRoot $Name) `
        -DestinationRelative (
            Join-Path `
                "phase_b\matrix_runs\$AggregationMatrixRunId" `
                $Name
        )
}

###########################################################################
# 4. Phase B selected aggregation runs - exact controlled runs only
###########################################################################

$SelectedPhaseBRuns = @(
    $AllToOneRunId,
    $IdentityRunId
)

foreach ($AggregationRunId in $SelectedPhaseBRuns) {
    $RunRoot = Join-Path $CaseRunsRoot $AggregationRunId

    Ensure-Directory -Path $RunRoot

    $RunPrefix = `
        "phase_b\cases\$CaseId\runs\$AggregationRunId"

    Copy-RequiredFile `
        -Source (Join-Path $RunRoot "aggregation_manifest.json") `
        -DestinationRelative (
            Join-Path $RunPrefix "aggregation_manifest.json"
        )

    $InputsRoot = Join-Path $RunRoot "inputs"
    Ensure-Directory -Path $InputsRoot

    foreach ($Name in @(
        "aggregation_plan.json",
        "source_generation_run.json",
        "source_run_manifest.json",
        "zone_mapping.csv"
    )) {
        Copy-RequiredFile `
            -Source (Join-Path $InputsRoot $Name) `
            -DestinationRelative (
                Join-Path `
                    (Join-Path $RunPrefix "inputs") `
                    $Name
            )
    }

    $ZonesRoot = Join-Path $RunRoot "zones"
    Ensure-Directory -Path $ZonesRoot

    $ZoneDirectories = @(
        Get-ChildItem `
            -LiteralPath $ZonesRoot `
            -Directory `
            -ErrorAction Stop
    )

    foreach ($ZoneDirectory in $ZoneDirectories) {
        Copy-OptionalFile `
            -Source (
                Join-Path `
                    $ZoneDirectory.FullName `
                    "zone_mapping.csv"
            ) `
            -DestinationRelative (
                Join-Path `
                    (Join-Path `
                        (Join-Path $RunPrefix "zones") `
                        $ZoneDirectory.Name
                    ) `
                    "zone_mapping.csv"
            )
    }
}

###########################################################################
# 5. Phase C parent campaign - exact controlled campaign only
###########################################################################

Copy-RequiredFile `
    -Source (
        Join-Path `
            $PhaseCCampaignRoot `
            "phase_c_campaign_plan.json"
    ) `
    -DestinationRelative (
        "phase_c\campaign_runs\$PhaseCCampaignRunId\phase_c_campaign_plan.json"
    )

Copy-RequiredFile `
    -Source (
        Join-Path `
            $PhaseCCampaignRoot `
            "phase_c_campaign_run_manifest.json"
    ) `
    -DestinationRelative (
        "phase_c\campaign_runs\$PhaseCCampaignRunId\phase_c_campaign_run_manifest.json"
    )

###########################################################################
# 6. Phase C C1 audit - root lineage + per-zone applicability
###########################################################################

Copy-RootMetadataFiles `
    -SourceRoot $PhaseCAuditRoot `
    -DestinationPrefix (
        "phase_c\audit_runs\$PhaseCAuditRunId"
    )

Copy-MatchingFiles `
    -SourceRoot (
        Join-Path $PhaseCAuditRoot "cases"
    ) `
    -DestinationPrefix (
        "phase_c\audit_runs\$PhaseCAuditRunId\cases"
    ) `
    -Names @(
        "zone_audit_manifest.json",
        "model_applicability.csv",
        "applicable_models.csv",
        "inapplicable_models.csv",
        "unavailable_models.csv",
        "heat_input_signal_catalog.csv"
    )

###########################################################################
# 7. Phase C C2 feature run - root lineage + per-zone feature contract
###########################################################################

Copy-RootMetadataFiles `
    -SourceRoot $PhaseCFeatureRoot `
    -DestinationPrefix (
        "phase_c\feature_runs\$PhaseCFeatureRunId"
    )

Copy-MatchingFiles `
    -SourceRoot (
        Join-Path $PhaseCFeatureRoot "cases"
    ) `
    -DestinationPrefix (
        "phase_c\feature_runs\$PhaseCFeatureRunId\cases"
    ) `
    -Names @(
        "zone_feature_manifest.json",
        "model_applicability_snapshot.csv",
        "applicable_models_snapshot.csv",
        "inapplicable_models_snapshot.csv",
        "derived_feature_catalog.csv",
        "derived_feature_validation.csv"
    )

###########################################################################
# 8. Phase C C3 split run - root lineage + per-zone split contract
###########################################################################

Copy-RootMetadataFiles `
    -SourceRoot $PhaseCSplitRoot `
    -DestinationPrefix (
        "phase_c\split_runs\$PhaseCSplitRunId"
    )

Copy-MatchingFiles `
    -SourceRoot (
        Join-Path $PhaseCSplitRoot "cases"
    ) `
    -DestinationPrefix (
        "phase_c\split_runs\$PhaseCSplitRunId\cases"
    ) `
    -Names @(
        "zone_split_manifest.json",
        "split_summary.csv",
        "split_diagnostics.csv"
    )

###########################################################################
# 9. Phase C C4 dataset run - root lineage + zone indexes only
#
# D5 does not need individual model train/validation/test Parquets or the
# hundreds of per-model dataset artifacts.
###########################################################################

Copy-RootMetadataFiles `
    -SourceRoot $PhaseCDatasetRoot `
    -DestinationPrefix (
        "phase_c\dataset_runs\$PhaseCDatasetRunId"
    )

Copy-MatchingFiles `
    -SourceRoot (
        Join-Path $PhaseCDatasetRoot "cases"
    ) `
    -DestinationPrefix (
        "phase_c\dataset_runs\$PhaseCDatasetRunId\cases"
    ) `
    -Names @(
        "model_dataset_index.csv",
        "model_applicability_snapshot.csv",
        "applicable_models_snapshot.csv",
        "inapplicable_models_snapshot.csv"
    )

###########################################################################
# 10. Phase C C6 training - root lineage only
###########################################################################

Copy-RootMetadataFiles `
    -SourceRoot $PhaseCTrainingRoot `
    -DestinationPrefix (
        "phase_c\training_runs\$PhaseCTrainingRunId"
    )

###########################################################################
# 11. Phase C C7 evaluation - root lineage only
###########################################################################

Copy-RootMetadataFiles `
    -SourceRoot $PhaseCEvaluationRoot `
    -DestinationPrefix (
        "phase_c\evaluation_runs\$PhaseCEvaluationRunId"
    )

###########################################################################
# 12. Phase C C8 inference - root lineage + final prediction manifests
#
# This is especially important for D5 because D5 must verify that a matched
# Phase B aggregation run has usable Phase C inference lineage.
###########################################################################

Copy-RootMetadataFiles `
    -SourceRoot $PhaseCInferenceRoot `
    -DestinationPrefix (
        "phase_c\inference_runs\$PhaseCInferenceRunId"
    )

Copy-MatchingFiles `
    -SourceRoot (
        Join-Path $PhaseCInferenceRoot "cases"
    ) `
    -DestinationPrefix (
        "phase_c\inference_runs\$PhaseCInferenceRunId\cases"
    ) `
    -Names @(
        "annual_component_predictions_manifest.json",
        "component_applicability.csv",
        "component_prediction_registry.csv",
        "component_prediction_summary.csv",
        "component_missing_value_timestamps.csv",
        "timestamp_component_availability.csv"
    )

$BuildingPhvacRoot = Join-Path `
    $PhaseCInferenceRoot `
    "building_phvac_reconstruction"

Ensure-Directory -Path $BuildingPhvacRoot

Copy-MatchingFiles `
    -SourceRoot $BuildingPhvacRoot `
    -DestinationPrefix (
        "phase_c\inference_runs\$PhaseCInferenceRunId\building_phvac_reconstruction"
    ) `
    -Names @(
        "building_phvac_reconstruction_index.csv",
        "annual_building_phvac_predictions_manifest.json"
    )

###########################################################################
# 13. Compact selected-tree report
###########################################################################

$TreeReportPath = Join-Path `
    $BundleRoot `
    "D5_SELECTED_TREE.txt"

$TreeLines = @(
    "ScaleBridge Phase D D5 selected upstream tree",
    "==============================================",
    "",
    "Testing campaign:",
    "  $CampaignId",
    "",
    "Phase B:",
    "  aggregation\matrix_runs\$AggregationMatrixRunId",
    "  aggregation\cases\$CaseId\runs\$AllToOneRunId",
    "  aggregation\cases\$CaseId\runs\$IdentityRunId",
    "",
    "Phase C:",
    "  heat_input_regression\campaign_runs\$PhaseCCampaignRunId",
    "  heat_input_regression\audit_runs\$PhaseCAuditRunId",
    "  heat_input_regression\feature_runs\$PhaseCFeatureRunId",
    "  heat_input_regression\split_runs\$PhaseCSplitRunId",
    "  heat_input_regression\dataset_runs\$PhaseCDatasetRunId",
    "  heat_input_regression\training_runs\$PhaseCTrainingRunId",
    "  heat_input_regression\evaluation_runs\$PhaseCEvaluationRunId",
    "  heat_input_regression\inference_runs\$PhaseCInferenceRunId",
    "",
    "D5 controlled counterpart expectations:",
    "  all-to-one/equal -> matched_self",
    "  identity/equal   -> matched_exact to all-to-one/equal",
    "",
    "No P1 campaign paths are inspected."
)

$TreeLines |
Set-Content `
    -LiteralPath $TreeReportPath `
    -Encoding ASCII

###########################################################################
# 14. Locked D5 contract
###########################################################################

$ContractPath = Join-Path `
    $BundleRoot `
    "D5_LOCKED_CONTRACT.txt"

$ContractLines = @(
    "ScaleBridge Phase D - D5 Locked Contract",
    "========================================",
    "",
    "D5: Aggregation Lineage and Exact All-to-One Counterpart Resolution",
    "",
    "Phase B is authoritative for aggregation identity and counterpart matching.",
    "Phase C is authoritative for regression/inference lineage and usability.",
    "",
    "Counterpart statuses:",
    "  matched_self",
    "  matched_exact",
    "  ambiguous_multiple_counterparts",
    "  unavailable_no_counterpart",
    "  invalid_configuration_mismatch",
    "",
    "ambiguous_multiple_counterparts:",
    "  preserve all compatible candidates;",
    "  choose the first deterministic candidate;",
    "  permit Dependent 2 when Phase C lineage is usable.",
    "",
    "invalid_configuration_mismatch:",
    "  Dependent 2 is not possible and is not created.",
    "",
    "Dependent 2 requires both:",
    "  1. compatible Phase B all-to-one lineage;",
    "  2. usable Phase C inference lineage.",
    "",
    "D5 persists metadata only.",
    "",
    "Only final Phase D silo time-series Parquets may persist:",
    "  Independent",
    "  Dependent 1",
    "  Dependent 2",
    "",
    "No aligned, assembled, canonical, preview, or other intermediate",
    "time-series representation is retained."
)

$ContractLines |
Set-Content `
    -LiteralPath $ContractPath `
    -Encoding ASCII

###########################################################################
# 15. Safety audit
###########################################################################

$ForbiddenExtensions = @(
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".onnx",
    ".joblib",
    ".h5",
    ".hdf5"
)

$Forbidden = @(
    Get-ChildItem `
        -LiteralPath $BundleRoot `
        -Recurse `
        -File `
        -ErrorAction Stop |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in $ForbiddenExtensions
    }
)

if ($Forbidden.Count -gt 0) {
    $Forbidden |
    ForEach-Object {
        Write-Host "FORBIDDEN: $($_.FullName)" -ForegroundColor Red
    }

    throw "Bundle safety audit failed."
}

###########################################################################
# 16. Bundle inventory
###########################################################################

$InventoryRows = @(
    Get-ChildItem `
        -LiteralPath $BundleRoot `
        -Recurse `
        -File `
        -ErrorAction Stop |
    ForEach-Object {
        [PSCustomObject]@{
            RelativePath = $_.FullName.Substring(
                $BundleRoot.TrimEnd("\").Length
            ).TrimStart("\")
            SizeBytes = [int64]$_.Length
        }
    }
)

$InventoryRows |
Sort-Object RelativePath |
Export-Csv `
    -LiteralPath (
        Join-Path $BundleRoot "D5_BUNDLE_INVENTORY.csv"
    ) `
    -NoTypeInformation `
    -Encoding ASCII

###########################################################################
# 17. ZIP
###########################################################################

Compress-Archive `
    -LiteralPath $BundleRoot `
    -DestinationPath $ZipPath `
    -CompressionLevel Optimal `
    -Force

$ZipInfo = Get-Item -LiteralPath $ZipPath

Write-Host ""
Write-Host "============================================================" `
    -ForegroundColor Cyan
Write-Host "D5 final development bundle created" `
    -ForegroundColor Green
Write-Host "============================================================" `
    -ForegroundColor Cyan
Write-Host ""
Write-Host "ZIP:"
Write-Host "  $ZipPath"
Write-Host ""
Write-Host "Size:"
Write-Host "  $([math]::Round($ZipInfo.Length / 1MB, 2)) MB"
Write-Host ""
Write-Host "No P1 campaign scanned."
Write-Host "No Parquet/model data copied."
