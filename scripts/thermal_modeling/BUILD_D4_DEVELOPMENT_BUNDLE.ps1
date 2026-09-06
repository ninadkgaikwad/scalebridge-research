###########################################################################
# Phase D D4 — Canonical Signal Assembly Development Bundle
#
# Run from:
#   ...\NewOrg\scalebridge-research
#
# Output:
#   _phase_d_d4_inventory\
#       PhaseD_D4_Canonical_Signal_Assembly_Bundle.zip
#
# This bundle does NOT copy full annual Parquet datasets.
###########################################################################

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

###########################################################################
# Configuration
###########################################################################

$RepoRoot = (Resolve-Path ".").Path

$D2ValidationRoot = Join-Path `
    $RepoRoot `
    "_phase_d_d2_validation"

$D3ValidationRoot = Join-Path `
    $RepoRoot `
    "_phase_d_d3_validation"

$OutputRoot = Join-Path `
    $RepoRoot `
    "_phase_d_d4_inventory"

$BundleName = "PhaseD_D4_Canonical_Signal_Assembly_Bundle"

$BundleRoot = Join-Path `
    $OutputRoot `
    $BundleName

$ZipPath = Join-Path `
    $OutputRoot `
    "$BundleName.zip"

$Zones = @(
    "RestaurantFastFood_All",
    "Dining",
    "Kitchen"
)

###########################################################################
# Helpers
###########################################################################

function Copy-BundleFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,

        [Parameter(Mandatory = $true)]
        [string]$DestinationRelativePath,

        [bool]$Required = $true
    )

    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        if ($Required) {
            throw "Required bundle file missing: $SourcePath"
        }

        Write-Host `
            "Optional file not found: $SourcePath" `
            -ForegroundColor Yellow

        return $false
    }

    $DestinationPath = Join-Path `
        $BundleRoot `
        $DestinationRelativePath

    $DestinationDirectory = Split-Path -Parent $DestinationPath

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $DestinationDirectory |
        Out-Null

    Copy-Item `
        -LiteralPath $SourcePath `
        -Destination $DestinationPath `
        -Force

    return $true
}

function Get-RequiredJsonValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string]$PropertyName,

        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $Property = $Object.PSObject.Properties[$PropertyName]

    if (
        $null -eq $Property -or
        $null -eq $Property.Value -or
        [string]::IsNullOrWhiteSpace(
            [string]$Property.Value
        )
    ) {
        throw (
            "Required JSON property '$PropertyName' missing " +
            "from $Context"
        )
    }

    return [string]$Property.Value
}

###########################################################################
# Validate repository state
###########################################################################

$RequiredRepositoryFiles = @(
    "src\scalebridge\data\thermal_modeling\__init__.py",
    "src\scalebridge\data\thermal_modeling\constants.py",
    "src\scalebridge\data\thermal_modeling\identities.py",
    "src\scalebridge\data\thermal_modeling\signals.py",
    "src\scalebridge\data\thermal_modeling\models.py",
    "src\scalebridge\data\thermal_modeling\manifests.py",
    "src\scalebridge\data\thermal_modeling\source_refs.py",
    "src\scalebridge\data\thermal_modeling\discovery.py",
    "src\scalebridge\data\thermal_modeling\alignment.py",

    "scripts\thermal_modeling\discover_phase_d_sources.py",
    "scripts\thermal_modeling\validate_phase_d_alignment.py",

    "tests\thermal_modeling\test_identities.py",
    "tests\thermal_modeling\test_signals.py",
    "tests\thermal_modeling\test_manifests.py",
    "tests\thermal_modeling\test_discovery.py",
    "tests\thermal_modeling\test_alignment.py"
)

foreach ($RelativePath in $RequiredRepositoryFiles) {
    $FullPath = Join-Path $RepoRoot $RelativePath

    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "Required repository file missing: $FullPath"
    }
}

foreach ($Zone in $Zones) {
    $D2Path = Join-Path `
        $D2ValidationRoot `
        "${Zone}_discovery.json"

    if (-not (Test-Path -LiteralPath $D2Path -PathType Leaf)) {
        throw "D2 discovery JSON missing: $D2Path"
    }

    $D3Path = Join-Path `
        $D3ValidationRoot `
        "${Zone}_alignment.json"

    if (-not (Test-Path -LiteralPath $D3Path -PathType Leaf)) {
        throw "D3 alignment JSON missing: $D3Path"
    }
}

###########################################################################
# Clean output
###########################################################################

New-Item `
    -ItemType Directory `
    -Force `
    -Path $OutputRoot |
    Out-Null

if (Test-Path -LiteralPath $BundleRoot) {
    Remove-Item `
        -LiteralPath $BundleRoot `
        -Recurse `
        -Force
}

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item `
        -LiteralPath $ZipPath `
        -Force
}

New-Item `
    -ItemType Directory `
    -Force `
    -Path $BundleRoot |
    Out-Null

###########################################################################
# Collection state
###########################################################################

$CollectedCount = 0
$OptionalMissingCount = 0

###########################################################################
# 1. Current D1–D3 implementation and tests
###########################################################################

foreach ($RelativePath in $RequiredRepositoryFiles) {
    $Copied = Copy-BundleFile `
        -SourcePath (
            Join-Path $RepoRoot $RelativePath
        ) `
        -DestinationRelativePath (
            Join-Path "repository" $RelativePath
        ) `
        -Required $true

    if ($Copied) {
        $CollectedCount++
    }
}

###########################################################################
# 2. Permanent PowerShell run scripts currently present
###########################################################################

$RunScripts = @(
    "scripts\thermal_modeling\D2_RUN_CONTROLLED_DISCOVERY.ps1",
    "scripts\thermal_modeling\D3_RUN_CONTROLLED_ALIGNMENT.ps1"
)

foreach ($RelativePath in $RunScripts) {
    $SourcePath = Join-Path $RepoRoot $RelativePath

    if (Test-Path -LiteralPath $SourcePath -PathType Leaf) {
        $Copied = Copy-BundleFile `
            -SourcePath $SourcePath `
            -DestinationRelativePath (
                Join-Path "repository" $RelativePath
            ) `
            -Required $false

        if ($Copied) {
            $CollectedCount++
        }
    }
    else {
        $OptionalMissingCount++
    }
}

# Also collect the older root-level files if they have not yet been moved.
$LegacyRunScripts = @(
    "D2_RUN_CONTROLLED_DISCOVERY.ps1",
    "D3_RUN_CONTROLLED_ALIGNMENT.ps1"
)

foreach ($RelativePath in $LegacyRunScripts) {
    $SourcePath = Join-Path $RepoRoot $RelativePath

    if (Test-Path -LiteralPath $SourcePath -PathType Leaf) {
        $Copied = Copy-BundleFile `
            -SourcePath $SourcePath `
            -DestinationRelativePath (
                Join-Path "legacy_root_run_scripts" $RelativePath
            ) `
            -Required $false

        if ($Copied) {
            $CollectedCount++
        }
    }
}

###########################################################################
# 3. D2 and D3 validation outputs
###########################################################################

foreach ($Zone in $Zones) {
    $D2Path = Join-Path `
        $D2ValidationRoot `
        "${Zone}_discovery.json"

    $D3Path = Join-Path `
        $D3ValidationRoot `
        "${Zone}_alignment.json"

    if (
        Copy-BundleFile `
            -SourcePath $D2Path `
            -DestinationRelativePath (
                "validation\d2\${Zone}_discovery.json"
            ) `
            -Required $true
    ) {
        $CollectedCount++
    }

    if (
        Copy-BundleFile `
            -SourcePath $D3Path `
            -DestinationRelativePath (
                "validation\d3\${Zone}_alignment.json"
            ) `
            -Required $true
    ) {
        $CollectedCount++
    }
}

###########################################################################
# 4. Resolve and collect exact Phase C metadata for each zone
###########################################################################

foreach ($Zone in $Zones) {
    $DiscoveryPath = Join-Path `
        $D2ValidationRoot `
        "${Zone}_discovery.json"

    $Discovery = Get-Content `
        -LiteralPath $DiscoveryPath `
        -Raw |
    ConvertFrom-Json

    $PhaseCZone = $Discovery.phase_c_zone

    if ($null -eq $PhaseCZone) {
        throw "phase_c_zone missing from: $DiscoveryPath"
    }

    $ZoneDestinationRoot = Join-Path `
        "upstream_metadata" `
        $Zone

    $RequiredPhaseCPaths = [ordered]@{
        "applicable_models.csv" = Get-RequiredJsonValue `
            -Object $PhaseCZone `
            -PropertyName "applicable_models_path" `
            -Context $DiscoveryPath

        "unavailable_models.csv" = Get-RequiredJsonValue `
            -Object $PhaseCZone `
            -PropertyName "unavailable_models_path" `
            -Context $DiscoveryPath

        "heat_input_signal_catalog.csv" = Get-RequiredJsonValue `
            -Object $PhaseCZone `
            -PropertyName "signal_catalog_path" `
            -Context $DiscoveryPath

        "annual_component_predictions_preview.csv" = Get-RequiredJsonValue `
            -Object $PhaseCZone `
            -PropertyName "predictions_preview_path" `
            -Context $DiscoveryPath
    }

    foreach ($OutputName in $RequiredPhaseCPaths.Keys) {
        $SourcePath = $RequiredPhaseCPaths[$OutputName]

        if (
            Copy-BundleFile `
                -SourcePath $SourcePath `
                -DestinationRelativePath (
                    Join-Path $ZoneDestinationRoot $OutputName
                ) `
                -Required $true
        ) {
            $CollectedCount++
        }
    }

    $OptionalPathMappings = [ordered]@{
        "component_prediction_summary.csv" =
            $PhaseCZone.component_prediction_summary_path

        "timestamp_component_availability.csv" =
            $PhaseCZone.timestamp_component_availability_path

        "split_assignments_preview.csv" =
            $PhaseCZone.split_assignments_preview_path
    }

    foreach ($OutputName in $OptionalPathMappings.Keys) {
        $SourcePath = [string]$OptionalPathMappings[$OutputName]

        if ([string]::IsNullOrWhiteSpace($SourcePath)) {
            $OptionalMissingCount++
            continue
        }

        $Copied = Copy-BundleFile `
            -SourcePath $SourcePath `
            -DestinationRelativePath (
                Join-Path $ZoneDestinationRoot $OutputName
            ) `
            -Required $false

        if ($Copied) {
            $CollectedCount++
        }
        else {
            $OptionalMissingCount++
        }
    }

    #######################################################################
    # 5. Collect small Phase B preview and metadata files
    #######################################################################

    $AggregationZone = $Discovery.aggregation_zone

    if ($null -eq $AggregationZone) {
        throw "aggregation_zone missing from: $DiscoveryPath"
    }

    $PhaseBOptionalMappings = [ordered]@{
        "phase_b_wide_preview.csv" =
            $AggregationZone.wide_preview_path

        "phase_b_zone_mapping.csv" =
            $AggregationZone.zone_mapping_path

        "phase_b_static_equipment.csv" =
            $AggregationZone.static_equipment_path
    }

    foreach ($OutputName in $PhaseBOptionalMappings.Keys) {
        $SourcePath = [string]$PhaseBOptionalMappings[$OutputName]

        if ([string]::IsNullOrWhiteSpace($SourcePath)) {
            $OptionalMissingCount++
            continue
        }

        $Copied = Copy-BundleFile `
            -SourcePath $SourcePath `
            -DestinationRelativePath (
                Join-Path $ZoneDestinationRoot $OutputName
            ) `
            -Required $false

        if ($Copied) {
            $CollectedCount++
        }
        else {
            $OptionalMissingCount++
        }
    }
}

###########################################################################
# 6. Collect duplicate diagnostics from D3
###########################################################################

$DuplicateDiagnosticRoot = Join-Path `
    $D3ValidationRoot `
    "duplicate_diagnostics"

if (Test-Path -LiteralPath $DuplicateDiagnosticRoot -PathType Container) {
    Get-ChildItem `
        -LiteralPath $DuplicateDiagnosticRoot `
        -File |
    ForEach-Object {
        $Copied = Copy-BundleFile `
            -SourcePath $_.FullName `
            -DestinationRelativePath (
                Join-Path `
                    "validation\d3\duplicate_diagnostics" `
                    $_.Name
            ) `
            -Required $false

        if ($Copied) {
            $CollectedCount++
        }
    }
}

###########################################################################
# 7. Generate compact CSV schema summary
###########################################################################

$SchemaSummaryPath = Join-Path `
    $BundleRoot `
    "D4_CSV_Schema_Summary.txt"

@(
    "Phase D D4 CSV Schema Summary"
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    ""
) | Set-Content `
    -LiteralPath $SchemaSummaryPath `
    -Encoding UTF8

$CsvFiles = @(
    Get-ChildItem `
        -LiteralPath (
            Join-Path $BundleRoot "upstream_metadata"
        ) `
        -Recurse `
        -File `
        -Filter "*.csv"
)

foreach ($CsvFile in $CsvFiles) {
    $RelativePath = $CsvFile.FullName.Substring(
        $BundleRoot.Length
    ).TrimStart("\")

    try {
        $Header = Get-Content `
            -LiteralPath $CsvFile.FullName `
            -TotalCount 1

        @(
            "FILE: $RelativePath"
            "HEADER: $Header"
            ""
        ) | Add-Content `
            -LiteralPath $SchemaSummaryPath `
            -Encoding UTF8
    }
    catch {
        @(
            "FILE: $RelativePath"
            "HEADER READ FAILED: $($_.Exception.Message)"
            ""
        ) | Add-Content `
            -LiteralPath $SchemaSummaryPath `
            -Encoding UTF8
    }
}

$CollectedCount++

###########################################################################
# 8. Generate D4 development contract
###########################################################################

$ContractPath = Join-Path `
    $BundleRoot `
    "D4_LOCKED_DEVELOPMENT_CONTRACT.txt"

@'
ScaleBridge Phase D — D4 Locked Development Contract
=====================================================

D4 scope
--------
D4 assembles the canonical Phase D signal table from the D3-aligned data.

Inputs
------
Phase B:
- timestamp
- zone_temperature
- outdoor_temperature

Phase C:
- predicted regression outputs only
- inherited applicability and unavailability metadata
- inherited split assignments

Signal treatment
----------------
- Time-varying applicable signal: retain values.
- Constant nonzero applicable signal: retain values.
- Complete-zero signal: canonical column becomes nullable.
- Phase C model not applicable: canonical column becomes nullable.
- Applicable signal with unexpected missing timestamps: validation failure.
- Null does not mean zero.

Canonical heat-input signals
----------------------------
- qsol1
- qsol2

- qzic_p
- qzic_l
- qzic_ee
- qzic_ge
- qzic_oe
- qzic_hwe
- qzic_se

- qzir_p
- qzir_l
- qzir_ee
- qzir_ge
- qzir_oe
- qzir_hwe
- qzir_se

- qzivr_l
- qac

Auxiliary/provenance
--------------------
- phvac
- phvac_oracle, if retained from Phase C

Grouped signals
---------------
zic:
- sum of retained applicable qzic_* components

zir:
- sum of retained applicable qzir_* components
- plus qzivr_l when include_visible_lighting_in_zir=true

Default:
- include_visible_lighting_in_zir=true

Grouping safety
---------------
- Do not blindly use skipna.
- Include only components explicitly marked active.
- Nullable complete-zero and not-applicable components do not contribute.
- Missing values in an active signal are a validation failure.

Timestamp behavior
------------------
- Use the D3 canonical Phase D timeline.
- Source years remain unchanged in Phase B and Phase C artifacts.
- Phase D calendar year is configurable, default 2001.
- Current controlled annual shape is non-leap.
- All aligned controlled zones contain 105,120 rows.

D4 output
---------
D4 should produce an in-memory canonical assembled table and serializable
diagnostics/manifests.

D4 should not yet:
- write final production Phase D Parquet datasets;
- create Independent/Dependent1/Dependent2 products;
- construct ML/SciML lagged tensors;
- construct Opt/Bayes seasonal profiles;
- train any thermal model.
'@ | Set-Content `
    -LiteralPath $ContractPath `
    -Encoding UTF8

$CollectedCount++

###########################################################################
# 9. Generate inventory
###########################################################################

$InventoryPath = Join-Path `
    $BundleRoot `
    "D4_Bundle_File_Inventory.csv"

Get-ChildItem `
    -LiteralPath $BundleRoot `
    -Recurse `
    -File |
ForEach-Object {
    [PSCustomObject]@{
        RelativePath = $_.FullName.Substring(
            $BundleRoot.Length
        ).TrimStart("\")

        SizeBytes = $_.Length

        ModifiedAt = $_.LastWriteTime.ToString("o")
    }
} |
Export-Csv `
    -LiteralPath $InventoryPath `
    -NoTypeInformation `
    -Encoding UTF8

$CollectedCount++

###########################################################################
# 10. Generate collection report
###########################################################################

$ReportPath = Join-Path `
    $BundleRoot `
    "D4_Collection_Report.txt"

$FinalFiles = @(
    Get-ChildItem `
        -LiteralPath $BundleRoot `
        -Recurse `
        -File
)

@(
    "Phase D D4 Canonical Signal Assembly Development Bundle"
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
    ""
    "Repository root: $RepoRoot"
    "Bundle root: $BundleRoot"
    ""
    "Zones:"
    "  RestaurantFastFood_All"
    "  Dining"
    "  Kitchen"
    ""
    "Collected files: $($FinalFiles.Count)"
    "Optional missing files: $OptionalMissingCount"
    ""
    "Included:"
    "  - current D1-D3 implementation and tests"
    "  - D2 discovery results"
    "  - D3 alignment diagnostics"
    "  - applicable and unavailable Phase C model catalogs"
    "  - Phase C signal catalogs"
    "  - annual prediction previews"
    "  - component summaries and availability metadata"
    "  - split previews"
    "  - Phase B temperature previews and zone metadata"
    "  - locked D4 development contract"
    ""
    "Excluded:"
    "  - full annual Phase B Parquet files"
    "  - full annual Phase C prediction Parquet files"
    "  - full split assignment Parquet files"
    "  - model checkpoints"
    "  - MLflow artifact stores"
) | Set-Content `
    -LiteralPath $ReportPath `
    -Encoding UTF8

###########################################################################
# 11. Create ZIP
###########################################################################

Compress-Archive `
    -LiteralPath $BundleRoot `
    -DestinationPath $ZipPath `
    -CompressionLevel Optimal `
    -Force

if (-not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) {
    throw "D4 ZIP was not created: $ZipPath"
}

$ZipInfo = Get-Item -LiteralPath $ZipPath

###########################################################################
# Final output
###########################################################################

Write-Host ""
Write-Host "============================================================" `
    -ForegroundColor Cyan

Write-Host "Phase D D4 development bundle created" `
    -ForegroundColor Green

Write-Host "============================================================" `
    -ForegroundColor Cyan

Write-Host ""
Write-Host "Files collected:"
Write-Host "  $($FinalFiles.Count)"

Write-Host ""
Write-Host "Optional missing files:"
Write-Host "  $OptionalMissingCount"

Write-Host ""
Write-Host "ZIP file:"
Write-Host "  $ZipPath"

Write-Host ""
Write-Host "ZIP size:"
Write-Host "  $([math]::Round($ZipInfo.Length / 1MB, 2)) MB"

Write-Host ""
Write-Host "Upload this ZIP for D4 development." `
    -ForegroundColor Green