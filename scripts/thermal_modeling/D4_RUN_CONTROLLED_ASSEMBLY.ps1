$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path ".").Path
$CampaignRoot = (
    Resolve-Path (
        Join-Path $RepoRoot `
        "..\..\Data\ScaleBridge\campaigns\p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
    )
).Path

$OutputRoot = Join-Path $RepoRoot "_phase_d_d4_validation"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

###########################################################################
# D4.2 storage policy
#
# Production-safe defaults retain only compact JSON manifests. Full annual
# D4 tables and previews are temporary controlled-debug artifacts and require
# explicit opt-in here. All stored time-series artifacts remain Parquet-only.
###########################################################################

$ValidationLevel = "standard"
$WriteAssembledTable = $false
$WritePreview = $false
$PreviewRows = 100
$CleanPreviousZoneArtifacts = $true

$Common = @(
    "--campaign-root", $CampaignRoot,
    "--matrix-run-id", "aggregation_matrix_20260715_114242",
    "--phase-c-campaign-run-id", "phase_c_full_updated_test_laptop_20260802_172455",
    "--phase-d-calendar-year", "2001",
    "--validation-level", $ValidationLevel,
    "--parquet-compression", "zstd"
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

foreach ($Case in $Cases) {
    $Zone = $Case.zone
    $Manifest = Join-Path $OutputRoot "${Zone}_assembly.json"

    if ($CleanPreviousZoneArtifacts) {
        $PreviousArtifacts = @(
            (Join-Path $OutputRoot "${Zone}_assembly.json"),
            (Join-Path $OutputRoot "${Zone}_assembly.parquet"),
            (Join-Path $OutputRoot "${Zone}_assembly_preview.parquet"),
            (Join-Path $OutputRoot "${Zone}_assembly_preview.csv")
        )

        foreach ($Artifact in $PreviousArtifacts) {
            if (Test-Path -LiteralPath $Artifact -PathType Leaf) {
                Remove-Item -LiteralPath $Artifact -Force
            }
        }
    }

    $Arguments = @(
        "scripts\thermal_modeling\validate_phase_d_assembly.py"
    ) + $Common + @(
        "--aggregation-run-id", $Case.aggregation_run_id,
        "--aggregate-zone-id", $Zone,
        "--output-manifest-json", $Manifest
    )

    if ($WriteAssembledTable) {
        $Table = Join-Path $OutputRoot "${Zone}_assembly.parquet"
        $Arguments += @("--output-table-parquet", $Table)
    }

    if ($WritePreview) {
        $Preview = Join-Path $OutputRoot "${Zone}_assembly_preview.parquet"
        $Arguments += @(
            "--output-preview-parquet", $Preview,
            "--preview-rows", "$PreviewRows"
        )
    }

    # A separate Python process per zone ensures all zone data is returned to
    # the operating system before the next zone begins.
    & python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "D4 assembly failed for $Zone"
    }
}

Write-Host ""
Write-Host "D4.2 controlled assembly completed." -ForegroundColor Green
Write-Host "Manifests: $OutputRoot"
Write-Host "Full assembled Parquets written: $WriteAssembledTable"
Write-Host "Preview Parquets written: $WritePreview"
