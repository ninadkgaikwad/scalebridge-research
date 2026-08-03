param(
    [string]$FeatureRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge\campaigns\p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3\heat_input_regression\feature_runs\heat_input_features_c2fix_20260719_142319",

    [string]$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $FeatureRoot)) {
    throw "Feature root does not exist: $FeatureRoot"
}

if (-not (Test-Path $RepoRoot)) {
    throw "Repository root does not exist: $RepoRoot"
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReportPath = Join-Path $RepoRoot "C2_Standard_Validation_Diagnostics_$Timestamp.txt"

$Lines = New-Object System.Collections.Generic.List[string]

function Add-Header {
    param([string]$Text)
    $Lines.Add("")
    $Lines.Add(("=" * 100))
    $Lines.Add($Text)
    $Lines.Add(("=" * 100))
    $Lines.Add("")
}

function Add-Text {
    param([object]$Value)
    if ($null -eq $Value) {
        return
    }

    $Text = $Value | Out-String
    foreach ($Line in ($Text -split "`r?`n")) {
        $Lines.Add($Line)
    }
}

Add-Header "SCALEBRIDGE C2 STANDARD VALIDATION DIAGNOSTIC REPORT"
$Lines.Add("GeneratedAt: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')")
$Lines.Add("FeatureRoot: $FeatureRoot")
$Lines.Add("RepositoryRoot: $RepoRoot")

Add-Header "1. FEATURE ROOT INVENTORY"

$AllFiles = Get-ChildItem -Path $FeatureRoot -Recurse -File |
    Sort-Object FullName

Add-Text (
    $AllFiles |
    Select-Object Length, LastWriteTime, FullName |
    Format-Table -Wrap -AutoSize
)

Add-Header "2. VALIDATION-RELATED FILES"

$ValidationFiles = $AllFiles |
    Where-Object {
        $_.Name -match "valid|diagnostic|result|summary|manifest"
    }

if ($ValidationFiles.Count -eq 0) {
    $Lines.Add("No validation-related files were found.")
}
else {
    Add-Text (
        $ValidationFiles |
        Select-Object Length, LastWriteTime, FullName |
        Format-Table -Wrap -AutoSize
    )
}

Add-Header "3. FAILED CSV ROWS"

$CsvFiles = $AllFiles |
    Where-Object {
        $_.Extension -ieq ".csv"
    }

$AnyFailedCsvRows = $false

foreach ($CsvFile in $CsvFiles) {
    try {
        $Rows = Import-Csv $CsvFile.FullName
    }
    catch {
        $Lines.Add("FAILED TO READ CSV: $($CsvFile.FullName)")
        $Lines.Add($_.Exception.Message)
        continue
    }

    if ($Rows.Count -eq 0) {
        continue
    }

    $CandidateFailures = $Rows |
        Where-Object {
            $StatusValues = @(
                $_.status,
                $_.validation_status,
                $_.check_status,
                $_.passed,
                $_.success,
                $_.is_valid
            ) |
            Where-Object {
                $null -ne $_ -and "$_".Trim() -ne ""
            } |
            ForEach-Object {
                "$_".Trim().ToLowerInvariant()
            }

            $ExplicitFailure = $StatusValues |
                Where-Object {
                    $_ -in @(
                        "failed",
                        "fail",
                        "false",
                        "invalid",
                        "error",
                        "0"
                    )
                }

            $FailedCount = 0
            foreach ($Name in @(
                "failed_check_count",
                "failure_count",
                "failed_count",
                "n_failed",
                "error_count"
            )) {
                if ($_.PSObject.Properties.Name -contains $Name) {
                    $Value = $_.$Name
                    if ($null -ne $Value -and "$Value".Trim() -ne "") {
                        [int]$Parsed = 0
                        if ([int]::TryParse("$Value", [ref]$Parsed)) {
                            $FailedCount += $Parsed
                        }
                    }
                }
            }

            $ExplicitFailure.Count -gt 0 -or $FailedCount -gt 0
        }

    if ($CandidateFailures.Count -gt 0) {
        $AnyFailedCsvRows = $true
        $Lines.Add("")
        $Lines.Add("FILE: $($CsvFile.FullName)")
        $Lines.Add(("-" * 100))
        Add-Text (
            $CandidateFailures |
            Format-List *
        )
    }
}

if (-not $AnyFailedCsvRows) {
    $Lines.Add("No rows with an explicit failed/false/error status were detected in CSV files.")
}

Add-Header "4. ALL SMALL VALIDATION CSV CONTENT"

foreach ($CsvFile in $CsvFiles) {
    try {
        $Rows = Import-Csv $CsvFile.FullName
    }
    catch {
        continue
    }

    if ($Rows.Count -le 100 -and $CsvFile.Name -match "valid|diagnostic|result|summary") {
        $Lines.Add("")
        $Lines.Add("FILE: $($CsvFile.FullName)")
        $Lines.Add(("-" * 100))
        Add-Text ($Rows | Format-List *)
    }
}

Add-Header "5. JSON MANIFESTS AND VALIDATION PAYLOADS"

$JsonFiles = $AllFiles |
    Where-Object {
        $_.Extension -ieq ".json" -and
        $_.Name -match "valid|manifest|summary|diagnostic"
    }

foreach ($JsonFile in $JsonFiles) {
    $Lines.Add("")
    $Lines.Add("FILE: $($JsonFile.FullName)")
    $Lines.Add(("-" * 100))

    try {
        $Raw = Get-Content $JsonFile.FullName -Raw
        $Parsed = $Raw | ConvertFrom-Json
        Add-Text ($Parsed | ConvertTo-Json -Depth 20)
    }
    catch {
        $Lines.Add("FAILED TO PARSE JSON:")
        $Lines.Add($_.Exception.Message)

        try {
            Add-Text (Get-Content $JsonFile.FullName -Raw)
        }
        catch {
            $Lines.Add("Unable to read file.")
        }
    }
}

Add-Header "6. ZONE FEATURE MANIFEST SUMMARY"

$ZoneManifests = $AllFiles |
    Where-Object {
        $_.Name -eq "zone_feature_manifest.json"
    }

foreach ($ManifestFile in $ZoneManifests) {
    $Lines.Add("")
    $Lines.Add("FILE: $($ManifestFile.FullName)")
    $Lines.Add(("-" * 100))

    try {
        $Manifest = Get-Content $ManifestFile.FullName -Raw |
            ConvertFrom-Json

        $Summary = [PSCustomObject]@{
            case_id = $Manifest.case_id
            aggregation_id = $Manifest.aggregation_id
            weight_mode = $Manifest.weight_mode
            aggregate_zone_id = $Manifest.aggregate_zone_id
            row_count = $Manifest.row_count
            timestamp_start = $Manifest.timestamp_start
            timestamp_end = $Manifest.timestamp_end
            duplicate_timestamp_count = $Manifest.duplicate_timestamp_count
            duplicate_parsed_timestamp_count = $Manifest.duplicate_parsed_timestamp_count
            canonical_row_count = $Manifest.canonical_row_count
            dropped_duplicate_row_count = $Manifest.dropped_duplicate_row_count
            unparsed_timestamp_count = $Manifest.unparsed_timestamp_count
            timestamp_monotonic = $Manifest.timestamp_monotonic
            status = $Manifest.status
        }

        Add-Text ($Summary | Format-List *)

        $Lines.Add("")
        $Lines.Add("FULL MANIFEST:")
        Add-Text ($Manifest | ConvertTo-Json -Depth 20)
    }
    catch {
        $Lines.Add("FAILED TO PARSE MANIFEST:")
        $Lines.Add($_.Exception.Message)
    }
}

Add-Header "7. PARQUET FILE INVENTORY"

$ParquetFiles = $AllFiles |
    Where-Object {
        $_.Extension -ieq ".parquet"
    }

Add-Text (
    $ParquetFiles |
    Select-Object Length, LastWriteTime, FullName |
    Format-Table -Wrap -AutoSize
)

Add-Header "8. NEXT INTERPRETATION"

$Lines.Add("The standard validator failed all three zones, while C2 feature construction completed all three zones.")
$Lines.Add("Use Sections 3-6 to distinguish:")
$Lines.Add("1. a real data-quality failure after canonicalization;")
$Lines.Add("2. a validator expectation that still assumes the pre-canonical schema or row count;")
$Lines.Add("3. a manifest-field mismatch introduced by the C2 correction;")
$Lines.Add("4. a required feature column removed or renamed during canonicalization.")

$Lines |
    Set-Content -Path $ReportPath -Encoding UTF8

Write-Host ""
Write-Host "C2 standard-validation diagnostic report written to:"
Write-Host $ReportPath
Write-Host ""

Get-Item $ReportPath |
    Select-Object FullName, Length, LastWriteTime |
    Format-List
