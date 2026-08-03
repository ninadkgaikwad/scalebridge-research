param(
    [string]$RepoRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\scalebridge-research",

    [string]$CampaignRoot = "C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge\campaigns\p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3",

    [string]$FeatureRunId = "heat_input_features_c2fix_20260719_153925",

    [string]$InferenceRunId = "c8_pytorch_cuda_c2fix_20260719_153925"
)

$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

$FeatureRoot = Join-Path `
    $CampaignRoot `
    "heat_input_regression\feature_runs\$FeatureRunId"

$InferenceRoot = Join-Path `
    $CampaignRoot `
    "heat_input_regression\inference_runs\$InferenceRunId"

$AuditRunId = "c8_residual_gap_audit_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

$OutputRoot = Join-Path `
    $CampaignRoot `
    "heat_input_regression\residual_gap_audits\$AuditRunId"

$ConsoleLog = Join-Path `
    $RepoRoot `
    "$AuditRunId.txt"

if (-not (Test-Path $FeatureRoot)) {
    throw "Feature root does not exist: $FeatureRoot"
}

if (-not (Test-Path $InferenceRoot)) {
    throw "Inference root does not exist: $InferenceRoot"
}

New-Item `
    -Path $OutputRoot `
    -ItemType Directory `
    -Force |
Out-Null

Write-Host ("=" * 100)
Write-Host "SCALEBRIDGE C8 RESIDUAL-GAP AUDIT"
Write-Host ("=" * 100)
Write-Host "FeatureRoot:   $FeatureRoot"
Write-Host "InferenceRoot: $InferenceRoot"
Write-Host "OutputRoot:    $OutputRoot"
Write-Host ""

$StdoutPath = Join-Path $env:TEMP "$AuditRunId.stdout.txt"
$StderrPath = Join-Path $env:TEMP "$AuditRunId.stderr.txt"

$Process = Start-Process `
    -FilePath "python" `
    -ArgumentList @(
        "scripts\heat_input_regression\audit_heat_input_regression_residual_gaps.py",
        "--inference-root", $InferenceRoot,
        "--feature-root", $FeatureRoot,
        "--output-root", $OutputRoot,
        "--neighbor-radius", "2"
    ) `
    -NoNewWindow `
    -Wait `
    -PassThru `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath

$CombinedLines = New-Object System.Collections.Generic.List[string]

if (Test-Path $StdoutPath) {
    Get-Content $StdoutPath | ForEach-Object {
        Write-Host $_
        $CombinedLines.Add($_)
    }
}

if (Test-Path $StderrPath) {
    Get-Content $StderrPath | ForEach-Object {
        Write-Host $_
        $CombinedLines.Add($_)
    }
}

$CombinedLines |
Set-Content `
    -Path $ConsoleLog `
    -Encoding UTF8

Remove-Item $StdoutPath, $StderrPath -Force -ErrorAction SilentlyContinue

if ($Process.ExitCode -ne 0) {
    throw "Residual-gap audit failed with exit code $($Process.ExitCode)"
}

$SummaryPath = Join-Path `
    $RepoRoot `
    "${AuditRunId}_summary.txt"

@(
    "status=completed"
    "audit_run_id=$AuditRunId"
    "feature_run_id=$FeatureRunId"
    "inference_run_id=$InferenceRunId"
    "feature_root=$FeatureRoot"
    "inference_root=$InferenceRoot"
    "output_root=$OutputRoot"
    "console_log=$ConsoleLog"
) |
Set-Content `
    -Path $SummaryPath `
    -Encoding UTF8

Write-Host ""
Write-Host "RESIDUAL-GAP AUDIT COMPLETED"
Write-Host "Output root: $OutputRoot"
Write-Host "Console log: $ConsoleLog"
Write-Host "Summary:     $SummaryPath"
