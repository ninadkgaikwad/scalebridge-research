$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path ".").Path
$OutputRoot = Join-Path $RepoRoot "_phase_d_d6_validation"

Remove-Item `
    -LiteralPath $OutputRoot `
    -Recurse `
    -Force `
    -ErrorAction SilentlyContinue

New-Item `
    -ItemType Directory `
    -Force `
    -Path $OutputRoot |
    Out-Null

python ".\scripts\thermal_modeling\validate_phase_d_silo_contracts.py" `
    --output-root $OutputRoot

if ($LASTEXITCODE -ne 0) {
    throw "D6 representative contract validation failed."
}

$Forbidden = @(
    Get-ChildItem `
        -LiteralPath $OutputRoot `
        -Recurse `
        -File |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in @(
            ".parquet",
            ".csv",
            ".pkl",
            ".pickle"
        )
    }
)

if (@($Forbidden).Count -gt 0) {
    throw "D6 storage policy failure: non-metadata output was written."
}

$JsonCount = @(
    Get-ChildItem `
        -LiteralPath $OutputRoot `
        -File `
        -Filter "*.json"
).Count

if ($JsonCount -ne 6) {
    throw "Expected 6 D6 representative contract JSON files; found $JsonCount."
}

Write-Host ""
Write-Host "D6 silo contract validation completed." -ForegroundColor Green
Write-Host "Representative contracts: 6"
Write-Host "Metadata only: $OutputRoot"
