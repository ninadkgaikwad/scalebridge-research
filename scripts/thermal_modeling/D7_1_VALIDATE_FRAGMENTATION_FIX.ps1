$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "Running D7.1 focused regression tests..."
pytest `
    tests\thermal_modeling\test_d7_builders.py `
    -v `
    -W error::pandas.errors.PerformanceWarning

if ($LASTEXITCODE -ne 0) {
    throw "D7.1 focused tests failed."
}

Write-Host ""
Write-Host "Running full thermal-modeling suite with PerformanceWarning as error..."
pytest `
    tests\thermal_modeling `
    -v `
    -W error::pandas.errors.PerformanceWarning

if ($LASTEXITCODE -ne 0) {
    throw "D7.1 full thermal-modeling suite failed."
}

Write-Host ""
Write-Host "D7.1 fragmentation fix validation passed." -ForegroundColor Green
Write-Host "Next run:"
Write-Host '& ".\scripts\thermal_modeling\D7_RUN_CONTROLLED_BUILDERS.ps1"'
