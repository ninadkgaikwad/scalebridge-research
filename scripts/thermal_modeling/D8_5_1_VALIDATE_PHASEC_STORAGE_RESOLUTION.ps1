$ErrorActionPreference = "Stop"

Write-Host "Running Phase D D8.5.1 discovery regression tests..." -ForegroundColor Cyan

pytest `
    tests\thermal_modeling\test_discovery.py `
    -v `
    -W error::pandas.errors.PerformanceWarning

if ($LASTEXITCODE -ne 0) {
    throw "D8.5.1 discovery regression tests failed."
}

Write-Host ""
Write-Host "Running full Phase D thermal-modeling suite..." -ForegroundColor Cyan

pytest `
    tests\thermal_modeling `
    -v `
    -W error::pandas.errors.PerformanceWarning

if ($LASTEXITCODE -ne 0) {
    throw "Full Phase D thermal-modeling suite failed."
}

Write-Host ""
Write-Host "PHASE D D8.5.1 PHASE C STORAGE RESOLUTION VALIDATED" -ForegroundColor Green
