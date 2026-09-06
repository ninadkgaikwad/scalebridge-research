$ErrorActionPreference = "Stop"

Write-Host "Running Phase D D8.5 focused regression tests..." -ForegroundColor Cyan

pytest `
    tests\thermal_modeling\test_discovery.py `
    tests\thermal_modeling\test_d6_silo_contracts.py `
    tests\thermal_modeling\test_d7_builders.py `
    -v `
    -W error::pandas.errors.PerformanceWarning

if ($LASTEXITCODE -ne 0) {
    throw "D8.5 focused regression tests failed."
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
Write-Host "PHASE D D8.5 CORRIDOR FIX VALIDATED" -ForegroundColor Green
