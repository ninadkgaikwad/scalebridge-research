$ErrorActionPreference = "Stop"

$Engine = Join-Path $PSScriptRoot "apply_phase02_step09b_mpc_geometry_vector_compatibility_v12.py"
if (-not (Test-Path -LiteralPath $Engine -PathType Leaf)) {
    throw "Missing sibling Step 9B V12 patch engine: $Engine"
}

python "$Engine"
if ($LASTEXITCODE -ne 0) {
    throw "Step 9B MPC geometry-vector compatibility V12 failed."
}

python -m py_compile "code\Experiments\Exp_Modules\Exp_MPC_RL_Helpers.py"
if ($LASTEXITCODE -ne 0) {
    throw "Compilation failed: Exp_MPC_RL_Helpers.py"
}

python -m py_compile "tests\integration\experiment_compatibility\test_step9b_mpc_geometry_vector_compatibility.py"
if ($LASTEXITCODE -ne 0) {
    throw "Compilation failed: V12 compatibility test"
}

Write-Host ""
Write-Host "Step 9B MPC geometry-vector compatibility V12 applied and compiled."
