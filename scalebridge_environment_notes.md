# ScaleBridge Environment Notes

Date: 2026-05-31  
Project: ScaleBridge Research  

## Environment Files

Two environment files are provided:

- `scalebridge.yaml`
- `scalebridge_cpu.yaml`

## GPU Environment

Use this when the machine has a compatible NVIDIA GPU and CUDA support:

```powershell
conda env create -f scalebridge.yaml
conda activate scalebridge-dev
python check_scalebridge_environment.py
```

If the environment already exists:

```powershell
conda activate scalebridge-dev
conda env update -f scalebridge.yaml --prune
python check_scalebridge_environment.py
```

## CPU Environment

Use this fallback when GPU package resolution fails:

```powershell
conda env create -f scalebridge_cpu.yaml
conda activate scalebridge-dev-cpu
python check_scalebridge_environment.py
```

If the environment already exists:

```powershell
conda activate scalebridge-dev-cpu
conda env update -f scalebridge_cpu.yaml --prune
python check_scalebridge_environment.py
```

## Editable Package Install

From the repository root:

```powershell
pip install -e .
python -c "import scalebridge; print(scalebridge.__version__ if hasattr(scalebridge, '__version__') else 'scalebridge import OK')"
```

## Expected Smoke-Test Imports

The smoke test checks:

- numpy
- pandas
- scipy
- matplotlib
- sklearn
- torch
- neuromancer
- casadi
- pyomo
- cvxpy
- stable_baselines3
- gymnasium
- mlflow
- optuna
- ray
- plotly
- dash
- opyplus
- eppy
- opendssdirect
- dss
- scalebridge

## Machine Status

| Machine | Status |
|---|---|
| dev-laptop | pending full environment test |
| home-pc | not tested |
| lab-pc | not tested |
| wsu-hpc | not tested |
