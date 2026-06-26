# P1 Campaign Commands

Run these commands from the ScaleBridge repository root in PowerShell.

## Prerequisites

Confirm the environment and campaign runner before execution:

```powershell
python -c "import opyplus; print('opyplus:', opyplus.__version__)"
$env:SCALEBRIDGE_GENERATED_DATA_ROOT
Test-Path .\scripts\energyplus\run_p1_campaign.py
```

## Single Machine: All 64 Cases

### Preview all assigned cases

This command does not run EnergyPlus:

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-id home-pc `
    --dry-run
```

### Test one case

Use this before starting the full campaign:

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-id home-pc `
    --case-limit 1 `
    --write-legacy-pickles
```

### Run all 64 cases

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-id home-pc `
    --write-legacy-pickles
```

## Four Machines: 16 Cases Each

Each machine must use a different machine number.

### Machine 1

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-number 1 `
    --machine-id laptop `
    --write-legacy-pickles
```

### Machine 2

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-number 2 `
    --machine-id home-pc `
    --write-legacy-pickles
```

### Machine 3

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-number 3 `
    --machine-id lab-pc `
    --write-legacy-pickles
```

### Machine 4

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-number 4 `
    --machine-id kamiak `
    --write-legacy-pickles
```

## Four-Machine Dry Run

Add `--dry-run` to any machine command to inspect its 16 cases without
running EnergyPlus:

```powershell
python .\scripts\energyplus\run_p1_campaign.py `
    --machine-number 1 `
    --machine-id laptop `
    --dry-run
```

## Restart And Retry Behavior

The runner skips a case when its latest run completed successfully. Re-run
the same command after an interruption to continue the campaign.

To intentionally execute completed cases again, add:

```powershell
--rerun-completed
```

MLflow tracking is optional. The campaign runs normally when
`MLFLOW_TRACKING_URI` is not configured.
