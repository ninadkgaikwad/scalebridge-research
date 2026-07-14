# ScaleBridge Research

**Repository:** `scalebridge-research`  
**Python package:** `scalebridge`  
**Project context:** PhD_Code_Framework / ScaleBridge research software stack  
**Primary current focus:** P1 compact EnergyPlus generation completed; next stage is aggregation  
**Current date context:** July 2026  

ScaleBridge is a professional research-software framework for scalable building thermal modeling, EnergyPlus data generation, one-zone commercial building datasets, grey-box and Bayesian estimation, scientific machine learning, PyTorch baselines, MLflow experiment tracking, automated hyperparameter tuning, and later building-grid co-simulation and control experiments.

The repository is being developed to support the PhD paper/dissertation workflow, especially the current P1 paper pipeline:

> Benchmarking black-box, sequence-learning, and scientific machine learning models for scalable one-zone commercial building thermal dynamics across DOE/PNNL ASHRAE 90.1-2013 prototype buildings.

---

## 1. Current Development Snapshot

The current validated milestone is **Stage A: P1 compact EnergyPlus variable-wise generation**.

The validated full compact campaign is:

```text
campaign_id: p1_compact_4b4c_labpc_1w_v1
experiment_name: p1_compact_4b4c_labpc_1w_v1_generation
machine_id: lab-pc
parallel_variable_workers: 1
write_legacy_pickles: true
```

Validated results:

```text
selected_cases: 16
completed_or_launched: 16
skipped: 0
failed: 0
latest_run.json count: 16
rdd_variable_intersection.json count: 16
canonical parquet count: 440
legacy pickle count: 440
traceback count: 0
MLflow export run_count: 16
MLflow merge: successful
```

The latest validated code state includes:

- Professional repository/package organization under `scalebridge-research`
- Python package `src/scalebridge`
- PyTorch-first modeling direction
- MLflow as first-class tracking infrastructure
- EnergyPlus v9.0.1 integration
- `opyplus`-based IDF preparation
- Variable-wise EnergyPlus generation
- Case-specific RDD probing and requested-variable filtering
- Canonical per-variable parquet output
- Optional per-variable legacy pickle output
- Raw EnergyPlus CSV deletion after successful canonical conversion
- Short EnergyPlus work root to avoid Windows path issues
- MLflow duplicate-run fix
- MLflow export and multi-machine merge workflow
- Compact P1 campaign runner
- Pre-opyplus IDF normalization for `ApartmentMidRise`
- Full compact 4-building x 4-climate lab-PC campaign validation

The next major development stage is **Stage B aggregation**, which must consume the validated variable-wise outputs and must not assume every case has all 35 requested variables.

---

## 2. Repository Purpose

ScaleBridge is intended to replace scattered legacy scripts with a modular, reproducible, paper-oriented research software package.

The repository supports:

- EnergyPlus data generation
- DOE/PNNL prototype building workflows
- One-zone building-level aggregation
- Canonical dataset generation
- PyTorch black-box baselines
- RNN/LSTM/GRU sequence models
- Scientific ML models
- Neural ODE / learned ODE / PINN / hybrid models
- Grey-box RC model estimation
- Bayesian estimation workflows
- MLflow tracking and result aggregation
- Optuna/Ray Tune hyperparameter tuning
- Future OpenDSS co-simulation
- Future grid-interactive control, MPC, and MARL experiments

---

## 3. Four-Machine Workflow

The project is designed to work across four machines.

| Machine | Role |
|---|---|
| `laptop` | Primary development, code editing, small validation, planning |
| `home-pc` | Windows GPU compute and medium smoke tests |
| `lab-pc` | Main Windows compute target for compact and future full campaign generation |
| `kamiak` | WSU HPC / SLURM / high-end GPU compute |

The intended workflow is:

1. Develop and commit code primarily on the laptop.
2. Pull the repo on home PC, lab PC, and Kamiak.
3. Keep generated data outside the repo.
4. Keep conda environments machine-local.
5. Use tracked scripts, knowledgebase files, and README instructions to reproduce validation.
6. Use MLflow to track machine/run identity.
7. Use lab PC for final compact campaign validation.
8. Use MLflow export/merge to centralize run metadata across machines.

---

## 4. Directory and Data-Root Policy

Generated data should **not** live inside the repo.

Avoid creating or committing:

```text
scalebridge-research/scratch/
scalebridge-research/data/
scalebridge-research/outputs/
scalebridge-research/mlruns/
scalebridge-research/mlartifacts/
```

Use the external ScaleBridge data root:

```text
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/...
```

Current validated lab-PC campaign root:

```text
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/p1_compact_4b4c_labpc_1w_v1
```

The preferred repo-relative contract is:

```text
repo root
  ../../Data
  ../../Data/ScaleBridge
```

Therefore, from repo root:

```text
SCALEBRIDGE_DATA_ROOT = ../../Data
SCALEBRIDGE_GENERATED_DATA_ROOT = ../../Data/ScaleBridge
```

Temporary EnergyPlus execution folders use a short local work root. For the validated lab-PC compact campaign:

```text
SCALEBRIDGE_EPLUS_WORK_ROOT = D:\ScaleBridge_EPlus_Work\p1_compact_4b4c_labpc_1w_v1
TEMP/TMP/TMPDIR = D:\Temp
```

These folders are temporary and can be cleaned when no EnergyPlus jobs are running, but do not remove the validated campaign output under `Data/ScaleBridge/campaigns` unless it has been archived or intentionally discarded.

---

## 5. Environment Variables

### Windows PowerShell setup

Run from repo root.

```powershell
chcp 65001

$repoRoot = (Resolve-Path ".").Path

$env:SCALEBRIDGE_DATA_ROOT = (Resolve-Path (Join-Path $repoRoot "..\..\Data")).Path
$env:SCALEBRIDGE_GENERATED_DATA_ROOT = (Resolve-Path (Join-Path $repoRoot "..\..\Data\ScaleBridge")).Path
$env:SCALEBRIDGE_EPLUS_WORK_ROOT = "D:\ScaleBridge_EPlus_Work\p1_compact_4b4c_labpc_1w_v1"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:SCALEBRIDGE_MACHINE_ID = "lab-pc"
$env:TEMP = "D:\Temp"
$env:TMP = "D:\Temp"
$env:TMPDIR = "D:\Temp"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

For laptop:

```powershell
$env:SCALEBRIDGE_MACHINE_ID = "laptop"
```

For home PC:

```powershell
$env:SCALEBRIDGE_MACHINE_ID = "home-pc"
```

### Kamiak / Linux setup

Run from repo root if the same relative layout exists:

```bash
export SCALEBRIDGE_DATA_ROOT="$(realpath ../../Data)"
export SCALEBRIDGE_GENERATED_DATA_ROOT="$(realpath ../../Data/ScaleBridge)"
export SCALEBRIDGE_EPLUS_WORK_ROOT="${TMPDIR:-/tmp/$USER}/ScaleBridge_EPlus_Work"
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export SCALEBRIDGE_MACHINE_ID="kamiak"
```

If the relative layout does not exist on Kamiak, set the data roots explicitly to the project storage paths.

### Variable meaning

| Variable | Meaning |
|---|---|
| `SCALEBRIDGE_DATA_ROOT` | General project data root, normally `../../Data` from repo root |
| `SCALEBRIDGE_GENERATED_DATA_ROOT` | Generated ScaleBridge artifact root, normally `../../Data/ScaleBridge` |
| `SCALEBRIDGE_EPLUS_WORK_ROOT` | Short temporary EnergyPlus work directory |
| `MLFLOW_TRACKING_URI` | MLflow tracking URI |
| `SCALEBRIDGE_MACHINE_ID` | Machine label: `laptop`, `home-pc`, `lab-pc`, or `kamiak` |

---

## 6. Current Environment and Tooling

Current validated local assumptions:

```text
Python: 3.10
EnergyPlus: 9.0.1
opyplus: 2.0.7
Primary conda env: scalebridge-dev-gpu
MLflow tracking URI: http://127.0.0.1:5000
```

Lab-PC MLflow validated setup:

```text
Backend store:
  D:\ScaleBridge_MLflow\backend\mlflow_labpc.sqlite

Artifact root:
  <SCALEBRIDGE_GENERATED_DATA_ROOT>\mlflow_artifacts

Export root:
  <SCALEBRIDGE_GENERATED_DATA_ROOT>\mlflow_exports\lab-pc

Merged registry:
  <SCALEBRIDGE_GENERATED_DATA_ROOT>\experiment_registry
```

Week 0 environment direction:

- PyTorch-first
- GPU-capable where possible
- Neuromancer support
- MLflow support
- Optuna/Ray Tune support
- Stable-Baselines3/Gymnasium support
- opyplus/EnergyPlus support
- OpenDSSDirect support
- Plotly/Dash support
- Scientific Python stack

Kamiak H100 GPU environment was validated previously with PyTorch, Neuromancer, MLflow, Optuna, Ray, Stable-Baselines3, Gymnasium, OpenDSSDirect, opyplus, plotting, GIS, and optimization packages.

---

## 7. Legacy Code Policy

Legacy code is **reference logic only**.

Do not directly copy old TensorFlow/Keras scripts into `src/scalebridge`.

Legacy interpretation:

| Legacy source | Policy |
|---|---|
| `Code/BuildingThermalModeling/` | Ignore as older/bad code unless specifically needed |
| `BuildingModeling_BlackGreyBox_Condensed/` | Ignore as older/bad code unless specifically needed |
| `BasicANN_Journal/` | Reference only for old basic ML pipeline behavior |
| `Code/` | Active conceptual source for current data pipeline and grey-box logic, but rewrite cleanly |

Correct migration process:

```text
legacy script
→ identify purpose
→ identify inputs
→ identify outputs
→ identify paper relevance
→ extract conceptual logic
→ rewrite cleanly inside ScaleBridge
→ test against legacy behavior where useful
→ track new experiments with MLflow
```

Production ML code should be PyTorch-first.

---

## 8. Package Architecture

Target package structure:

```text
src/scalebridge/
├── core/
├── data/
├── integration/
├── db/
├── models/
├── training/
├── tracking/
├── tuning/
├── evaluation/
├── simulators/
└── control/
```

Responsibilities:

| Module | Responsibility |
|---|---|
| `core/` | Config, paths, logging, typing, registries, utilities |
| `data/` | Schemas, loaders, preprocessing, splitting, scaling, validation, manifests |
| `integration/` | EnergyPlus, OpenDSS, weather, external data |
| `db/` | Database layer, ORM models, schemas, repositories |
| `models/` | PyTorch model definitions |
| `training/` | Dataset wrappers, dataloaders, trainers, losses, checkpoints |
| `tracking/` | MLflow and run metadata |
| `tuning/` | Optuna/Ray Tune utilities |
| `evaluation/` | Metrics, diagnostics, plots, tables, reports |
| `simulators/` | Commercial/residential/community/co-simulation environments |
| `control/` | MPC, optimization, RL, distributed control |

Experiment folders are paper-specific and should call reusable package code from `src/scalebridge`.

---

## 9. P1 Paper and Data Pipeline Context

Primary current paper:

```text
P1: Benchmarking black-box and scientific ML models for scalable one-zone commercial building thermal dynamics
```

P1 dataset target:

- DOE/PNNL commercial prototype buildings
- ASHRAE 90.1-2013 vintage
- 16 commercial prototype building types
- 4 climate/weather locations
- 64 building-weather cases in the full campaign
- EnergyPlus simulations
- 5-minute timestep
- One full year
- One-zone equivalent building-level representation
- Default split: 70% train, 15% validation, 15% test

Current compact Stage A subset:

- 4 selected commercial prototype building types
- 4 climate/weather locations
- 16 building-weather cases
- RDD-filtered variable-wise EnergyPlus generation
- 440 canonical per-variable parquet outputs

P1 model families to support:

1. ANN / MLP
2. RNN / LSTM / GRU
3. Learned ODE / Neural ODE
4. PINN
5. Hybrid NeuralODE + PINN
6. Grey-box deterministic estimation
7. Grey-box Bayesian estimation

The data pipeline must be shared across these model families. Model-specific adapters are allowed, but the canonical dataset should not be model-specific.

---

## 10. Canonical P1 Data Pipeline Direction

The intended P1 lifecycle is:

```text
EnergyPlus IDF/EPW inventory
→ case manifest
→ EnergyPlus RDD probe and variable availability manifest
→ EnergyPlus variable-wise simulation outputs
→ raw signal catalog
→ one-zone aggregation
→ canonical one-zone timeseries
→ split manifest
→ scaler artifacts
→ model-specific adapters
→ training/evaluation pipelines
```

Canonical schema concepts:

### `case_manifest`

One row per building-weather case.

Important fields:

```text
case_id
building_type
prototype_vintage
climate_zone
weather_city
idf_path
epw_path
simulation_status
raw_output_path
processed_output_path
timestep_minutes
start_time
end_time
n_timesteps_expected
n_timesteps_actual
notes
```

### `rdd_variable_intersection`

One manifest per case, stored at:

```text
generation/cases/<case_id>/rdd_probe/rdd_variable_intersection.json
```

Important fields:

```text
case_id
rdd_path
requested_variable_count
rdd_available_variable_count
rdd_unavailable_variable_count
available_variables
unavailable_variables
```

### `raw_signal_catalog`

One row per EnergyPlus output variable.

Important fields:

```text
case_id
source_file
raw_column
variable_group
zone_name
unit
include_flag
aggregation_rule
canonical_column
notes
```

### `one_zone_timeseries`

One row per timestamp per case.

Important fields:

```text
case_id
timestamp
step_index
split
T_zone
P_hvac
P_building_total
u_heat
u_cool
T_outdoor
solar_global
occupancy_schedule
lighting_schedule
equipment_schedule
internal_gain
solar_gain
heating_setpoint
cooling_setpoint
```

### `dataset_manifest`

Tracks generated model-ready artifacts.

### `split_manifest`

Tracks train/validation/test split boundaries.

### `scaler_artifacts`

Scalers must be fit on training data only.

---

## 11. EnergyPlus Variable-Wise Generation

The core module is:

```text
src/scalebridge/integration/energyplus/generation/variable_wise.py
```

Purpose:

Instead of running one huge all-variable EnergyPlus simulation and creating enormous raw files, variable-wise generation runs one EnergyPlus simulation per requested `Output:Variable`.

This avoids the previous all-variable output problem where large cases produced enormous files, such as:

```text
eplusout.eso ~19 GB
eplusout.csv ~13 GB
```

High-level variable-wise lifecycle:

```text
For each selected OutputVariableRequest:
    create one-variable CaseSpec
    prepare one-variable IDF
    run EnergyPlus in short work directory
    move eplusout.csv to raw/variable_csv/<variable_id>.csv
    convert CSV to canonical/variables/<variable_id>.parquet
    optionally write legacy/per_variable_pickle/<variable_id>.pickle
    delete raw CSV after successful canonical conversion
    retain eio/err diagnostics

After all variables:
    write variable_manifest.json
    write variable_manifest.csv
    write metadata.json
    write eio_tables.json
    write legacy_manifest.json if enabled
    write run_manifest.json
    write latest_run.json
    finish MLflow run
```

Output layout:

```text
runs/<run_id>/
  inputs/
  raw/
    variable_csv/
    eio/
    err/
  variable_runs/
  canonical/
    variables/
    variable_manifest.json
    variable_manifest.csv
    metadata.json
    eio_tables.json
  legacy/
    per_variable_pickle/
    legacy_manifest.json
  run_manifest.json
```

Parallel variable workers run independent EnergyPlus simulations concurrently. They do not make a single EnergyPlus simulation multithreaded.

The validated compact lab-PC campaign used:

```text
parallel_variable_workers = 1
```

---

## 12. RDD-Filtered Variable Generation

The P1 requested output-variable list is a maximum desired vocabulary, not a guarantee that every building can produce every variable.

Some equipment-related variables are only available when the corresponding equipment type exists in the IDF. For example, RestaurantFastFood can produce 29 of the 35 requested variables, while OfficeSmall, RetailStripmall, and ApartmentMidRise produce 27 of the 35 requested variables.

New reusable modules:

```text
src/scalebridge/integration/energyplus/generation/rdd.py
```

Parses `eplusout.rdd`, normalizes EnergyPlus variable names, and filters requested variables against case-specific RDD availability.

```text
src/scalebridge/integration/energyplus/generation/rdd_probe.py
```

Creates a temporary RDD probe IDF with:

```idf
Output:VariableDictionary,
    Regular;
```

Then runs EnergyPlus once and returns the produced `eplusout.rdd`.

Updated campaign lifecycle:

```text
normalized IDF
→ RDD probe run
→ parse eplusout.rdd
→ effective_variables = requested_variables ∩ rdd_available_variables
→ variable-wise generation only for effective_variables
→ validation against effective_variables count
```

Per-case RDD manifest:

```text
generation/cases/<case_id>/rdd_probe/rdd_variable_intersection.json
```

Validation rule:

```text
produced_variable_count == rdd_available_variable_count
```

not:

```text
produced_variable_count == requested_max_variable_count
```

Validated RDD counts in the compact campaign:

| Building | Available variables per climate | Climate count | Generated files |
|---|---:|---:|---:|
| `RestaurantFastFood` | 29 | 4 | 116 |
| `OfficeSmall` | 27 | 4 | 108 |
| `RetailStripmall` | 27 | 4 | 108 |
| `ApartmentMidRise` | 27 | 4 | 108 |
| **Total** |  |  | **440** |

---

## 13. MLflow Architecture

MLflow is first-class experiment tracking.

Design decision:

```text
Machine identity is metadata/tag, not a top-level artifact folder.
```

Current local Windows tracking URI:

```text
http://127.0.0.1:5000
```

Important MLflow files/modules:

```text
src/scalebridge/tracking/mlflow/generation.py
src/scalebridge/tracking/mlflow/semantic.py
scripts/mlflow/export_mlflow_runs.py
scripts/mlflow/merge_mlflow_exports.py
scripts/mlflow/start_local_tracking_server.ps1
```

MLflow duplicate-run fix:

- The runner creates one `MLflowGenerationTracker`
- The actual generation function owns start/finish
- The runner does not manually start/finish MLflow runs

Validated lab-PC MLflow server pattern:

```powershell
$dataRoot = "F:\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge"

mlflow server `
  --backend-store-uri "sqlite:///D:/ScaleBridge_MLflow/backend/mlflow_labpc.sqlite" `
  --default-artifact-root "file:///$dataRoot/mlflow_artifacts" `
  --host 127.0.0.1 `
  --port 5000
```

MLflow export path:

```text
Data/ScaleBridge/mlflow_exports/<machine_id>/
```

Merged registry path:

```text
Data/ScaleBridge/experiment_registry/
```

Validated export and merge commands:

```powershell
python scripts\mlflow\export_mlflow_runs.py `
  --experiment-name p1_compact_4b4c_labpc_1w_v1_generation

python scripts\mlflow\merge_mlflow_exports.py
```

Validated export result:

```text
experiment_count: 1
run_count: 16
```

---

## 14. Pre-opyplus IDF Normalization

The module is:

```text
src/scalebridge/integration/energyplus/idf/pre_opyplus_normalization.py
```

Current supported patch:

```text
Insert ScheduleTypeLimits, Control Type
when an IDF references Control Type but does not define it.
```

Known affected prototype:

```text
ApartmentMidRise
```

The original DOE/PNNL IDFs are never modified.

Normalized copies are written under:

```text
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/normalization/idfs/<case_id>/normalized.idf
```

No mandatory `__init__.py` update is required because the compact runner imports the normalization module directly:

```python
from scalebridge.integration.energyplus.idf.pre_opyplus_normalization import (
    normalize_idf_before_opyplus,
)
```

---

## 15. P1 Compact Campaign

The compact campaign runner is:

```text
scripts/energyplus/run_p1_compact_campaign.py
```

Validated compact lab-PC campaign ID:

```text
p1_compact_4b4c_labpc_1w_v1
```

Validated MLflow experiment name:

```text
p1_compact_4b4c_labpc_1w_v1_generation
```

Final compact building set:

```text
RestaurantFastFood
OfficeSmall
RetailStripmall
ApartmentMidRise
```

Weather locations:

```text
Buffalo
Seattle
Tampa
Tucson
```

Expected campaign size:

```text
4 buildings × 4 weather locations = 16 cases
```

Building progression:

```text
3 zones  -> RestaurantFastFood -> food service / high internal gains
6 zones  -> OfficeSmall        -> simple office baseline
10 zones -> RetailStripmall    -> retail / schedule and lighting diversity
27 zones -> ApartmentMidRise   -> multifamily residential-like diversity
```

Validated full command:

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --machine-id lab-pc `
  --parallel-variable-workers 1 `
  --write-legacy-pickles `
  --mlflow-experiment-name p1_compact_4b4c_labpc_1w_v1_generation `
  --mlflow-strict 2>&1 | Tee-Object -FilePath "p1_compact_4b4c_labpc_1w_v1_log.txt"
```

Validated summary:

```text
selected_cases: 16
completed_or_launched: 16
skipped: 0
failed: 0
```

---

## 16. Building Selection Evidence

Local ASHRAE 2013 building-complexity audit:

| Building Type | Zones | People | Lights | Electric Equipment | Thermostats |
|---|---:|---:|---:|---:|---:|
| `Warehouse` | 3 | 1 | 3 | 2 | 3 |
| `RestaurantFastFood` | 3 | 2 | 2 | 4 | 2 |
| `RestaurantSitDown` | 3 | 2 | 2 | 4 | 2 |
| `RetailStandalone` | 5 | 5 | 5 | 4 | 5 |
| `OfficeSmall` | 6 | 5 | 5 | 5 | 5 |
| `RetailStripmall` | 10 | 10 | 20 | 10 | 10 |
| `OfficeMedium` | 18 | 15 | 15 | 17 | 15 |
| `HotelLarge` | 22 | 18 | 22 | 22 | 22 |
| `OfficeLarge` | 23 | 16 | 20 | 20 | 20 |
| `SchoolPrimary` | 25 | 23 | 25 | 27 | 25 |
| `ApartmentMidRise` | 27 | 24 | 50 | 26 | 24 |
| `SchoolSecondary` | 46 | 42 | 46 | 50 | 46 |
| `Hospital` | 55 | 55 | 55 | 50 | 55 |
| `HotelSmall` | 67 | 67 | 67 | 68 | 54 |
| `ApartmentHighRise` | 90 | 80 | 169 | 82 | 80 |
| `OutPatientHealthCare` | 118 | 59 | 118 | 119 | 118 |

Seattle one-day smoke-test results:

| Building | Result | Decision |
|---|---|---|
| `RestaurantFastFood` | Completed, 33 warnings, 0 severe, 0 fatal | Keep |
| `OfficeSmall` | Completed, 19 warnings, 0 severe, 0 fatal | Keep |
| `RetailStandalone` | Completed, 19 warnings, 0 severe, 0 fatal | Safe, but less diverse |
| `RetailStripmall` | Completed, 883 warnings, 0 severe, 0 fatal | Keep |
| `ApartmentMidRise` | Completed after `Control Type` patch, 27 warnings, 0 severe, 0 fatal | Keep with normalization |
| `SchoolPrimary` | Completed but 16 severe warmup errors | Avoid first compact campaign |
| `HotelLarge` | Completed but 10 severe errors and very slow | Avoid first compact campaign |

---

## 17. Validation Commands

### Compile all

```powershell
python -m compileall src scripts
```

### Full compact campaign command already validated

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --machine-id lab-pc `
  --parallel-variable-workers 1 `
  --write-legacy-pickles `
  --mlflow-experiment-name p1_compact_4b4c_labpc_1w_v1_generation `
  --mlflow-strict 2>&1 | Tee-Object -FilePath "p1_compact_4b4c_labpc_1w_v1_log.txt"
```

### Final artifact audit

```powershell
$campaignId = "p1_compact_4b4c_labpc_1w_v1"
$campaignRoot = Join-Path $env:SCALEBRIDGE_GENERATED_DATA_ROOT "campaigns\$campaignId"

Write-Host "latest_run.json count:"
(Get-ChildItem $campaignRoot -Recurse -Filter "latest_run.json").Count

Write-Host "RDD intersection manifest count:"
(Get-ChildItem $campaignRoot -Recurse -Filter "rdd_variable_intersection.json").Count

Write-Host "Parquet count:"
(Get-ChildItem $campaignRoot -Recurse -Filter "*.parquet").Count

Write-Host "Legacy pickle count:"
(Get-ChildItem $campaignRoot -Recurse -Filter "*.pickle").Count

Write-Host "Traceback count:"
(
  Get-ChildItem $campaignRoot -Recurse -File |
    Where-Object { $_.Name -match "trace" -and $_.Extension -in ".txt", ".log" }
).Count
```

Expected:

```text
latest_run.json count: 16
rdd_variable_intersection.json count: 16
parquet count: 440
pickle count: 440
traceback count: 0
```

---

## 18. Lab-PC Cleanup Policy

Clean only disposable generated/test artifacts.

Do **not** clean:

- repo source files
- knowledgebase files
- validated scripts
- original DOE/PNNL IDFs
- weather/source data
- environment lock documentation that you intend to keep
- validated campaign output unless it has been archived or intentionally discarded

Validated campaign output to preserve:

```text
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/p1_compact_4b4c_labpc_1w_v1
```

Clean only if confirmed disposable:

- old failed compact campaign test folders
- temporary EnergyPlus work root contents from failed tests
- local MLflow test data that is not part of validated registry

---

## 19. Git Policy

Commit:

```text
source files
scripts
knowledgebase files
README.md
small validation text files if useful
environment summary/lock reports if intentionally preserved
```

Do not commit:

```text
generated campaign outputs
normalized IDF copies
parquet files
pickle files
EnergyPlus raw outputs
MLflow artifacts
temporary work folders
repo-root scratch folders
large environment folders
```

Code files to commit for Stage A RDD-filtered generation:

```text
src/scalebridge/integration/energyplus/generation/rdd.py
src/scalebridge/integration/energyplus/generation/rdd_probe.py
scripts/energyplus/run_p1_compact_campaign.py
```

Suggested commit message:

```text
Add RDD-filtered EnergyPlus compact campaign generation
```

Suggested commit body:

```text
- Add reusable EnergyPlus RDD parser and variable matching utilities
- Add reusable RDD probe runner for case-specific output availability
- Filter requested P1 output variables using eplusout.rdd before variable-wise generation
- Validate generated artifacts against RDD-available variable count instead of maximum requested count
- Preserve per-case rdd_variable_intersection.json under generation/cases/<case_id>/rdd_probe
- Validate full compact 4-building x 4-climate lab-PC campaign with 16/16 cases, 440 parquet files, 440 legacy pickles, and 0 tracebacks
```

Before committing:

```powershell
git status --short
```

Remove repo-root scratch if present:

```powershell
Remove-Item -Recurse -Force scratch
```

---

## 20. Important Current Source Files

| File | Purpose |
|---|---|
| `scripts/energyplus/run_p1_campaign.py` | Main P1 campaign runner with standard/variable-wise modes |
| `scripts/energyplus/run_p1_compact_campaign.py` | Compact 4-building x 4-weather campaign runner with RDD-filtered variable-wise generation |
| `src/scalebridge/integration/energyplus/generation/variable_wise.py` | Variable-wise EnergyPlus generation |
| `src/scalebridge/integration/energyplus/generation/rdd.py` | Parses EnergyPlus `eplusout.rdd` and filters requested variables by case-specific availability |
| `src/scalebridge/integration/energyplus/generation/rdd_probe.py` | Runs a case-level EnergyPlus probe to produce `eplusout.rdd` before variable-wise generation |
| `src/scalebridge/integration/energyplus/generation/orchestrator.py` | Standard generation orchestrator |
| `src/scalebridge/integration/energyplus/idf/pre_opyplus_normalization.py` | Pre-opyplus IDF normalization |
| `src/scalebridge/tracking/mlflow/generation.py` | MLflow generation tracking |
| `src/scalebridge/tracking/mlflow/semantic.py` | Semantic MLflow utilities |
| `scripts/mlflow/export_mlflow_runs.py` | Export machine MLflow runs |
| `scripts/mlflow/merge_mlflow_exports.py` | Merge MLflow exports |
| `knowledgebase/p1_compact_campaign_selection_notes.txt` | Compact campaign rationale |
| `knowledgebase/scalebridge_four_machine_environment_variables_and_validation.txt` | Four-machine environment contract |

---

## 21. New Chat Operating Instructions

A new development chat should follow these rules:

1. Work one task at a time.
2. Do not assume exact current file contents if code has changed.
3. Ask for a focused audit file when uncertain.
4. Avoid full long-running EnergyPlus tests unless explicitly requested.
5. Use `--variable-limit 1` or `--variable-limit 2` for first validations.
6. Use lab PC for compact-campaign validation and generated-output checks.
7. Use `--mlflow-strict` during validation.
8. Do not create repo-root scratch/data/output folders.
9. Do not commit generated artifacts.
10. Treat `p1_compact_4b4c_labpc_1w_v1` as the validated Stage A compact campaign.
11. Stage B aggregation must use RDD manifests and must not assume all 35 variables exist for every case.

---

## 22. Next Development Stage: Stage B Aggregation

Stage A generation is complete for the compact P1 campaign.

Next stage:

```text
Stage B aggregation
```

Stage B objective:

Convert the case-wise variable parquet outputs into aggregated one-zone building-level timeseries suitable for downstream regression/modeling.

Stage B inputs:

```text
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/p1_compact_4b4c_labpc_1w_v1/generation/cases/<case_id>/latest_run.json
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/p1_compact_4b4c_labpc_1w_v1/generation/cases/<case_id>/rdd_probe/rdd_variable_intersection.json
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/p1_compact_4b4c_labpc_1w_v1/generation/cases/<case_id>/runs/<run_id>/canonical/variables/*.parquet
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/p1_compact_4b4c_labpc_1w_v1/generation/cases/<case_id>/runs/<run_id>/canonical/variable_manifest.json
```

Stage B non-negotiable rule:

```text
Use RDD/manifest availability. Never assume all 35 requested variables exist.
```

Recommended first aggregation smoke:

```text
RestaurantFastFood / Buffalo
```

Then run all 16 compact cases after schema and logic are validated.

---

## 23. Current Status Statement

ScaleBridge has completed the P1 compact Stage A EnergyPlus generation milestone. The generation pipeline now includes case-specific RDD probing, requested-variable intersection, variable-wise EnergyPlus generation, strict artifact validation, MLflow tracking, MLflow export/merge, short-path Windows execution, and pre-opyplus normalization for `ApartmentMidRise`.

The validated campaign is:

```text
p1_compact_4b4c_labpc_1w_v1
```

It produced:

```text
16 successful cases
440 canonical parquet variable files
440 legacy pickle variable files
16 RDD intersection manifests
0 tracebacks
```

The next development focus is **Stage B aggregation**.

---

## 24. Useful Handoff Files

The following historical handoffs and reports support this README and broader project context:

```text
Scalebridge-Main-Context-Latest.txt
scalebridge_energyplus_variable_wise_handoff_2026-06-30.txt
scalebridge_week1_day1_data_pipeline_execution_context.txt
scalebridge_week1_master_data_pipeline_context.txt
scalebridge_week0_master_coding_context_2026-05-28.txt
scalebridge_week0_environment_folder_reorg_2026-05-31.txt
scalebridge_week0_environment_source_inspection_2026-05-31.txt
scalebridge_week0_github_setup_verification_2026-05-31.txt
scalebridge_week0_step2_created_missing_files_2026-05-31.txt
scalebridge_week0_step2_create_missing_files_commands_2026-05-31.txt
scalebridge_week0_final_repo_inspection_2026-05-31.txt
scalebridge_structure_week0_final_2026-05-31.txt
scalebridge_next_day_coding_context_2026-05-29.txt
ScaleBridge_Week0_End_of_Week_Report.txt
```
