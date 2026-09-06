# ScaleBridge Research

**Repository:** `scalebridge-research`  
**Python package:** `scalebridge`  
**Project context:** PhD_Code_Framework / ScaleBridge research software stack  
**Primary current focus:** Stage A generation, the complete 240-object Stage B aggregation matrix, Phase C availability-aware heat-input regression, and Phase D canonical thermal-model data assembly are implemented. Phase D D8.2 is fully validated on the controlled testing campaign with all seven temporal policies (92/92 tests and 49/49 real-data policy realizations, zero failures). The next production target is the full P1 Phase D lab-PC campaign over `p1_compact_4b4c_labpc_1w_v1`, using MDH for ML/SciML with lags 1/3/6 and one-step target, and seasonal-distributed Opt/Bayes windows with 30-day seasonal offset, 21 training days, and 7 test days. A D8.3 repeatable-ML-lag CLI extension is required/applied before this P1 launch.
**Current date context:** August 11, 2026  

ScaleBridge is a professional research-software framework for scalable building thermal modeling, EnergyPlus data generation, one-zone commercial building datasets, grey-box and Bayesian estimation, scientific machine learning, PyTorch baselines, MLflow experiment tracking, automated hyperparameter tuning, and later building-grid co-simulation and control experiments.

The repository is being developed to support the PhD paper/dissertation workflow, especially the current P1 paper pipeline:

> Benchmarking black-box, sequence-learning, and scientific machine learning models for scalable one-zone commercial building thermal dynamics across DOE/PNNL ASHRAE 90.1-2013 prototype buildings.

---

## 1. Current Development Snapshot

The current validated milestones are **Stage A: P1 compact EnergyPlus variable-wise generation**, **Stage B: the complete 240-object multi-level aggregation matrix**, and **Phase C: a fully validated C1–C8 QAC/PHVAC smoke campaign with all validators enabled**. C9 MLflow registration is implemented and had passed an earlier smoke registration, but the latest authoritative run intentionally stopped at C8 with MLflow disabled so the computational pipeline could be validated independently.

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

The complete **240-plan Stage B aggregation matrix** on `p1_compact_4b4c_labpc_1w_v1` has been completed and validated. The next campaign execution is unrestricted Phase C heat-input regression on lab-PC, followed by Phase D canonical thermal-model dataset assembly.

---


> **Authoritative status note — July 31, 2026:** Sections describing Stage B as pending are historical development records. The full Stage B matrix completed with **240/240 successful aggregation objects and 0 failures**. The latest authoritative Phase C test is `phase_c_qac_phvac_test_20260727_200800`: C1–C8 completed with **17/17 commands passed**, **33/33 datasets, models, and evaluations passed**, **3/3 full-year inference zones passed**, and all C2/C3/C4/C6/C7/C8 validators passed. C9 MLflow registration remains implemented but was intentionally excluded from this run.


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
| `scripts/aggregation/run_p1_aggregation.py` | Production Stage B aggregation runner |
| `scripts/aggregation/build_p1_aggregation_plan.py` | Build P1 aggregation plans |
| `src/scalebridge/data/aggregation/engine.py` | Production aggregation engine |
| `src/scalebridge/data/aggregation/rules.py` | Cleaned legacy_v1 aggregation rules |
| `src/scalebridge/tracking/mlflow/aggregation.py` | Aggregation MLflow tracking helpers |
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

---

## 22. Stage B Aggregation Status

Stage B aggregation has been implemented and validated through single-case production runs.
The aggregation pipeline consumes the variable-wise Stage A generation outputs and produces one-zone building-level time-series outputs for downstream regression, dataset construction, and modeling.

Validated smoke campaign/case:

```text
campaign_id: p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3
case_id: epcase_827ca4812c0199221d031e59
source_generation_run_id: epvwr_a8695a44ed3f
building: RestaurantFastFood
weather: Buffalo
strategy: all_thermal_zones_to_one
aggregate zone: Aggregated_Zone_1
source zones: DINING + KITCHEN
excluded zone: ATTIC
```

Validated aggregation output shape:

```text
5-minute annual timestamps: 105,120
aggregated time-series variables: 32
long-format rows: 3,363,840
wide preview shape: 100 rows × 33 columns
```

The row count is correct because:

```text
365 days × 24 hours/day × 12 samples/hour = 105,120 timestamps
105,120 timestamps × 32 variables = 3,363,840 long rows
```

Validated weighting modes:

```text
equal
floor_area
volume
```

Validated static equipment outputs for the RestaurantFastFood/Buffalo smoke case:

```text
People_Level              46.9000
Lights_Level            1162.9275
ElectricEquipment_Level 11515.8835
GasEquipment_Level      91932.6010
```

Validated diagnostics are expected and not failures:

```text
DINING missing gas-equipment convective heating key
DINING missing gas-equipment radiant heating key
OtherEquipment schedule not present
HotWaterEquipment schedule not present
SteamEquipment schedule not present
```

---

## 23. Aggregation Modules and Scripts

Reusable aggregation modules:

| File | Purpose |
|---|---|
| `src/scalebridge/data/aggregation/models.py` | Shared dataclasses/enums for generation refs, RDD intersections, aggregation strategies, weight modes, rule sets, and plans |
| `src/scalebridge/data/aggregation/discovery.py` | Campaign root resolution, generation run discovery, optional RDD manifest loading |
| `src/scalebridge/data/aggregation/eio.py` | EIO zone information and schedule/equipment mapping extraction |
| `src/scalebridge/data/aggregation/audit.py` | Audits generation outputs for aggregation readiness |
| `src/scalebridge/data/aggregation/plans.py` | Builds aggregation plans and zone mappings |
| `src/scalebridge/data/aggregation/loaders.py` | Loads canonical variable-wise Parquet outputs |
| `src/scalebridge/data/aggregation/rules.py` | Cleaned `legacy_v1` aggregation rules |
| `src/scalebridge/data/aggregation/writers.py` | JSON/CSV/Parquet/provenance writers |
| `src/scalebridge/data/aggregation/engine.py` | Production aggregation campaign runner |
| `src/scalebridge/tracking/mlflow/aggregation.py` | Aggregation MLflow tracking helpers |

Aggregation scripts:

| Script | Purpose |
|---|---|
| `scripts/aggregation/audit_generation_run_for_aggregation.py` | Audit generation runs before aggregation |
| `scripts/aggregation/build_p1_aggregation_plan.py` | Build P1 aggregation plans |
| `scripts/aggregation/probe_aggregation_variable_loader.py` | Probe canonical variable loading |
| `scripts/aggregation/probe_aggregation_rules.py` | Probe aggregation rules before production writes |
| `scripts/aggregation/run_p1_aggregation.py` | Production aggregation runner |

---

## 24. Aggregation Inputs and Outputs

Primary aggregation inputs:

```text
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/generation/cases/<case_id>/latest_run.json
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/generation/cases/<case_id>/rdd_probe/rdd_variable_intersection.json
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/generation/cases/<case_id>/runs/<run_id>/canonical/variables/*.parquet
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/generation/cases/<case_id>/runs/<run_id>/canonical/variable_manifest.json
<SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/generation/cases/<case_id>/runs/<run_id>/canonical/eio_tables.json
```

Production aggregation output layout:

```text
<campaign_root>/aggregation/cases/<case_id>/runs/<aggregation_run_id>/
  aggregation_manifest.json
  inputs/
    aggregation_plan.json
    zone_mapping.csv
    source_run_manifest.json
    source_generation_run.json
  diagnostics/
    loaded_variables.csv
    rule_summary.csv
    rule_diagnostics.csv
    schedule_equipment_mapping_used.csv
    equipment_contributions.csv
  zones/
    Aggregated_Zone_1/
      aggregated_timeseries_wide.parquet
      aggregated_timeseries_wide_preview.csv
      aggregated_timeseries_long.parquet
      aggregated_timeseries_long_preview.csv
      aggregated_static_equipment.parquet
      aggregated_static_equipment.csv
      equipment_contributions.csv
      equipment_contributions.parquet
      zone_mapping.csv
  legacy/
    Aggregation_Dict_1Zone.pickle
```

Campaign-level output layout:

```text
<campaign_root>/aggregation/campaign_runs/<aggregation_campaign_run_id>/
  aggregation_campaign_manifest.json
  aggregation_case_runs.csv
  aggregation_outputs.csv
  discovery_missing_rows.csv
```

---

## 25. Aggregation Rule Behavior

Aggregation strategies:

| Strategy | Meaning |
|---|---|
| `all_thermal_zones_to_one` | All included thermal zones map to `Aggregated_Zone_1`; default P1 mode |
| `identity` | Each included source zone maps to its own aggregate zone |
| `custom_groups` | Reserved for future user-defined grouping |

Thermal-zone discovery:

```text
Source: canonical/eio_tables.json → Zone Information
Include: Part of Total Building Area == Yes
Exclude: non-building-area zones such as ATTIC
```

Weight modes:

| Mode | Meaning |
|---|---|
| `equal` | Equal averaging across source zones |
| `floor_area` | Floor-area weighted averaging |
| `volume` | Volume weighted averaging |

Rule families:

| Family | Behavior |
|---|---|
| `Site` / `Facility` | Copy signal directly into each aggregate zone |
| `Zone` | Exact source-zone key matching and weighted aggregation |
| `Surface` | Safe source-zone surface matching; prevents excluded-zone leakage such as `ATTIC-FLOOR-KITCHEN` |
| `System` | Token-safe source-zone matching plus `DIRECT AIR INLET NODE` pattern |
| `Schedule` | EIO schedule/equipment mapping with exact-normalized schedule matching |

Schedule aggregation produces equipment-specific schedule columns such as:

```text
Schedule_Value_People
Schedule_Value_Lights
Schedule_Value_ElectricEquipment
Schedule_Value_GasEquipment
```

It also writes `equipment_contributions.csv`, which exposes every contributing EIO equipment object before scalar reduction.

---

## 26. Aggregation Commands

Compile:

```powershell
python -m compileall src scripts
```

Build aggregation plans for a campaign:

```powershell
python scripts\aggregation\build_p1_aggregation_plan.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --weight-mode equal
```

```powershell
python scripts\aggregation\build_p1_aggregation_plan.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --weight-mode floor_area
```

```powershell
python scripts\aggregation\build_p1_aggregation_plan.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --weight-mode volume
```

Run production aggregation with MLflow:

```powershell
python scripts\aggregation\run_p1_aggregation.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --weight-mode equal `
  --continue-on-error `
  --write-legacy-pickle `
  --mlflow `
  --mlflow-experiment-name ScaleBridge_P1_Aggregation_4b4c_1w
```

After equal passes, repeat with:

```text
--weight-mode floor_area
--weight-mode volume
```

Export aggregation MLflow runs on lab-PC:

```powershell
python scripts\mlflow\export_mlflow_runs.py `
  --experiment-name ScaleBridge_P1_Aggregation_4b4c_1w `
  --machine-id lab-pc
```

Merge MLflow exports:

```powershell
python scripts\mlflow\merge_mlflow_exports.py
```

PowerShell path note:

```text
Use $env:SCALEBRIDGE_GENERATED_DATA_ROOT in PowerShell.
Do not use %SCALEBRIDGE_GENERATED_DATA_ROOT%, which is cmd.exe syntax.
```

---

## 27. Aggregation MLflow Tracking

Aggregation tracking helper:

```text
src/scalebridge/tracking/mlflow/aggregation.py
```

Validated test experiment:

```text
ScaleBridge_P1_Aggregation_Test
```

Recommended 4×4 compact aggregation experiment:

```text
ScaleBridge_P1_Aggregation_4b4c_1w
```

Aggregation MLflow logs:

```text
params:
  campaign_id
  strategy
  rule_set
  weight_mode
  case_count
  write_legacy_pickle
  continue_on_error

metrics:
  case_count
  successful_case_count
  failed_case_count
  per-case loaded_variable_count
  per-case aggregate_zone_count
  per-case aggregated_long_rows
  per-case static_equipment_rows
  per-case equipment_contribution_rows
  per-case diagnostic_rows
  per-case runtime_seconds

artifacts:
  aggregation_campaign_summary/
  aggregation_cases/<case_id>/
```

Validated MLflow export/merge after the aggregation smoke test:

```text
laptop aggregation test export:
  experiment_name: ScaleBridge_P1_Aggregation_Test
  machine_id: laptop
  experiment_count: 1
  run_count: 2

merged registry:
  included_export_count: 3
  run_count_raw: 21
  run_count_merged: 21
  machine_ids: home-pc, lab-pc, laptop
```

---

## 28. Immediate Next Step

The next step is campaign-scale Stage B aggregation for:

```text
campaign_id: p1_compact_4b4c_labpc_1w_v1
```

Run on lab-PC:

1. Build equal, floor_area, and volume aggregation plans.
2. Run equal aggregation with MLflow first.
3. Inspect campaign-level aggregation summary files.
4. If all 16 cases pass, run floor_area and volume.
5. Export and merge MLflow aggregation runs.

After equal run, inspect/upload first:

```text
<campaign_root>/aggregation/campaign_runs/<aggregation_campaign_run_id>/aggregation_campaign_manifest.json
<campaign_root>/aggregation/campaign_runs/<aggregation_campaign_run_id>/aggregation_case_runs.csv
<campaign_root>/aggregation/campaign_runs/<aggregation_campaign_run_id>/aggregation_outputs.csv
```

---

## 29. Previous Updated Current Status Statement

ScaleBridge has completed the P1 compact Stage A EnergyPlus generation milestone and has developed/validated the Stage B aggregation pipeline through single-case production runs.

Validated Stage A campaign:

```text
p1_compact_4b4c_labpc_1w_v1
```

Validated Stage A outputs:

```text
16 successful cases
440 canonical parquet variable files
440 legacy pickle variable files
16 RDD intersection manifests
0 tracebacks
```

Validated Stage B smoke results:

```text
29 loaded variables
1 aggregate zone
32 aggregated time-series variables
105,120 timestamps
3,363,840 long rows
4 static equipment rows
9 equipment contribution rows
5 expected diagnostics
equal/floor_area/volume modes validated
MLflow aggregation export/merge validated
```

Current next focus:

```text
Run campaign-wide Stage B aggregation on p1_compact_4b4c_labpc_1w_v1.
```

---

## 30. July 12 Stage B Update: Multi-Level Compact Aggregation

The Stage B aggregation pipeline has now moved beyond single-case all-to-one aggregation. It supports a paper-ready multi-resolution aggregation ladder for the compact P1 campaign:

```text
campaign_id: p1_compact_4b4c_labpc_1w_v1
buildings: 4
weather/climate cases: 16
aggregation levels: 5
weight modes: 3
full matrix size: 240 aggregation plans/runs
```

Final aggregation levels:

| Aggregation ID | Label | Meaning |
|---|---:|---|
| `p1_l01_all_to_one` | L1 | Whole-building response |
| `p1_l02_functional` | L2 | Major functional/use grouping |
| `p1_l03_intermediate` | L3 | Coarse spatial/function grouping |
| `p1_l04_spatial_detailed` | L4 | Fine spatial grouping |
| `p1_l05_identity` | L5 | No aggregation / approved-zone identity |

Weight modes:

```text
equal
floor_area
volume
```

Final aggregation experiment:

```text
ScaleBridge_P1_Aggregation_4b4c_1w
```

---

## 31. Approved Zones and Aggregation Ladder

Approved source-zone counts in the compact campaign:

| Building | Approved source zones |
|---|---:|
| `RestaurantFastFood` | 2 |
| `OfficeSmall` | 5 |
| `RetailStripmall` | 10 |
| `ApartmentMidRise` | 27 |

High-level ladder:

| Level | RestaurantFastFood | OfficeSmall | RetailStripmall | ApartmentMidRise |
|---|---|---|---|---|
| L1 | All zones | All zones | All stores | All zones |
| L2 | Dining; kitchen | Core; perimeter | Left block; right block | Residential; common/non-residential |
| L3 | Identity | Core; two perimeter pairs | Five adjacent store pairs | Office; corridors; floor residential groups |
| L4 | Identity | Identity | Identity | Office, individual corridors, floor-by-row residential groups |
| L5 | Identity | Identity | Identity | Identity |

A Google-Sheets-ready grouping record was created with:

```text
Aggregation_Groups_Detailed
Constituent_Long
Building_Level_Summary
Approved_Source_Zones
```

The detailed grouping table records every aggregate zone and its constituent source zones.

---

## 32. Custom Grouping and Plan Building

The compact 4x4 aggregation ladder is generated by:

```text
scripts/aggregation/build_p1_4b4c_custom_grouping_csv.py
```

It writes:

```text
<campaign_root>/aggregation/custom_grouping_levels/p1_4b4c_custom_groups.csv
<campaign_root>/aggregation/custom_grouping_levels/p1_4b4c_custom_groups_case_summary.csv
<campaign_root>/aggregation/custom_grouping_levels/p1_4b4c_custom_groups_manifest.json
```

The plan builder now supports custom groupings:

```powershell
python scripts\aggregation\build_p1_aggregation_plan.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --strategy custom_groups `
  --rule-set legacy_v1 `
  --weight-mode equal `
  --custom-zone-groups "$env:SCALEBRIDGE_GENERATED_DATA_ROOT\campaigns\p1_compact_4b4c_labpc_1w_v1\aggregation\custom_grouping_levels\p1_4b4c_custom_groups.csv"
```

Repeat with:

```text
--weight-mode floor_area
--weight-mode volume
```

Validated plan-build result per weight mode:

```text
plan_count: 80
zone_mapping_row_count: 880
included_thermal_zone_row_count: 176
excluded_zone_row_count: 8
missing_plan_input_row_count: 0
```

Total formal compact aggregation plans:

```text
80 plans/weight × 3 weights = 240 plans
```

---

## 33. Aggregation Matrix Runner

The preferred runner for the full compact aggregation is:

```text
scripts/aggregation/run_p1_aggregation_matrix.py
```

It selects exact `aggregation_plan.json` paths from `plan_build_*` folders and runs combinations over:

```text
case_id × aggregation_id × weight_mode
```

It writes:

```text
<campaign_root>/aggregation/matrix_runs/<matrix_run_id>/
  selected_aggregation_plans.csv
  aggregation_matrix_case_runs.csv
  aggregation_matrix_outputs.csv
  missing_generation_rows.csv
  aggregation_matrix_manifest.json
```

Full lab-PC matrix command with legacy pickles:

```powershell
python scripts\aggregation\run_p1_aggregation_matrix.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --aggregation-id p1_l01_all_to_one `
  --aggregation-id p1_l02_functional `
  --aggregation-id p1_l03_intermediate `
  --aggregation-id p1_l04_spatial_detailed `
  --aggregation-id p1_l05_identity `
  --weight-mode equal `
  --weight-mode floor_area `
  --weight-mode volume `
  --continue-on-error `
  --write-legacy-pickle `
  --mlflow `
  --mlflow-experiment-name ScaleBridge_P1_Aggregation_4b4c_1w `
  --mlflow-run-name p1_4b4c_all_levels_all_weights_one_parquet_with_pickle_labpc
```

Expected full matrix:

```text
selected_plan_count: 240
successful_plan_count: 240
failed_plan_count: 0
```

---

## 34. Memory-Safe System Node Mass Flow Rate Handling

`System Node Mass Flow Rate` is downstream-important and is not skipped.

New module:

```text
src/scalebridge/data/aggregation/system_node_mass_flow.py
```

Why it exists:

- `System Node Mass Flow Rate` is node-level, not directly zone-level.
- ApartmentMidRise mass-flow files are very large, around 47.3 million rows per climate case.
- Loading it as a normal pandas dataframe after many other variables caused memory pressure.

Memory-safe behavior:

```text
read only required columns
stream parquet in pyarrow batches
map node key_value to source zone using normalized prefix rules
map source zone to aggregate zone using the aggregation plan
sum mass flow by timestamp and aggregate zone
write mapping and unmapped-node diagnostics
```

Currently supported node naming conventions:

```text
<Approved Zone Name> DIRECT AIR INLET NODE NAME
<Approved Zone Name> ZONE EQUIP INLET
```

Examples:

| Node key | Source zone |
|---|---|
| `CORE_ZN DIRECT AIR INLET NODE NAME` | `Core_ZN` |
| `LGSTORE1 DIRECT AIR INLET NODE NAME` | `LGstore1` |
| `DINING DIRECT AIR INLET NODE NAME` | `Dining` |
| `G N1 APARTMENT ZONE EQUIP INLET` | `G N1 Apartment` |

If an approved zone has no mapped mass-flow node, the aggregation still succeeds. The thermal variables remain present and diagnostics record the missing node mapping.

New diagnostics:

```text
diagnostics/system_node_mass_flow_summary.csv
diagnostics/system_node_mass_flow_mapping.csv
diagnostics/system_node_mass_flow_unmapped_nodes.csv
```

---

## 35. One-Parquet-at-a-Time Aggregation Engine

The aggregation engine was updated to reduce memory pressure.

Updated file:

```text
src/scalebridge/data/aggregation/engine.py
```

Old behavior:

```text
load all canonical variable parquet files into memory
then apply aggregation rules once
```

New behavior:

```text
for each variable:
  load one parquet
  aggregate that one variable
  merge compact aggregated output into accumulator
  delete raw dataframe
  garbage collect
  move to next parquet
```

Special handling:

```text
System Node Mass Flow Rate is streamed through system_node_mass_flow.py
instead of loaded as a normal dataframe.
```

The engine should now hold at most:

```text
one normal raw parquet dataframe
+ accumulated aggregated outputs
```

instead of all raw variables at once.

Validated smoke command:

```powershell
python scripts\aggregation\run_p1_aggregation_matrix.py `
  --campaign-id p1_compact_4b4c_labpc_1w_v1 `
  --aggregation-id p1_l01_all_to_one `
  --weight-mode equal `
  --continue-on-error `
  --write-legacy-pickle `
  --mlflow `
  --mlflow-experiment-name ScaleBridge_P1_Aggregation_4b4c_1w `
  --mlflow-run-name p1_4b4c_smoke_l01_equal_one_parquet_labpc
```

Validated smoke result:

```text
selected_plan_count: 16
successful_plan_count: 16
failed_plan_count: 0
```

This validated:

- one-parquet-at-a-time aggregation
- System Node Mass Flow Rate streaming
- all four ApartmentMidRise cases
- all four OfficeSmall cases
- all four RestaurantFastFood cases
- all four RetailStripmall cases
- legacy pickle writing
- MLflow matrix tracking

---

## 36. Storage and Machine Data Policy

Observed L1 equal smoke storage:

```text
approximately 1 GB for 16 cases, one aggregation level, one weight mode, with legacy pickles
```

Full matrix estimate:

| Estimate type | Approximate storage |
|---|---:|
| Run-count lower bound | ~15 GB |
| Aggregate-zone scaled estimate | ~70–80 GB |
| Conservative practical estimate | ~100–120 GB |
| Safe free-space target | ~150 GB |

Machine policy:

| Machine | Data role |
|---|---|
| `lab-pc` | Full generation and aggregation storage; preferred full matrix execution |
| `home-pc` | Also has enough space for full aggregation outputs |
| `laptop` | Code development only; Dropbox data folder should remain online-only |
| `kamiak` | Later ANN/ML training target after curated dataset copy |

Recommended later Kamiak export:

```text
<SCALEBRIDGE_GENERATED_DATA_ROOT>/training_exports/p1_4b4c_1w_aggregation_v1/
```

That export should contain model-training artifacts only, for example:

```text
aggregated_timeseries_wide.parquet
aggregation_manifest.json
zone_mapping.csv
selected provenance/diagnostics
```

Do not copy all previews, diagnostics, and legacy pickles to Kamiak unless required.

---

## 37. Current Status After July 12 Stage B Work

Current validated state:

```text
Stage A compact EnergyPlus generation: complete
Stage B custom grouping ladder: complete
Stage B custom plan building: complete for 240 plans
Stage B matrix runner: implemented
Stage B MLflow matrix logging: implemented
System Node Mass Flow Rate streaming: implemented
one-parquet-at-a-time engine: implemented
L1 equal 16-case smoke: passed
```

Latest validated smoke:

```text
campaign_id: p1_compact_4b4c_labpc_1w_v1
aggregation_id: p1_l01_all_to_one
weight_mode: equal
selected_plan_count: 16
successful_plan_count: 16
failed_plan_count: 0
write_legacy_pickle: true
mlflow: true
```

Current next execution:

```text
Run the full 240-plan compact Stage B aggregation matrix on lab-PC.
```

After full run, export and merge MLflow:

```powershell
python scripts\mlflow\export_mlflow_runs.py `
  --experiment-name ScaleBridge_P1_Aggregation_4b4c_1w `
  --machine-id lab-pc

python scripts\mlflow\merge_mlflow_exports.py
```



---

## 42. July 2026 Four-Machine Environment Rebuild and Validation

This section is the authoritative environment record for the current ScaleBridge development stack. It supersedes older generic references to a single `scalebridge-dev-gpu` environment and supersedes the earlier statement that the final Kamiak validation used an H100. The final validated Kamiak rebuild ran on an **NVIDIA A100-PCIE-40GB** compute node.

### 42.1 Environment architecture

ScaleBridge now uses a two-layer environment strategy on every machine:

```text
Layer 1: Conda-managed scientific/system foundation
Layer 2: pip-managed GPU and project-specific overlay
Layer 3: editable ScaleBridge repository install
```

The design goals are:

- Preserve the old `scalebridge-dev-gpu` environments until the new environments pass.
- Create a new machine-specific environment instead of mutating the historical environment.
- Use Conda for the broad compiled scientific stack and platform-specific native libraries.
- Use pip for the exact PyTorch CUDA wheel set and packages that were unavailable, more reliable, or intentionally controlled through pip.
- Install `scalebridge` in editable mode from the current repository.
- Validate imports, dependency consistency, CUDA visibility, and optimization solvers before accepting an environment.
- Export machine-specific lock files after validation.
- Keep environment folders and large lock artifacts outside Git.

### 42.2 Final environment names and machine roles

| Machine | Final environment | Primary role | Validated GPU |
|---|---|---|---|
| Laptop | `scalebridge-dev-gpu-laptop` | Primary development, planning, small tests | NVIDIA GeForce MX150 |
| Home PC | `scalebridge-dev-gpu-homepc` | Windows GPU compute and medium smoke tests | NVIDIA GeForce GTX 1050 Ti |
| Lab PC | `scalebridge-dev-gpu-labpc` | Main Windows generation/aggregation compute target | NVIDIA RTX A4000 |
| Kamiak | `/home/ninad.gaikwad/conda_envs/scalebridge-dev-gpu-kamiak` | SLURM GPU training and large compute | NVIDIA A100-PCIE-40GB |

Historical environments were not overwritten during development. On Windows, the original `scalebridge-dev-gpu` environment was preserved. On Kamiak, the original environment remained at:

```text
/home/ninad.gaikwad/conda_envs/scalebridge-dev-gpu
```

Temporary Kamiak test environments were removed after they were no longer needed:

```text
/home/ninad.gaikwad/conda_envs/scalebridge-torch-gpu-test-cu118
/home/ninad.gaikwad/conda_envs/scalebridge-torch-gpu-test-pip-cu118
```

### 42.3 Common validated software baseline

The three Windows environments share the following validated core:

```text
Python              3.10.20
NumPy               1.26.3
SciPy               1.15.2
pandas              2.3.3
scikit-learn        1.7.2
matplotlib          3.10.9
seaborn             0.13.2
PyArrow             23.0.1

PyTorch             2.5.1+cu118
Torchvision         0.20.1+cu118
Torchaudio          2.5.1+cu118
CUDA runtime        11.8
Pillow              12.2.0

MLflow              3.13.0
Optuna              4.9.0
CVXPY               1.7.5
Pyomo               6.10.1
CasADi              3.7.2
opyplus             2.0.7
python-slugify      5.0.2
text-unidecode      1.3
Unidecode           1.4.0
ScaleBridge         0.1.0 editable
```

The final Kamiak environment validated:

```text
Python              3.10.12
NumPy               2.2.6
PyTorch             2.5.1+cu118
Torchvision         0.20.1+cu118
Torchaudio          2.5.1+cu118
CUDA runtime        11.8
Pillow              12.2.0
MLflow              3.13.0
Optuna              4.9.0
CVXPY               1.7.5
CasADi              3.7.2
opyplus             2.0.7
Gymnasium           1.2.3
Stable-Baselines3   2.8.0
ScaleBridge         0.1.0 editable
GPU                  NVIDIA A100-PCIE-40GB
```

The Linux NumPy version differs from Windows because the Kamiak environment was reconstructed from a Linux-native explicit lock and portable pip overlay. This difference is accepted because the environment passed its import, dependency, CUDA, CVXPY, and CasADi validation suite.

### 42.4 Windows Conda foundation

The Windows environments were created from the laptop's validated explicit Conda lock:

```text
..\Environments\locks\windows\laptop\conda-explicit-spec.txt
```

Example environment creation:

```powershell
conda create --name scalebridge-dev-gpu-labpc `
  --file "..\Environments\locks\windows\laptop\conda-explicit-spec.txt"
```

or in CMD:

```cmd
conda create --name scalebridge-dev-gpu-homepc --file "..\Environments\locks\windows\laptop\conda-explicit-spec.txt"
```

The Conda layer includes the compiled scientific stack, OpenBLAS, MLflow, Optuna, CVXPY, Pyomo, GIS/plotting dependencies, and related native libraries.

Validated Windows numerical backend:

```text
_openmp_mutex  4.5  20_gnu
libopenblas    0.3.32 pthreads
libblas        3.11.0 openblas
libcblas       3.11.0 openblas
liblapack      3.11.0 openblas
```

Do not use:

```text
KMP_DUPLICATE_LIB_OK=TRUE
```

That variable masks OpenMP conflicts rather than fixing the environment and is not part of the validated configuration.

### 42.5 Exact PyTorch CUDA overlay

PyTorch was intentionally installed through pip on all machines using the CUDA 11.8 wheel index:

```powershell
python -m pip install `
  torch==2.5.1+cu118 `
  torchvision==0.20.1+cu118 `
  torchaudio==2.5.1+cu118 `
  --index-url https://download.pytorch.org/whl/cu118
```

CMD equivalent:

```cmd
python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Linux/Kamiak equivalent:

```bash
"$TARGET_ENV/bin/python" -m pip install \
  torch==2.5.1+cu118 \
  torchvision==0.20.1+cu118 \
  torchaudio==2.5.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

This isolates the exact GPU framework version from the Conda solver and gives the same PyTorch/CUDA runtime combination across all four machines.

### 42.6 Windows Pillow DLL workaround

The Conda Pillow 12.2.0 build caused a Windows DLL/import-order issue when Torch was imported before Torchvision/Pillow. The validated fix was to replace the Conda-installed Pillow package with the official pip wheel of the same version:

```powershell
python -m pip install --force-reinstall --no-deps pillow==12.2.0
```

This was required on laptop, home PC, and lab PC. It is a Windows-specific binary compatibility workaround and was not required on Kamiak.

Validation must preserve the torch-first order:

```python
import torch
import torchvision
import torchaudio
import PIL
```

The accepted Windows result is:

```text
PyTorch: 2.5.1+cu118
Torchvision: 0.20.1+cu118
Torchaudio: 2.5.1+cu118
Pillow: 12.2.0
CUDA available: True
```

### 42.7 pip-only package overlay

The following packages were deliberately installed through pip after the Conda foundation:

```text
casadi==3.7.2
opyplus==2.0.7
python-slugify==5.0.2
text-unidecode==1.3
Unidecode==1.4.0
```

Windows command:

```powershell
python -m pip install --no-deps `
  casadi==3.7.2 `
  opyplus==2.0.7 `
  python-slugify==5.0.2 `
  text-unidecode==1.3 `
  Unidecode==1.4.0
```

`--no-deps` was used because the dependency foundation was already controlled through the Conda lock, and reinstalling transitive dependencies through pip could destabilize compiled packages.

Kamiak additionally used a generated portable pip overlay derived from the existing validated Linux environment:

```text
/home/ninad.gaikwad/projects/Environments/locks/kamiak/pip-portable-requirements.txt
```

The portable overlay intentionally excluded:

```text
torch
torchvision
torchaudio
triton
nvidia-*
scalebridge
pip
wheel
```

PyTorch and its NVIDIA runtime packages were installed from the CUDA 11.8 wheel index. ScaleBridge was installed editable from the repository. Packaging tools were bootstrapped separately.

### 42.8 Editable ScaleBridge installation

Every final environment installs the repository in editable mode:

```powershell
python -m pip install --no-deps -e .
```

Validated Windows paths:

```text
Laptop:
  <laptop repo>\src\scalebridge\__init__.py

Home PC:
  D:\Dropbox (Personal)\NinadGaikwad_PhD\Gaikwad_Research\
  From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\
  scalebridge-research\src\scalebridge\__init__.py

Lab PC:
  F:\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\
  From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\
  scalebridge-research\src\scalebridge\__init__.py
```

Validated Kamiak path:

```text
/home/ninad.gaikwad/projects/scalebridge-research/src/scalebridge/__init__.py
```

Editable installation means source-code changes in the repository are immediately visible to the active environment without rebuilding a wheel.

### 42.9 Dependency and import validation

Every accepted environment passed:

```text
python -m pip check
```

Expected result:

```text
No broken requirements found.
```

The consolidated validation imports:

```python
import numpy
import scipy
import pandas
import sklearn
import matplotlib
import seaborn
import torch
import torchvision
import torchaudio
import PIL
import pyarrow
import mlflow
import optuna
import cvxpy
import pyomo.environ
import casadi
import opyplus
import scalebridge
```

Kamiak additionally validated:

```python
import gymnasium
import stable_baselines3
import neuromancer
```

### 42.10 Optimization solver validation

CVXPY was tested with:

```python
x = cp.Variable()
problem = cp.Problem(cp.Minimize((x - 3) ** 2), [x >= 0])
problem.solve()
```

Accepted result:

```text
status = optimal
x = 3.0
objective = 0.0
```

Lab-PC installed CVXPY solvers:

```text
CLARABEL
ECOS
ECOS_BB
OSQP
SCIPY
SCS
```

CasADi's bundled IPOPT was tested on Windows and Kamiak with:

```python
x = ca.MX.sym("x")
nlp = {"x": x, "f": (x - 3) ** 2}
solver = ca.nlpsol(
    "solver",
    "ipopt",
    nlp,
    {"ipopt.print_level": 0, "print_time": 0},
)
solution = solver(x0=0)
```

Accepted result:

```text
x = 3.0
objective = 0.0
```

For Pyomo on Windows, an external AMPL-compatible Ipopt executable is required. The validated binary is:

```text
Ipopt 3.10.1
Microsoft cl 15.00.21022.08 for x64
ASL(20111018)
```

Validated path:

```text
C:\software\Ipopt-3.10.1-win64-intel11.1\bin\ipopt.exe
```

The archive is stored outside the repository in the shared external environment tree:

```text
..\Environments\software\Ipopt-3.10.1-win64-intel11.1.zip
```

The lab-PC user `PATH` was updated to include:

```text
C:\software\Ipopt-3.10.1-win64-intel11.1\bin
```

Validated Pyomo result on laptop and lab PC:

```text
Available: True
Status: ok
Termination: optimal
x: 3.0
objective: 0.0
```

The home-PC environment passed import and dependency validation, but an external Pyomo-Ipopt solve was not separately recorded during this rebuild. CasADi and CVXPY remain available through the environment.

### 42.11 Lock-file policy

Each machine has three final environment records:

```text
conda-explicit-spec.txt
conda-env-export.yml
pip-freeze.txt
```

Purpose:

| File | Purpose |
|---|---|
| `conda-explicit-spec.txt` | Exact platform-specific Conda artifact URLs and builds; strongest same-platform reproduction |
| `conda-env-export.yml` | Human-readable environment specification without build pins |
| `pip-freeze.txt` | Complete installed Python distribution record, including pip-installed packages |

Windows external lock layout:

```text
..\Environments\locks\windows\
  laptop\
    conda-explicit-spec.txt
    conda-env-export.yml
    pip-freeze.txt
  home_pc\
    conda-explicit-spec.txt
    conda-env-export.yml
    pip-freeze.txt
  lab_pc\
    conda-explicit-spec.txt
    conda-env-export.yml
    pip-freeze.txt
```

Kamiak external lock layout:

```text
/home/ninad.gaikwad/projects/Environments/locks/
  kamiak/
    conda-explicit-spec.txt
    conda-env-export.yml
    pip-freeze.txt
    pip-portable-requirements.txt
  kamiak_gpu_full/
    conda-explicit-spec.txt
    conda-env-export.yml
    pip-freeze.txt
```

The `kamiak` directory contains the source/bootstrap records derived from the original Linux environment. The `kamiak_gpu_full` directory contains the final rebuilt and validated environment locks.

### 42.12 UTF-8 BOM issue and lock export

PowerShell `Out-File -Encoding utf8` on older Windows PowerShell versions writes a UTF-8 BOM. Conda explicit files with a BOM can fail because the first line is no longer parsed correctly.

The original Windows lock files were rewritten as UTF-8 without BOM. Older Windows PowerShell also does not recognize:

```text
-Encoding utf8NoBOM
```

Therefore, all three Windows lock files should be exported through Python:

```powershell
python -c "import subprocess, pathlib; out=pathlib.Path(r'..\Environments\locks\windows\lab_pc'); out.mkdir(parents=True, exist_ok=True); cmds={'conda-explicit-spec.txt':['conda','list','--explicit'],'conda-env-export.yml':['conda','env','export','--no-builds'],'pip-freeze.txt':['python','-m','pip','freeze']}; [(out/name).write_text(subprocess.check_output(cmd, text=True), encoding='utf-8', newline='\n') for name,cmd in cmds.items()]"
```

Use the corresponding output directory for `laptop`, `home_pc`, or `lab_pc`.

Bash redirection on Kamiak did not introduce the Windows BOM problem:

```bash
conda list --prefix "$TARGET_ENV" --explicit \
  > "$FINAL_LOCK/conda-explicit-spec.txt"

conda env export --prefix "$TARGET_ENV" --no-builds \
  > "$FINAL_LOCK/conda-env-export.yml"

"$TARGET_ENV/bin/python" -m pip freeze \
  > "$FINAL_LOCK/pip-freeze.txt"
```

### 42.13 Important pip-freeze limitation

Conda-installed Python packages can appear in `pip freeze` as build-machine URLs such as:

```text
file:///home/conda/feedstock_root/build_artifacts/...
file:///D:/bld/...
file:///C:/bld/...
```

These paths are not portable and must not be installed wholesale with:

```text
pip install -r pip-freeze.txt
```

The authoritative reconstruction order is:

```text
1. Conda explicit lock for the platform
2. exact PyTorch CUDA pip wheels
3. controlled pip-only overlay
4. editable ScaleBridge install
5. validation
```

`pip-freeze.txt` is an audit record, not the primary all-in-one installer.

### 42.14 Kamiak module and SLURM requirements

The login node is for inspection and job submission only. Heavy environment creation and GPU validation must run through SLURM.

Required modules:

```bash
module purge
module load StdEnv
module load miniconda3/3.10
module load cuda/11.8.0
```

Validated Conda executable:

```text
/opt/apps/miniconda3/3.10/bin/conda
```

SLURM account and partition:

```text
account: dubey
partition: vcea
qos: normal
GPU node used: sn14
```

Validated module availability:

```text
miniconda3/3.10 requires StdEnv
cuda/11.8.0 is available
```

External Kamiak environment tree:

```text
/home/ninad.gaikwad/projects/
  scalebridge-research/
  Environments/
    locks/
    logs/
    scripts/
    software/
```

The external `Environments` directory is intentionally outside the Git repository and mirrors the Windows external environment organization.

### 42.15 Kamiak environment-build script

Authoritative script:

```text
/home/ninad.gaikwad/projects/Environments/scripts/create_scalebridge_kamiak_env.sbatch
```

Primary target:

```text
/home/ninad.gaikwad/conda_envs/scalebridge-dev-gpu-kamiak
```

The script performs:

```text
1. Allocate a vcea Tesla GPU through SLURM.
2. Load StdEnv, miniconda3/3.10, and cuda/11.8.0.
3. Recreate the Linux Conda foundation from the explicit lock.
4. Bootstrap pip without a full Conda re-solve.
5. Install exact PyTorch 2.5.1 CUDA 11.8 wheels.
6. Install the portable Linux pip overlay.
7. Install ScaleBridge editable.
8. Run pip check.
9. Run the full import/CUDA validation.
10. Run CVXPY and CasADi-IPOPT tests.
11. Export final Kamiak locks.
```

Final successful SLURM job:

```text
job_id: 27929819
state: COMPLETED
exit_code: 0:0
elapsed: 00:05:03
node: sn14
GPU observed by PyTorch: NVIDIA A100-PCIE-40GB
```

### 42.16 Kamiak failure history and fixes

The first submission failed immediately because the SLURM output/error directory did not exist before SLURM attempted to open the files. A script cannot create its own log directory after SLURM has already tried to open the output path.

Fix:

```bash
mkdir -p /home/ninad.gaikwad/projects/Environments/logs
```

The next build recreated the explicit Conda layer but failed because the target environment did not contain a usable `pip` module:

```text
No module named pip
```

A subsequent script revision attempted:

```bash
conda install --prefix "$TARGET_ENV" pip setuptools wheel
```

That forced Conda 23.3.1 to re-solve the large environment, fell back from frozen to flexible solving, and ran for over an hour.

That job was canceled. The partial target environment was removed, and the packaging-tools block was replaced with:

```bash
"$TARGET_ENV/bin/python" -m ensurepip --upgrade
"$TARGET_ENV/bin/python" -m pip install --upgrade \
  pip==26.1.2 \
  setuptools==83.0.0 \
  wheel==0.47.0
```

This avoided a second full Conda solve. The final job completed in approximately five minutes.

### 42.17 Machine-specific operational notes

#### Laptop

```text
Environment: scalebridge-dev-gpu-laptop
GPU: NVIDIA GeForce MX150
Use: development, code review, small validation
Data policy: Dropbox-generated data should remain online-only when space is limited
```

#### Home PC

```text
Environment: scalebridge-dev-gpu-homepc
GPU: NVIDIA GeForce GTX 1050 Ti
Repository shell used during rebuild: CMD
Use: medium compute and smoke testing
Lock export: Python-based UTF-8-no-BOM writer
```

#### Lab PC

```text
Environment: scalebridge-dev-gpu-labpc
GPU: NVIDIA RTX A4000
Repository root:
  F:\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\
  From_WSU_OneDrive\BuildingModelingProject_Condensed\NewOrg\
  scalebridge-research
Use: primary Windows generation and aggregation
MLflow and EnergyPlus scratch policy: use D: drive to avoid C: space exhaustion
External Ipopt: validated and added to user PATH
```

#### Kamiak

```text
Environment:
  /home/ninad.gaikwad/conda_envs/scalebridge-dev-gpu-kamiak
Repository:
  /home/ninad.gaikwad/projects/scalebridge-research
GPU validation:
  NVIDIA A100-PCIE-40GB
Execution model:
  login node for light inspection/submission only
  SLURM sbatch for environment creation and GPU validation
```

### 42.18 Rebuild acceptance checklist

An environment is accepted only after all applicable checks pass:

```text
[ ] New environment has a machine-specific name/prefix.
[ ] Historical environment remains untouched until validation passes.
[ ] Conda foundation installs successfully.
[ ] Exact PyTorch 2.5.1+cu118 stack installs successfully.
[ ] Windows Pillow pip-wheel override is applied.
[ ] Controlled pip-only overlay installs.
[ ] ScaleBridge editable path points to the current repository.
[ ] python -m pip check reports no broken requirements.
[ ] Consolidated scientific/ML imports pass.
[ ] torch.cuda.is_available() is True on a GPU machine.
[ ] CUDA runtime reports 11.8.
[ ] Expected GPU is visible.
[ ] CVXPY solve passes.
[ ] CasADi IPOPT solve passes.
[ ] Pyomo external Ipopt passes where configured.
[ ] Machine-specific locks are exported.
[ ] Lock files are UTF-8 without BOM.
[ ] Final paths and machine details are recorded.
```

### 42.19 Environment status summary

As of the final July 2026 rebuild:

```text
Laptop environment: complete
Home-PC environment: complete
Lab-PC environment: complete
Kamiak environment: complete

Windows pip check: passed
Kamiak pip check: passed
Windows CUDA validation: passed
Kamiak CUDA validation: passed
CVXPY validation: passed
CasADi IPOPT validation: passed
Pyomo external Ipopt: passed on laptop and lab PC
Machine-specific locks: exported
```

---

## 43. July 26, 2026 Authoritative Stage B Completion Record

The compact P1 Stage B aggregation campaign is complete and is the fixed upstream source for Phase C.

```text
campaign_id: p1_compact_4b4c_labpc_1w_v1
matrix_run_id: aggregation_matrix_20260712_215839
building-weather cases: 16
aggregation levels: 5
weight modes: 3
selected aggregation objects: 240
successful aggregation objects: 240
failed aggregation objects: 0
```

Matrix definition:

```text
4 buildings
× 4 climates/weather cases
× 5 aggregation levels
× 3 aggregation weighting styles
= 240 Phase B aggregation objects
```

The five aggregation levels are:

```text
p1_l01_all_to_one
p1_l02_functional
p1_l03_intermediate
p1_l04_spatial_detailed
p1_l05_identity
```

The three weighting modes are:

```text
equal
floor_area
volume
```

Stage B provides more than 240 physical zone outputs because each aggregation object can contain one or more aggregate zones. Phase C therefore discovers and processes aggregate zones, not merely the 240 plan rows.

The Stage B matrix is memory-safe by design:

- one aggregation plan is executed at a time;
- normal variables are loaded one parquet at a time;
- `System Node Mass Flow Rate` is streamed with PyArrow batches;
- raw frames are deleted and garbage-collected between variables;
- exact plan paths are selected by case, aggregation ID, and weight mode.

The full Stage B matrix is now considered immutable upstream provenance for the first P1 Phase C/D/E campaign unless a documented re-aggregation is intentionally performed.

---

## 44. Phase C Purpose and Scientific Role

Phase C learns intermediate heat-input models from Stage B aggregated EnergyPlus data.

The main purpose is to reconstruct or predict heat-input components needed by downstream thermal models without requiring the thermal-model stage to reproduce every EnergyPlus internal-gain and HVAC calculation directly.

The core mapping is:

```text
available aggregated predictors
    → component-specific heat-input regression model
    → persisted model
    → full-year predicted heat-input component
```

Phase C outputs are not the final building thermal models. They are reusable learned input blocks for Phase D and Phase E.

Downstream uses include:

- assembling the complete thermal-model state/input/target table;
- supplying component heat inputs to ANN/RNN/SciML thermal models;
- supplying physically interpretable exogenous inputs to grey-box RC models;
- enabling full-year inference with a common model artifact interface;
- supporting future Gymnasium simulation and sensitivity workflows.

Current component vocabulary observed in the validated smoke campaign includes:

```text
QSol1
QSol2
QZic_P
QZir_P
QZic_L
QZir_L
QZic_EE
QZir_EE
QZir_GE
QZivr_L
QAC
```

Not every aggregate zone has every component. Dataset and task counts are therefore discovered from available signals rather than hard-coded.

---

## 45. Phase C C1–C9 Pipeline

The Phase C pipeline is organized into nine ordered stages.

| Stage | Name | Responsibility |
|---|---|---|
| C1 | Aggregation readiness audit | Discover selected Stage B aggregation-zone outputs and verify required inputs |
| C2 | Canonical feature construction | Build aligned full-year predictors and heat-input targets |
| C3 | Split construction | Build train/validation/test assignments |
| C4 | Model-dataset construction | Materialize one model-ready dataset per zone/component |
| C5 | Model API validation | Validate estimator factory, persistence, reload, inference, and common API behavior |
| C6 | Model training | Train and persist component regression models |
| C7 | Persisted-model evaluation | Reload saved models and evaluate train/validation/test behavior |
| C8 | Full-year component inference | Produce full-year component predictions by aggregate zone |
| C9 | MLflow registration | Register parent, stage, training-task, evaluation-task, and inference-task runs |

A single timestamp suffix is shared across all stages:

```text
phase_c_YYYYMMDD_HHMMSS

heat_input_audit_YYYYMMDD_HHMMSS
heat_input_features_YYYYMMDD_HHMMSS
heat_input_splits_YYYYMMDD_HHMMSS
heat_input_datasets_YYYYMMDD_HHMMSS
c5_YYYYMMDD_HHMMSS
c6_pytorch_YYYYMMDD_HHMMSS
c7_pytorch_YYYYMMDD_HHMMSS
c8_pytorch_YYYYMMDD_HHMMSS
```

This shared suffix is the primary provenance link across C1–C9.

---

## 46. Phase C Important Modules and Scripts

Primary campaign runner:

```text
scripts/heat_input_regression/run_phase_c_campaign.py
```

Primary stage scripts:

```text
scripts/heat_input_regression/audit_aggregation_for_heat_input_regression.py
scripts/heat_input_regression/build_heat_input_regression_features.py
scripts/heat_input_regression/build_heat_input_regression_splits.py
scripts/heat_input_regression/build_heat_input_regression_datasets.py
scripts/heat_input_regression/validate_heat_input_regression_model_api.py
scripts/heat_input_regression/train_heat_input_regression_models.py
scripts/heat_input_regression/evaluate_heat_input_regression_models.py
scripts/heat_input_regression/run_heat_input_regression_full_year_inference.py
scripts/heat_input_regression/register_phase_c_run_with_mlflow.py
scripts/heat_input_regression/validate_phase_c_mlflow_tracking.py
```

Reusable package areas developed during Phase C include:

```text
src/scalebridge/data/heat_input_regression/
src/scalebridge/models/heat_input_regression/
src/scalebridge/tracking/mlflow/
```

The implementation includes:

- aggregation discovery and signal cataloging;
- canonical time alignment;
- HVAC target construction;
- feature engineering;
- split construction;
- dataset manifests and validation;
- estimator base/factory API;
- closed-form linear estimator;
- PyTorch linear estimator;
- persisted inference;
- evaluation metrics and artifacts;
- full-year component inference;
- hierarchical MLflow registration.

The current production estimator for the P1 full campaign is:

```text
pytorch_linear
```

The validated Windows GPU stack is:

```text
torch: 2.5.1+cu118
CUDA runtime: 11.8
lab-PC GPU: NVIDIA RTX A4000
```

---

## 47. Phase C Feature and Target Semantics

Validated C2 defaults:

```text
internal_gain_predictor_method: aggregate_average
hvac_target_method: signed_zone_sensible
```

C2 consumes Stage B zone-level outputs and constructs canonical aligned data keyed by:

```text
case_id
aggregation_id
weight_mode
aggregate_zone_id
timestamp
```

The feature builder preserves provenance back to:

```text
campaign_id
matrix_run_id
case_id
source generation run
aggregation run
aggregation level
weight mode
source-zone grouping
```

Expected Phase D work must not discard this provenance.

C3 validated split behavior:

```text
split_strategy: monthly_distributed_holdout
train_fraction: 0.70
validation_fraction: 0.15
test_fraction: 0.15
```

The monthly distributed split is intended to distribute seasonal conditions across train, validation, and test rather than creating one single chronological tail holdout.

Phase E may require additional sequence-aware windows or scenario-specific splits, but those must be derived from the canonical C3/Phase D records and recorded explicitly.

---

## 48. Phase C Estimator Contract

All Phase C estimators must expose a common lifecycle:

```text
construct
fit
predict
save
load
report metadata
```

Implemented estimator types:

```text
closed_form_linear
pytorch_linear
```

The PyTorch linear model is the selected production model for the real P1 campaign.

The estimator interface is designed so later regression methods can be added without changing C1–C4 or C7–C9. Potential future additions include:

```text
regularized linear regression
small MLP
monotonic or constrained regression
probabilistic regression
multi-output regression
```

These are not required before Phase D begins.

---

## 49. Phase C Output Layout

Campaign-level orchestration output:

```text
<campaign_root>/
  heat_input_regression/
    campaign_runs/
      <phase_c_run_id>/
        phase_c_campaign_plan.json
        phase_c_campaign_run_manifest.json
        logs/
          01_*.log
          02_*.log
          ...
```

Stage outputs:

```text
<campaign_root>/heat_input_regression/
  audit_runs/<audit_run_id>/
  feature_runs/<feature_run_id>/
  split_runs/<split_run_id>/
  dataset_runs/<dataset_run_id>/
  model_api_validation/<c5_run_id>/
  training_runs/<training_run_id>/
  evaluation_runs/<evaluation_run_id>/
  inference_runs/<inference_run_id>/
  mlflow_registration_runs/<phase_c_run_id>/
```

Key manifests:

```text
heat_input_regression_audit_manifest.json
heat_input_feature_run_manifest.json
split_run_manifest.json
dataset_run_manifest.json
c5_model_api_validation_manifest.json
training_run_manifest.json
evaluation_run_manifest.json
inference_run_manifest.json
phase_c_mlflow_registration_manifest.json
phase_c_campaign_run_manifest.json
```

Phase D should discover Phase C artifacts from these manifests. It should not reconstruct paths from assumptions when a manifest path is available.

---

## 50. Phase C Validation Modes

The campaign runner supports:

```text
full
some
none
```

Meaning:

```text
full:
    Run all available separate stage validators and final MLflow validation.

some:
    Run selected high-value validators for C2, C4, C6, C7, C8, and C9.

none:
    Run C1–C8 and C9 MLflow registration, but skip separate validator scripts.
```

Important nuance:

- C5 remains part of the core pipeline even under `--validation none`.
- Stage-internal assertions and failure handling remain active.
- `--validation none` does not mean “ignore errors.”
- MLflow registration remains enabled unless `--disable-mlflow` is supplied.
- The C9 registration script has its own `--validation-mode` setting, exposed by the runner as `--mlflow-validation-mode`.

For a truly no-separate-validation production run, use:

```text
--validation none
--mlflow-validation-mode none
```

---

## 51. Phase C Campaign Runner Development and Final Validation Fixes

The campaign runner was developed after the individual C1–C9 scripts existed. It now provides:

- required `--campaign-root` and optional exact `--matrix-run-id`;
- automatic latest-successful matrix discovery when the matrix ID is omitted;
- one shared timestamp suffix across C1–C9 artifacts;
- `--start-stage` and `--stop-stage` resume controls;
- `--validation full|some|none` profiles;
- `--dry-run`, `--overwrite-existing`, and per-command logs;
- adaptive `--help` inspection so historical CLI aliases can be resolved;
- MLflow enabled by default unless `--disable-mlflow` is supplied.

The final runner/validator repair sequence established these important contracts:

1. **Source syntax validation**
   - `validate_python_source_syntax.py` requires one `--paths` option followed by multiple paths.
   - The runner now emits the full Phase C source/module/script path set correctly.

2. **C3 validator provenance**
   - The C3 validator requires `campaign_root`, `matrix_run_id`, `feature_run_id`, and `split_run_id`.
   - Unsupported root aliases were removed.

3. **C4 validator provenance**
   - The C4 validator requires `campaign_root`, `matrix_run_id`, `audit_run_id`, `feature_run_id`, `split_run_id`, and `dataset_run_id`.

4. **C6–C8 validator roots**
   - C6 uses `--training-root`.
   - C7 uses `--evaluation-root`.
   - C8 uses `--inference-root`.

5. **Canonical-aware C2 validator forwarding**
   - The wrapper now accepts and forwards `campaign_root`, `matrix_run_id`, `audit_run_id`, and `feature_run_id` to the deterministic legacy validator.

6. **C2 deterministic recomputation alignment**
   - The validator canonicalizes the Stage B wide frame before recomputation, matching the builder.
   - It passes the required `aggregate_zone_count` used by PHVAC feature construction.
   - It checks canonical row counts and aggregate-zone-count consistency.

7. **Canonical timestamp and duplicate coalescence checks**
   - Separate validators verify annual 5-minute row counts, parsed timestamps, monotonic order, canonical cadence, duplicate removal, complementary-value coalescence, and absence of unresolved source conflicts.

These fixes are part of the current validated source state and should not be reverted to the older feature-root-only validator interface.

## 52. Authoritative Fully Validated Phase C QAC/PHVAC Smoke Campaign

Controlled upstream campaign:

```text
campaign_id:
p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3

matrix_run_id:
aggregation_matrix_20260715_114242
```

Authoritative completed run:

```text
phase_c_run_id: phase_c_qac_phvac_test_20260727_200800
validation: full
start_stage: C1
stop_stage: C8
estimator_type: pytorch_linear
pytorch_device: cuda
mlflow: disabled for this run
selected aggregate zones: 3
status: completed
passed command count: 17
failed command count: 0
```

Selected zones:

```text
RestaurantFastFood_All
Dining
Kitchen
```

The validated component vocabulary now includes PHVAC in addition to the earlier heat-input components:

```text
QSol1
QSol2
QZic_P
QZir_P
QZic_L
QZir_L
QZic_EE
QZir_EE
QZir_GE
QZivr_L
QAC
PHVAC
```

Availability remains structurally zone-dependent. The authoritative run produced:

```text
RestaurantFastFood_All: 12 component datasets
Dining:                11 component datasets
Kitchen:               10 component datasets
Total:                  33 component datasets
```

Validated results:

```text
C1 readiness audit:
    successful zones: 3
    failed zones: 0

Source syntax validation:
    checked files: 62
    passed files: 62
    failed files: 0

C2 feature construction:
    successful zones: 3
    failed zones: 0
    internal_gain_predictor_method: aggregate_average
    hvac_target_method: signed_zone_sensible

C2 deterministic/canonical-aware feature validation:
    passed zones: 3
    failed zones: 0

C2 canonical timestamp validation:
    annual rows per zone: 105120
    unparsed timestamps: 0
    duplicate timestamps after canonicalization: 0
    timestamp monotonic: true
    noncanonical cadence count: 0
    passed zones: 3

C2 timestamp coalescence validation:
    missing source values after coalescence: 0
    conflicting source values: 0
    passed zones: 3

C3 split construction:
    split strategy: monthly_distributed_holdout
    train/validation/test: 0.70/0.15/0.15
    successful zones: 3
    split validation passed zones: 3

C4 dataset construction:
    successful zones: 3
    successful model datasets: 33
    failed model datasets: 0

C4 dataset validation:
    passed model datasets: 33
    failed model datasets: 0

C5 model API validation:
    checks: 224
    passed: 224
    failed: 0
    estimators exercised: closed_form_linear and pytorch_linear[cpu]

C6 CUDA training:
    completed training tasks: 33
    failed training tasks: 0
    estimator: pytorch_linear
    requested device: cuda

C6 training validation:
    passed artifacts: 33
    failed artifacts: 0

C7 persisted-model evaluation:
    completed evaluations: 33
    failed evaluations: 0

C7 evaluation validation:
    passed artifacts: 33
    failed artifacts: 0

C8 full-year inference:
    completed zones: 3
    failed zones: 0
    inferred component counts: 12, 11, and 10

C8 inference validation:
    passed artifacts: 3
    failed artifacts: 0
```

Campaign manifest:

```text
<campaign_root>/heat_input_regression/campaign_runs/
phase_c_qac_phvac_test_20260727_200800/
phase_c_campaign_run_manifest.json
```

This run is the authoritative computational validation of Phase C C1–C8. It supersedes the older 30-model smoke as the primary development record.

## 53. Phase C Sequential Execution and Memory Policy

The campaign runner launches one stage subprocess at a time:

```text
C1 completes
→ C2 completes
→ C3 completes
→ C4 completes
→ C5 completes
→ C6 completes
→ C7 completes
→ C8 completes
→ C9 completes
```

Within validated stage scripts, work is also iterated sequentially:

```text
C1: one aggregation zone at a time
C2: one aggregation zone at a time
C3: one aggregation zone at a time
C4: one aggregation zone, then one component dataset at a time
C6: one model dataset at a time
C7: one persisted model at a time
C8: one aggregate zone at a time
C9: one MLflow run registration at a time
```

The design does not place all 240 Stage B objects, all aggregate zones, or all component models into GPU memory simultaneously.

Expected memory scaling is approximately with the largest current zone/model dataset plus stage-local intermediates, rather than the full campaign.

Highest-risk memory stages are:

```text
C2: full-year feature construction
C4: model-dataset materialization
C6: CPU/GPU tensor conversion and training
C8: full-year inference output assembly
```

Phase D must preserve this streaming/sequential philosophy. It should not concatenate the entire campaign into one in-memory pandas dataframe.

---

## 54. Phase C MLflow Status and Full P1 Lab-PC Execution

### Latest authoritative test-run MLflow status

The fully validated run `phase_c_qac_phvac_test_20260727_200800` intentionally used:

```text
--disable-mlflow
--stop-stage C8
```

This isolated C1–C8 pipeline correctness from tracking registration while the runner and validators were being repaired. Therefore:

```text
C1–C8 for the latest run: fully validated
C9 for the latest run: not executed
```

C9 is still implemented, and an earlier 30-model smoke run successfully registered the expected Phase C parent/stage/task hierarchy. That earlier MLflow result is retained as historical evidence, but it must not be reported as the C9 result for the latest 33-model run.

### Full P1 Phase C target

Validated lab-PC environment:

```text
environment: scalebridge-dev-gpu-labpc
machine_id: lab-pc
generated data root:
F:\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge
MLflow tracking URI: http://127.0.0.1:5000
GPU: NVIDIA RTX A4000
```

Full upstream campaign:

```text
campaign_id: p1_compact_4b4c_labpc_1w_v1
matrix_run_id: aggregation_matrix_20260712_215839
Stage B aggregation objects: 240/240 successful
```

The full campaign should first be dry-run with the corrected runner, then executed without limits. For a full validation run without C9:

```powershell
$CampaignRoot = "$env:SCALEBRIDGE_GENERATED_DATA_ROOT\campaigns\p1_compact_4b4c_labpc_1w_v1"
$MatrixRunId = "aggregation_matrix_20260712_215839"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PhaseCRunId = "phase_c_p1_full_$Timestamp"

python `
  ".\scripts\heat_input_regression\run_phase_c_campaign.py" `
  --campaign-root "$CampaignRoot" `
  --matrix-run-id "$MatrixRunId" `
  --phase-c-run-id "$PhaseCRunId" `
  --start-stage C1 `
  --stop-stage C8 `
  --validation full `
  --estimator-type pytorch_linear `
  --pytorch-device cuda `
  --disable-mlflow `
  --overwrite-existing
```

After C1–C8 passes, C9 should be run or resumed with the same Phase C run ID and MLflow enabled. Do not claim the unrestricted full Phase C campaign complete until its final campaign manifest and, when used, C9 registration manifest have been inspected.

Do not use these limits for the unrestricted campaign:

```text
--max-zones
--max-model-datasets
--max-artifacts
```

The final Phase C model count is not 240. It is the sum of all available component datasets over all discovered aggregate zones.

## 55. Phase D Development Objective

Phase D is the canonical thermal-model data assembly layer.

Its purpose is to combine:

1. Stage B aggregated physical/environmental signals;
2. Phase C measured and predicted heat-input components;
3. thermal state variables;
4. control variables and setpoints;
5. weather and schedule inputs;
6. split/provenance metadata;

into one consistent, model-family-neutral dataset contract.

Phase D must not train final thermal models. It prepares authoritative data packages that all Phase E methods consume.

Conceptual lifecycle:

```text
Stage B aggregation manifests and zone outputs
+ Phase C feature/split/dataset/training/evaluation/inference manifests
→ resolve one canonical record per case/aggregation/weight/zone
→ align timestamps and units
→ select measured versus Phase-C-predicted heat inputs
→ construct thermal-model state/input/target variables
→ derive lagged and sequence-ready views without duplicating truth
→ fit training-only scalers
→ write canonical thermal-model datasets and manifests
→ validate completeness, leakage, units, and provenance
```

---

## 56. Proposed Phase D Canonical Keys and Schemas

Every Phase D row or sequence must retain the identity key:

```text
campaign_id
matrix_run_id
phase_c_run_id
case_id
building_type
climate/weather
aggregation_id
aggregation_level
weight_mode
aggregate_zone_id
timestamp
```

Recommended canonical long/wide data fields:

```text
state:
    T_zone
    optional latent/auxiliary temperatures when available

weather:
    T_outdoor
    solar variables
    humidity and other selected weather variables

control/HVAC:
    QAC measured or Phase-C-predicted
    heating setpoint
    cooling setpoint
    operating mode
    mass flow / supply conditions when retained

internal/solar heat inputs:
    QSol1
    QSol2
    QZic_P
    QZir_P
    QZic_L
    QZir_L
    QZic_EE
    QZir_EE
    QZir_GE
    QZivr_L

metadata:
    split
    timestep_minutes
    source signal provenance
    missingness flags
    measured/predicted source flag per heat-input component
```

Recommended Phase D artifacts:

```text
thermal_model_dataset_manifest.json
thermal_model_zone_index.csv
thermal_model_signal_catalog.csv
thermal_model_timeseries.parquet
thermal_model_timeseries_preview.csv
thermal_model_split_manifest.json
thermal_model_scaler_manifest.json
thermal_model_quality_report.json
thermal_model_provenance.json
```

For memory safety, write one zone package at a time and a lightweight campaign index.

---

## 57. Phase D Required Design Decisions

The next development chat should resolve these explicitly before implementation:

1. **Thermal prediction target**
   ```text
   one-step T_zone(t+1)
   derivative dT/dt
   multi-step sequence
   continuous-time state derivative
   ```
   A canonical dataset may support multiple targets, but one authoritative base representation is required.

2. **Measured versus Phase-C-predicted heat inputs**
   - preserve both when available;
   - mark source explicitly;
   - do not silently overwrite measured components;
   - support controlled experiments comparing oracle/measured and learned-input settings.

3. **Time representation**
   - maintain the native 5-minute timeline;
   - record timezone and EnergyPlus timestamp semantics;
   - define sequence windows without leakage across split boundaries.

4. **Scaling**
   - fit scalers on training rows only;
   - store per-zone, per-building, and global scaler options separately;
   - never infer scaler policy from filenames.

5. **Missing components**
   - absence can be structurally valid;
   - distinguish unavailable equipment from missing/corrupt data;
   - define zero-fill, omission, and mask policies explicitly.

6. **Static features**
   - floor area;
   - volume;
   - aggregation compression ratio;
   - building and climate identifiers;
   - equipment levels;
   - zone-group metadata.

7. **Sequence adapters**
   - derive ANN tabular, RNN windowed, and SciML continuous-time adapters from one canonical truth;
   - do not create separate independent preprocessing pipelines.

---

## 58. Phase D Proposed Package Architecture

Recommended reusable modules:

```text
src/scalebridge/data/thermal_modeling/
  models.py
  discovery.py
  signal_catalog.py
  alignment.py
  heat_inputs.py
  targets.py
  splits.py
  scaling.py
  windows.py
  writers.py
  validation.py
  engine.py
```

Recommended scripts:

```text
scripts/thermal_modeling/audit_phase_d_inputs.py
scripts/thermal_modeling/build_phase_d_signal_catalog.py
scripts/thermal_modeling/build_phase_d_datasets.py
scripts/thermal_modeling/validate_phase_d_datasets.py
scripts/thermal_modeling/run_phase_d_campaign.py
```

Recommended campaign output:

```text
<campaign_root>/thermal_modeling/
  phase_d_runs/<phase_d_run_id>/
    phase_d_campaign_plan.json
    phase_d_campaign_manifest.json
    zone_index.csv
    logs/
  datasets/<dataset_run_id>/
    cases/<case_id>/<aggregation_id>/<weight_mode>/<aggregate_zone_id>/
      thermal_model_timeseries.parquet
      signal_catalog.csv
      split_manifest.json
      scaler_manifest.json
      provenance.json
```

---

## 59. Phase E Model-Family Scope

Phase E contains final thermal-model development and benchmarking.

Required families:

### E1. Classical feedforward neural networks

```text
linear baseline
MLP / ANN
residual MLP where useful
```

Use lagged state/input features or a defined autoregressive representation.

### E2. Sequence models

```text
vanilla RNN
LSTM
GRU
```

Potential later additions:

```text
temporal convolution
transformer/attention baseline
```

These should not displace the core RNN/LSTM/GRU benchmark unless justified.

### E3. Scientific machine learning

```text
learned ODE / Neural ODE
PINN
hybrid Neural ODE + physics constraints
Neuromancer-based constrained dynamics where appropriate
```

The physics formulation must use consistent units, timestep semantics, and explicit state/input equations from Phase D.

### E4. Explicit deterministic optimization / grey-box estimation

Potential model structures:

```text
1R1C
2R2C
3R2C
4R3C
```

Potential estimation methods:

```text
nonlinear least squares
maximum likelihood
constrained optimization
multiple shooting
CasADi/IPOPT
Pyomo/IPOPT where suitable
```

### E5. Bayesian inference

Potential methods:

```text
Bayesian parameter estimation
Metropolis-Hastings / MCMC
extended Kalman filter
extended Kalman smoother
expectation-maximization
particle methods if needed
posterior predictive uncertainty
```

The exact P1/P2 split must remain clear:

- P1 emphasizes black-box and SciML comparison.
- P2 emphasizes grey-box model structures, deterministic estimation, filtering/smoothing, Bayesian inference, and uncertainty.

Shared Phase D data and evaluation infrastructure should serve both papers.

---

## 60. Phase E Common Runtime Contract

Every Phase E model should support a shared high-level interface:

```text
fit
predict one step
roll out multiple steps
save
load
report metadata
report parameter count
report runtime
report device
```

Where scientifically meaningful, also support:

```text
state initialization
continuous-time derivative
uncertainty prediction
parameter posterior
constraint diagnostics
```

The runtime contract should allow later loading into a Gymnasium environment.

Planned simulator direction:

- one or multiple building thermal models;
- exposed controllable inputs;
- common model loading;
- scenario testing;
- sensitivity analysis;
- later MPC, RL, and co-simulation workflows.

Model-specific training remains separate. The Gymnasium environment is the common simulation/control layer.

---

## 61. Phase E Experiment Fairness Rules

All compared model families must align on:

```text
same Phase D source records
same train/validation/test assignments
same aggregation object and zone identity
same weather and input variables
same prediction horizon
same initialization rules
same missing-data policy
same metrics
same measured-versus-predicted heat-input scenario
```

Recommended evaluation categories:

```text
one-step accuracy
multi-step rollout accuracy
seasonal performance
operating-regime performance
setpoint-change performance
climate transfer
aggregation-level sensitivity
weight-mode sensitivity
training runtime
inference runtime
peak CPU memory
peak GPU memory
parameter count
model artifact size
physical constraint violations
uncertainty calibration where applicable
```

Do not compare methods using different hidden preprocessing choices.

---

## 62. Phase D/E MLflow Hierarchy

Recommended hierarchy:

```text
Phase D parent run
  stage/data-build runs
  per-zone dataset audit runs if needed

Phase E experiment parent
  model-family run
    hyperparameter/tuning child runs
    final training run
    evaluation/rollout child runs
```

Required tags:

```text
campaign_id
matrix_run_id
phase_c_run_id
phase_d_run_id
case_id
building_type
climate
aggregation_id
aggregation_level
weight_mode
aggregate_zone_id
model_family
model_name
estimation_method
machine_id
device
git_commit
```

Keep machine identity as metadata, not as a top-level artifact folder.

---

## 63. Phase D/E Immediate New-Chat Starting Tasks

A new development chat should begin with Phase D, not directly with model training.

Recommended first sequence:

1. Audit the exact current Phase C inference, feature, split, and dataset manifests.
2. Audit representative Stage B zone files and Phase C full-year inference files.
3. Define the canonical Phase D identity key and signal catalog.
4. Decide the first thermal target representation.
5. Implement discovery and manifest models.
6. Build one-zone Phase D smoke output without loading the campaign globally.
7. Validate timestamp alignment, splits, units, and measured/predicted heat-input provenance.
8. Expand to a multi-building/multi-aggregation smoke.
9. Add the Phase D campaign runner.
10. Only after Phase D validation, begin Phase E with the simplest linear/MLP baseline.

Files that the new chat should request first if exact current code is needed:

```text
Phase C inference manifest and one zone output
Phase C dataset manifest and one component dataset
Phase C feature manifest
Phase C split manifest
Stage B aggregation manifest and one aggregated_timeseries_wide.parquet schema
current heat_input_regression package tree
current model base/factory implementation
```

---

## 64. Authoritative Current Status for New Development Chats

```text
Stage A:
    complete and validated
    16/16 generation cases
    440 canonical variable parquet files
    440 legacy pickles

Stage B:
    complete and validated
    matrix_run_id: aggregation_matrix_20260712_215839
    240/240 aggregation objects successful
    0 failures

Phase C:
    C1–C9 implemented
    PyTorch linear estimator implemented
    campaign runner implemented
    resume and dry-run supported
    MLflow hierarchical registration implemented
    3-zone / 30-model smoke completed end to end
    38/38 C5 API checks passed
    30/30 CUDA training tasks passed
    30/30 evaluation tasks passed
    3/3 full-year inference zones passed
    8 stage + 63 task MLflow registration passed
    full P1 command dry run passed on lab-PC
    unrestricted full P1 execution ready, not yet claimed complete

Phase D:
    next code-development phase
    canonical thermal-model data assembly

Phase E:
    follows validated Phase D
    ANN, RNN/LSTM/GRU, SciML, explicit optimization, Bayesian inference
```

This section supersedes earlier “next step” statements elsewhere in this document.


---

## 56. August 2, 2026 Authoritative Phase C Availability-Aware Completion

This section supersedes earlier Phase C status statements in this README.

### 56.1 Controlled end-to-end validation

The authoritative controlled Phase C run is:

```text
campaign_id:
p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3

matrix_run_id:
aggregation_matrix_20260715_114242

phase_c_run_id:
phase_c_full_updated_test_laptop_20260802_172455

machine:
laptop

environment:
scalebridge-dev-gpu-laptop

estimator:
pytorch_linear

training device:
cuda

validation profile:
full

MLflow:
enabled
```

Final orchestration result:

```text
status: completed
passed_command_count: 19
failed_command_count: 0
```

The 19 commands consist of C1-C9 plus all configured source, feature,
timestamp, coalescence, split, dataset, training, evaluation, inference, and
MLflow validators.

### 56.2 Final availability-aware counts

```text
candidate_model_count:                  57
applicable_model_count:                 33
structurally_inapplicable_model_count:  20
invalid_model_count:                     4
missing_expected_data_model_count:       0

created_dataset_count:                  33
trained_model_count:                    33
evaluated_model_count:                  33
inferred_component_count:               33

inference_zone_count:                    3
zero_component_zone_count:               0
```

The core cross-stage invariant passed:

```text
C1 applicable models
= C4 model datasets
= C6 trained models
= C7 evaluated models
= C8 inferred components
= 33
```

Controlled per-zone component counts:

```text
RestaurantFastFood_All: 12
Dining:                11
Kitchen:               10
```

### 56.3 Structural model availability is now part of the Phase C contract

A candidate Phase C component model is no longer assumed to exist for every
aggregate zone.

Each candidate relationship is classified independently as one of:

```text
applicable
structurally_inapplicable
invalid
missing_expected_data
fatal/unexpected failure
```

Examples of structural availability differences include:

- a zone with no People object has no People convective/radiant model;
- a zone with no Lights object has no Lights convective/radiant/visible model;
- a zone may lack one or more electric, gas, other, hot-water, or steam
  equipment components;
- a zone with no mapped HVAC supply/system node cannot support QAC;
- PHVAC depends on QAC and is structurally inapplicable when QAC is unavailable;
- an unconditioned or common zone can still have valid solar or equipment
  components even when QAC and PHVAC are absent.

Phase C must never create fake models or silently substitute zeros merely to
force a fixed model count.

### 56.4 Availability propagation through C1-C9

C1 writes the authoritative model-applicability inventory, including:

```text
model_applicability.csv
applicable_models.csv
inapplicable_models.csv
```

Important fields include:

```text
applicability_class
reason_code
missing_required_signals
dependency_status
fatal_for_zone
```

C2 snapshots this availability beside each zone's features and builds only
feature families required by applicable models.

C3 carries the zone-level availability counts and IDs through split provenance.

C4 materializes only C1-approved model datasets. A valid zero-model zone may
complete with zero model datasets.

C5 validates the estimator API against discovered C4 datasets.

C6 trains only discovered C4 datasets and supports valid zero-task runs.

C7 evaluates only completed C6 artifacts and supports valid zero-artifact runs.

C8 uses the C2 feature inventory as the authoritative zone inventory. It
therefore preserves zones even when they have zero evaluated components. Each
zone receives a `component_applicability.csv` snapshot beside its prediction
package.

C9 logs availability-aware metrics, creates the parent/stage/task hierarchy,
and validates that C6, C7, and C8 task runs are nested under the correct stage
runs.

### 56.5 Final C9 MLflow validation

The authoritative controlled run registered:

```text
stage_run_count:             8
training_task_run_count:    33
evaluation_task_run_count:  33
inference_task_run_count:    3
total_task_run_count:       69
misplaced_task_run_count:    0
failed_registration_count:   0
```

The C9 validator derives expected C6-C8 task counts at runtime from the
completed registration manifest. The campaign runner must not freeze these
counts while planning a fresh run.

### 56.6 Authoritative controlled artifacts

```text
<testing_campaign_root>/heat_input_regression/campaign_runs/
  phase_c_full_updated_test_laptop_20260802_172455/
    phase_c_campaign_run_manifest.json

<testing_campaign_root>/heat_input_regression/mlflow_registration_runs/
  phase_c_full_updated_test_laptop_20260802_172455/
    phase_c_mlflow_registration_manifest.json
```

### 56.7 Main lab-PC Phase C production run

Main campaign:

```text
campaign_id:
p1_compact_4b4c_labpc_1w_v1

matrix_run_id:
aggregation_matrix_20260712_215839

machine:
lab-pc

environment:
scalebridge-dev-gpu-labpc
```

The production run should use:

```text
validation: none
MLflow validation mode: none
estimator: pytorch_linear
device: cuda
MLflow: enabled
```

`--validation none` disables separate stage validator subprocesses. It does not
disable stage-internal assertions or ordinary failure handling. MLflow remains
enabled because `--disable-mlflow` is not supplied.

```powershell
conda activate scalebridge-dev-gpu-labpc

chcp 65001

$CampaignRoot = "F:\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge\campaigns\p1_compact_4b4c_labpc_1w_v1"
$MatrixRunId = "aggregation_matrix_20260712_215839"

$env:SCALEBRIDGE_MACHINE_ID = "lab-pc"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$PhaseCRunId = "phase_c_full_main_labpc_$Timestamp"
$LogFile = ".\Phase_C_Full_Main_LabPC_$Timestamp.log"

python `
  ".\scripts\heat_input_regression\run_phase_c_campaign.py" `
  --campaign-root "$CampaignRoot" `
  --matrix-run-id "$MatrixRunId" `
  --phase-c-run-id "$PhaseCRunId" `
  --start-stage C1 `
  --stop-stage C9 `
  --validation none `
  --mlflow-validation-mode none `
  --estimator-type pytorch_linear `
  --pytorch-device cuda `
  --overwrite-existing `
  2>&1 | Tee-Object -FilePath "$LogFile"
```

Do not add `--disable-mlflow`. Do not add `--continue-on-error` for the first
authoritative production run; fail fast on an unexpected error.

The authoritative production manifest will be:

```text
<main_campaign_root>/heat_input_regression/campaign_runs/<phase_c_run_id>/
phase_c_campaign_run_manifest.json
```

---

## 57. Phase D Interface Consequence of Phase C Availability

Phase D must not assume a fixed 12-component input vector for every zone.

For every aggregate zone, Phase D must consume:

```text
component_applicability.csv
annual_component_predictions_manifest.json
annual_component_predictions.parquet
zone_feature_manifest.json
model_applicability_snapshot.csv
```

when available.

For each component, Phase D must preserve an explicit state such as:

```text
available_and_predicted
structurally_inapplicable
invalid_upstream_relationship
missing_expected_data
not_trained
training_failed
evaluation_failed
inference_failed
```

Structural absence is not equivalent to zero and is not equivalent to data
corruption.

Grouped totals such as `QZic_total` and `QZir_total` must record:

- which component terms were available;
- which terms were structurally absent;
- whether the grouped total is complete relative to the zone's applicable
  component set;
- the exact aggregation formula;
- measured versus predicted source;
- whether any missing applicable term forced the grouped total to be marked
  incomplete.

QAC and PHVAC require special handling:

```text
QAC:
    available only when the required mapped system-node mass-flow and
    temperature signals exist for that aggregate zone.

PHVAC:
    depends on QAC and must be marked structurally inapplicable when QAC is
    unavailable.
```

No Phase D or Phase E model may interpret an absent QAC/PHVAC model as a
physical zero without an explicit, scientifically justified transformation.

---

## 63. August 11, 2026 — Authoritative Phase D Implementation and Validation

This section supersedes the earlier **proposed** Phase D package/output design in Sections 55–58. Phase D is now implemented through D8.2 and validated end-to-end on the controlled testing campaign. D8.3 adds repeatable ML/SciML lag selection for the first full P1 production configuration.

### 63.1 Phase D purpose

Phase D is the canonical thermal-model data assembly layer between Stage B/Phase C and final thermal-model development. It does **not** train the final thermal models.

Canonical roles:

```text
state:
    zone_temperature

control:
    qac

common disturbance:
    outdoor_temperature

solar disturbances:
    qsol1
    qsol2

internal-heat disturbances:
    applicable qzic_*
    applicable qzir_*
    qzivr_l

auxiliary HVAC electrical quantity:
    phvac
```

Component availability is zone-specific. Phase D consumes Phase C availability metadata and must not assume a fixed People/Lights/equipment/QAC/PHVAC model vector for every zone.

### 63.2 Implemented package structure

```text
src/scalebridge/data/thermal_modeling/
    __init__.py
    alignment.py
    assembly.py
    builders.py
    campaign_runner.py
    constants.py
    discovery.py
    identities.py
    lineage.py
    manifests.py
    models.py
    policies.py
    signals.py
    silo_contracts.py
    source_refs.py
```

Primary production scripts:

```text
scripts/thermal_modeling/
    audit_phase_d_source_schema.py
    build_phase_d_assembly.py
    build_phase_d_final_datasets.py
    inspect_phase_d_sources.py
    inspect_phase_d_source_schema.py
    probe_phase_d_alignment.py
    run_phase_d_campaign.py
    validate_phase_d_campaign.py
    validate_phase_d_all_policies.py
```

### 63.3 Canonical D3/D4 alignment and assembly

Canonical year is configurable and defaults to `2001`.

Current annual source contract:

```text
5-minute timestep
non-leap year
105,120 canonical rows
```

EnergyPlus `12/31 24:00:00` is normalized to next-year midnight before canonical-year mapping.

Duplicate groups are merged column-by-column:

```text
complete + null remnant -> keep complete
identical duplicate rows -> collapse
complementary duplicate rows -> coalesce
conflicting non-null required values -> fail
```

D4 canonical assembly preserves the 30-column contract:

```text
timestamp, zone_temperature, outdoor_temperature,
qsol1, qsol2, qac, phvac,
qzic_p, qzic_l, qzic_ee, qzic_ge, qzic_oe, qzic_hwe, qzic_se,
qzir_p, qzir_l, qzir_ee, qzir_ge, qzir_oe, qzir_hwe, qzir_se,
qzivr_l, phvac_oracle, zic, zir,
split, split_index, included, exclusion_reason, source_row_index
```

Signal rules:

```text
varying -> retain
constant nonzero -> retain
complete zero -> nullable, complete_zero_signal
structurally not applicable -> nullable with source reason
applicable but missing prediction/timestamps -> fail
```

Grouped heat:

```text
zic = active qzic_* only
zir = active qzir_* + qzivr_l by default
```

`qzivr_l` can be kept separate through `--qzivr-separate`.

### 63.4 D5 lineage and name-agnostic Dependent-2

`aggregation_run_id` is opaque and is never parsed for authoritative semantics.

Phase D preserves campaign/case/generation/matrix/aggregation/weight/zone lineage from Stage B and Phase C manifests.

Dependent-2 counterpart discovery is **structural**, not based on P1 aggregation names. A candidate is eligible when the realized aggregation has exactly one aggregate zone covering the complete source-zone set. `aggregation_id`, level/family names, aggregate-zone names, and grouping strategy labels are not Dep2 eligibility requirements.

Dep2 statuses:

```text
matched_self
matched_exact
unavailable_no_counterpart
ambiguous_multiple_counterparts
invalid_configuration_mismatch
```

If no usable counterpart exists, `ind` and `dep1` still build and `dep2` is omitted without failing the aggregation run.

### 63.5 D6/D7 silo, spatial, heat, and storage contracts

Silos:

```text
ml = ML/SciML
ob = Optimization/Bayesian
```

Spatial modes:

```text
ind  = one realization per current aggregate zone
dep1 = wide current-zone coupled realization
dep2 = current states/QAC + compatible one-zone disturbances
```

Heat representations:

```text
grp_vrin  = grouped zic/zir with qzivr_l included in zir
grp_vrsep = grouped with qzivr_l separate
cmp       = active components retained separately
```

Temporal folder:

```text
l<L>_h<H>/<policy>[_rNN]/
```

Exactly one `data.parquet` and one adjacent `manifest.json` are written per realization. Train/test/validation split files are forbidden. Partition assignment remains inside `data.parquet`.

Production hierarchy:

```text
<campaign_root>/phase_d/
    campaign_runs/<phase_d_run_id>/
        phase_d_campaign_plan.json
        phase_d_campaign_run_manifest.json
        aggregation_run_registry.csv
        dataset_registry.csv
        failures.csv
        logs/

    cases/<case_id>/aggregation_runs/<aggregation_run_id>/
        aggregation_manifest.json
        silos/
            ml/
                ind|dep1|dep2/...
            ob/
                ind|dep1|dep2/...
```

D3/D4 intermediate time-series are not persisted by default.

### 63.6 Complete temporal policy catalog

ML/SciML:

```text
mdh = monthly_distributed_holdout
ch  = chronological_holdout
sh  = seasonal_holdout
```

Opt/Bayes:

```text
sd  = seasonal_distributed
sbh = seasonal_block_holdout
ci  = contiguous_identification
cdr = custom_datetime_ranges
```

Predefined meteorological seasons:

```text
winter = Dec/Jan/Feb
spring = Mar/Apr/May
summer = Jun/Jul/Aug
fall   = Sep/Oct/Nov
```

Policy knobs:

```text
MDH / CH:
    --ml-train-fraction
    --ml-test-fraction
    --ml-validation-fraction

SH:
    --ml-sh-train-seasons
    --ml-sh-test-seasons
    --ml-sh-validation-seasons

SD:
    --sd-season-offset-days
    --sd-train-days
    --sd-test-days

SBH:
    --sbh-train-seasons
    --sbh-test-seasons

CI:
    --ci-start-datetime
    --ci-train-days
    --ci-test-days

CDR:
    --cdr-train-range START/END   # repeatable
    --cdr-test-range START/END    # repeatable
```

`--ml-policy` and `--ob-policy` are repeatable. Defaults remain `mdh` and `sd`.

ML/SciML supports configurable lag and target horizon. Only state targets are generated. Opt/Bayes remains fixed at lag/horizon `1/1`.

### 63.7 D8 runner behavior

Main runner:

```text
scripts/thermal_modeling/run_phase_d_campaign.py
```

It can auto-resolve the latest fully successful aggregation matrix and the latest completed Phase C campaign matching that exact matrix, or exact IDs can be supplied.

Matrix selection/filter knobs:

```text
--aggregation-id        # repeatable
--weight-mode           # repeatable
--case-id               # repeatable
--max-aggregation-runs
```

Execution knobs:

```text
--resume
--overwrite-existing
--continue-on-error
--dry-run
```

Resume is configuration-aware. Completed outputs are skipped only when the persisted runner configuration matches the requested configuration.

### 63.8 Authoritative validation evidence

Name-agnostic D8 code validation:

```text
82/82 thermal-modeling tests passed
```

D8 controlled production run:

```text
aggregation runs: 2
datasets: 14
phase_d_parquets: 14
unexpected_parquets: 0
failed aggregation runs: 0
intermediate_time_series_persisted: False
resume skipped: 2/2
```

D8.2 all-policy code validation:

```text
92/92 thermal-modeling tests passed
PerformanceWarning treated as error
```

All-seven-policy real-data validation:

```text
phase_d_run_id:
    phase_d_all_policies_test_20260811_115228

aggregation runs:
    2/2 completed

failed:
    0

datasets:
    49

phase_d_parquets:
    49

unexpected_parquets:
    0

intermediate_time_series_persisted:
    False
```

Validated policies:

```text
ML:
    monthly_distributed_holdout
    chronological_holdout
    seasonal_holdout

Opt/Bayes:
    seasonal_distributed
    seasonal_block_holdout
    contiguous_identification
    custom_datetime_ranges
```

Resume validation:

```text
completed = 0
skipped_completed = 2
failed = 0
datasets = 49
```

Validation sentinels:

```text
ALL_PHASE_D_POLICIES_VALIDATED
ALL PHASE D POLICIES VALIDATED ON TESTING CAMPAIGN
```

### 63.9 D8.3 repeatable ML/SciML lag extension

The first P1 production realization requires three lag values in one canonical Phase D run:

```text
lag 1 = 5 minutes
lag 3 = 15 minutes
lag 6 = 30 minutes
```

D8.3 makes `--ml-input-lag` repeatable while retaining `12` as the compatibility default if no value is supplied.

Example:

```text
--ml-input-lag 1
--ml-input-lag 3
--ml-input-lag 6
```

The ML realization set becomes:

```text
selected ML policies x selected lags
```

and remains collision-free through existing folders:

```text
l1_h1/
l3_h1/
l6_h1/
```

The complete lag list is part of the configuration-aware resume identity.

Focused reconstructed test result before delivery:

```text
28 passed
1 skipped
```

The full thermal-modeling suite must be rerun after D8.3 is applied to the working repository.

### 63.10 First full P1 Phase D production configuration

Target:

```text
machine:
    lab-pc

environment:
    scalebridge-dev-gpu-labpc

campaign:
    p1_compact_4b4c_labpc_1w_v1

matrix:
    aggregation_matrix_20260712_215839
```

ML/SciML:

```text
policy:
    mdh

train/test/validation:
    0.70 / 0.15 / 0.15

lags:
    1, 3, 6

target horizon:
    1
```

Opt/Bayes:

```text
policy:
    sd

predefined seasons:
    winter, spring, summer, fall

global season offset:
    30 days

training:
    21 contiguous days/season

testing:
    next 7 contiguous days/season

lag/horizon:
    1/1
```

Full command is included in the Phase D complete handoff file and should only be launched after D8.3 full-suite validation.

