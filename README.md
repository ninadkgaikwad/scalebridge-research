# ScaleBridge Research

**Repository:** `scalebridge-research`  
**Python package:** `scalebridge`  
**Project context:** PhD_Code_Framework / ScaleBridge research software stack  
**Primary current focus:** P1 EnergyPlus data generation and compact campaign validation  
**Current date context:** July 2026  

ScaleBridge is a professional research-software framework for scalable building thermal modeling, EnergyPlus data generation, one-zone commercial building datasets, grey-box and Bayesian estimation, scientific machine learning, PyTorch baselines, MLflow experiment tracking, automated hyperparameter tuning, and later building-grid co-simulation and control experiments.

The repository is being developed to support the PhD paper/dissertation workflow, especially the current P1 paper pipeline:

> Benchmarking black-box, sequence-learning, and scientific machine learning models for scalable one-zone commercial building thermal dynamics across DOE/PNNL ASHRAE 90.1-2013 prototype buildings.

---

## 1. Current Development Snapshot

The current validated milestone is the **P1 compact EnergyPlus variable-wise generation campaign foundation**.

The latest validated code state includes:

- Professional repository/package organization under `scalebridge-research`
- Python package `src/scalebridge`
- PyTorch-first modeling direction
- MLflow as first-class tracking infrastructure
- EnergyPlus v9.0.1 integration
- `opyplus`-based IDF preparation
- Variable-wise EnergyPlus generation
- Canonical per-variable parquet output
- Optional per-variable legacy pickle output
- Raw EnergyPlus CSV deletion after successful canonical conversion
- Short EnergyPlus work root to avoid Windows path issues
- MLflow duplicate-run fix
- Parallel variable-wise workers
- Compact P1 campaign runner
- Pre-opyplus IDF normalization for `ApartmentMidRise`
- Four-machine environment-variable contract
- Successful compileall and dry-run validation for compact campaign with:
  - 16 cases
  - 2 variables
  - 2 parallel variable workers

Latest validated dry-run result:

```text
compileall_exit_code = 0
dry_run_exit_code = 0
selected_case_count = 16
variable_limit = 2
parallel_variable_workers = 2
```

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
| `lab-pc` | Main Windows compute target for the final compact campaign |
| `kamiak` | WSU HPC / SLURM / high-end GPU compute |

The intended workflow is:

1. Develop and commit code primarily on the laptop.
2. Pull the repo on home PC, lab PC, and Kamiak.
3. Keep generated data outside the repo.
4. Keep conda environments machine-local.
5. Use tracked scripts, knowledgebase files, and README instructions to reproduce validation.
6. Use MLflow to track machine/run identity.
7. Use lab PC for the final compact campaign once dry-run and smoke tests pass.

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

Current Windows laptop path example:

```text
C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge
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

Temporary EnergyPlus execution folders use a short local work root:

```text
SCALEBRIDGE_EPLUS_WORK_ROOT = C:\ScaleBridge_EPlus_Work
```

This folder is temporary and can be cleaned when no EnergyPlus jobs are running.

---

## 5. Environment Variables

### Windows PowerShell setup

Run from repo root.

For lab PC final campaign work:

```powershell
$repoRoot = (Resolve-Path ".").Path

$env:SCALEBRIDGE_DATA_ROOT = (Resolve-Path (Join-Path $repoRoot "..\..\Data")).Path
$env:SCALEBRIDGE_GENERATED_DATA_ROOT = (Resolve-Path (Join-Path $repoRoot "..\..\Data\ScaleBridge")).Path
$env:SCALEBRIDGE_EPLUS_WORK_ROOT = "C:\ScaleBridge_EPlus_Work"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:SCALEBRIDGE_MACHINE_ID = "lab-pc"
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
→ EnergyPlus simulation outputs
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
  traceback.txt
```

Parallel variable workers run independent EnergyPlus simulations concurrently. They do not make a single EnergyPlus simulation multithreaded.

Recommended current workers:

| Machine | Workers |
|---|---:|
| laptop | 2 |
| home-pc | 2 initially |
| lab-pc | 2 initially |
| kamiak | 1 initially until EnergyPlus/HPC behavior is tested |

---

## 12. MLflow Architecture

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

MLflow export path:

```text
Data/ScaleBridge/mlflow_exports/<machine_id>/
```

Merged registry path:

```text
Data/ScaleBridge/experiment_registry/
```

For lab PC final campaign, there are two acceptable choices:

1. Use MLflow server at `http://127.0.0.1:5000`
2. Use a file store under generated data root:

```powershell
$env:MLFLOW_TRACKING_URI = "file:///$($env:SCALEBRIDGE_GENERATED_DATA_ROOT.Replace('\','/'))/mlflow"
```

This stores MLflow data under:

```text
Data/ScaleBridge/mlflow
```

---

## 13. Pre-opyplus IDF Normalization

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

## 14. P1 Compact Campaign

The compact campaign runner is:

```text
scripts/energyplus/run_p1_compact_campaign.py
```

Original compact campaign ID:

```text
p1_ashrae2013_one_zone_compact_4b4c
```

Recommended lab-PC final campaign ID:

```text
p1_ashrae2013_one_zone_compact_4b4c_labpc_v1
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

---

## 15. Building Selection Evidence

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

## 16. Validated Compact Dry Run

Latest validated dry run:

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --machine-id $env:SCALEBRIDGE_MACHINE_ID `
  --dry-run `
  --variable-limit 2 `
  --parallel-variable-workers 2
```

Result:

```text
compileall_exit_code = 0
dry_run_exit_code = 0
selected_case_count = 16
variable_limit = 2
parallel_variable_workers = 2
```

Selected cases:

```text
RestaurantFastFood: Buffalo, Seattle, Tampa, Tucson
OfficeSmall: Buffalo, Seattle, Tampa, Tucson
RetailStripmall: Buffalo, Seattle, Tampa, Tucson
ApartmentMidRise: Buffalo, Seattle, Tampa, Tucson
```

`ApartmentMidRise` received:

```text
inserted ScheduleTypeLimits: Control Type
```

Other compact buildings required no patch.

Normalized IDF paths were correctly written under:

```text
Data/ScaleBridge/campaigns/p1_ashrae2013_one_zone_compact_4b4c/normalization/idfs
```

---

## 17. Validation Commands

### Compile all

```powershell
python -m compileall src scripts
```

### Compact campaign dry run with 2 workers and 2 variables

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --machine-id $env:SCALEBRIDGE_MACHINE_ID `
  --dry-run `
  --variable-limit 2 `
  --parallel-variable-workers 2
```

Expected:

```text
selected_case_count: 16
variable_limit: 2
parallel_variable_workers: 2
```

### Lab-PC real 2-variable smoke test

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_v1 `
  --machine-id $env:SCALEBRIDGE_MACHINE_ID `
  --case-limit 1 `
  --variable-limit 2 `
  --parallel-variable-workers 2 `
  --write-legacy-pickles `
  --mlflow-strict
```

Expected first case:

```text
RestaurantFastFood / Buffalo
```

### Lab-PC first full 35-variable compact case

Only after the 2-variable smoke passes:

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_v1 `
  --machine-id $env:SCALEBRIDGE_MACHINE_ID `
  --case-limit 1 `
  --parallel-variable-workers 2 `
  --write-legacy-pickles `
  --mlflow-strict `
  --rerun-completed
```

### Full compact campaign

Only after the first full case passes:

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_v1 `
  --machine-id $env:SCALEBRIDGE_MACHINE_ID `
  --parallel-variable-workers 2 `
  --write-legacy-pickles `
  --mlflow-strict
```

---

## 18. Lab-PC Cleanup Policy Before Final Campaign

Clean only disposable generated/test artifacts.

Do **not** clean:

- repo source files
- knowledgebase files
- validated scripts
- original DOE/PNNL IDFs
- weather/source data
- environment lock documentation that you intend to keep

Clean only if confirmed disposable:

- old compact campaign test folder
- temporary EnergyPlus work root contents
- local MLflow test data

Preview campaign folder:

```powershell
$campaignId = "p1_ashrae2013_one_zone_compact_4b4c"
$campaignRoot = Join-Path $env:SCALEBRIDGE_GENERATED_DATA_ROOT "campaigns\$campaignId"

Write-Host $campaignRoot
Write-Host "Exists:" (Test-Path $campaignRoot)
```

Remove old test campaign folder:

```powershell
$campaignId = "p1_ashrae2013_one_zone_compact_4b4c"
$campaignRoot = Join-Path $env:SCALEBRIDGE_GENERATED_DATA_ROOT "campaigns\$campaignId"

if (Test-Path $campaignRoot) {
    Remove-Item -Recurse -Force $campaignRoot
}
```

Clear temporary EnergyPlus work root:

```powershell
if (Test-Path $env:SCALEBRIDGE_EPLUS_WORK_ROOT) {
    Get-ChildItem -Force $env:SCALEBRIDGE_EPLUS_WORK_ROOT | Remove-Item -Recurse -Force
} else {
    New-Item -ItemType Directory -Force -Path $env:SCALEBRIDGE_EPLUS_WORK_ROOT | Out-Null
}
```

MLflow cleanup if test-only:

```powershell
foreach ($folder in @("mlruns", "mlartifacts")) {
    if (Test-Path $folder) {
        Remove-Item -Recurse -Force $folder
    }
}
```

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

Before committing:

```powershell
git status --short
```

Remove repo-root scratch if present:

```powershell
Remove-Item -Recurse -Force scratch
```

Known recent `git status` item to inspect:

```text
D block_e_direct_validation.txt
```

This may be fine if the file was intentionally moved into `knowledgebase/`.

---

## 20. Important Current Source Files

| File | Purpose |
|---|---|
| `scripts/energyplus/run_p1_campaign.py` | Main P1 campaign runner with standard/variable-wise modes |
| `scripts/energyplus/run_p1_compact_campaign.py` | Compact 4-building × 4-weather campaign runner |
| `src/scalebridge/integration/energyplus/generation/variable_wise.py` | Variable-wise EnergyPlus generation |
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
6. Use lab PC for compact-campaign smoke and full campaign.
7. Use `--parallel-variable-workers 2` initially.
8. Use `--mlflow-strict` during validation.
9. Do not create repo-root scratch/data/output folders.
10. Do not commit generated artifacts.
11. Prefer `OfficeSmall/Seattle` or compact dry-run for quick validation.
12. For final compact lab-PC campaign, use:
    `p1_ashrae2013_one_zone_compact_4b4c_labpc_v1`.

---

## 22. Next Three Tasks

The user explicitly split the next work into three tasks:

```text
Task 1: Create a detailed README.md and context handoff
Task 2: Cleanup generated/test artifacts safely
Task 3: Run compact campaign smoke test on lab-PC with 2 workers and 2 variables
```

This README completes Task 1.

Task 2 should be done under user guidance and should clean only disposable test artifacts.

Task 3 should be run on lab-PC only after cleanup and environment validation.

---

## 23. Recommended Next Lab-PC Sequence

On lab-PC:

1. Pull latest repo.
2. Activate `scalebridge-dev-gpu`.
3. Set env vars from repo root.
4. Run `python -m compileall src scripts`.
5. Run compact dry-run:

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --machine-id lab-pc `
  --dry-run `
  --variable-limit 2 `
  --parallel-variable-workers 2
```

6. If dry-run passes, run real smoke:

```powershell
python scripts\energyplus\run_p1_compact_campaign.py `
  --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_v1 `
  --machine-id lab-pc `
  --case-limit 1 `
  --variable-limit 2 `
  --parallel-variable-workers 2 `
  --write-legacy-pickles `
  --mlflow-strict
```

7. Upload:
   - run terminal output
   - `latest_run.json`
   - canonical metadata for the run
   - MLflow run status if available

---

## 24. Useful Handoff Files

The following historical handoffs and reports were used to create this README:

```text
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

---

## 25. Current Status Statement

ScaleBridge has moved from repository/environment foundation into a validated EnergyPlus P1 compact-campaign foundation. The repo now has the key pieces needed to run a controlled compact campaign: variable-wise EnergyPlus generation, MLflow tracking, short-path work roots, pre-opyplus IDF normalization, and a compact campaign runner. The next execution target is lab-PC, where the final compact campaign should be validated first with a 2-variable, 2-worker smoke test before launching a full case or the full 16-case compact campaign.
