# ScaleBridge Package Design

Date: 2026-05-31  
Project: ScaleBridge Research  
Repository: scalebridge-research  
Python package: scalebridge  

## Purpose

ScaleBridge is a research software package for scalable building thermal modeling, grey-box and Bayesian estimation, building-grid co-simulation, and grid-interactive control.

The package supports PhD papers P1-P6 and the dissertation software workflow.

## Core Design Principles

- PyTorch-first modeling stack.
- Neuromancer-compatible scientific ML and control workflows.
- MLflow-based experiment tracking.
- Optuna and Ray Tune for automated hyperparameter tuning.
- EnergyPlus, OpenDSS, weather, and database integration.
- Reproducible paper-specific experiment pipelines.
- Legacy code used only as reference logic.

## Repository Structure

```text
scalebridge-research/
├── src/
├── experiments/
├── configs/
├── scripts/
├── docs/
├── legacy_reference/
├── tests/
├── notebooks/
├── data/
└── outputs/
```

## Package Structure

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

## Module Responsibilities

### core/

Shared infrastructure:

- configuration,
- path handling,
- logging,
- typing,
- registries,
- general utilities.

### data/

Data workflow utilities:

- schemas,
- loaders,
- preprocessing,
- splitting,
- scaling,
- feature construction,
- validation,
- dataset manifests.

### integration/

External tool and data integrations:

- EnergyPlus,
- EPW/weather,
- OpenDSS,
- external datasets,
- pricing and grid data.

### db/

Database layer:

- connection management,
- ORM models,
- schemas,
- repositories,
- services,
- migrations.

### models/

Model definitions only:

- black-box models,
- grey-box models,
- Bayesian models,
- scientific ML models,
- baselines.

Training loops should not live here.

### training/

Reusable training code:

- PyTorch datasets,
- DataLoaders,
- trainers,
- losses,
- optimizers,
- schedulers,
- callbacks,
- checkpointing.

### tracking/

Experiment tracking:

- MLflow clients,
- artifact logging,
- run metadata,
- experiment registry utilities.

### tuning/

Hyperparameter tuning:

- Optuna objectives,
- Ray Tune integration,
- search spaces,
- pruning,
- schedulers.

### evaluation/

Post-training analysis:

- metrics,
- diagnostics,
- visualization,
- benchmarking,
- paper tables,
- paper reports.

### simulators/

Simulation environments:

- residential simulator,
- commercial simulator,
- community simulator,
- co-simulation workflows,
- Gymnasium-compatible wrappers.

### control/

Control and optimization:

- MPC,
- optimization,
- distributed control,
- reinforcement learning,
- Neuromancer-compatible control.

## Experiments

Paper-specific workflows live under `experiments/`.

Experiment folders may contain:

- configs,
- scripts,
- notebooks,
- results,
- figures,
- tables,
- local MLflow runs.

Experiment scripts should call reusable package code from `src/scalebridge/`.

They should not contain production model definitions.

## Planned Paper Folders

```text
experiments/
├── p1_one_zone_commercial_benchmark/
├── p2_greybox_bayesian_estimation/
├── p3_building_grid_cosimulation/
├── p4_grid_interactive_control/
├── p5_review_assets/
└── p6_distributed_mpc_marl/
```

## Week 1 Development Direction

Week 1 should focus on:

- package import validation,
- core path/config utilities,
- P1 data-pipeline skeleton,
- minimal dataset schema,
- minimal PyTorch Dataset/DataLoader design,
- first smoke tests.

Deep model implementation should wait until the foundation is importable and testable.
