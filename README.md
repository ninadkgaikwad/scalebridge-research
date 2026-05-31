# ScaleBridge Research Platform

PyTorch-first research software for scalable building thermal modeling,
grey-box/Bayesian estimation, building-grid co-simulation, control, and
Neuromancer-compatible scientific ML workflows.

## Core rule

Legacy scripts are reference logic only. Production code is rebuilt as a clean,
modular package under `src/scalebridge`.

## Main technology choices

- PyTorch for ML/SciML models
- Neuromancer-compatible model/control components
- MLflow for experiment tracking
- Optuna/Ray Tune-compatible hyperparameter tuning
- Paper-driven experiment workspaces
- Database-ready input/output architecture
