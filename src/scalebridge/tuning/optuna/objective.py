from __future__ import annotations

from typing import Callable

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


def run_optuna_study(
    objective: Callable,
    study_name: str,
    direction: str = "minimize",
    n_trials: int = 50,
):
    """Run an Optuna study for automated hyperparameter tuning."""
    if optuna is None:
        raise ImportError("Optuna is not installed. Install with: pip install optuna")
    study = optuna.create_study(study_name=study_name, direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return study
