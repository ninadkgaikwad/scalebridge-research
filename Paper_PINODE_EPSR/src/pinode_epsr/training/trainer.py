from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from ..core.common import set_deterministic
from ..backends.neuromancer import scalar_objective_problem


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    max_epochs: int = 1000
    patience: int = 100
    min_delta: float = 1e-7
    gradient_clip_norm: float | None = 10.0
    seed: int = 42


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 20
    representative_max_windows: int = 256
    seed: int = 42
    direction: str = "minimize"
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class FrozenHyperparameters:
    method: str
    values: dict[str, Any]
    tuning_scope: str
    study_best_value: float
    seed: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FrozenHyperparameters":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def build_optimizer(parameters, config: OptimizationConfig):
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=config.learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer {config.optimizer!r}")


def optimize_steps(
    model: nn.Module,
    loss_closure: Callable[[], torch.Tensor],
    *,
    config: OptimizationConfig,
    steps: int | None = None,
    dataset_name: str = "train",
) -> list[float]:
    """Small deterministic trainer used by Patch-2 smoke tests and tuning trials."""

    set_deterministic(config.seed)
    # NeuroMANCER owns the trainable objective graph (Problem + PenaltyLoss);
    # PyTorch owns the numerical optimizer acting on that Problem's parameters.
    problem, data_loader = scalar_objective_problem(
        model, loss_closure, dataset_name=dataset_name
    )
    problem_input = next(iter(data_loader))
    expected_loss_key = f"{dataset_name}_loss"
    if problem_input.get("name") != dataset_name:
        raise RuntimeError(
            "NeuroMANCER named-data contract failed: expected batch name "
            f"{dataset_name!r}, received {problem_input.get('name')!r}"
        )
    optimizer = build_optimizer(problem.parameters(), config)
    n = config.max_epochs if steps is None else min(int(steps), config.max_epochs)
    history: list[float] = []
    best = float("inf")
    stale = 0
    for _ in range(n):
        optimizer.zero_grad(set_to_none=True)
        output = problem(problem_input)
        if expected_loss_key not in output:
            raise KeyError(
                f"NeuroMANCER Problem did not return {expected_loss_key!r}; "
                f"available keys: {sorted(output)}"
            )
        loss = output[expected_loss_key]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append(value)
        if value < best - config.min_delta:
            best = value
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    return history


def suggest_inverse_pinn_hyperparameters(trial) -> dict[str, Any]:
    """Day-3 representative-data Optuna search space for Part-2."""

    return {
        "hidden_layers": trial.suggest_int("hidden_layers", 1, 4),
        "hidden_width": trial.suggest_categorical("hidden_width", [16, 32, 64, 128]),
        "activation": trial.suggest_categorical("activation", ["tanh", "silu", "gelu"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
        "lambda_y": 1.0,
        "lambda_f": trial.suggest_float("lambda_f", 1e-2, 10.0, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop"]),
    }


def suggest_node_hyperparameters(trial, *, rc_order: int) -> dict[str, Any]:
    """Part-3 search space; architecture size is explicitly tunable and later frozen."""

    values = {
        "hidden_layers": trial.suggest_int("hidden_layers", 1, 4),
        "hidden_width": trial.suggest_categorical("hidden_width", [16, 32, 64, 128]),
        "activation": trial.suggest_categorical("activation", ["tanh", "silu", "gelu"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop"]),
        "N_r": trial.suggest_categorical("N_r", [1, 3, 6, 12]),
        "N_s": trial.suggest_categorical("N_s", [1, 2, 5, 10]),
        "lambda_wd": trial.suggest_float("lambda_wd", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
    }
    if rc_order == 2:
        values.update(
            {
                "L_e": trial.suggest_categorical("L_e", [3, 6, 12]),
                "delta_T_m_max": trial.suggest_float("delta_T_m_max", 2.0, 15.0),
            }
        )
    return values


def suggest_base_pinode_hyperparameters(trial, *, rc_order: int) -> dict[str, Any]:
    """Part-4 Base-PINODE representative-data Optuna search space.

    The search keeps ``lambda_y=1`` as the reference data-loss scale and tunes
    the soft-physics weight ``lambda_f`` together with the neural architecture,
    rollout horizon, RK4 substeps, optimizer, and 2C causal-encoder settings.
    Selected values are frozen before later full-data training.
    """

    if rc_order not in (1, 2):
        raise ValueError("rc_order must be 1 or 2")
    values = {
        "hidden_layers": trial.suggest_int("hidden_layers", 1, 4),
        "hidden_width": trial.suggest_categorical("hidden_width", [16, 32, 64, 128]),
        "activation": trial.suggest_categorical("activation", ["tanh", "silu", "gelu"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop"]),
        "N_r": trial.suggest_categorical("N_r", [1, 3, 6, 12]),
        "N_s": trial.suggest_categorical("N_s", [1, 2, 5, 10]),
        "lambda_f": trial.suggest_float("lambda_f", 1e-4, 10.0, log=True),
        "lambda_wd": trial.suggest_float("lambda_wd", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
    }
    if rc_order == 2:
        values.update(
            {
                "L_e": trial.suggest_categorical("L_e", [3, 6, 12]),
                "delta_T_m_max": trial.suggest_float("delta_T_m_max", 2.0, 15.0),
            }
        )
    return values


def suggest_ebp_pinode_hyperparameters(trial, *, rc_order: int) -> dict[str, Any]:
    """Part-5 EBP-PINODE representative-training-only Optuna search space.

    The primary capacitance-squared projection metric is fixed by the Part-5
    contract.  Tuning therefore covers neural architecture, rollout/integration
    settings, optimizer, weight decay, and EBP-specific ``lambda_int`` /
    ``lambda_corr``.  ``lambda_int`` is only used for 2C because 1C has no
    remaining internal derivative freedom after hard projection.
    """

    if rc_order not in (1, 2):
        raise ValueError("rc_order must be 1 or 2")
    values = {
        "hidden_layers": trial.suggest_int("hidden_layers", 1, 4),
        "hidden_width": trial.suggest_categorical("hidden_width", [16, 32, 64, 128]),
        "activation": trial.suggest_categorical("activation", ["tanh", "silu", "gelu"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop"]),
        "N_r": trial.suggest_categorical("N_r", [1, 3, 6, 12]),
        "N_s": trial.suggest_categorical("N_s", [1, 2, 5, 10]),
        "lambda_corr": trial.suggest_float("lambda_corr", 1e-6, 1.0, log=True),
        "lambda_wd": trial.suggest_float("lambda_wd", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
    }
    if rc_order == 2:
        values.update(
            {
                "lambda_int": trial.suggest_float("lambda_int", 1e-4, 10.0, log=True),
                "L_e": trial.suggest_categorical("L_e", [3, 6, 12]),
                "delta_T_m_max": trial.suggest_float("delta_T_m_max", 2.0, 15.0),
            }
        )
    return values


def run_optuna_tuning(
    objective: Callable[[Any], float],
    *,
    method: str,
    tuning_scope: str,
    config: TuningConfig,
) -> tuple[Any, FrozenHyperparameters]:
    """Run Optuna only on the supplied representative TRAINING data objective.

    The caller owns construction of the training-only subset and validation
    slice. Test data must not enter this objective. The returned values are
    serialized/frozen before full-data training.
    """

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("The PINODE/EPSR paper workflow requires Optuna for hyperparameter tuning") from exc

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction=config.direction, sampler=sampler)
    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds)
    frozen = FrozenHyperparameters(
        method=method,
        values=dict(study.best_trial.params),
        tuning_scope=tuning_scope,
        study_best_value=float(study.best_value),
        seed=config.seed,
    )
    return study, frozen


def assert_training_only_indices(indices: np.ndarray, partition: np.ndarray) -> None:
    inds = np.asarray(indices, dtype=int)
    part = np.asarray(partition, dtype=str)
    bad = inds[part[inds] != "train"]
    if bad.size:
        raise ValueError(f"Optuna representative subset leaked non-training rows: {bad[:10].tolist()}")
