from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from ..core.common import write_json
from ..training.experiment import build_paper_model, suggest_method_hyperparameters
from ..evaluation.runtime import PaperModelRuntime
from ..training.trainer import FrozenHyperparameters, OptimizationConfig
from .contracts import HPOConfig
from .matrix import ExperimentSpec
from .objectives import score_temperature_objective
from .sampling import HPO_SAMPLING_PROTOCOL_VERSION, MonthBalancedHPOSample, select_month_balanced_hpo_sample
from .provenance import stable_id
from .training import fit_model


class _GeometryRestrictedTrial:
    """Optuna Trial proxy that narrows only N_r/L_e categorical geometry.

    Method-owned search spaces stay authoritative.  The proxy merely intersects
    the existing categorical choices with the HPO protocol's declared maximum
    rollout/history geometry. This is used by micro32 to keep 0.5% HPO feasible
    without changing production search-space defaults.
    """

    def __init__(self, trial, *, max_rollout_steps: int, max_encoder_history_steps: int):
        self._trial = trial
        self.max_rollout_steps = int(max_rollout_steps)
        self.max_encoder_history_steps = int(max_encoder_history_steps)

    def suggest_categorical(self, name, choices):
        filtered = list(choices)
        if name == "N_r":
            filtered = [v for v in filtered if int(v) <= self.max_rollout_steps]
        elif name == "L_e":
            filtered = [v for v in filtered if int(v) <= self.max_encoder_history_steps]
        if not filtered:
            raise ValueError(
                f"HPO geometry restriction removed every choice for {name}: "
                f"original={list(choices)!r}, max_rollout_steps={self.max_rollout_steps}, "
                f"max_encoder_history_steps={self.max_encoder_history_steps}"
            )
        return self._trial.suggest_categorical(name, filtered)

    def __getattr__(self, name):
        return getattr(self._trial, name)


def hpo_protocol_id(spec: ExperimentSpec, config: HPOConfig) -> str:
    """Stable study identity for a fixed scientific HPO protocol.

    Trial count and timeout are execution budgets and intentionally excluded so
    the same study can be resumed with more trials.  Dataset percentage, inner
    holdout, objective, per-trial training budget and sampling geometry are part
    of the identity and therefore cannot be mixed silently.
    """
    payload = config.to_dict()
    payload["sampling_protocol_version"] = HPO_SAMPLING_PROTOCOL_VERSION
    payload.pop("n_trials", None)
    payload.pop("timeout_seconds", None)
    return stable_id("hpo_protocol", {"spec": spec.to_dict(), "protocol": payload})


def run_persistent_hpo(
    trajectory,
    spec: ExperimentSpec,
    *,
    output_dir: Path,
    config: HPOConfig,
) -> tuple[Any, FrozenHyperparameters, MonthBalancedHPOSample]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for production HPO") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    sample = select_month_balanced_hpo_sample(
        trajectory,
        train_percentage=config.train_percentage,
        holdout_percentage=config.holdout_percentage,
        conservative_N_r=config.max_rollout_steps,
        conservative_L_e=config.max_encoder_history_steps,
        blocks_per_month=config.sampling_blocks_per_month,
    )
    write_json(output_dir / "hpo_sample_manifest.json", sample.to_dict())

    db_path = output_dir / "study.db"
    storage = f"sqlite:///{db_path.as_posix()}"
    protocol_id = hpo_protocol_id(spec, config)
    study_name = f"{spec.configuration_id}__{protocol_id}"
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction="minimize", sampler=sampler,
    )

    def objective(trial):
        started = time.perf_counter()
        restricted_trial = _GeometryRestrictedTrial(
            trial,
            max_rollout_steps=config.max_rollout_steps,
            max_encoder_history_steps=config.max_encoder_history_steps,
        )
        hp = suggest_method_hyperparameters(
            spec.method, restricted_trial, rc_order=spec.rc_order
        )
        trial.set_user_attr(
            "hpo_geometry",
            {
                "max_rollout_steps": int(config.max_rollout_steps),
                "max_encoder_history_steps": int(config.max_encoder_history_steps),
            },
        )
        model, _ = build_paper_model(
            spec.method, trajectory, rc_order=spec.rc_order,
            train_indices=sample.fit_indices, hyperparameters=hp, seed=config.seed,
        )
        N_r = int(getattr(model.config, "N_r", 1))
        L_e = int(getattr(model.config, "L_e", 1)) if spec.rc_order == 2 and spec.method != "inverse_pinn" else 1
        fit_windows = sample.rollout_windows("fit", N_r=N_r, L_e=L_e, rc_order=spec.rc_order)
        hold_windows = sample.rollout_windows("holdout", N_r=N_r, L_e=L_e, rc_order=spec.rc_order)
        if not hold_windows:
            raise RuntimeError("No legal HPO holdout windows for sampled hyperparameters")
        opt = OptimizationConfig(
            learning_rate=float(hp.get("learning_rate", 1e-3)),
            optimizer=str(hp.get("optimizer", "adam")),
            max_epochs=config.max_epochs_per_trial,
            patience=config.patience,
            seed=config.seed + int(trial.number),
        )
        batch_size = int(hp.get("batch_size", config.max_batch_windows))
        outcome = fit_model(
            model, trajectory, fit_indices=sample.fit_indices,
            fit_windows=fit_windows, validation_windows=hold_windows,
            objective=config.objective, optimization=opt,
            batch_size=min(batch_size, config.max_batch_windows),
        )
        score = score_temperature_objective(
            PaperModelRuntime(model, trajectory), hold_windows, config.objective,
        )
        trial.set_user_attr("fit_outcome", outcome.to_dict())
        trial.set_user_attr("wall_seconds", float(time.perf_counter() - started))
        return float(score)

    already = len([t for t in study.trials if t.state.name == "COMPLETE"])
    remaining = max(0, int(config.n_trials) - already)
    if remaining:
        study.optimize(
            objective, n_trials=remaining, timeout=config.timeout_seconds,
            catch=(RuntimeError, ValueError, FloatingPointError),
        )
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError(f"No completed HPO trial for {study_name}")

    frozen = FrozenHyperparameters(
        method=spec.method, values=dict(study.best_trial.params),
        tuning_scope=(f"phase_d_train_month_balanced_{config.train_percentage:g}pct__"
                      f"objective_{config.objective}"),
        study_best_value=float(study.best_value), seed=config.seed,
    )
    frozen.save(output_dir / "frozen_hyperparameters.json")
    write_json(output_dir / "study_manifest.json", {
        "study_name": study_name,
        "protocol_id": protocol_id,
        "storage": str(db_path),
        "spec": spec.to_dict(),
        "config": config.to_dict(),
        "sample": sample.to_dict(),
        "completed_trials": len(completed),
        "best_trial": int(study.best_trial.number),
        "best_value": float(study.best_value),
    })
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)
    return study, frozen, sample
