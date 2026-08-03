# -*- coding: utf-8 -*-
"""Reusable Stage C6 training functions for model-specific C4 datasets.

C6 fits one C5 estimator to one C4 ``train.parquet`` dataset, saves the
artifact, reloads it, and verifies prediction identity. It intentionally does
not perform C7 train/validation/test evaluation or model selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scalebridge.models.heat_input_regression import (
    create_heat_input_regression_model,
    load_heat_input_regression_model,
)


@dataclass(frozen=True)
class EstimatorTrainingConfig:
    """Configuration for one C5 estimator family."""

    estimator_type: str
    fit_intercept: bool | None = None
    ridge_alpha: float = 0.0
    learning_rate: float = 0.03
    max_epochs: int = 3000
    tolerance: float = 1e-10
    patience: int = 200
    seed: int = 42
    device: str = "cpu"

    def model_kwargs(self, *, model_id: str, metadata: dict[str, Any], fit_intercept: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "fit_intercept": bool(fit_intercept),
            "model_id": model_id,
            "metadata": metadata,
        }
        if self.estimator_type == "closed_form_linear":
            kwargs["ridge_alpha"] = self.ridge_alpha
        elif self.estimator_type == "pytorch_linear":
            kwargs.update(
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                tolerance=self.tolerance,
                patience=self.patience,
                seed=self.seed,
                device=self.device,
            )
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelDatasetReference:
    """Resolved reference to one C4 model-specific regression dataset."""

    manifest_path: Path
    dataset_root: Path
    model_dir: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    identity: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.identity.get("case_id", "unknown_case"))

    @property
    def aggregation_id(self) -> str:
        return str(
            self.identity.get("aggregation_id")
            or self.identity.get("aggregation_run_id")
            or "unknown_aggregation"
        )

    @property
    def aggregate_zone_id(self) -> str:
        return str(self.identity.get("aggregate_zone_id", "unknown_zone"))

    @property
    def weight_mode(self) -> str:
        return str(self.identity.get("weight_mode", "unknown_weight"))

    @property
    def model_id(self) -> str:
        return str(self.identity.get("model_id", self.model_dir.name))

    def identity_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "aggregation_id": self.aggregation_id,
            "aggregate_zone_id": self.aggregate_zone_id,
            "weight_mode": self.weight_mode,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class TrainingResult:
    """Structured outcome from one model-dataset-estimator training task."""

    row: dict[str, Any]
    output_dir: Path


def _safe_name(value: Any) -> str:
    text = str(value).strip() or "unnamed"
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in text)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_list).to_csv(path, index=False)


def _rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2)))


def discover_model_datasets(
    dataset_root: str | Path,
    *,
    model_ids: set[str] | None = None,
    aggregate_zone_ids: set[str] | None = None,
    max_model_datasets: int | None = None,
) -> list[ModelDatasetReference]:
    """Discover completed C4 model datasets from their manifests."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"C4 dataset root does not exist: {root}")

    references: list[ModelDatasetReference] = []
    for manifest_path in sorted(root.rglob("model_dataset_manifest.json")):
        manifest = _read_json(manifest_path)
        if str(manifest.get("status", "completed")) != "completed":
            continue
        model_dir = manifest_path.parent
        identity = dict(manifest)
        model_id = str(identity.get("model_id", model_dir.name))
        zone_id = str(identity.get("aggregate_zone_id", ""))
        if model_ids and model_id not in model_ids:
            continue
        if aggregate_zone_ids and zone_id not in aggregate_zone_ids:
            continue
        outputs = manifest.get("outputs", {})
        train_path = Path(outputs.get("train", model_dir / "train.parquet"))
        validation_path = Path(outputs.get("validation", model_dir / "validation.parquet"))
        test_path = Path(outputs.get("test", model_dir / "test.parquet"))
        for required in (train_path, validation_path, test_path):
            if not required.is_file():
                raise FileNotFoundError(f"C4 split dataset is missing: {required}")
        references.append(
            ModelDatasetReference(
                manifest_path=manifest_path,
                dataset_root=root,
                model_dir=model_dir,
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                identity=identity,
                manifest=manifest,
            )
        )
        if max_model_datasets is not None and len(references) >= max_model_datasets:
            break
    return references


def build_training_output_dir(
    training_root: Path,
    reference: ModelDatasetReference,
    estimator_type: str,
    device: str = "cpu",
) -> Path:
    """Build a stable, collision-resistant C6 output location."""

    return (
        training_root
        / "cases"
        / _safe_name(reference.case_id)
        / _safe_name(reference.aggregation_id)
        / _safe_name(reference.weight_mode)
        / _safe_name(reference.aggregate_zone_id)
        / _safe_name(reference.model_id)
        / _safe_name(
            f"{estimator_type}_{device}" if estimator_type == "pytorch_linear" else estimator_type
        )
    )


def train_model_dataset(
    reference: ModelDatasetReference,
    config: EstimatorTrainingConfig,
    *,
    training_root: str | Path,
    training_run_id: str,
    overwrite_existing: bool = False,
    reload_atol: float = 1e-12,
    reload_rtol: float = 1e-12,
    prediction_preview_rows: int = 100,
) -> TrainingResult:
    """Fit, save, reload, and verify one estimator on one C4 training set."""

    started = time.perf_counter()
    output_dir = build_training_output_dir(
        Path(training_root), reference, config.estimator_type, config.device
    )
    artifact_dir = output_dir / "model_artifact"
    training_manifest_path = output_dir / "training_manifest.json"
    if training_manifest_path.exists() and not overwrite_existing:
        raise FileExistsError(
            f"Training output already exists: {training_manifest_path}. "
            "Use --overwrite-existing only for an intentional replacement."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(reference.train_path, columns=["timestamp_raw", "timestamp", "x", "y"])
    if train.empty:
        raise ValueError(f"Training dataset is empty: {reference.train_path}")
    x = train["x"].to_numpy(dtype=np.float64)
    y = train["y"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("C4 train.parquet contains non-finite x or y values.")

    manifest_fit_intercept = bool(reference.manifest.get("fit_intercept", False))
    resolved_fit_intercept = (
        manifest_fit_intercept
        if config.fit_intercept is None
        else bool(config.fit_intercept)
    )
    intercept_policy_source = (
        "model_dataset_manifest"
        if config.fit_intercept is None
        else "explicit_training_override"
    )

    provenance = {
        **reference.identity_dict(),
        "training_run_id": training_run_id,
        "source_dataset_root": str(reference.dataset_root),
        "source_dataset_manifest": str(reference.manifest_path),
        "source_train_path": str(reference.train_path),
        "predictor_column": reference.manifest.get("predictor_column", "x"),
        "target_column": reference.manifest.get("target_column", "y"),
        "predictor_units": reference.manifest.get("predictor_units", ""),
        "target_units": reference.manifest.get("target_units", ""),
        "fit_intercept": resolved_fit_intercept,
        "intercept_policy_source": intercept_policy_source,
        "model_role": reference.manifest.get("model_role", ""),
        "input_transform": reference.manifest.get("input_transform", "identity"),
        "dependency_model_id": reference.manifest.get("dependency_model_id", ""),
        "target_allocation": reference.manifest.get("target_allocation", "none"),
    }
    model = create_heat_input_regression_model(
        config.estimator_type,
        **config.model_kwargs(model_id=reference.model_id, metadata=provenance, fit_intercept=resolved_fit_intercept),
    )
    model.fit(x, y)
    in_memory_prediction = model.predict(x)
    manifest_path = model.save(artifact_dir)

    reloaded = load_heat_input_regression_model(artifact_dir)
    reloaded_prediction = reloaded.predict(x)
    max_abs_difference = float(np.max(np.abs(in_memory_prediction - reloaded_prediction)))
    reload_predictions_match = bool(
        np.allclose(in_memory_prediction, reloaded_prediction, atol=reload_atol, rtol=reload_rtol)
    )
    if not reload_predictions_match:
        raise ValueError(
            "Saved/reloaded predictions do not match: "
            f"maximum absolute difference={max_abs_difference}."
        )

    training_rmse = _rmse(y, in_memory_prediction)
    fit_summary = model.fit_summary.to_dict()
    device = str(getattr(model, "resolved_device", "cpu") or "cpu")

    prediction_preview = train.head(prediction_preview_rows).copy()
    prediction_preview["prediction"] = in_memory_prediction[: len(prediction_preview)]
    prediction_preview["residual"] = prediction_preview["y"] - prediction_preview["prediction"]
    prediction_preview.to_csv(output_dir / "training_prediction_preview.csv", index=False)

    history_rows: list[dict[str, Any]] = []
    if config.estimator_type == "pytorch_linear":
        history_rows = [
            {"epoch": index + 1, "training_loss_standardized": float(loss)}
            for index, loss in enumerate(getattr(model, "loss_history", []))
        ]
    _write_csv(output_dir / "training_history.csv", history_rows)

    reload_rows = [
        {
            "check_name": "artifact_manifest_exists",
            "status": "passed" if manifest_path.is_file() else "failed",
            "observed_value": str(manifest_path),
            "expected_value": "existing file",
        },
        {
            "check_name": "reload_predictions_match",
            "status": "passed" if reload_predictions_match else "failed",
            "observed_value": max_abs_difference,
            "expected_value": f"allclose(atol={reload_atol}, rtol={reload_rtol})",
        },
        {
            "check_name": "reload_coefficient_match",
            "status": "passed" if np.isclose(model.coefficient, reloaded.coefficient, atol=0.0, rtol=0.0) else "failed",
            "observed_value": reloaded.coefficient,
            "expected_value": model.coefficient,
        },
        {
            "check_name": "reload_intercept_match",
            "status": "passed" if np.isclose(model.intercept, reloaded.intercept, atol=0.0, rtol=0.0) else "failed",
            "observed_value": reloaded.intercept,
            "expected_value": model.intercept,
        },
    ]
    _write_csv(output_dir / "reload_validation.csv", reload_rows)

    summary = {
        **reference.identity_dict(),
        "training_run_id": training_run_id,
        "estimator_type": config.estimator_type,
        "requested_device": config.device if config.estimator_type == "pytorch_linear" else "cpu",
        "fit_intercept": resolved_fit_intercept,
        "manifest_fit_intercept": manifest_fit_intercept,
        "intercept_policy_source": intercept_policy_source,
        "model_role": reference.manifest.get("model_role", ""),
        "input_transform": reference.manifest.get("input_transform", "identity"),
        "dependency_model_id": reference.manifest.get("dependency_model_id", ""),
        "target_allocation": reference.manifest.get("target_allocation", "none"),
        "device": device,
        "sample_count": int(len(train)),
        "coefficient": float(model.coefficient),
        "intercept": float(model.intercept),
        "training_rmse": training_rmse,
        "training_loss": float(fit_summary["training_loss"]),
        "converged": bool(fit_summary["converged"]),
        "epochs_completed": int(fit_summary["epochs_completed"]),
        "reload_predictions_match": reload_predictions_match,
        "reload_max_absolute_difference": max_abs_difference,
        "runtime_seconds": time.perf_counter() - started,
        "artifact_dir": str(artifact_dir),
        "model_manifest": str(manifest_path),
        "source_train_path": str(reference.train_path),
    }
    _write_csv(output_dir / "training_summary.csv", [summary])

    training_manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        **summary,
        "estimator_config": config.to_dict(),
        "fit_summary": fit_summary,
        "source_dataset_manifest_payload": reference.manifest,
        "outputs": {
            "model_artifact_dir": str(artifact_dir),
            "model_manifest": str(manifest_path),
            "training_summary": str(output_dir / "training_summary.csv"),
            "training_history": str(output_dir / "training_history.csv"),
            "training_prediction_preview": str(output_dir / "training_prediction_preview.csv"),
            "reload_validation": str(output_dir / "reload_validation.csv"),
        },
    }
    _write_json(training_manifest_path, training_manifest)
    return TrainingResult(row=summary, output_dir=output_dir)
