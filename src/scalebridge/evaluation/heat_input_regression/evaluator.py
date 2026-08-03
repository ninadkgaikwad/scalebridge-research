# -*- coding: utf-8 -*-
"""Stage C7 evaluation for persisted heat-input regression models.

C7 never refits. It loads the C6 model artifact, evaluates the original C4
train/validation/test Parquets, writes metrics, prediction/residual tables,
and a complete provenance manifest.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scalebridge.models.heat_input_regression import load_heat_input_regression_model


@dataclass(frozen=True)
class TrainingArtifactReference:
    training_manifest_path: Path
    training_dir: Path
    model_artifact_dir: Path
    case_id: str
    aggregation_id: str
    weight_mode: str
    aggregate_zone_id: str
    model_id: str
    estimator_type: str
    requested_device: str
    resolved_device: str
    train_path: Path
    validation_path: Path
    test_path: Path
    fit_intercept: bool
    model_role: str
    input_transform: str
    dependency_model_id: str
    target_allocation: str
    source_dataset_manifest_payload: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    evaluation_dir: Path
    metrics_path: Path
    manifest_path: Path
    estimator_type: str
    model_id: str
    aggregate_zone_id: str
    requested_device: str
    resolved_device: str
    error_type: str = ""
    error_message: str = ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(raw_value: Any, *, base_dir: Path, label: str, require_file: bool) -> Path:
    """Resolve a manifest path and reject empty values/directories early."""
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError(f"Missing path value for {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if require_file and not path.is_file():
        raise IsADirectoryError(f"Expected {label} to be a file, got: {path}")
    if not require_file and not path.is_dir():
        raise NotADirectoryError(f"Expected {label} to be a directory, got: {path}")
    return path


def discover_training_artifacts(
    training_root: Path,
    model_ids: Iterable[str] | None = None,
    aggregate_zone_ids: Iterable[str] | None = None,
    estimator_types: Iterable[str] | None = None,
    requested_devices: Iterable[str] | None = None,
    max_artifacts: int | None = None,
) -> list[TrainingArtifactReference]:
    model_filter = set(model_ids or [])
    zone_filter = set(aggregate_zone_ids or [])
    estimator_filter = set(estimator_types or [])
    device_filter = set(requested_devices or [])
    refs: list[TrainingArtifactReference] = []

    for manifest_path in sorted(training_root.rglob("training_manifest.json")):
        manifest = _load_json(manifest_path)
        model_id = str(manifest.get("model_id", ""))
        zone_id = str(manifest.get("aggregate_zone_id", ""))
        estimator = str(manifest.get("estimator_type", ""))
        requested_device = str(manifest.get("requested_device", manifest.get("device", "cpu")))
        resolved_device = str(manifest.get("resolved_device", manifest.get("device", requested_device)))
        if model_filter and model_id not in model_filter:
            continue
        if zone_filter and zone_id not in zone_filter:
            continue
        if estimator_filter and estimator not in estimator_filter:
            continue
        if device_filter and requested_device not in device_filter:
            continue

        training_dir = manifest_path.parent
        training_outputs = manifest.get("outputs", {}) or {}
        dataset_payload = manifest.get("source_dataset_manifest_payload", {}) or {}
        dataset_outputs = dataset_payload.get("outputs", {}) or {}

        model_artifact_raw = (
            training_outputs.get("model_artifact_dir")
            or manifest.get("artifact_dir")
            or manifest.get("model_artifact_dir")
            or (training_dir / "model_artifact")
        )
        model_artifact_dir = _resolve_manifest_path(
            model_artifact_raw,
            base_dir=training_dir,
            label="C6 model artifact directory",
            require_file=False,
        )

        train_raw = (
            dataset_outputs.get("train")
            or manifest.get("source_train_path")
            or dataset_payload.get("source_train_path")
        )
        validation_raw = (
            dataset_outputs.get("validation")
            or manifest.get("source_validation_path")
            or dataset_payload.get("source_validation_path")
        )
        test_raw = (
            dataset_outputs.get("test")
            or manifest.get("source_test_path")
            or dataset_payload.get("source_test_path")
        )
        dataset_base_dir = Path(manifest.get("source_dataset_manifest", training_dir)).parent
        train_path = _resolve_manifest_path(
            train_raw, base_dir=dataset_base_dir, label="C4 train Parquet", require_file=True
        )
        validation_path = _resolve_manifest_path(
            validation_raw, base_dir=dataset_base_dir, label="C4 validation Parquet", require_file=True
        )
        test_path = _resolve_manifest_path(
            test_raw, base_dir=dataset_base_dir, label="C4 test Parquet", require_file=True
        )

        refs.append(TrainingArtifactReference(
            training_manifest_path=manifest_path,
            training_dir=training_dir,
            model_artifact_dir=model_artifact_dir,
            case_id=str(manifest.get("case_id", "")),
            aggregation_id=str(manifest.get("aggregation_id", "")),
            weight_mode=str(manifest.get("weight_mode", "")),
            aggregate_zone_id=zone_id,
            model_id=model_id,
            estimator_type=estimator,
            requested_device=requested_device,
            resolved_device=resolved_device,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            fit_intercept=bool(dataset_payload.get("fit_intercept", manifest.get("fit_intercept", False))),
            model_role=str(dataset_payload.get("model_role", manifest.get("model_role", ""))),
            input_transform=str(dataset_payload.get("input_transform", manifest.get("input_transform", "identity"))),
            dependency_model_id=str(dataset_payload.get("dependency_model_id", manifest.get("dependency_model_id", ""))),
            target_allocation=str(dataset_payload.get("target_allocation", manifest.get("target_allocation", "none"))),
            source_dataset_manifest_payload=dict(dataset_payload),
        ))
        if max_artifacts is not None and len(refs) >= max_artifacts:
            break
    return refs


def _safe_r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    residual = p - y
    mse = float(np.mean(residual ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(residual)))
    mbe = float(np.mean(residual))
    max_abs = float(np.max(np.abs(residual)))
    y_range = float(np.max(y) - np.min(y))
    y_mean_abs = float(np.mean(np.abs(y)))
    nrmse_range = rmse / y_range if y_range > 0 else float("nan")
    nrmse_mean = rmse / y_mean_abs if y_mean_abs > 0 else float("nan")
    return {
        "row_count": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "r2": _safe_r2(y, p),
        "mean_bias_error": mbe,
        "max_absolute_error": max_abs,
        "nrmse_by_range": nrmse_range,
        "nrmse_by_mean_abs_target": nrmse_mean,
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "target_mean": float(np.mean(y)),
        "prediction_min": float(np.min(p)),
        "prediction_max": float(np.max(p)),
        "prediction_mean": float(np.mean(p)),
    }


def _load_split(path: Path) -> pd.DataFrame:
    columns = ["x", "y"]
    if not path.is_file():
        raise FileNotFoundError(f"Expected exact split Parquet file, got: {path}")
    frame = pd.read_parquet(path)
    for col in columns:
        if col not in frame.columns:
            raise KeyError(f"Required column {col!r} missing from {path}")
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    if len(frame) == 0:
        raise ValueError(f"No rows in split file: {path}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite x/y values found in {path}")
    return frame



def _find_matching_dependency_artifact(ref: TrainingArtifactReference) -> tuple[Any, dict[str, Path]]:
    """Resolve the matching QAC artifact and split paths for PHVAC chaining."""
    if not ref.dependency_model_id:
        raise ValueError(f"No dependency_model_id recorded for {ref.model_id}")
    root = ref.training_manifest_path
    while root.parent != root and not (root / "training_run_manifest.json").is_file():
        root = root.parent
    candidates: list[Path] = []
    for path in root.rglob("training_manifest.json"):
        payload = _load_json(path)
        if (
            str(payload.get("case_id", "")) == ref.case_id
            and str(payload.get("aggregation_id", "")) == ref.aggregation_id
            and str(payload.get("weight_mode", "")) == ref.weight_mode
            and str(payload.get("aggregate_zone_id", "")) == ref.aggregate_zone_id
            and str(payload.get("model_id", "")) == ref.dependency_model_id
            and str(payload.get("estimator_type", "")) == ref.estimator_type
            and str(payload.get("requested_device", payload.get("device", "cpu"))) == ref.requested_device
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one matching {ref.dependency_model_id} artifact for PHVAC chaining; "
            f"found {len(candidates)}"
        )
    payload = _load_json(candidates[0])
    outputs = payload.get("outputs", {}) or {}
    artifact = Path(outputs.get("model_artifact_dir") or payload.get("artifact_dir"))
    dataset = payload.get("source_dataset_manifest_payload", {}) or {}
    doutputs = dataset.get("outputs", {}) or {}
    paths = {name: Path(doutputs[name]) for name in ("train", "validation", "test")}
    return load_heat_input_regression_model(artifact), paths


def build_phvac_building_reconstruction(evaluation_root: str | Path) -> Path | None:
    """Sum aggregate-zone PHVAC outputs to reconstruct building HVAC power."""
    root = Path(evaluation_root)
    manifests = []
    for path in root.rglob("evaluation_manifest.json"):
        payload = _load_json(path)
        if str(payload.get("model_id", "")) == "PHVAC":
            manifests.append((path, payload))
    if not manifests:
        return None
    records: list[dict[str, Any]] = []
    group_keys: dict[tuple[str, ...], list[tuple[Path, dict[str, Any]]]] = {}
    for item in manifests:
        payload = item[1]
        key = tuple(str(payload.get(k, "")) for k in (
            "case_id", "aggregation_id", "weight_mode", "estimator_type", "requested_device"
        ))
        group_keys.setdefault(key, []).append(item)
    out_root = root / "building_phvac_reconstruction"
    out_root.mkdir(parents=True, exist_ok=True)
    for key, items in group_keys.items():
        for split in ("train", "validation", "test"):
            frames = []
            for manifest_path, payload in items:
                pred_path = Path((payload.get("prediction_paths", {}) or {}).get(split, ""))
                if not pred_path.is_file():
                    continue
                frame = pd.read_parquet(pred_path)
                keep = [c for c in ("timestamp_raw", "timestamp", "y", "prediction", "prediction_chained") if c in frame]
                current = frame[keep].copy()
                current = current.rename(columns={
                    "y": f"y__{payload.get('aggregate_zone_id')}",
                    "prediction": f"oracle__{payload.get('aggregate_zone_id')}",
                    "prediction_chained": f"chained__{payload.get('aggregate_zone_id')}",
                })
                frames.append(current)
            if not frames:
                continue
            merged = frames[0]
            for frame in frames[1:]:
                merged = merged.merge(frame, on=["timestamp_raw", "timestamp"], how="inner", validate="one_to_one")
            ycols = [c for c in merged if c.startswith("y__")]
            ocols = [c for c in merged if c.startswith("oracle__")]
            ccols = [c for c in merged if c.startswith("chained__")]
            merged["building_target"] = merged[ycols].sum(axis=1)
            merged["building_prediction_oracle"] = merged[ocols].sum(axis=1)
            if ccols:
                merged["building_prediction_chained"] = merged[ccols].sum(axis=1)
            group_dir = out_root.joinpath(*[_safe_token(x) for x in key])
            group_dir.mkdir(parents=True, exist_ok=True)
            path = group_dir / f"{split}_building_phvac.parquet"
            merged.to_parquet(path, index=False)
            for mode, col in (("oracle", "building_prediction_oracle"), ("chained", "building_prediction_chained")):
                if col not in merged:
                    continue
                metrics = _metrics(merged["building_target"].to_numpy(float), merged[col].to_numpy(float))
                records.append({
                    "case_id": key[0], "aggregation_id": key[1], "weight_mode": key[2],
                    "estimator_type": key[3], "requested_device": key[4], "split": split,
                    "evaluation_mode": mode, "aggregate_zone_count": len(items), **metrics,
                    "predictions_path": str(path),
                })
    metrics_path = out_root / "building_phvac_metrics.csv"
    pd.DataFrame(records).to_csv(metrics_path, index=False)
    return metrics_path


def _safe_token(value: Any) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def evaluate_training_artifact(
    ref: TrainingArtifactReference,
    evaluation_dir: Path,
    prediction_rows: int = 200,
    write_full_predictions: bool = True,
) -> EvaluationResult:
    started = time.perf_counter()
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    model = load_heat_input_regression_model(ref.model_artifact_dir)

    split_paths = {
        "train": ref.train_path,
        "validation": ref.validation_path,
        "test": ref.test_path,
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_paths: dict[str, str] = {}
    preview_paths: dict[str, str] = {}

    for split, path in split_paths.items():
        frame = _load_split(path)
        pred = model.predict(frame["x"].to_numpy(dtype=float))
        if len(pred) != len(frame) or not np.all(np.isfinite(pred)):
            raise ValueError(f"Invalid predictions for {split}: {path}")
        metrics = _metrics(frame["y"].to_numpy(dtype=float), pred)
        metric_rows.append({"split": split, "evaluation_mode": "oracle" if ref.model_id == "PHVAC" else "direct", **metrics})

        chained_prediction = None
        if ref.model_id == "PHVAC":
            dependency_model, dependency_paths = _find_matching_dependency_artifact(ref)
            dependency_frame = _load_split(dependency_paths[split])
            if "timestamp_raw" in frame.columns and "timestamp_raw" in dependency_frame.columns:
                if not frame["timestamp_raw"].astype(str).reset_index(drop=True).equals(dependency_frame["timestamp_raw"].astype(str).reset_index(drop=True)):
                    raise ValueError("PHVAC and QAC split timestamps do not align")
            qhvac_predicted = dependency_model.predict(dependency_frame["x"].to_numpy(dtype=float))
            chained_prediction = model.predict(np.abs(qhvac_predicted))
            chained_metrics = _metrics(frame["y"].to_numpy(dtype=float), chained_prediction)
            metric_rows.append({"split": split, "evaluation_mode": "chained", **chained_metrics})

        out = pd.DataFrame({
            "timestamp_raw": frame["timestamp_raw"] if "timestamp_raw" in frame.columns else "",
            "timestamp": frame["timestamp"] if "timestamp" in frame.columns else pd.NaT,
            "x": frame["x"].to_numpy(dtype=float),
            "y": frame["y"].to_numpy(dtype=float),
            "prediction": pred,
            "residual": pred - frame["y"].to_numpy(dtype=float),
            "absolute_error": np.abs(pred - frame["y"].to_numpy(dtype=float)),
        })
        if chained_prediction is not None:
            out["prediction_chained"] = chained_prediction
            out["residual_chained"] = chained_prediction - frame["y"].to_numpy(dtype=float)
            out["absolute_error_chained"] = np.abs(out["residual_chained"])
        preview_path = evaluation_dir / f"{split}_prediction_preview.csv"
        out.head(prediction_rows).to_csv(preview_path, index=False)
        preview_paths[split] = str(preview_path)
        if write_full_predictions:
            full_path = evaluation_dir / f"{split}_predictions.parquet"
            out.to_parquet(full_path, index=False)
            prediction_paths[split] = str(full_path)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_path = evaluation_dir / "split_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    generalization = pd.DataFrame([
        {
            "metric": metric,
            "train": float(metrics_frame.loc[metrics_frame["split"] == "train", metric].iloc[0]),
            "validation": float(metrics_frame.loc[metrics_frame["split"] == "validation", metric].iloc[0]),
            "test": float(metrics_frame.loc[metrics_frame["split"] == "test", metric].iloc[0]),
        }
        for metric in ["rmse", "mae", "r2", "mean_bias_error", "nrmse_by_range"]
    ])
    generalization["validation_minus_train"] = generalization["validation"] - generalization["train"]
    generalization["test_minus_validation"] = generalization["test"] - generalization["validation"]
    generalization_path = evaluation_dir / "generalization_summary.csv"
    generalization.to_csv(generalization_path, index=False)

    model_manifest_path = ref.model_artifact_dir / "model_manifest.json"
    model_manifest = _load_json(model_manifest_path)
    manifest = {
        "evaluation_schema_version": 1,
        "case_id": ref.case_id,
        "aggregation_id": ref.aggregation_id,
        "weight_mode": ref.weight_mode,
        "aggregate_zone_id": ref.aggregate_zone_id,
        "model_id": ref.model_id,
        "estimator_type": ref.estimator_type,
        "requested_device": ref.requested_device,
        "resolved_device": ref.resolved_device,
        "coefficient": model_manifest.get("coefficient"),
        "intercept": model_manifest.get("intercept"),
        "fit_intercept": ref.fit_intercept,
        "model_role": ref.model_role,
        "input_transform": ref.input_transform,
        "dependency_model_id": ref.dependency_model_id,
        "target_allocation": ref.target_allocation,
        "evaluation_modes": ["oracle", "chained"] if ref.model_id == "PHVAC" else ["direct"],
        "source_training_manifest": str(ref.training_manifest_path),
        "source_model_artifact": str(ref.model_artifact_dir),
        "source_split_paths": {k: str(v) for k, v in split_paths.items()},
        "split_metrics_path": str(metrics_path),
        "generalization_summary_path": str(generalization_path),
        "prediction_paths": prediction_paths,
        "preview_paths": preview_paths,
        "write_full_predictions": bool(write_full_predictions),
        "runtime_seconds": time.perf_counter() - started,
    }
    manifest_path = evaluation_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return EvaluationResult(
        status="completed",
        evaluation_dir=evaluation_dir,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        estimator_type=ref.estimator_type,
        model_id=ref.model_id,
        aggregate_zone_id=ref.aggregate_zone_id,
        requested_device=ref.requested_device,
        resolved_device=ref.resolved_device,
    )
