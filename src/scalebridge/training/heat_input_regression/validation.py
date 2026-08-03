# -*- coding: utf-8 -*-
"""Independent validation for Stage C6 persisted training artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalebridge.models.heat_input_regression import load_heat_input_regression_model


def _check(name: str, passed: bool, observed: Any = "", expected: Any = "", message: str = "") -> dict[str, Any]:
    return {
        "check_name": name,
        "status": "passed" if passed else "failed",
        "observed_value": observed,
        "expected_value": expected,
        "message": message,
    }


def validate_training_artifact(
    training_manifest_path: str | Path,
    *,
    coefficient_atol: float = 0.0,
    prediction_atol: float = 1e-12,
    prediction_rtol: float = 1e-12,
) -> list[dict[str, Any]]:
    """Reload one C6 artifact and validate it against the original C4 train data."""

    path = Path(training_manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    checks.append(_check("training_status_completed", manifest.get("status") == "completed", manifest.get("status"), "completed"))

    source_train = Path(manifest["source_train_path"])
    artifact_dir = Path(manifest["artifact_dir"])
    checks.append(_check("source_train_exists", source_train.is_file(), str(source_train), "existing file"))
    checks.append(_check("artifact_directory_exists", artifact_dir.is_dir(), str(artifact_dir), "existing directory"))
    if not source_train.is_file() or not artifact_dir.is_dir():
        return checks

    frame = pd.read_parquet(source_train, columns=["x", "y"])
    x = frame["x"].to_numpy(dtype=np.float64)
    y = frame["y"].to_numpy(dtype=np.float64)
    model = load_heat_input_regression_model(artifact_dir)
    prediction_a = model.predict(x)
    model_again = load_heat_input_regression_model(artifact_dir)
    prediction_b = model_again.predict(x)

    checks.append(_check("training_row_count_matches", len(frame) == int(manifest["sample_count"]), len(frame), manifest["sample_count"]))
    checks.append(_check("training_x_finite", bool(np.all(np.isfinite(x))), int(np.isfinite(x).sum()), len(x)))
    checks.append(_check("training_y_finite", bool(np.all(np.isfinite(y))), int(np.isfinite(y).sum()), len(y)))
    checks.append(_check("prediction_finite", bool(np.all(np.isfinite(prediction_a))), int(np.isfinite(prediction_a).sum()), len(prediction_a)))
    checks.append(_check("repeat_load_prediction_match", bool(np.allclose(prediction_a, prediction_b, atol=prediction_atol, rtol=prediction_rtol)), float(np.max(np.abs(prediction_a - prediction_b))), 0.0))
    checks.append(_check("coefficient_matches_manifest", bool(np.isclose(model.coefficient, float(manifest["coefficient"]), atol=coefficient_atol, rtol=0.0)), model.coefficient, manifest["coefficient"]))
    checks.append(_check("intercept_matches_manifest", bool(np.isclose(model.intercept, float(manifest["intercept"]), atol=coefficient_atol, rtol=0.0)), model.intercept, manifest["intercept"]))
    rmse = float(np.sqrt(np.mean((y - prediction_a) ** 2)))
    checks.append(_check("training_rmse_matches_manifest", bool(np.isclose(rmse, float(manifest["training_rmse"]), atol=prediction_atol, rtol=prediction_rtol)), rmse, manifest["training_rmse"]))
    checks.append(_check("model_id_matches", model.model_id == str(manifest["model_id"]), model.model_id, manifest["model_id"]))
    checks.append(_check("estimator_type_matches", model.estimator_type == str(manifest["estimator_type"]), model.estimator_type, manifest["estimator_type"]))
    dataset_payload = manifest.get("source_dataset_manifest_payload", {}) or {}
    expected_fit_intercept = bool(dataset_payload.get("fit_intercept", manifest.get("fit_intercept", False)))
    checks.append(_check("fit_intercept_matches_dataset_policy", bool(model.fit_intercept) == expected_fit_intercept, model.fit_intercept, expected_fit_intercept))
    if not expected_fit_intercept:
        checks.append(_check("origin_constrained_intercept_is_zero", float(model.intercept) == 0.0, model.intercept, 0.0))
    checks.append(_check("intercept_policy_source_recorded", bool(str(manifest.get("intercept_policy_source", "")).strip()), manifest.get("intercept_policy_source", ""), "non-empty"))
    expected_device = str(manifest.get("device", "cpu"))
    observed_device = str(getattr(model, "resolved_device", "cpu") or "cpu")
    checks.append(_check("resolved_device_matches_manifest", observed_device == expected_device, observed_device, expected_device))
    if model.estimator_type == "pytorch_linear":
        requested_device = str(manifest.get("requested_device", "cpu"))
        observed_requested = str(getattr(model, "requested_device", "cpu"))
        checks.append(_check("requested_device_matches_manifest", observed_requested == requested_device, observed_requested, requested_device))
        artifact_manifest = json.loads((artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
        checks.append(_check("artifact_resolved_device_matches", str(artifact_manifest.get("resolved_device", "")) == expected_device, artifact_manifest.get("resolved_device", ""), expected_device))
        if expected_device == "cuda":
            try:
                import torch
                cuda_ok = bool(torch.cuda.is_available())
                cuda_name = torch.cuda.get_device_name(0) if cuda_ok else ""
            except Exception as exc:
                cuda_ok = False
                cuda_name = f"{type(exc).__name__}: {exc}"
            checks.append(_check("cuda_available_for_cuda_artifact", cuda_ok, cuda_name, "available CUDA device"))
    return checks
