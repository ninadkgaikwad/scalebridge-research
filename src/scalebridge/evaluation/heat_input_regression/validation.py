# -*- coding: utf-8 -*-
"""Independent validation of Stage C7 evaluation artifacts."""
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


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def validate_evaluation_artifact(
    manifest_path: Path,
    metric_atol: float = 1e-12,
    metric_rtol: float = 1e-12,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    model_dir = Path(manifest["source_model_artifact"])
    metrics_path = Path(manifest["split_metrics_path"])
    checks.append(_check("model_artifact_exists", model_dir.exists(), str(model_dir), "existing directory"))
    checks.append(_check("metrics_file_exists", metrics_path.exists(), str(metrics_path), "existing file"))
    model = load_heat_input_regression_model(model_dir)
    metrics = pd.read_csv(metrics_path)
    checks.append(_check("three_splits_present", set(metrics["split"]) == {"train", "validation", "test"}, sorted(metrics["split"].tolist()), ["train", "validation", "test"]))
    checks.append(_check("estimator_type_matches", model.estimator_type == manifest["estimator_type"], model.estimator_type, manifest["estimator_type"]))
    checks.append(_check("model_id_matches", model.model_id == manifest["model_id"], model.model_id, manifest["model_id"]))

    for split, source in manifest["source_split_paths"].items():
        source_path = Path(source)
        frame = pd.read_parquet(source_path)
        x = frame["x"].to_numpy(dtype=float)
        y = frame["y"].to_numpy(dtype=float)
        p = model.predict(x)
        row = metrics.loc[metrics["split"] == split].iloc[0]
        checks.append(_check(f"{split}_row_count", int(row["row_count"]) == len(frame), int(row["row_count"]), len(frame)))
        checks.append(_check(f"{split}_finite_predictions", np.all(np.isfinite(p)), bool(np.all(np.isfinite(p))), True))
        checks.append(_check(
            f"{split}_rmse_recomputed",
            np.isclose(float(row["rmse"]), _rmse(y, p), atol=metric_atol, rtol=metric_rtol),
            float(row["rmse"]),
            _rmse(y, p),
        ))
        checks.append(_check(
            f"{split}_mae_recomputed",
            np.isclose(float(row["mae"]), _mae(y, p), atol=metric_atol, rtol=metric_rtol),
            float(row["mae"]),
            _mae(y, p),
        ))
        full_path = manifest.get("prediction_paths", {}).get(split)
        if manifest.get("write_full_predictions"):
            checks.append(_check(f"{split}_prediction_file_exists", bool(full_path) and Path(full_path).exists(), full_path, "existing file"))
            if full_path and Path(full_path).exists():
                saved = pd.read_parquet(full_path)
                checks.append(_check(f"{split}_prediction_row_count", len(saved) == len(frame), len(saved), len(frame)))
                checks.append(_check(
                    f"{split}_saved_predictions_match",
                    np.allclose(saved["prediction"].to_numpy(dtype=float), p, atol=metric_atol, rtol=metric_rtol),
                    float(np.max(np.abs(saved["prediction"].to_numpy(dtype=float) - p))),
                    0.0,
                ))
    return checks
