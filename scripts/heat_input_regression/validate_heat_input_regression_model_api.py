# -*- coding: utf-8 -*-
"""Validate C5 heat-input regression APIs on closed-form, PyTorch CPU, and PyTorch GPU."""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.models.heat_input_regression import (  # noqa: E402
    create_heat_input_regression_model,
    load_heat_input_regression_model,
)


def _check(name: str, passed: bool, observed: Any = "", expected: Any = "", message: str = "") -> dict[str, Any]:
    return {"check_name": name, "status": "passed" if passed else "failed", "observed_value": observed, "expected_value": expected, "message": message}


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def validate_estimator(estimator_type: str, output_root: Path, *, device: str = "cpu") -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    x = np.linspace(-10.0, 20.0, 2000)
    y = 125.0 + 3.75 * x
    kwargs: dict[str, Any] = {"fit_intercept": True, "model_id": "synthetic_exact", "metadata": {"purpose": "C5 validation"}}
    if estimator_type == "pytorch_linear":
        kwargs.update({"learning_rate": 0.03, "max_epochs": 5000, "tolerance": 1e-12, "patience": 300, "seed": 42, "device": device})
    model = create_heat_input_regression_model(estimator_type, **kwargs).fit(x, y)
    pred = model.predict(x)
    coefficient_tol = 1e-10 if estimator_type == "closed_form_linear" else 2e-3
    intercept_tol = 1e-10 if estimator_type == "closed_form_linear" else 2e-2
    rmse_tol = 1e-9 if estimator_type == "closed_form_linear" else 2e-2
    checks.extend([
        _check("coefficient_recovery", abs(model.coefficient - 3.75) <= coefficient_tol, model.coefficient, 3.75),
        _check("intercept_recovery", abs(model.intercept - 125.0) <= intercept_tol, model.intercept, 125.0),
        _check("exact_linear_prediction_rmse", _rmse(y, pred) <= rmse_tol, _rmse(y, pred), f"<= {rmse_tol}"),
        _check("predict_one_matches_batch", math.isclose(model.predict_one(2.5), float(model.predict([2.5])[0]), rel_tol=0, abs_tol=1e-12)),
        _check("predict_batch_shape", model.predict_batch([1.0, 2.0, 3.0]).shape == (3,), model.predict_batch([1.0, 2.0, 3.0]).shape, (3,)),
        _check("fit_summary_available", model.fit_summary.sample_count == len(x), model.fit_summary.sample_count, len(x)),
    ])
    if estimator_type == "pytorch_linear":
        checks.append(_check("resolved_device", model.resolved_device == device or (device == "auto" and model.resolved_device in {"cpu", "cuda"}), model.resolved_device, device))
        if device == "cuda":
            checks.append(_check("cuda_device_name_recorded", bool(model.cuda_device_name), model.cuda_device_name, "non-empty"))

    label = estimator_type if estimator_type != "pytorch_linear" else f"{estimator_type}_{device}"
    artifact_dir = output_root / label / "artifact"
    manifest_path = model.save(artifact_dir)
    loaded = load_heat_input_regression_model(artifact_dir)
    loaded_pred = loaded.predict(x)
    checks.extend([
        _check("manifest_written", manifest_path.exists(), str(manifest_path), "existing path"),
        _check("round_trip_prediction", np.allclose(pred, loaded_pred, atol=1e-12, rtol=1e-12), float(np.max(np.abs(pred - loaded_pred))), 0.0),
        _check("round_trip_metadata", loaded.metadata.get("purpose") == "C5 validation", loaded.metadata.get("purpose"), "C5 validation"),
    ])
    if estimator_type == "pytorch_linear":
        checks.append(_check("round_trip_resolved_device", loaded.resolved_device == model.resolved_device, loaded.resolved_device, model.resolved_device))

    x0 = np.linspace(0.1, 100.0, 1000)
    y0 = 6.5 * x0
    kwargs0: dict[str, Any] = {"fit_intercept": False, "model_id": "synthetic_origin"}
    if estimator_type == "pytorch_linear":
        kwargs0.update({"learning_rate": 5e-4, "max_epochs": 8000, "tolerance": 1e-11, "patience": 500, "seed": 42, "device": device})
    origin_model = create_heat_input_regression_model(estimator_type, **kwargs0).fit(x0, y0)
    origin_tol = 1e-10 if estimator_type == "closed_form_linear" else 5e-2
    checks.extend([
        _check("origin_model_zero_intercept", origin_model.intercept == 0.0, origin_model.intercept, 0.0),
        _check("origin_model_coefficient", abs(origin_model.coefficient - 6.5) <= origin_tol, origin_model.coefficient, 6.5),
    ])

    noisy_y = 10.0 + 2.25 * x + rng.normal(0.0, 0.5, size=x.size)
    noisy_model = create_heat_input_regression_model(estimator_type, **kwargs).fit(x, noisy_y)
    checks.append(_check("noisy_model_finite_parameters", np.isfinite(noisy_model.coefficient) and np.isfinite(noisy_model.intercept), f"coef={noisy_model.coefficient}, intercept={noisy_model.intercept}", "finite"))
    try:
        create_heat_input_regression_model(estimator_type, **kwargs).predict([1.0]); unfitted_ok = False
    except RuntimeError:
        unfitted_ok = True
    checks.append(_check("unfitted_prediction_rejected", unfitted_ok))
    try:
        create_heat_input_regression_model(estimator_type, **kwargs).fit([1.0, 1.0], [2.0, 3.0]); constant_ok = False
    except ValueError:
        constant_ok = True
    checks.append(_check("constant_predictor_rejected", constant_ok))
    return checks


def validate_c4_compatibility(dataset_root: Path, estimator_type: str, max_models: int | None, *, device: str = "cpu") -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    manifests = sorted(dataset_root.rglob("model_dataset_manifest.json"))
    if max_models is not None:
        manifests = manifests[:max_models]
    checks.append(_check("c4_manifests_found", len(manifests) > 0, len(manifests), "> 0"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_dir = manifest_path.parent
        label = f"{manifest.get('aggregate_zone_id', 'zone')}::{manifest.get('model_id', model_dir.name)}"
        try:
            train = pd.read_parquet(model_dir / "train.parquet", columns=["x", "y"])
            valid = pd.read_parquet(model_dir / "validation.parquet", columns=["x", "y"])
            fit_intercept = bool(manifest.get("fit_intercept", False))
            kwargs: dict[str, Any] = {"fit_intercept": fit_intercept, "model_id": str(manifest.get("model_id", "")), "metadata": {"source_manifest": str(manifest_path), "model_role": manifest.get("model_role", ""), "input_transform": manifest.get("input_transform", "identity")}}
            if estimator_type == "pytorch_linear":
                kwargs.update({"learning_rate": 0.03, "max_epochs": 3000, "tolerance": 1e-10, "patience": 200, "seed": 42, "device": device})
            model = create_heat_input_regression_model(estimator_type, **kwargs).fit(train["x"].to_numpy(), train["y"].to_numpy())
            pred = model.predict(valid["x"].to_numpy())
            checks.append(_check(f"c4_compatibility::{label}", len(pred) == len(valid) and np.all(np.isfinite(pred)), f"predictions={len(pred)}; fit_intercept={fit_intercept}", f"{len(valid)} finite predictions"))
            checks.append(_check(f"c4_intercept_policy::{label}", model.fit_intercept == fit_intercept, model.fit_intercept, fit_intercept))
            if not fit_intercept:
                checks.append(_check(f"c4_zero_intercept::{label}", model.intercept == 0.0, model.intercept, 0.0))
        except Exception as exc:
            checks.append(_check(f"c4_compatibility::{label}", False, message=f"{type(exc).__name__}: {exc}"))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-c4-models", type=int, default=None, help="Default: validate every C4 model dataset.")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--pytorch-device", action="append", choices=["cpu", "cuda", "auto"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root or Path(tempfile.mkdtemp(prefix="scalebridge_c5_validation_"))
    output_root.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION MODEL API VALIDATION")
    print("=" * 100)
    print(f"output_root: {output_root}")
    tasks: list[tuple[str, str]] = [("closed_form_linear", "cpu")]
    if not args.skip_pytorch:
        tasks.extend(("pytorch_linear", device) for device in (args.pytorch_device or ["cpu"]))
    all_rows: list[dict[str, Any]] = []
    for estimator, device in tasks:
        label = estimator if estimator != "pytorch_linear" else f"{estimator}[{device}]"
        print(f"\nValidating estimator: {label}")
        rows = validate_estimator(estimator, output_root, device=device)
        if args.dataset_root is not None:
            rows.extend(validate_c4_compatibility(args.dataset_root, estimator, args.max_c4_models, device=device))
        for row in rows:
            row["estimator_type"] = estimator
            row["device"] = device
        all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    csv_path = output_root / "c5_model_api_validation.csv"
    frame.to_csv(csv_path, index=False)
    failed = frame.loc[frame["status"] == "failed"]
    manifest = {"validation_status": "passed" if failed.empty else "failed", "tasks": [{"estimator_type": e, "device": d} for e, d in tasks], "check_count": int(len(frame)), "passed_check_count": int((frame["status"] == "passed").sum()), "failed_check_count": int(len(failed)), "dataset_root": str(args.dataset_root) if args.dataset_root else "", "validation_csv": str(csv_path)}
    (output_root / "c5_model_api_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print("C5 VALIDATION SUMMARY")
    print("=" * 100)
    for key in ("check_count", "passed_check_count", "failed_check_count", "validation_status"):
        print(f"{key}: {manifest[key]}")
    print(f"output_root: {output_root}")
    if not failed.empty:
        print("\nFAILED CHECKS")
        print(failed[["estimator_type", "device", "check_name", "message"]].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
