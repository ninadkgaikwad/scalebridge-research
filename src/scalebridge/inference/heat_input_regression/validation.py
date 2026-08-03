# -*- coding: utf-8 -*-
"""Independent validation for Stage C8 annual inference artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scalebridge.models.heat_input_regression import load_heat_input_regression_model


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_existing_path(raw: Any, *, base: Path, label: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"Missing path for {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def validate_zone_inference_artifact(
    manifest_path: str | Path,
    *,
    prediction_atol: float = 1e-12,
    prediction_rtol: float = 1e-12,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a C8 zone artifact while allowing intentional component NaNs.

    A missing prediction is valid only when the corresponding full-year
    predictor is non-finite. The timestamp must remain present, and the saved
    NaN mask must exactly match the predictor-availability mask.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    diagnostics: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        diagnostics.append({
            "check_name": name,
            "status": "passed" if bool(passed) else "failed",
            "detail": detail,
        })

    outputs = manifest.get("outputs", {}) or {}
    try:
        pred_path = _resolve_existing_path(
            outputs.get("annual_component_predictions"),
            base=manifest_path.parent,
            label="annual component predictions",
        )
        registry_path = _resolve_existing_path(
            outputs.get("component_prediction_registry"),
            base=manifest_path.parent,
            label="component prediction registry",
        )
        feature_path = _resolve_existing_path(
            manifest.get("source_feature_parquet"),
            base=manifest_path.parent,
            label="source feature parquet",
        )
        check("predictions_file_exists", True, str(pred_path))
        check("registry_file_exists", True, str(registry_path))
        check("feature_file_exists", True, str(feature_path))
    except Exception as exc:
        check("required_paths_resolve", False, f"{type(exc).__name__}: {exc}")
        return ({
            "status": "failed",
            "failed_check_count": 1,
            "check_count": len(diagnostics),
            "manifest_path": str(manifest_path),
        }, diagnostics)

    saved = pd.read_parquet(pred_path)
    features = pd.read_parquet(feature_path)
    registry = pd.read_csv(registry_path)

    check("row_count_matches_manifest", len(saved) == int(manifest.get("row_count", -1)), f"saved={len(saved)}")
    check("row_count_matches_features", len(saved) == len(features), f"saved={len(saved)}, features={len(features)}")
    check("component_count_matches_manifest", len(registry) == int(manifest.get("component_count", -1)), f"registry={len(registry)}")
    check("timestamp_raw_column_exists", "timestamp_raw" in saved.columns and "timestamp_raw" in features.columns)
    check("timestamp_column_exists", "timestamp" in saved.columns)
    if "timestamp_raw" in saved.columns and "timestamp_raw" in features.columns:
        saved_ts = saved["timestamp_raw"].astype(str).reset_index(drop=True)
        feature_ts = features["timestamp_raw"].astype(str).reset_index(drop=True)
        check("timestamp_raw_unique", not saved_ts.duplicated().any())
        check("timestamp_raw_matches_features", np.array_equal(saved_ts.to_numpy(), feature_ts.to_numpy()))
    if "timestamp" in saved.columns:
        check("timestamp_parse_complete", pd.to_datetime(saved["timestamp"], errors="coerce").notna().all())

    total_expected_unavailable = 0
    total_saved_unavailable = 0
    for _, row in registry.iterrows():
        model_id = str(row["model_id"])
        output_col = str(row["output_prediction_column"])
        predictor_col = str(row["predictor_column"])
        prefix = f"{model_id}:"
        check(prefix + "output_column_exists", output_col in saved.columns, output_col)
        check(prefix + "predictor_column_exists", predictor_col in features.columns, predictor_col)
        if output_col not in saved.columns or predictor_col not in features.columns:
            continue

        predictor_mode = str(row.get("predictor_mode", "direct"))
        if model_id == "PHVAC" and predictor_mode == "chained_from_predicted_QAC":
            check(prefix + "dependency_prediction_exists", "predicted_QAC" in saved.columns, "predicted_QAC")
            if "predicted_QAC" not in saved.columns:
                continue
            x = np.abs(pd.to_numeric(saved["predicted_QAC"], errors="coerce").to_numpy(dtype=float))
        else:
            x = pd.to_numeric(features[predictor_col], errors="coerce").to_numpy(dtype=float)
        y_saved = pd.to_numeric(saved[output_col], errors="coerce").to_numpy(dtype=float)
        predictor_valid = np.isfinite(x)
        saved_valid = np.isfinite(y_saved)
        expected_valid_count = int(predictor_valid.sum())
        expected_invalid_count = int((~predictor_valid).sum())
        saved_invalid_count = int((~saved_valid).sum())
        total_expected_unavailable += expected_invalid_count
        total_saved_unavailable += saved_invalid_count

        check(prefix + "has_finite_predictors", expected_valid_count > 0, f"valid={expected_valid_count}")
        check(
            prefix + "prediction_availability_matches_predictor",
            np.array_equal(saved_valid, predictor_valid),
            f"predictor_valid={expected_valid_count}, prediction_valid={int(saved_valid.sum())}",
        )
        check(
            prefix + "unavailable_rows_are_nan",
            bool(np.isnan(y_saved[~predictor_valid]).all()),
            f"expected_unavailable={expected_invalid_count}, saved_unavailable={saved_invalid_count}",
        )
        check(prefix + "valid_predictions_are_finite", bool(np.isfinite(y_saved[predictor_valid]).all()))

        for field, expected in (
            ("full_year_row_count", len(saved)),
            ("valid_predictor_count", expected_valid_count),
            ("invalid_predictor_count", expected_invalid_count),
            ("valid_prediction_count", expected_valid_count),
            ("unavailable_prediction_count", expected_invalid_count),
        ):
            if field in registry.columns and pd.notna(row.get(field)):
                check(prefix + f"registry_{field}", int(row[field]) == int(expected), f"saved={row[field]}, expected={expected}")

        model_dir = Path(str(row["model_artifact_dir"])).expanduser()
        if not model_dir.is_absolute():
            model_dir = (registry_path.parent / model_dir).resolve()
        check(prefix + "model_artifact_dir_exists", model_dir.is_dir(), str(model_dir))
        if not model_dir.is_dir():
            continue
        model = load_heat_input_regression_model(model_dir)
        y_expected_valid = np.asarray(model.predict(x[predictor_valid]), dtype=float).reshape(-1)
        max_abs_diff = float(np.max(np.abs(y_saved[predictor_valid] - y_expected_valid))) if expected_valid_count else 0.0
        matches = bool(np.allclose(
            y_saved[predictor_valid],
            y_expected_valid,
            atol=prediction_atol,
            rtol=prediction_rtol,
            equal_nan=False,
        ))
        check(prefix + "prediction_recomputed_on_valid_rows", matches, f"max_abs_diff={max_abs_diff}")
        oracle_output_col = str(row.get("oracle_output_prediction_column", "")).strip()
        if model_id == "PHVAC" and oracle_output_col:
            check(prefix + "oracle_output_column_exists", oracle_output_col in saved.columns, oracle_output_col)
            if oracle_output_col in saved.columns:
                oracle_x = pd.to_numeric(features[predictor_col], errors="coerce").to_numpy(dtype=float)
                oracle_valid = np.isfinite(oracle_x)
                oracle_saved = pd.to_numeric(saved[oracle_output_col], errors="coerce").to_numpy(dtype=float)
                oracle_expected = model.predict(oracle_x[oracle_valid])
                check(prefix + "oracle_prediction_recomputed", np.allclose(oracle_saved[oracle_valid], oracle_expected, atol=prediction_atol, rtol=prediction_rtol), "oracle PHVAC")
        check(prefix + "coefficient_matches", bool(np.isclose(float(model.coefficient), float(row["coefficient"]), atol=prediction_atol, rtol=prediction_rtol)))
        check(prefix + "intercept_matches", bool(np.isclose(float(model.intercept), float(row["intercept"]), atol=prediction_atol, rtol=prediction_rtol)))

    check(
        "total_unavailable_component_values_matches_manifest",
        int(manifest.get("total_unavailable_component_values", -1)) == total_expected_unavailable,
        f"manifest={manifest.get('total_unavailable_component_values')}, expected={total_expected_unavailable}",
    )
    check(
        "total_saved_unavailable_matches_expected",
        total_saved_unavailable == total_expected_unavailable,
        f"saved={total_saved_unavailable}, expected={total_expected_unavailable}",
    )

    failed = sum(d["status"] == "failed" for d in diagnostics)
    result = {
        "case_id": manifest.get("case_id", ""),
        "aggregation_id": manifest.get("aggregation_id", ""),
        "weight_mode": manifest.get("weight_mode", ""),
        "aggregate_zone_id": manifest.get("aggregate_zone_id", ""),
        "component_count": manifest.get("component_count", 0),
        "row_count": manifest.get("row_count", 0),
        "expected_unavailable_component_values": total_expected_unavailable,
        "saved_unavailable_component_values": total_saved_unavailable,
        "check_count": len(diagnostics),
        "failed_check_count": failed,
        "status": "passed" if failed == 0 else "failed",
        "manifest_path": str(manifest_path),
    }
    return result, diagnostics
