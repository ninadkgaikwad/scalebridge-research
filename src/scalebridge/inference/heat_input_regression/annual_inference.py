# -*- coding: utf-8 -*-
"""Stage C8 full-year inference for heat-input regression models.

C8 consumes validated C7 evaluation artifacts and their persisted C6 models.
It locates the matching C2 zone feature table, preserves its complete timestamp
axis, predicts every available heat-input component, and writes one annual
component table per aggregate zone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scalebridge.models.heat_input_regression import load_heat_input_regression_model


@dataclass(frozen=True)
class EvaluationArtifactReference:
    evaluation_manifest_path: Path
    evaluation_dir: Path
    training_manifest_path: Path
    model_artifact_dir: Path
    case_id: str
    aggregation_id: str
    weight_mode: str
    aggregate_zone_id: str
    model_id: str
    estimator_type: str
    requested_device: str
    resolved_device: str
    predictor_column: str
    predictor_units: str
    target_units: str
    output_prediction_column: str
    feature_run_id: str
    dataset_manifest_path: Path
    raw_evaluation_manifest: dict[str, Any]
    raw_training_manifest: dict[str, Any]
    raw_dataset_manifest: dict[str, Any]
    fit_intercept: bool
    model_role: str
    input_transform: str
    dependency_model_id: str
    target_allocation: str

    @property
    def zone_key(self) -> tuple[str, str, str, str]:
        return (self.case_id, self.aggregation_id, self.weight_mode, self.aggregate_zone_id)


@dataclass(frozen=True)
class ZoneInferenceResult:
    row: dict[str, Any]
    output_dir: Path
    manifest_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_name(value: Any) -> str:
    text = str(value).strip() or "unnamed"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def _resolve_path(raw: Any, *, base: Path, label: str, file: bool | None = None) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"Missing path for {label}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if file is True and not path.is_file():
        raise IsADirectoryError(f"Expected file for {label}: {path}")
    if file is False and not path.is_dir():
        raise NotADirectoryError(f"Expected directory for {label}: {path}")
    return path


def _find_heat_input_root(path: Path) -> Path:
    for candidate in [path, *path.parents]:
        if candidate.name == "heat_input_regression":
            return candidate
    raise ValueError(f"Could not locate heat_input_regression ancestor from: {path}")


def discover_evaluation_artifacts(
    evaluation_root: str | Path,
    *,
    model_ids: Iterable[str] | None = None,
    aggregate_zone_ids: Iterable[str] | None = None,
    estimator_types: Iterable[str] | None = None,
    requested_devices: Iterable[str] | None = None,
    max_artifacts: int | None = None,
) -> list[EvaluationArtifactReference]:
    """Discover completed C7 artifacts and resolve C6/C4 provenance."""
    root = Path(evaluation_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"C7 evaluation root does not exist: {root}")
    mf, zf, ef, df = map(set, (model_ids or [], aggregate_zone_ids or [], estimator_types or [], requested_devices or []))
    refs: list[EvaluationArtifactReference] = []
    for eval_manifest_path in sorted(root.rglob("evaluation_manifest.json")):
        em = _read_json(eval_manifest_path)
        if str(em.get("status", "completed")) not in {"completed", "passed"}:
            continue
        model_id = str(em.get("model_id", ""))
        zone_id = str(em.get("aggregate_zone_id", ""))
        estimator = str(em.get("estimator_type", ""))
        requested = str(em.get("requested_device", em.get("device", "cpu")))
        if mf and model_id not in mf or zf and zone_id not in zf or ef and estimator not in ef or df and requested not in df:
            continue
        source = em.get("source", {}) or {}
        outputs = em.get("outputs", {}) or {}
        training_raw = source.get("training_manifest") or em.get("source_training_manifest")
        training_manifest_path = _resolve_path(training_raw, base=eval_manifest_path.parent, label="C6 training manifest", file=True)
        tm = _read_json(training_manifest_path)
        t_outputs = tm.get("outputs", {}) or {}
        model_artifact_dir = _resolve_path(
            t_outputs.get("model_artifact_dir") or tm.get("artifact_dir") or tm.get("model_artifact_dir") or training_manifest_path.parent / "model_artifact",
            base=training_manifest_path.parent,
            label="C6 model artifact",
            file=False,
        )
        dm = tm.get("source_dataset_manifest_payload", {}) or {}

        # C6 always embeds the C4 manifest payload, but older C6 artifacts may
        # not include a separate source_dataset_manifest path. Resolve the
        # physical C4 manifest through the strongest available provenance.
        dataset_manifest_candidates: list[Any] = [
            tm.get("source_dataset_manifest"),
            tm.get("source_dataset_manifest_path"),
            dm.get("manifest_path"),
            dm.get("model_dataset_manifest"),
        ]
        dataset_output_root = str(dm.get("output_root", "")).strip()
        if dataset_output_root:
            dataset_manifest_candidates.append(Path(dataset_output_root) / "model_dataset_manifest.json")
        dataset_outputs = dm.get("outputs", {}) or {}
        train_raw = str(dataset_outputs.get("train", "")).strip()
        if train_raw:
            train_candidate = Path(train_raw).expanduser()
            if not train_candidate.is_absolute():
                train_candidate = (training_manifest_path.parent / train_candidate).resolve()
            dataset_manifest_candidates.append(train_candidate.parent / "model_dataset_manifest.json")

        dataset_manifest_path: Path | None = None
        candidate_errors: list[str] = []
        for candidate in dataset_manifest_candidates:
            if not str(candidate or "").strip():
                continue
            try:
                dataset_manifest_path = _resolve_path(
                    candidate,
                    base=training_manifest_path.parent,
                    label="C4 model dataset manifest",
                    file=True,
                )
                break
            except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
                candidate_errors.append(str(exc))
        if dataset_manifest_path is None:
            detail = "; ".join(candidate_errors) or "no non-empty provenance candidates"
            raise FileNotFoundError(
                f"Could not resolve C4 model dataset manifest from C6 provenance: {training_manifest_path}. {detail}"
            )
        if not dm:
            dm = _read_json(dataset_manifest_path)
        predictor_column = str(dm.get("predictor_column", tm.get("predictor_column", "")))
        output_col = str(dm.get("output_prediction_column", ""))
        if not predictor_column or not output_col:
            raise ValueError(f"C4 manifest lacks predictor/output column: {dataset_manifest_path}")
        refs.append(EvaluationArtifactReference(
            evaluation_manifest_path=eval_manifest_path,
            evaluation_dir=eval_manifest_path.parent,
            training_manifest_path=training_manifest_path,
            model_artifact_dir=model_artifact_dir,
            case_id=str(em.get("case_id", tm.get("case_id", ""))),
            aggregation_id=str(em.get("aggregation_id", tm.get("aggregation_id", ""))),
            weight_mode=str(em.get("weight_mode", tm.get("weight_mode", ""))),
            aggregate_zone_id=zone_id,
            model_id=model_id,
            estimator_type=estimator,
            requested_device=requested,
            resolved_device=str(em.get("resolved_device", tm.get("resolved_device", requested))),
            predictor_column=predictor_column,
            predictor_units=str(dm.get("predictor_units", "")),
            target_units=str(dm.get("target_units", "W")),
            output_prediction_column=output_col,
            feature_run_id=str(dm.get("source_feature_run_id", tm.get("source_feature_run_id", ""))),
            dataset_manifest_path=dataset_manifest_path,
            raw_evaluation_manifest=em,
            raw_training_manifest=tm,
            raw_dataset_manifest=dm,
            fit_intercept=bool(dm.get("fit_intercept", False)),
            model_role=str(dm.get("model_role", "")),
            input_transform=str(dm.get("input_transform", "identity")),
            dependency_model_id=str(dm.get("dependency_model_id", "")),
            target_allocation=str(dm.get("target_allocation", "none")),
        ))
        if max_artifacts is not None and len(refs) >= max_artifacts:
            break
    return refs


def _locate_zone_feature_manifest(ref: EvaluationArtifactReference) -> Path:
    heat_root = _find_heat_input_root(ref.dataset_manifest_path)
    feature_root = heat_root / "feature_runs" / ref.feature_run_id
    if not feature_root.is_dir():
        raise FileNotFoundError(f"C2 feature run root is missing: {feature_root}")
    matches: list[Path] = []
    for path in feature_root.rglob("zone_feature_manifest.json"):
        payload = _read_json(path)
        if (
            str(payload.get("case_id", "")) == ref.case_id
            and str(payload.get("aggregate_zone_id", "")) == ref.aggregate_zone_id
            and str(payload.get("weight_mode", "")) == ref.weight_mode
            and str(payload.get("aggregation_id", payload.get("aggregation_level", ""))) == ref.aggregation_id
        ):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one matching C2 feature manifest for {ref.zone_key}, found {len(matches)} under {feature_root}")
    return matches[0]


def build_zone_output_dir(inference_root: Path, ref: EvaluationArtifactReference) -> Path:
    return inference_root / "cases" / _safe_name(ref.case_id) / _safe_name(ref.aggregation_id) / _safe_name(ref.weight_mode) / _safe_name(ref.aggregate_zone_id)



def build_building_phvac_inference(inference_root: str | Path) -> Path | None:
    """Sum aggregate-zone PHVAC predictions into one building-level table."""
    root = Path(inference_root)
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in root.rglob("annual_component_predictions_manifest.json"):
        payload = _read_json(path)
        prediction_path = Path((payload.get("outputs", {}) or {}).get("annual_component_predictions", ""))
        if prediction_path.is_file():
            frame = pd.read_parquet(prediction_path, columns=None)
            if "predicted_PHVAC" in frame.columns:
                manifests.append((path, payload))
    if not manifests:
        return None
    groups: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for item in manifests:
        payload = item[1]
        key = (str(payload.get("case_id", "")), str(payload.get("aggregation_id", "")), str(payload.get("weight_mode", "")))
        groups.setdefault(key, []).append(item)
    out_root = root / "building_phvac_reconstruction"
    out_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for key, items in groups.items():
        merged = None
        for path, payload in items:
            pred_path = Path((payload.get("outputs", {}) or {}).get("annual_component_predictions"))
            frame = pd.read_parquet(pred_path)
            zone = str(payload.get("aggregate_zone_id", "zone"))
            cols = ["timestamp_raw", "timestamp", "predicted_PHVAC"]
            if "predicted_PHVAC_oracle" in frame.columns:
                cols.append("predicted_PHVAC_oracle")
            current = frame[cols].copy().rename(columns={
                "predicted_PHVAC": f"chained__{zone}",
                "predicted_PHVAC_oracle": f"oracle__{zone}",
            })
            merged = current if merged is None else merged.merge(current, on=["timestamp_raw", "timestamp"], how="inner", validate="one_to_one")
        chained_cols = [c for c in merged.columns if c.startswith("chained__")]
        oracle_cols = [c for c in merged.columns if c.startswith("oracle__")]
        merged["predicted_PHVAC_building"] = merged[chained_cols].sum(axis=1, min_count=len(chained_cols))
        if oracle_cols:
            merged["predicted_PHVAC_building_oracle"] = merged[oracle_cols].sum(axis=1, min_count=len(oracle_cols))
        group_dir = out_root / _safe_name(key[0]) / _safe_name(key[1]) / _safe_name(key[2])
        group_dir.mkdir(parents=True, exist_ok=True)
        output_path = group_dir / "annual_building_phvac_predictions.parquet"
        merged.to_parquet(output_path, index=False)
        preview_path = group_dir / "annual_building_phvac_predictions_preview.csv"
        merged.head(100).to_csv(preview_path, index=False)
        manifest = {
            "schema_version": "0.1.0", "stage": "C8_building_phvac", "status": "completed",
            "case_id": key[0], "aggregation_id": key[1], "weight_mode": key[2],
            "aggregate_zone_count": len(items), "row_count": len(merged),
            "zone_manifests": [str(path) for path, _ in items],
            "outputs": {"annual_building_phvac_predictions": str(output_path), "preview": str(preview_path)},
        }
        manifest_path = group_dir / "annual_building_phvac_predictions_manifest.json"
        _write_json(manifest_path, manifest)
        index_rows.append({**{k:v for k,v in zip(("case_id","aggregation_id","weight_mode"),key)}, "aggregate_zone_count":len(items), "row_count":len(merged), "manifest_path":str(manifest_path), "predictions_path":str(output_path)})
    index_path = out_root / "building_phvac_reconstruction_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return index_path


def run_zone_inference(
    refs: list[EvaluationArtifactReference],
    *,
    inference_root: str | Path,
    inference_run_id: str,
    preview_rows: int = 100,
    overwrite_existing: bool = False,
) -> ZoneInferenceResult:
    """Generate one complete annual component-prediction table for one zone."""
    if not refs:
        raise ValueError("No evaluation artifacts supplied for zone inference")
    zone_keys = {r.zone_key for r in refs}
    if len(zone_keys) != 1:
        raise ValueError(f"run_zone_inference requires one zone, got: {sorted(zone_keys)}")
    started = time.perf_counter()
    ref0 = refs[0]
    output_dir = build_zone_output_dir(Path(inference_root), ref0)
    manifest_path = output_dir / "annual_component_predictions_manifest.json"
    if manifest_path.exists() and not overwrite_existing:
        raise FileExistsError(f"C8 output exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_manifest_path = _locate_zone_feature_manifest(ref0)
    fm = _read_json(feature_manifest_path)
    feature_path = _resolve_path((fm.get("outputs", {}) or {}).get("derived_features_parquet"), base=feature_manifest_path.parent, label="C2 full-year feature Parquet", file=True)
    features = pd.read_parquet(feature_path)
    required_time = ["timestamp_raw", "timestamp"]
    missing_time = [c for c in required_time if c not in features.columns]
    if missing_time:
        raise KeyError(f"C2 full-year feature table lacks timestamp columns {missing_time}: {feature_path}")
    if features.empty:
        raise ValueError(f"C2 full-year feature table is empty: {feature_path}")
    if features["timestamp_raw"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate timestamp_raw values in C2 feature table: {feature_path}")

    out = pd.DataFrame({
        "timestamp_raw": features["timestamp_raw"].astype(str),
        "timestamp": pd.to_datetime(features["timestamp"], errors="coerce"),
    })
    if out["timestamp"].isna().any():
        raise ValueError(f"Unparsed timestamps in C2 feature table: {feature_path}")

    registry_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    seen_output_columns: set[str] = set()
    ordered_refs = sorted(refs, key=lambda r: (r.model_id == "PHVAC", r.model_id))
    for ref in ordered_refs:
        if ref.output_prediction_column in seen_output_columns:
            raise ValueError(f"Duplicate output prediction column: {ref.output_prediction_column}")
        seen_output_columns.add(ref.output_prediction_column)
        if ref.predictor_column not in features.columns:
            raise KeyError(f"Predictor {ref.predictor_column!r} missing from {feature_path}")
        x_oracle = pd.to_numeric(features[ref.predictor_column], errors="coerce").to_numpy(dtype=float)
        x = x_oracle
        predictor_mode = "direct"
        if ref.model_id == "PHVAC":
            dependency_output = "predicted_QAC"
            if dependency_output not in out.columns:
                raise KeyError("PHVAC requires predicted_QAC from dependency-ordered C8 inference")
            x = np.abs(pd.to_numeric(out[dependency_output], errors="coerce").to_numpy(dtype=float))
            predictor_mode = "chained_from_predicted_QAC"
        predictor_valid = np.isfinite(x)
        valid_count = int(predictor_valid.sum())
        invalid_count = int((~predictor_valid).sum())
        if valid_count == 0:
            raise ValueError(f"No finite full-year predictor values for {ref.model_id}")

        # Preserve the complete C2 timestamp axis. Predict only where the
        # component-specific predictor is available; leave unavailable rows as
        # NaN rather than dropping timestamps or silently imputing values.
        model = load_heat_input_regression_model(ref.model_artifact_dir)
        prediction = np.full(len(out), np.nan, dtype=float)
        predicted_valid = np.asarray(model.predict(x[predictor_valid]), dtype=float).reshape(-1)
        if len(predicted_valid) != valid_count or not np.all(np.isfinite(predicted_valid)):
            raise ValueError(f"Invalid annual predictions for {ref.model_id}")
        prediction[predictor_valid] = predicted_valid
        if ref.model_id == "PHVAC":
            oracle_valid = np.isfinite(x_oracle)
            oracle_prediction = np.full(len(out), np.nan, dtype=float)
            oracle_prediction[oracle_valid] = np.asarray(model.predict(x_oracle[oracle_valid]), dtype=float).reshape(-1)
            out[f"{ref.output_prediction_column}_oracle"] = oracle_prediction
        out[ref.output_prediction_column] = prediction
        if invalid_count:
            invalid_indices = np.flatnonzero(~predictor_valid)
            for idx in invalid_indices:
                missing_rows.append({
                    "row_index": int(idx),
                    "timestamp_raw": str(out.iloc[idx]["timestamp_raw"]),
                    "timestamp": str(out.iloc[idx]["timestamp"]),
                    "model_id": ref.model_id,
                    "predictor_column": ref.predictor_column,
                    "output_prediction_column": ref.output_prediction_column,
                    "missing_reason": "predictor_non_finite",
                    "predictor_value": None,
                })
        registry_rows.append({
            "model_id": ref.model_id,
            "output_prediction_column": ref.output_prediction_column,
            "predictor_column": ref.predictor_column,
            "predictor_units": ref.predictor_units,
            "prediction_units": ref.target_units,
            "estimator_type": ref.estimator_type,
            "requested_device": ref.requested_device,
            "resolved_device": ref.resolved_device,
            "coefficient": float(model.coefficient),
            "intercept": float(model.intercept),
            "model_artifact_dir": str(ref.model_artifact_dir),
            "training_manifest": str(ref.training_manifest_path),
            "evaluation_manifest": str(ref.evaluation_manifest_path),
            "dataset_manifest": str(ref.dataset_manifest_path),
            "feature_manifest": str(feature_manifest_path),
            "full_year_row_count": int(len(out)),
            "valid_predictor_count": valid_count,
            "invalid_predictor_count": invalid_count,
            "valid_prediction_count": int(np.isfinite(prediction).sum()),
            "unavailable_prediction_count": int((~np.isfinite(prediction)).sum()),
            "missing_value_policy": "preserve_timestamp_and_write_nan",
            "fit_intercept": ref.fit_intercept,
            "model_role": ref.model_role,
            "input_transform": ref.input_transform,
            "dependency_model_id": ref.dependency_model_id,
            "target_allocation": ref.target_allocation,
            "predictor_mode": predictor_mode,
            "oracle_output_prediction_column": (f"{ref.output_prediction_column}_oracle" if ref.model_id == "PHVAC" else ""),
        })

    predictions_path = output_dir / "annual_component_predictions.parquet"
    preview_path = output_dir / "annual_component_predictions_preview.csv"
    registry_path = output_dir / "component_prediction_registry.csv"
    summary_path = output_dir / "component_prediction_summary.csv"
    missing_path = output_dir / "component_missing_value_timestamps.csv"
    availability_path = output_dir / "timestamp_component_availability.csv"
    out.to_parquet(predictions_path, index=False)
    out.head(preview_rows).to_csv(preview_path, index=False)
    pd.DataFrame(registry_rows).to_csv(registry_path, index=False)
    missing_frame = pd.DataFrame(missing_rows, columns=[
        "row_index", "timestamp_raw", "timestamp", "model_id",
        "predictor_column", "output_prediction_column", "missing_reason",
        "predictor_value",
    ])
    missing_frame.to_csv(missing_path, index=False)
    availability = pd.DataFrame({
        "row_index": np.arange(len(out), dtype=int),
        "timestamp_raw": out["timestamp_raw"].astype(str),
        "timestamp": out["timestamp"].astype(str),
    })
    prediction_columns = [r["output_prediction_column"] for r in registry_rows]
    availability["available_component_count"] = out[prediction_columns].notna().sum(axis=1).astype(int)
    availability["unavailable_component_count"] = out[prediction_columns].isna().sum(axis=1).astype(int)
    availability = availability.loc[availability["unavailable_component_count"] > 0]
    availability.to_csv(availability_path, index=False)
    summary_rows = []
    for row in registry_rows:
        values = out[row["output_prediction_column"]].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        summary_rows.append({
            **{k: row[k] for k in ("model_id", "output_prediction_column", "prediction_units")},
            "row_count": int(len(values)),
            "valid_prediction_count": int(len(finite_values)),
            "unavailable_prediction_count": int((~np.isfinite(values)).sum()),
            "availability_fraction": float(len(finite_values) / len(values)),
            "minimum": float(np.min(finite_values)) if len(finite_values) else np.nan,
            "maximum": float(np.max(finite_values)) if len(finite_values) else np.nan,
            "mean": float(np.mean(finite_values)) if len(finite_values) else np.nan,
            "standard_deviation": float(np.std(finite_values)) if len(finite_values) else np.nan,
            "missing_value_policy": row["missing_value_policy"],
        })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    manifest = {
        "schema_version": "0.1.0",
        "stage": "C8",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_run_id": inference_run_id,
        "case_id": ref0.case_id,
        "aggregation_id": ref0.aggregation_id,
        "weight_mode": ref0.weight_mode,
        "aggregate_zone_id": ref0.aggregate_zone_id,
        "row_count": int(len(out)),
        "component_count": len(registry_rows),
        "timestamp_start": str(out["timestamp"].min()),
        "timestamp_end": str(out["timestamp"].max()),
        "duplicate_timestamp_count": int(out["timestamp_raw"].duplicated().sum()),
        "missing_value_policy": "component_specific_predictor_unavailable_rows_are_nan",
        "total_unavailable_component_values": int(sum(r["unavailable_prediction_count"] for r in registry_rows)),
        "source_feature_manifest": str(feature_manifest_path),
        "source_feature_parquet": str(feature_path),
        "source_evaluation_manifests": [str(r.evaluation_manifest_path) for r in refs],
        "outputs": {
            "annual_component_predictions": str(predictions_path),
            "annual_component_predictions_preview": str(preview_path),
            "component_prediction_registry": str(registry_path),
            "component_prediction_summary": str(summary_path),
            "component_missing_value_timestamps": str(missing_path),
            "timestamp_component_availability": str(availability_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(manifest_path, manifest)
    row = {k: manifest[k] for k in ("case_id", "aggregation_id", "weight_mode", "aggregate_zone_id", "row_count", "component_count", "status", "runtime_seconds")}
    row["output_dir"] = str(output_dir)
    row["manifest_path"] = str(manifest_path)
    return ZoneInferenceResult(row=row, output_dir=output_dir, manifest_path=manifest_path)
