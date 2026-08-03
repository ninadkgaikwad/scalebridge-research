# -*- coding: utf-8 -*-
"""Orchestrate deterministic Stage C2 feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from scalebridge.data.heat_input_regression.alignment import (
    align_predictor_target,
    build_timestamp_frame,
)
from scalebridge.data.heat_input_regression.hvac import (
    HVAC_PREDICTOR_OUTPUT_COLUMN,
    HVAC_TARGET_OUTPUT_COLUMN,
    PHVAC_PREDICTOR_OUTPUT_COLUMN,
    PHVAC_TARGET_OUTPUT_COLUMN,
    build_hvac_predictor,
    build_hvac_target,
    build_phvac_features,
)
from scalebridge.data.heat_input_regression.schedules import (
    build_corrected_schedule,
)
from scalebridge.data.heat_input_regression.signal_catalog import (
    get_signal_definition,
    resolve_present_column,
)
from scalebridge.data.heat_input_regression.solar import (
    GHI_OUTPUT_COLUMN,
    build_ghi,
)
from scalebridge.models.heat_input_regression.registry import (
    get_model_specification,
)


@dataclass(frozen=True)
class ModelFeatureResult:
    """One model-specific deterministic feature dataset."""

    model_id: str
    predictor_column: str
    target_column: str
    frame: pd.DataFrame
    feature_metadata: dict[str, Any]
    alignment_metadata: dict[str, Any]


def build_model_feature_result(
    *,
    model_id: str,
    wide_frame: pd.DataFrame,
    static_equipment_frame: pd.DataFrame,
    contribution_frame: pd.DataFrame,
    internal_gain_predictor_method: str,
    hvac_target_method: str,
    drop_invalid: bool = False,
) -> ModelFeatureResult:
    """Build deterministic predictor and target columns for one model."""
    spec = get_model_specification(model_id)
    timestamp_frame, timestamp_metadata = build_timestamp_frame(wide_frame)

    if spec.predictor_kind == "ghi":
        predictor, feature_metadata = build_ghi(wide_frame)
        predictor_column = GHI_OUTPUT_COLUMN
    elif spec.predictor_kind == "corrected_schedule":
        predictor, feature_metadata = build_corrected_schedule(
            frame=wide_frame,
            static_equipment_frame=static_equipment_frame,
            contribution_frame=contribution_frame,
            equipment_type=spec.source_family,
            predictor_method=internal_gain_predictor_method,
        )
        predictor_column = predictor.name
    elif spec.predictor_kind == "hvac_thermodynamic":
        predictor, feature_metadata = build_hvac_predictor(wide_frame)
        predictor_column = HVAC_PREDICTOR_OUTPUT_COLUMN
    else:
        raise ValueError(f"Unsupported predictor_kind: {spec.predictor_kind}")

    if spec.model_id == "QAC":
        target, target_metadata = build_hvac_target(
            wide_frame,
            method=hvac_target_method,
        )
        target_column = HVAC_TARGET_OUTPUT_COLUMN
    else:
        physical_target = resolve_present_column(
            spec.target_semantic_name,
            {str(column) for column in wide_frame.columns},
        )
        if physical_target is None:
            definition = get_signal_definition(spec.target_semantic_name)
            raise KeyError(
                f"Target is absent for model_id={model_id}: "
                f"{definition.canonical_column}"
            )
        target = pd.to_numeric(wide_frame[physical_target], errors="coerce")
        target_column = f"target_{model_id}_W"
        target = pd.Series(
            target,
            index=wide_frame.index,
            name=target_column,
            dtype="float64",
        )
        target_metadata = {
            "feature_name": target_column,
            "feature_family": "regression_target",
            "units": spec.expected_target_units,
            "source_columns": [physical_target],
            "formula": physical_target,
        }

    aligned_frame, alignment_metadata = align_predictor_target(
        timestamp_frame=timestamp_frame,
        predictor=predictor,
        target=target,
        predictor_column=str(predictor_column),
        target_column=target_column,
        drop_invalid=drop_invalid,
    )
    feature_metadata = {
        **feature_metadata,
        "model_id": model_id,
        "display_name": spec.display_name,
        "source_family": spec.source_family,
        "component": spec.component,
        "predictor_kind": spec.predictor_kind,
        "output_prediction_column": spec.output_prediction_column,
        "target_metadata": target_metadata,
        "timestamp_metadata": timestamp_metadata,
    }
    return ModelFeatureResult(
        model_id=model_id,
        predictor_column=str(predictor_column),
        target_column=target_column,
        frame=aligned_frame,
        feature_metadata=feature_metadata,
        alignment_metadata=alignment_metadata,
    )


def build_feature_summary_row(result: ModelFeatureResult) -> dict[str, Any]:
    """Return a compact CSV-ready summary row."""
    frame = result.frame
    predictor = pd.to_numeric(frame[result.predictor_column], errors="coerce")
    target = pd.to_numeric(frame[result.target_column], errors="coerce")
    valid = frame["pair_valid"].astype(bool)
    return {
        "model_id": result.model_id,
        "predictor_column": result.predictor_column,
        "target_column": result.target_column,
        "row_count": int(len(frame)),
        "valid_pair_count": int(valid.sum()),
        "invalid_pair_count": int((~valid).sum()),
        "predictor_min": _number_or_blank(predictor.min()),
        "predictor_max": _number_or_blank(predictor.max()),
        "predictor_mean": _number_or_blank(predictor.mean()),
        "predictor_std": _number_or_blank(predictor.std(ddof=0)),
        "predictor_unique_count": int(predictor.dropna().nunique()),
        "target_min": _number_or_blank(target.min()),
        "target_max": _number_or_blank(target.max()),
        "target_mean": _number_or_blank(target.mean()),
        "target_std": _number_or_blank(target.std(ddof=0)),
        "target_unique_count": int(target.dropna().nunique()),
    }


def validate_model_feature_result(
    result: ModelFeatureResult,
    *,
    minimum_sample_count: int,
) -> dict[str, Any]:
    """Validate one deterministic feature dataset before splitting/training."""
    summary = build_feature_summary_row(result)
    status = "valid"
    reasons: list[str] = []
    if summary["valid_pair_count"] < minimum_sample_count:
        status = "invalid"
        reasons.append(
            f"valid_pair_count {summary['valid_pair_count']} < {minimum_sample_count}"
        )
    if summary["predictor_unique_count"] <= 1:
        status = "invalid"
        reasons.append("predictor is constant")
    if summary["target_unique_count"] <= 1:
        status = "invalid"
        reasons.append("target is constant")
    return {
        **summary,
        "validation_status": status,
        "validation_reason": " | ".join(reasons) if reasons else "feature pair is valid",
    }


def _number_or_blank(value: Any) -> float | str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return numeric if pd.notna(numeric) else ""


def build_zone_derived_features(
    *,
    wide_frame: pd.DataFrame,
    static_equipment_frame: pd.DataFrame,
    contribution_frame: pd.DataFrame,
    applicable_model_ids: list[str],
    internal_gain_predictor_method: str,
    hvac_target_method: str,
    aggregate_zone_count: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Build one reusable zone-level deterministic feature dataframe.

    The output contains timestamps and unique derived predictors. It does not
    create train/validation/test splits or model-specific regression pair files.
    """
    timestamp_frame, timestamp_metadata = build_timestamp_frame(wide_frame)
    output = timestamp_frame.copy()
    catalog_rows: list[dict[str, Any]] = []

    model_specs = [get_model_specification(model_id) for model_id in applicable_model_ids]
    predictor_kinds = {spec.predictor_kind for spec in model_specs}

    if "ghi" in predictor_kinds:
        ghi, metadata = build_ghi(wide_frame)
        output[ghi.name] = ghi
        catalog_rows.append(_feature_catalog_row(metadata, model_specs))

    internal_sources = sorted(
        {
            spec.source_family
            for spec in model_specs
            if spec.predictor_kind == "corrected_schedule"
        }
    )
    for equipment_type in internal_sources:
        feature, metadata = build_corrected_schedule(
            frame=wide_frame,
            static_equipment_frame=static_equipment_frame,
            contribution_frame=contribution_frame,
            equipment_type=equipment_type,
            predictor_method=internal_gain_predictor_method,
        )
        output[feature.name] = feature
        catalog_rows.append(_feature_catalog_row(metadata, model_specs))

    needs_qhvac = bool({"hvac_thermodynamic", "hvac_power_from_qhvac"} & predictor_kinds)
    if needs_qhvac:
        hvac_predictor, predictor_metadata = build_hvac_predictor(wide_frame)
        hvac_target, target_metadata = build_hvac_target(
            wide_frame,
            method=hvac_target_method,
        )
        output[hvac_predictor.name] = hvac_predictor
        output[hvac_target.name] = hvac_target
        catalog_rows.append(_feature_catalog_row(predictor_metadata, model_specs))
        catalog_rows.append(_feature_catalog_row(target_metadata, model_specs))

    if "hvac_power_from_qhvac" in predictor_kinds:
        phvac_predictor, phvac_target, predictor_metadata, target_metadata = build_phvac_features(
            wide_frame,
            qhvac_target=output[HVAC_TARGET_OUTPUT_COLUMN],
            aggregate_zone_count=aggregate_zone_count,
        )
        output[phvac_predictor.name] = phvac_predictor
        output[phvac_target.name] = phvac_target
        catalog_rows.append(_feature_catalog_row(predictor_metadata, model_specs))
        catalog_rows.append(_feature_catalog_row(target_metadata, model_specs))

    manifest = {
        "row_count": int(len(output)),
        "column_count": int(len(output.columns)),
        "timestamp_metadata": timestamp_metadata,
        "applicable_model_ids": list(applicable_model_ids),
        "derived_feature_columns": [
            column
            for column in output.columns
            if column not in {"timestamp_raw", "timestamp"}
        ],
        "internal_gain_predictor_method": internal_gain_predictor_method,
        "hvac_target_method": hvac_target_method,
        "aggregate_zone_count": int(aggregate_zone_count),
    }
    return output, catalog_rows, manifest


def validate_zone_derived_features(
    frame: pd.DataFrame,
    *,
    minimum_sample_count: int,
) -> list[dict[str, Any]]:
    """Validate every numeric derived feature in a zone-level output."""
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        if column in {"timestamp_raw", "timestamp"}:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        nonnull = numeric.dropna()
        status = "valid"
        reasons: list[str] = []
        if len(nonnull) < minimum_sample_count:
            status = "invalid"
            reasons.append(
                f"non_null_count {len(nonnull)} < {minimum_sample_count}"
            )
        if nonnull.nunique() <= 1:
            status = "invalid"
            reasons.append("feature is constant")
        rows.append(
            {
                "feature_column": column,
                "validation_status": status,
                "validation_reason": (
                    "feature contains sufficient varying data"
                    if not reasons
                    else " | ".join(reasons)
                ),
                "row_count": int(len(numeric)),
                "non_null_count": int(len(nonnull)),
                "nan_count": int(numeric.isna().sum()),
                "nan_fraction": float(numeric.isna().mean()) if len(numeric) else "",
                "zero_count": int((nonnull == 0.0).sum()),
                "zero_fraction": (
                    float((nonnull == 0.0).mean()) if len(nonnull) else ""
                ),
                "minimum": _number_or_blank(nonnull.min()),
                "maximum": _number_or_blank(nonnull.max()),
                "mean": _number_or_blank(nonnull.mean()),
                "standard_deviation": _number_or_blank(nonnull.std(ddof=0)),
                "unique_non_null_count": int(nonnull.nunique()),
            }
        )
    return rows


def _feature_catalog_row(
    metadata: dict[str, Any],
    model_specs: list[Any],
) -> dict[str, Any]:
    feature_name = str(metadata.get("feature_name", ""))
    if feature_name == GHI_OUTPUT_COLUMN:
        model_ids = [spec.model_id for spec in model_specs if spec.predictor_kind == "ghi"]
    elif feature_name == HVAC_PREDICTOR_OUTPUT_COLUMN:
        model_ids = [
            spec.model_id
            for spec in model_specs
            if spec.predictor_kind == "hvac_thermodynamic"
        ]
    elif feature_name == HVAC_TARGET_OUTPUT_COLUMN:
        model_ids = ["QAC"]
    elif feature_name in {PHVAC_PREDICTOR_OUTPUT_COLUMN, PHVAC_TARGET_OUTPUT_COLUMN}:
        model_ids = ["PHVAC"]
    else:
        equipment_type = str(metadata.get("equipment_type", ""))
        model_ids = [
            spec.model_id
            for spec in model_specs
            if spec.predictor_kind == "corrected_schedule"
            and spec.source_family == equipment_type
        ]
    return {
        "feature_name": feature_name,
        "feature_family": metadata.get("feature_family", ""),
        "units": metadata.get("units", ""),
        "formula": metadata.get("formula", ""),
        "source_columns": " | ".join(
            str(item) for item in metadata.get("source_columns", [])
        ),
        "equipment_type": metadata.get("equipment_type", ""),
        "predictor_method": metadata.get("predictor_method", ""),
        "target_method": metadata.get("target_method", ""),
        "applicable_model_ids": " | ".join(model_ids),
    }
