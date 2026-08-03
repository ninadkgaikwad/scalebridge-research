# -*- coding: utf-8 -*-
"""Model-specific regression-pair dataset construction for Stage C4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scalebridge.data.heat_input_regression.alignment import (
    canonicalize_wide_frame,
)
from scalebridge.data.heat_input_regression.hvac import (
    HVAC_PREDICTOR_OUTPUT_COLUMN,
    HVAC_TARGET_OUTPUT_COLUMN,
    PHVAC_PREDICTOR_OUTPUT_COLUMN,
    PHVAC_TARGET_OUTPUT_COLUMN,
)
from scalebridge.data.heat_input_regression.schedules import (
    corrected_schedule_column_name,
)
from scalebridge.data.heat_input_regression.signal_catalog import (
    get_signal_definition,
    resolve_present_column,
)
from scalebridge.data.heat_input_regression.solar import GHI_OUTPUT_COLUMN
from scalebridge.models.heat_input_regression.registry import (
    get_model_specification,
)

PAIR_COLUMNS = (
    "timestamp_raw",
    "timestamp",
    "source_row_index",
    "split",
    "split_index",
    "split_included",
    "x",
    "y",
    "pair_valid",
    "pair_exclusion_reason",
)


@dataclass(frozen=True)
class RegressionPairDataset:
    """One model-specific, split-aware regression-pair dataset."""

    model_id: str
    predictor_column: str
    target_column: str
    predictor_units: str
    target_units: str
    output_prediction_column: str
    frame: pd.DataFrame
    metadata: dict[str, Any]


def predictor_column_for_model(model_id: str) -> str:
    """Return the C2 predictor column required by a model specification."""
    spec = get_model_specification(model_id)
    if spec.predictor_kind == "ghi":
        return GHI_OUTPUT_COLUMN
    if spec.predictor_kind == "corrected_schedule":
        return corrected_schedule_column_name(spec.source_family)
    if spec.predictor_kind == "hvac_thermodynamic":
        return HVAC_PREDICTOR_OUTPUT_COLUMN
    if spec.predictor_kind == "hvac_power_from_qhvac":
        return PHVAC_PREDICTOR_OUTPUT_COLUMN
    raise ValueError(f"Unsupported predictor_kind: {spec.predictor_kind}")


def target_column_for_model(
    *,
    model_id: str,
    stage_b_columns: set[str],
) -> tuple[str, str]:
    """Return target source kind and physical/derived target column."""
    spec = get_model_specification(model_id)
    if model_id == "QAC":
        return "c2_derived", HVAC_TARGET_OUTPUT_COLUMN
    if model_id == "PHVAC":
        return "c2_derived", PHVAC_TARGET_OUTPUT_COLUMN

    physical = resolve_present_column(spec.target_semantic_name, stage_b_columns)
    if physical is None:
        definition = get_signal_definition(spec.target_semantic_name)
        raise KeyError(
            f"Target is absent for model_id={model_id}: "
            f"{definition.canonical_column}"
        )
    return "stage_b", physical


def build_regression_pair_dataset(
    *,
    model_id: str,
    feature_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    stage_b_frame: pd.DataFrame,
) -> RegressionPairDataset:
    """Join C2 predictor, C3 split assignment, and Stage B/C2 target.

    The returned frame preserves every C2 row. Rows that are excluded by C3 or
    have an invalid predictor/target are retained with pair_valid=False and an
    explicit exclusion reason. Training-ready split files are selected later
    from pair_valid=True rows only.
    """
    spec = get_model_specification(model_id)
    predictor_column = predictor_column_for_model(model_id)
    if predictor_column not in feature_frame.columns:
        raise KeyError(
            f"C2 predictor column is absent for model_id={model_id}: "
            f"{predictor_column}"
        )

    _require_columns(
        feature_frame,
        {"timestamp_raw", "timestamp", predictor_column},
        "C2 feature frame",
    )
    _require_columns(
        split_frame,
        {
            "timestamp_raw",
            "timestamp",
            "source_row_index",
            "split",
            "split_index",
            "included",
            "exclusion_reason",
        },
        "C3 split frame",
    )

    target_source, target_column = target_column_for_model(
        model_id=model_id,
        stage_b_columns={str(column) for column in stage_b_frame.columns},
    )

    features = feature_frame.reset_index(drop=True).copy()
    features["source_row_index"] = np.arange(len(features), dtype="int64")

    split_subset = split_frame[
        [
            "source_row_index",
            "split",
            "split_index",
            "included",
            "exclusion_reason",
        ]
    ].copy()
    if split_subset["source_row_index"].duplicated().any():
        raise ValueError("C3 split assignments contain duplicate source_row_index values")

    joined = features[
        ["timestamp_raw", "timestamp", "source_row_index", predictor_column]
    ].merge(
        split_subset,
        on="source_row_index",
        how="left",
        validate="one_to_one",
    )

    if joined["split"].isna().any():
        raise ValueError("Some C2 rows have no matching C3 split assignment")

    if target_source == "c2_derived":
        if target_column not in features.columns:
            raise KeyError(f"C2 target column is absent: {target_column}")
        joined[target_column] = pd.to_numeric(
            features[target_column], errors="coerce"
        )
    else:
        _require_columns(
            stage_b_frame,
            {"timestamp_raw", target_column},
            "Stage B frame",
        )
        target_lookup = _build_unique_timestamp_target(
            stage_b_frame=stage_b_frame,
            target_column=target_column,
        )

        joined["timestamp"] = pd.to_datetime(
            joined["timestamp"],
            errors="coerce",
        )
        joined = joined.merge(
            target_lookup,
            on="timestamp",
            how="left",
            validate="many_to_one",
        )

    output = pd.DataFrame(index=joined.index)
    output["timestamp_raw"] = joined["timestamp_raw"].astype(str)
    output["timestamp"] = pd.to_datetime(joined["timestamp"], errors="coerce")
    output["source_row_index"] = pd.to_numeric(
        joined["source_row_index"], errors="raise"
    ).astype("int64")
    output["split"] = joined["split"].astype(str)
    output["split_index"] = pd.to_numeric(
        joined["split_index"], errors="raise"
    ).astype("int64")
    output["split_included"] = joined["included"].astype(bool)
    output["x"] = pd.to_numeric(joined[predictor_column], errors="coerce")
    output["y"] = pd.to_numeric(joined[target_column], errors="coerce")

    finite_x = pd.Series(np.isfinite(output["x"]), index=output.index)
    finite_y = pd.Series(np.isfinite(output["y"]), index=output.index)
    output["pair_valid"] = output["split_included"] & finite_x & finite_y
    output["pair_exclusion_reason"] = _build_pair_exclusion_reasons(
        split_included=output["split_included"],
        split_reason=joined["exclusion_reason"],
        x=output["x"],
        y=output["y"],
    )

    output = output[list(PAIR_COLUMNS)]
    metadata = {
        "model_id": model_id,
        "display_name": spec.display_name,
        "source_family": spec.source_family,
        "component": spec.component,
        "predictor_kind": spec.predictor_kind,
        "predictor_column": predictor_column,
        "predictor_units": spec.expected_predictor_units,
        "target_source": target_source,
        "target_column": target_column,
        "target_units": spec.expected_target_units,
        "output_prediction_column": spec.output_prediction_column,
        "fit_intercept": spec.fit_intercept,
        "input_transform": spec.input_transform,
        "model_role": spec.model_role,
        "dependency_model_id": spec.dependency_model_id,
        "target_allocation": spec.target_allocation,
        "source_row_count": int(len(output)),
        "valid_pair_count": int(output["pair_valid"].sum()),
        "invalid_pair_count": int((~output["pair_valid"]).sum()),
    }
    return RegressionPairDataset(
        model_id=model_id,
        predictor_column=predictor_column,
        target_column=target_column,
        predictor_units=spec.expected_predictor_units,
        target_units=spec.expected_target_units,
        output_prediction_column=spec.output_prediction_column,
        frame=output,
        metadata=metadata,
    )


def split_valid_pairs(dataset: RegressionPairDataset) -> dict[str, pd.DataFrame]:
    """Return direct training-ready valid rows for train/validation/test."""
    valid = dataset.frame[dataset.frame["pair_valid"]].copy()
    return {
        split: valid[valid["split"] == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }


def build_dataset_summary(dataset: RegressionPairDataset) -> list[dict[str, Any]]:
    """Build one summary row for each split plus total."""
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "excluded", "total"):
        if split == "total":
            current = dataset.frame
        elif split == "excluded":
            current = dataset.frame[~dataset.frame["pair_valid"]]
        else:
            current = dataset.frame[
                (dataset.frame["split"] == split) & dataset.frame["pair_valid"]
            ]
        x = pd.to_numeric(current["x"], errors="coerce")
        y = pd.to_numeric(current["y"], errors="coerce")
        rows.append(
            {
                "model_id": dataset.model_id,
                "split": split,
                "row_count": int(len(current)),
                "fraction_of_source": (
                    float(len(current) / len(dataset.frame))
                    if len(dataset.frame)
                    else ""
                ),
                "x_non_null_count": int(x.notna().sum()),
                "y_non_null_count": int(y.notna().sum()),
                "x_min": _number_or_blank(x.min()),
                "x_max": _number_or_blank(x.max()),
                "x_mean": _number_or_blank(x.mean()),
                "x_std": _number_or_blank(x.std(ddof=0)),
                "y_min": _number_or_blank(y.min()),
                "y_max": _number_or_blank(y.max()),
                "y_mean": _number_or_blank(y.mean()),
                "y_std": _number_or_blank(y.std(ddof=0)),
            }
        )
    return rows


def build_exclusion_summary(dataset: RegressionPairDataset) -> list[dict[str, Any]]:
    """Summarize model-specific row exclusions."""
    invalid = dataset.frame[~dataset.frame["pair_valid"]].copy()
    if invalid.empty:
        return [
            {
                "model_id": dataset.model_id,
                "exclusion_reason": "none",
                "row_count": 0,
                "fraction_of_source": 0.0,
            }
        ]
    counts = invalid["pair_exclusion_reason"].value_counts(dropna=False)
    return [
        {
            "model_id": dataset.model_id,
            "exclusion_reason": str(reason),
            "row_count": int(count),
            "fraction_of_source": float(count / len(dataset.frame)),
        }
        for reason, count in counts.items()
    ]


def _build_unique_timestamp_target(
    *, stage_b_frame: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    """Return one Stage B target value per canonical physical timestamp.

    C4 intentionally reuses C2's shared timestamp canonicalization utility.
    It does not implement its own timestamp parser or duplicate policy.
    """
    target = stage_b_frame[["timestamp_raw", target_column]].copy()
    target[target_column] = pd.to_numeric(
        target[target_column],
        errors="coerce",
    )

    canonical, metadata = canonicalize_wide_frame(
        target,
        timestamp_column="timestamp_raw",
    )

    conflict_count = int(
        metadata.get("conflicting_source_value_count", 0)
    )
    if conflict_count:
        examples = metadata.get("conflict_audit_rows", [])[:10]
        raise ValueError(
            "Conflicting Stage B target values were found during shared "
            f"timestamp canonicalization: count={conflict_count}, "
            f"examples={examples}"
        )

    lookup = canonical[["timestamp", target_column]].copy()
    lookup["timestamp"] = pd.to_datetime(
        lookup["timestamp"],
        errors="coerce",
    )

    if lookup["timestamp"].isna().any():
        raise ValueError(
            "Shared timestamp canonicalization produced unparsed Stage B "
            f"timestamps: count={int(lookup['timestamp'].isna().sum())}"
        )
    if lookup["timestamp"].duplicated().any():
        raise ValueError(
            "Shared timestamp canonicalization did not produce unique "
            "Stage B target timestamps"
        )

    return lookup


def _build_pair_exclusion_reasons(
    *,
    split_included: pd.Series,
    split_reason: pd.Series,
    x: pd.Series,
    y: pd.Series,
) -> pd.Series:
    reasons: list[str] = []
    for included, existing_reason, x_value, y_value in zip(
        split_included, split_reason, x, y
    ):
        current: list[str] = []
        if not bool(included):
            reason = str(existing_reason).strip()
            current.append(reason or "excluded_by_split")
        if pd.isna(x_value):
            current.append("missing_predictor")
        elif not np.isfinite(float(x_value)):
            current.append("nonfinite_predictor")
        if pd.isna(y_value):
            current.append("missing_target")
        elif not np.isfinite(float(y_value)):
            current.append("nonfinite_target")
        reasons.append(" | ".join(current))
    return pd.Series(reasons, index=x.index, dtype="object")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(str(column) for column in frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _number_or_blank(value: Any) -> float | str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return numeric if np.isfinite(numeric) else ""
