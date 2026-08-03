# -*- coding: utf-8 -*-
"""Internal-gain schedule feature construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scalebridge.data.heat_input_regression.signal_catalog import resolve_present_column

SUPPORTED_INTERNAL_GAIN_PREDICTOR_METHODS = (
    "aggregate_average",
    "contribution_sum",
)


def corrected_schedule_column_name(equipment_type: str) -> str:
    """Return the canonical corrected-schedule feature column."""
    return f"derived_Corrected_Schedule_{equipment_type}"


def build_aggregate_average_corrected_schedule(
    *,
    frame: pd.DataFrame,
    static_equipment_frame: pd.DataFrame,
    equipment_type: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Build aggregate-average corrected schedule.

    X_s(t) = Schedule_Value_s(t) * Aggregated_Static_Level_s

    Both quantities inherit the Stage B aggregate-zone interpretation and
    selected aggregation weight mode.
    """
    semantic_name = f"schedule_{equipment_type.lower()}"
    schedule_column = resolve_present_column(
        semantic_name,
        {str(column) for column in frame.columns},
    )
    if schedule_column is None:
        raise KeyError(
            f"Aggregate schedule is absent for equipment_type={equipment_type}"
        )

    if static_equipment_frame.empty:
        raise ValueError("Static equipment table is empty")

    current = static_equipment_frame[
        static_equipment_frame["equipment_type"].astype(str) == equipment_type
    ].copy()
    if current.empty:
        raise KeyError(
            f"Static equipment level is absent for equipment_type={equipment_type}"
        )

    level_values = pd.to_numeric(current["value"], errors="coerce").dropna()
    if level_values.empty:
        raise ValueError(
            f"Static equipment level is non-numeric for equipment_type={equipment_type}"
        )

    # The zone-level static table should contain one row per equipment type.
    # Mean is defensive if duplicate rows are present.
    static_level = float(level_values.mean())
    schedule = pd.to_numeric(frame[schedule_column], errors="coerce")
    output_column = corrected_schedule_column_name(equipment_type)
    corrected = pd.Series(
        schedule * static_level,
        index=frame.index,
        name=output_column,
        dtype="float64",
    )

    metadata = {
        "feature_name": output_column,
        "feature_family": "internal_gain",
        "equipment_type": equipment_type,
        "predictor_method": "aggregate_average",
        "formula": "aggregate_schedule * aggregate_static_level",
        "units": _corrected_schedule_units(equipment_type),
        "source_columns": [schedule_column],
        "static_level": static_level,
        "static_level_output_name": str(
            current.iloc[0].get("output_variable_name", "")
        ),
    }
    return corrected, metadata


def build_corrected_schedule(
    *,
    frame: pd.DataFrame,
    static_equipment_frame: pd.DataFrame,
    contribution_frame: pd.DataFrame,
    equipment_type: str,
    predictor_method: str,
) -> tuple[pd.Series, dict[str, Any]]:
    """Build one corrected schedule using the selected method."""
    if predictor_method not in SUPPORTED_INTERNAL_GAIN_PREDICTOR_METHODS:
        raise ValueError(
            f"Unsupported internal-gain predictor method: {predictor_method}"
        )

    if predictor_method == "aggregate_average":
        return build_aggregate_average_corrected_schedule(
            frame=frame,
            static_equipment_frame=static_equipment_frame,
            equipment_type=equipment_type,
        )

    raise NotImplementedError(
        "contribution_sum requires contribution-specific source Schedule Value "
        "time series. Stage B currently exposes only the aggregate schedule in "
        "the zone-wide output. The method is reserved in the public interface "
        "but cannot be reconstructed exactly from current zone artifacts."
    )


def audit_static_level(
    *,
    static_equipment_frame: pd.DataFrame,
    contribution_frame: pd.DataFrame,
    equipment_type: str,
) -> dict[str, Any]:
    """Return transparent aggregate-average versus contribution statistics."""
    static_rows = static_equipment_frame[
        static_equipment_frame.get("equipment_type", pd.Series(dtype=str)).astype(str)
        == equipment_type
    ]
    contribution_rows = contribution_frame[
        contribution_frame.get("equipment_type", pd.Series(dtype=str)).astype(str)
        == equipment_type
    ]

    stored = pd.to_numeric(static_rows.get("value"), errors="coerce").dropna()
    levels = pd.to_numeric(
        contribution_rows.get("equipment_level"), errors="coerce"
    ).dropna()
    return {
        "equipment_type": equipment_type,
        "stored_static_level": float(stored.mean()) if not stored.empty else "",
        "contribution_level_sum": float(levels.sum()) if not levels.empty else "",
        "contribution_level_mean": float(levels.mean()) if not levels.empty else "",
        "contribution_count": int(len(levels)),
    }


def _corrected_schedule_units(equipment_type: str) -> str:
    return "people" if equipment_type == "People" else "W"
