# -*- coding: utf-8 -*-
"""Detect aggregate-zone heat sources and static-level semantics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scalebridge.models.heat_input_regression.registry import INTERNAL_SOURCES
from scalebridge.data.heat_input_regression.signal_catalog import resolve_present_column


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_heat_source_inventory(*, wide_columns: set[str], static_frame: pd.DataFrame, contribution_frame: pd.DataFrame, predictor_method: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    static_audit: list[dict[str, Any]] = []
    for source, _ in INTERNAL_SOURCES:
        static_rows = static_frame[static_frame.get("equipment_type", pd.Series(dtype=str)).astype(str) == source] if not static_frame.empty else pd.DataFrame()
        contrib_rows = contribution_frame[contribution_frame.get("equipment_type", pd.Series(dtype=str)).astype(str) == source] if not contribution_frame.empty else pd.DataFrame()
        stored = float(_numeric(static_rows["value"]).dropna().iloc[0]) if (not static_rows.empty and "value" in static_rows and not _numeric(static_rows["value"]).dropna().empty) else None
        levels = _numeric(contrib_rows["equipment_level"]).dropna() if (not contrib_rows.empty and "equipment_level" in contrib_rows) else pd.Series(dtype=float)
        schedule_semantic = f"schedule_{source.lower()}"
        schedule_column = resolve_present_column(schedule_semantic, wide_columns)
        convective = resolve_present_column(f"target_convective_{source.lower()}", wide_columns)
        radiant = resolve_present_column(f"target_radiant_{source.lower()}", wide_columns)
        visible = resolve_present_column("target_visible_lights", wide_columns) if source == "Lights" else None
        unique_schedules = int(contrib_rows["schedule_name"].dropna().astype(str).nunique()) if "schedule_name" in contrib_rows else 0
        source_present = bool(len(contrib_rows) > 0 or (stored is not None and stored != 0.0))
        inventory.append({
            "equipment_type": source, "source_present": source_present,
            "schedule_present": schedule_column is not None, "schedule_column": schedule_column or "",
            "static_level_present": stored is not None, "static_level_value": stored if stored is not None else "",
            "contribution_count": int(len(contrib_rows)),
            "source_zone_count": int(contrib_rows["source_zone"].dropna().astype(str).nunique()) if "source_zone" in contrib_rows else 0,
            "unique_schedule_count": unique_schedules,
            "convective_target_present": convective is not None, "convective_target_column": convective or "",
            "radiant_target_present": radiant is not None, "radiant_target_column": radiant or "",
            "visible_target_present": visible is not None, "visible_target_column": visible or "",
            "aggregate_average_supported": bool(schedule_column and stored is not None),
            "contribution_sum_supported": bool(len(contrib_rows) > 0 and unique_schedules > 0),
            "selected_predictor_method": predictor_method,
        })
        static_audit.append({
            "equipment_type": source, "stored_static_level": stored if stored is not None else "",
            "contribution_level_sum": float(levels.sum()) if not levels.empty else "",
            "contribution_level_mean": float(levels.mean()) if not levels.empty else "",
            "contribution_count": int(len(levels)), "unique_schedule_count": unique_schedules,
            "selected_predictor_method": predictor_method,
            "note": "Aggregate-average is the default Stage C interpretation; contribution values are retained for optional total-source formulations.",
        })
    return inventory, static_audit
