# -*- coding: utf-8 -*-
"""Legacy-v1 aggregation rules for ScaleBridge EnergyPlus variables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from scalebridge.data.aggregation.eio import (
    EQUIPMENT_TYPES,
)


ADDITIVE_NAME_TOKENS = (
    "Heat",
    "Gain",
    "Rate",
    "Power",
    "Energy",
)


@dataclass(frozen=True)
class AggregatedSeriesResult:
    """One aggregated time-series result."""

    aggregate_zone_id: str
    output_variable_name: str
    source_variable_name: str
    rule_family: str
    values: pd.Series
    units: str
    semantic_role: str
    source_key_count: int
    source_keys: tuple[str, ...]


@dataclass(frozen=True)
class StaticEquipmentResult:
    """One static equipment-level result."""

    aggregate_zone_id: str
    equipment_type: str
    output_variable_name: str
    value: float | None
    source_zone_count: int
    source_zones: tuple[str, ...]
    contribution_count: int


@dataclass(frozen=True)
class EquipmentContributionResult:
    """One equipment-level contribution row before final scalar reduction."""

    aggregate_zone_id: str
    equipment_type: str
    source_zone: str
    schedule_name: str
    equipment_level: float | None
    eio_table: str
    matched_schedule_keys: tuple[str, ...]


@dataclass(frozen=True)
class RuleDiagnostic:
    """One rule diagnostic row."""

    severity: str
    rule_family: str
    aggregate_zone_id: str
    source_variable_name: str
    output_variable_name: str
    message: str
    source_zones: str
    matched_key_count: int
    matched_keys: str


@dataclass(frozen=True)
class AggregationRuleOutputs:
    """Complete output from applying rules to a plan."""

    wide_by_zone: dict[str, pd.DataFrame]
    long_frame: pd.DataFrame
    static_equipment_frame: pd.DataFrame
    equipment_contribution_frame: pd.DataFrame
    diagnostics_frame: pd.DataFrame
    rule_summary_frame: pd.DataFrame


def apply_legacy_v1_rules(
    *,
    plan: dict[str, Any],
    variable_frames_by_name: dict[str, pd.DataFrame],
    schedule_equipment_rows: list[dict[str, Any]],
    zone_mapping_rows: list[dict[str, Any]],
    excluded_zone_names: set[str] | None = None,
) -> AggregationRuleOutputs:
    """Apply legacy_v1 aggregation rules to loaded canonical variables."""
    aggregate_groups = extract_aggregate_groups(plan)
    zone_metadata = build_zone_metadata(zone_mapping_rows)
    schedule_rows_by_equipment = group_schedule_rows(schedule_equipment_rows)
    excluded_zone_names = excluded_zone_names or set()

    diagnostics: list[RuleDiagnostic] = []
    series_results: list[AggregatedSeriesResult] = []
    static_results: list[StaticEquipmentResult] = []
    contribution_results: list[EquipmentContributionResult] = []

    schedule_frame = variable_frames_by_name.get("Schedule Value")

    for source_variable_name, long_frame in variable_frames_by_name.items():
        if source_variable_name == "Schedule Value":
            continue

        family = classify_variable_family(source_variable_name)

        for aggregate_zone_id, source_zones in aggregate_groups.items():
            if family in {"Site", "Facility"}:
                result = aggregate_site_or_facility_variable(
                    aggregate_zone_id=aggregate_zone_id,
                    source_zones=source_zones,
                    source_variable_name=source_variable_name,
                    long_frame=long_frame,
                )
                series_results.append(result)

            elif family == "Zone":
                result, diag = aggregate_zone_variable(
                    aggregate_zone_id=aggregate_zone_id,
                    source_zones=source_zones,
                    source_variable_name=source_variable_name,
                    long_frame=long_frame,
                    weight_mode=str(plan.get("weight_mode", "equal")),
                    zone_metadata=zone_metadata,
                )
                if result is not None:
                    series_results.append(result)
                diagnostics.extend(diag)

            elif family == "Surface":
                result, diag = aggregate_surface_variable(
                    aggregate_zone_id=aggregate_zone_id,
                    source_zones=source_zones,
                    source_variable_name=source_variable_name,
                    long_frame=long_frame,
                    weight_mode=str(plan.get("weight_mode", "equal")),
                    zone_metadata=zone_metadata,
                    excluded_zone_names=excluded_zone_names,
                )
                if result is not None:
                    series_results.append(result)
                diagnostics.extend(diag)

            elif family == "System":
                result, diag = aggregate_system_variable(
                    aggregate_zone_id=aggregate_zone_id,
                    source_zones=source_zones,
                    source_variable_name=source_variable_name,
                    long_frame=long_frame,
                    weight_mode=str(plan.get("weight_mode", "equal")),
                    zone_metadata=zone_metadata,
                    system_node_name_pattern=str(
                        plan.get("system_node_name_pattern", "DIRECT AIR INLET NODE")
                    ),
                )
                if result is not None:
                    series_results.append(result)
                diagnostics.extend(diag)

            else:
                diagnostics.append(
                    RuleDiagnostic(
                        severity="warning",
                        rule_family=family,
                        aggregate_zone_id=aggregate_zone_id,
                        source_variable_name=source_variable_name,
                        output_variable_name=legacy_output_name(source_variable_name),
                        message="unsupported variable family; variable skipped",
                        source_zones=" | ".join(source_zones),
                        matched_key_count=0,
                        matched_keys="",
                    )
                )

    if schedule_frame is not None:
        schedule_series, schedule_static, schedule_contrib, schedule_diag = (
            aggregate_schedule_variables(
                schedule_long_frame=schedule_frame,
                aggregate_groups=aggregate_groups,
                schedule_rows_by_equipment=schedule_rows_by_equipment,
            )
        )
        series_results.extend(schedule_series)
        static_results.extend(schedule_static)
        contribution_results.extend(schedule_contrib)
        diagnostics.extend(schedule_diag)
    else:
        for aggregate_zone_id, source_zones in aggregate_groups.items():
            diagnostics.append(
                RuleDiagnostic(
                    severity="warning",
                    rule_family="Schedule",
                    aggregate_zone_id=aggregate_zone_id,
                    source_variable_name="Schedule Value",
                    output_variable_name="Schedule_Value_*",
                    message="Schedule Value variable not generated; schedule aggregation skipped",
                    source_zones=" | ".join(source_zones),
                    matched_key_count=0,
                    matched_keys="",
                )
            )

    wide_by_zone = build_wide_frames_by_zone(series_results)
    long_frame = build_long_frame(series_results)
    static_equipment_frame = build_static_equipment_frame(static_results)
    equipment_contribution_frame = build_equipment_contribution_frame(contribution_results)
    diagnostics_frame = build_diagnostics_frame(diagnostics)
    rule_summary_frame = build_rule_summary_frame(series_results, diagnostics)

    return AggregationRuleOutputs(
        wide_by_zone=wide_by_zone,
        long_frame=long_frame,
        static_equipment_frame=static_equipment_frame,
        equipment_contribution_frame=equipment_contribution_frame,
        diagnostics_frame=diagnostics_frame,
        rule_summary_frame=rule_summary_frame,
    )


def extract_aggregate_groups(plan: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Extract aggregate zone groups from aggregation_plan.json."""
    groups: dict[str, tuple[str, ...]] = {}

    for group in plan.get("aggregate_zones", []):
        aggregate_zone_id = str(group.get("aggregate_zone_id", "")).strip()
        source_zones = tuple(str(item).strip() for item in group.get("source_zones", []))
        if aggregate_zone_id:
            groups[aggregate_zone_id] = source_zones

    return groups


def build_zone_metadata(
    zone_mapping_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Build source-zone metadata for area/volume weighting."""
    metadata: dict[str, dict[str, float | None]] = {}

    for row in zone_mapping_rows:
        source_zone = str(row.get("source_zone", "")).strip()
        if not source_zone:
            continue

        metadata[source_zone] = {
            "floor_area": optional_float(row.get("floor_area_m2")),
            "volume": optional_float(row.get("volume_m3")),
        }

    return metadata


def group_schedule_rows(
    schedule_equipment_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group all schedule mapping rows by equipment type.

    Do not deduplicate here. Multiple EIO equipment objects may legitimately
    exist in the same zone and should appear in the contribution table.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for row in schedule_equipment_rows:
        equipment_type = str(row.get("equipment_type", "")).strip()
        zone_name = str(row.get("zone_name", "")).strip()
        schedule_name = str(row.get("schedule_name", "")).strip()

        if not equipment_type or not zone_name or not schedule_name:
            continue

        grouped.setdefault(equipment_type, []).append(row)

    return grouped


def classify_variable_family(variable_name: str) -> str:
    """Classify EnergyPlus variable into legacy rule family."""
    first_token = variable_name.strip().split(" ", maxsplit=1)[0]

    if first_token == "Site":
        return "Site"
    if first_token == "Facility":
        return "Facility"
    if first_token == "Zone":
        return "Zone"
    if first_token == "Surface":
        return "Surface"
    if first_token == "System":
        return "System"
    if first_token == "Schedule":
        return "Schedule"

    return "Other"


def canonical_long_to_wide(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical long variable dataframe to timestamp/key_value wide form."""
    required = {"timestamp_raw", "key_value", "value"}
    missing = sorted(required.difference(long_frame.columns))
    if missing:
        raise ValueError(f"Canonical long frame missing columns: {missing}")

    wide = long_frame.pivot_table(
        index="timestamp_raw",
        columns="key_value",
        values="value",
        aggfunc="mean",
    )
    wide = wide.sort_index()
    wide.columns = [str(col) for col in wide.columns]
    return wide


def aggregate_site_or_facility_variable(
    *,
    aggregate_zone_id: str,
    source_zones: tuple[str, ...],
    source_variable_name: str,
    long_frame: pd.DataFrame,
) -> AggregatedSeriesResult:
    """Copy site/facility variable directly into aggregate zone."""
    wide = canonical_long_to_wide(long_frame)
    keys = tuple(str(item) for item in wide.columns)

    if len(keys) == 0:
        values = pd.Series(index=wide.index, dtype="float64")
    elif len(keys) == 1:
        values = wide.iloc[:, 0]
    else:
        values = wide.mean(axis=1)

    values = values.copy()
    values.index.name = "timestamp_raw"

    return AggregatedSeriesResult(
        aggregate_zone_id=aggregate_zone_id,
        output_variable_name=legacy_output_name(source_variable_name),
        source_variable_name=source_variable_name,
        rule_family=classify_variable_family(source_variable_name),
        values=values,
        units=first_value(long_frame, "units"),
        semantic_role=first_value(long_frame, "semantic_role"),
        source_key_count=len(keys),
        source_keys=keys,
    )


def aggregate_zone_variable(
    *,
    aggregate_zone_id: str,
    source_zones: tuple[str, ...],
    source_variable_name: str,
    long_frame: pd.DataFrame,
    weight_mode: str,
    zone_metadata: dict[str, dict[str, float | None]],
) -> tuple[AggregatedSeriesResult | None, list[RuleDiagnostic]]:
    """Aggregate Zone-family variable with exact zone key matching."""
    wide = canonical_long_to_wide(long_frame)
    per_zone_series: dict[str, pd.Series] = {}
    matched_keys: list[str] = []
    diagnostics: list[RuleDiagnostic] = []

    for source_zone in source_zones:
        zone_keys = match_zone_keys_exact(wide.columns, source_zone)
        matched_keys.extend(zone_keys)

        if not zone_keys:
            diagnostics.append(
                unmatched_diag(
                    rule_family="Zone",
                    aggregate_zone_id=aggregate_zone_id,
                    source_variable_name=source_variable_name,
                    source_zones=(source_zone,),
                    message="no exact zone key_values matched source zone",
                )
            )
            continue

        per_zone_series[source_zone] = wide[zone_keys].mean(axis=1)

    if not per_zone_series:
        return None, diagnostics

    values = combine_source_zone_series(
        per_zone_series=per_zone_series,
        weight_mode=weight_mode,
        zone_metadata=zone_metadata,
    )

    return (
        AggregatedSeriesResult(
            aggregate_zone_id=aggregate_zone_id,
            output_variable_name=legacy_output_name(source_variable_name),
            source_variable_name=source_variable_name,
            rule_family="Zone",
            values=values,
            units=first_value(long_frame, "units"),
            semantic_role=first_value(long_frame, "semantic_role"),
            source_key_count=len(set(matched_keys)),
            source_keys=tuple(sorted(set(matched_keys))),
        ),
        diagnostics,
    )


def aggregate_surface_variable(
    *,
    aggregate_zone_id: str,
    source_zones: tuple[str, ...],
    source_variable_name: str,
    long_frame: pd.DataFrame,
    weight_mode: str,
    zone_metadata: dict[str, dict[str, float | None]],
    excluded_zone_names: set[str],
) -> tuple[AggregatedSeriesResult | None, list[RuleDiagnostic]]:
    """Aggregate Surface-family variable with excluded-zone leakage protection."""
    wide = canonical_long_to_wide(long_frame)
    per_zone_series: dict[str, pd.Series] = {}
    matched_keys: list[str] = []
    diagnostics: list[RuleDiagnostic] = []
    additive = is_additive_surface_variable(source_variable_name)

    for source_zone in source_zones:
        zone_keys = match_surface_keys(
            wide.columns,
            source_zone=source_zone,
            excluded_zone_names=excluded_zone_names,
        )
        matched_keys.extend(zone_keys)

        if not zone_keys:
            diagnostics.append(
                unmatched_diag(
                    rule_family="Surface",
                    aggregate_zone_id=aggregate_zone_id,
                    source_variable_name=source_variable_name,
                    source_zones=(source_zone,),
                    message="no safe surface key_values matched source zone",
                )
            )
            continue

        if additive:
            per_zone_series[source_zone] = wide[zone_keys].sum(axis=1)
        else:
            per_zone_series[source_zone] = wide[zone_keys].mean(axis=1)

    if not per_zone_series:
        return None, diagnostics

    values = combine_source_zone_series(
        per_zone_series=per_zone_series,
        weight_mode=weight_mode,
        zone_metadata=zone_metadata,
    )

    return (
        AggregatedSeriesResult(
            aggregate_zone_id=aggregate_zone_id,
            output_variable_name=legacy_output_name(source_variable_name),
            source_variable_name=source_variable_name,
            rule_family="Surface",
            values=values,
            units=first_value(long_frame, "units"),
            semantic_role=first_value(long_frame, "semantic_role"),
            source_key_count=len(set(matched_keys)),
            source_keys=tuple(sorted(set(matched_keys))),
        ),
        diagnostics,
    )


def aggregate_system_variable(
    *,
    aggregate_zone_id: str,
    source_zones: tuple[str, ...],
    source_variable_name: str,
    long_frame: pd.DataFrame,
    weight_mode: str,
    zone_metadata: dict[str, dict[str, float | None]],
    system_node_name_pattern: str,
) -> tuple[AggregatedSeriesResult | None, list[RuleDiagnostic]]:
    """Aggregate System-family variable with token-safe zone matching."""
    wide = canonical_long_to_wide(long_frame)
    per_zone_series: dict[str, pd.Series] = {}
    matched_keys: list[str] = []
    diagnostics: list[RuleDiagnostic] = []

    for source_zone in source_zones:
        zone_keys = [
            str(key)
            for key in wide.columns
            if key_contains_tokens(str(key), tokenize_identifier(source_zone))
            and key_contains_tokens(str(key), tokenize_identifier(system_node_name_pattern))
        ]
        matched_keys.extend(zone_keys)

        if not zone_keys:
            diagnostics.append(
                unmatched_diag(
                    rule_family="System",
                    aggregate_zone_id=aggregate_zone_id,
                    source_variable_name=source_variable_name,
                    source_zones=(source_zone,),
                    message=(
                        "no system key_values matched source zone and "
                        f"system node pattern '{system_node_name_pattern}'"
                    ),
                )
            )
            continue

        per_zone_series[source_zone] = wide[zone_keys].mean(axis=1)

    if not per_zone_series:
        return None, diagnostics

    values = combine_source_zone_series(
        per_zone_series=per_zone_series,
        weight_mode=weight_mode,
        zone_metadata=zone_metadata,
    )

    return (
        AggregatedSeriesResult(
            aggregate_zone_id=aggregate_zone_id,
            output_variable_name=legacy_output_name(source_variable_name),
            source_variable_name=source_variable_name,
            rule_family="System",
            values=values,
            units=first_value(long_frame, "units"),
            semantic_role=first_value(long_frame, "semantic_role"),
            source_key_count=len(set(matched_keys)),
            source_keys=tuple(sorted(set(matched_keys))),
        ),
        diagnostics,
    )


def aggregate_schedule_variables(
    *,
    schedule_long_frame: pd.DataFrame,
    aggregate_groups: dict[str, tuple[str, ...]],
    schedule_rows_by_equipment: dict[str, list[dict[str, Any]]],
) -> tuple[
    list[AggregatedSeriesResult],
    list[StaticEquipmentResult],
    list[EquipmentContributionResult],
    list[RuleDiagnostic],
]:
    """Aggregate Schedule Value into equipment-specific schedule columns."""
    schedule_wide = canonical_long_to_wide(schedule_long_frame)

    series_results: list[AggregatedSeriesResult] = []
    static_results: list[StaticEquipmentResult] = []
    contribution_results: list[EquipmentContributionResult] = []
    diagnostics: list[RuleDiagnostic] = []

    for aggregate_zone_id, source_zones in aggregate_groups.items():
        for equipment_type in EQUIPMENT_TYPES:
            equipment_rows = schedule_rows_by_equipment.get(equipment_type, [])

            schedule_keys: list[str] = []
            equipment_levels: list[float] = []
            contributing_zones: list[str] = []

            for row in equipment_rows:
                source_zone = str(row.get("zone_name", "")).strip()
                if source_zone not in source_zones:
                    continue

                schedule_name = str(row.get("schedule_name", "")).strip()
                if not schedule_name:
                    continue

                matched = match_schedule_keys_exact(schedule_wide.columns, schedule_name)
                schedule_keys.extend(matched)

                level = optional_float(row.get("equipment_level"))
                if level is not None:
                    equipment_levels.append(level)

                contributing_zones.append(source_zone)

                contribution_results.append(
                    EquipmentContributionResult(
                        aggregate_zone_id=aggregate_zone_id,
                        equipment_type=equipment_type,
                        source_zone=source_zone,
                        schedule_name=schedule_name,
                        equipment_level=level,
                        eio_table=str(row.get("eio_table", "")),
                        matched_schedule_keys=tuple(matched),
                    )
                )

            output_variable_name = f"Schedule_Value_{equipment_type}"

            if not schedule_keys:
                diagnostics.append(
                    RuleDiagnostic(
                        severity="warning",
                        rule_family="Schedule",
                        aggregate_zone_id=aggregate_zone_id,
                        source_variable_name="Schedule Value",
                        output_variable_name=output_variable_name,
                        message="no exact Schedule Value key_values matched EIO schedule names",
                        source_zones=" | ".join(source_zones),
                        matched_key_count=0,
                        matched_keys="",
                    )
                )
                continue

            unique_schedule_keys = sorted(set(schedule_keys))
            values = schedule_wide[unique_schedule_keys].mean(axis=1)

            series_results.append(
                AggregatedSeriesResult(
                    aggregate_zone_id=aggregate_zone_id,
                    output_variable_name=output_variable_name,
                    source_variable_name="Schedule Value",
                    rule_family="Schedule",
                    values=values,
                    units=first_value(schedule_long_frame, "units"),
                    semantic_role=first_value(schedule_long_frame, "semantic_role"),
                    source_key_count=len(unique_schedule_keys),
                    source_keys=tuple(unique_schedule_keys),
                )
            )

            static_value: float | None
            if equipment_levels:
                static_value = float(sum(equipment_levels) / len(equipment_levels))
            else:
                static_value = None

            static_results.append(
                StaticEquipmentResult(
                    aggregate_zone_id=aggregate_zone_id,
                    equipment_type=equipment_type,
                    output_variable_name=f"{equipment_type}_Level",
                    value=static_value,
                    source_zone_count=len(set(contributing_zones)),
                    source_zones=tuple(sorted(set(contributing_zones))),
                    contribution_count=len(equipment_levels),
                )
            )

    return series_results, static_results, contribution_results, diagnostics


def combine_source_zone_series(
    *,
    per_zone_series: dict[str, pd.Series],
    weight_mode: str,
    zone_metadata: dict[str, dict[str, float | None]],
) -> pd.Series:
    """Combine source-zone series using equal, floor-area, or volume weighting."""
    zone_names = list(per_zone_series.keys())
    frame = pd.DataFrame({zone: per_zone_series[zone] for zone in zone_names})

    normalized_weight_mode = weight_mode.casefold()

    if normalized_weight_mode == "equal":
        return frame.mean(axis=1)

    if normalized_weight_mode == "floor_area":
        weights = [zone_metadata.get(zone, {}).get("floor_area") for zone in zone_names]
    elif normalized_weight_mode == "volume":
        weights = [zone_metadata.get(zone, {}).get("volume") for zone in zone_names]
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    if any(weight is None for weight in weights):
        return frame.mean(axis=1)

    numeric_weights = pd.Series(
        [float(weight) for weight in weights],
        index=zone_names,
        dtype="float64",
    )
    total_weight = float(numeric_weights.sum())
    if total_weight == 0.0:
        return frame.mean(axis=1)

    return frame.mul(numeric_weights, axis=1).sum(axis=1) / total_weight


def build_wide_frames_by_zone(
    series_results: list[AggregatedSeriesResult],
) -> dict[str, pd.DataFrame]:
    """Build one wide aggregated dataframe per aggregate zone."""
    grouped: dict[str, list[AggregatedSeriesResult]] = {}

    for result in series_results:
        grouped.setdefault(result.aggregate_zone_id, []).append(result)

    wide_by_zone: dict[str, pd.DataFrame] = {}

    for aggregate_zone_id, results in grouped.items():
        frame = pd.DataFrame(
            {result.output_variable_name: result.values for result in results}
        )
        frame.index.name = "timestamp_raw"
        wide_by_zone[aggregate_zone_id] = frame.reset_index()

    return wide_by_zone


def build_long_frame(
    series_results: list[AggregatedSeriesResult],
) -> pd.DataFrame:
    """Build long-form aggregated dataframe."""
    frames: list[pd.DataFrame] = []

    for result in series_results:
        current = result.values.rename("value").reset_index()
        current.insert(0, "aggregate_zone_id", result.aggregate_zone_id)
        current.insert(2, "output_variable_name", result.output_variable_name)
        current.insert(3, "source_variable_name", result.source_variable_name)
        current.insert(4, "rule_family", result.rule_family)
        current.insert(5, "units", result.units)
        current.insert(6, "semantic_role", result.semantic_role)
        frames.append(current)

    if not frames:
        return pd.DataFrame(
            columns=[
                "aggregate_zone_id",
                "timestamp_raw",
                "output_variable_name",
                "source_variable_name",
                "rule_family",
                "units",
                "semantic_role",
                "value",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def build_static_equipment_frame(
    static_results: list[StaticEquipmentResult],
) -> pd.DataFrame:
    """Build static equipment output dataframe."""
    return pd.DataFrame(
        [
            {
                "aggregate_zone_id": result.aggregate_zone_id,
                "equipment_type": result.equipment_type,
                "output_variable_name": result.output_variable_name,
                "value": result.value,
                "source_zone_count": result.source_zone_count,
                "source_zones": " | ".join(result.source_zones),
                "contribution_count": result.contribution_count,
            }
            for result in static_results
        ]
    )


def build_equipment_contribution_frame(
    contribution_results: list[EquipmentContributionResult],
) -> pd.DataFrame:
    """Build equipment contribution dataframe."""
    return pd.DataFrame(
        [
            {
                "aggregate_zone_id": result.aggregate_zone_id,
                "equipment_type": result.equipment_type,
                "source_zone": result.source_zone,
                "schedule_name": result.schedule_name,
                "equipment_level": result.equipment_level,
                "eio_table": result.eio_table,
                "matched_schedule_key_count": len(result.matched_schedule_keys),
                "matched_schedule_keys": " | ".join(result.matched_schedule_keys),
            }
            for result in contribution_results
        ]
    )


def build_diagnostics_frame(
    diagnostics: list[RuleDiagnostic],
) -> pd.DataFrame:
    """Build diagnostics dataframe."""
    return pd.DataFrame(
        [
            {
                "severity": diag.severity,
                "rule_family": diag.rule_family,
                "aggregate_zone_id": diag.aggregate_zone_id,
                "source_variable_name": diag.source_variable_name,
                "output_variable_name": diag.output_variable_name,
                "message": diag.message,
                "source_zones": diag.source_zones,
                "matched_key_count": diag.matched_key_count,
                "matched_keys": diag.matched_keys,
            }
            for diag in diagnostics
        ]
    )


def build_rule_summary_frame(
    series_results: list[AggregatedSeriesResult],
    diagnostics: list[RuleDiagnostic],
) -> pd.DataFrame:
    """Build rule application summary dataframe."""
    rows: list[dict[str, Any]] = []

    for result in series_results:
        rows.append(
            {
                "aggregate_zone_id": result.aggregate_zone_id,
                "source_variable_name": result.source_variable_name,
                "output_variable_name": result.output_variable_name,
                "rule_family": result.rule_family,
                "status": "aggregated",
                "source_key_count": result.source_key_count,
                "source_keys": " | ".join(result.source_keys[:100]),
            }
        )

    for diag in diagnostics:
        rows.append(
            {
                "aggregate_zone_id": diag.aggregate_zone_id,
                "source_variable_name": diag.source_variable_name,
                "output_variable_name": diag.output_variable_name,
                "rule_family": diag.rule_family,
                "status": diag.severity,
                "source_key_count": diag.matched_key_count,
                "source_keys": diag.matched_keys,
                "message": diag.message,
            }
        )

    return pd.DataFrame(rows)


def match_zone_keys_exact(columns: Any, source_zone: str) -> list[str]:
    """Match zone variables using exact normalized zone name."""
    source_norm = normalize_identifier(source_zone)
    return [
        str(key)
        for key in columns
        if normalize_identifier(str(key)) == source_norm
    ]


def match_surface_keys(
    columns: Any,
    *,
    source_zone: str,
    excluded_zone_names: set[str],
) -> list[str]:
    """Match surface keys containing source-zone tokens but no excluded-zone tokens."""
    source_tokens = tokenize_identifier(source_zone)
    excluded_token_sets = [
        tokenize_identifier(zone_name)
        for zone_name in excluded_zone_names
        if str(zone_name).strip()
    ]

    matched: list[str] = []
    for key in columns:
        key_text = str(key)
        if not key_contains_tokens(key_text, source_tokens):
            continue

        if any(key_contains_tokens(key_text, token_set) for token_set in excluded_token_sets):
            continue

        matched.append(key_text)

    return matched


def match_schedule_keys_exact(columns: Any, schedule_name: str) -> list[str]:
    """Match Schedule Value keys using exact normalized schedule name."""
    schedule_norm = normalize_identifier(schedule_name)
    return [
        str(key)
        for key in columns
        if normalize_identifier(str(key)) == schedule_norm
    ]


def key_contains_tokens(key_text: str, required_tokens: tuple[str, ...]) -> bool:
    """Return True when all required tokens are present as tokens in key_text."""
    key_tokens = set(tokenize_identifier(key_text))
    return all(token in key_tokens for token in required_tokens)


def tokenize_identifier(value: str) -> tuple[str, ...]:
    """Tokenize EnergyPlus names safely for zone/system/surface matching."""
    text = normalize_identifier(value)
    tokens = tuple(token for token in re.split(r"[^A-Z0-9]+", text) if token)
    return tokens


def normalize_identifier(value: str) -> str:
    """Normalize an EnergyPlus identifier for exact comparisons."""
    return " ".join(str(value).strip().upper().split())


def is_additive_surface_variable(variable_name: str) -> bool:
    """Return True if surface variable should be summed within source zone."""
    return any(token in variable_name for token in ADDITIVE_NAME_TOKENS)


def legacy_output_name(variable_name: str) -> str:
    """Convert EnergyPlus variable name to legacy-style output column name."""
    return variable_name.strip().replace(" ", "_") + "_"


def unmatched_diag(
    *,
    rule_family: str,
    aggregate_zone_id: str,
    source_variable_name: str,
    source_zones: tuple[str, ...],
    message: str,
) -> RuleDiagnostic:
    """Create unmatched-key diagnostic."""
    return RuleDiagnostic(
        severity="warning",
        rule_family=rule_family,
        aggregate_zone_id=aggregate_zone_id,
        source_variable_name=source_variable_name,
        output_variable_name=legacy_output_name(source_variable_name),
        message=message,
        source_zones=" | ".join(source_zones),
        matched_key_count=0,
        matched_keys="",
    )


def first_value(frame: pd.DataFrame, column: str) -> str:
    """Return first non-null value from a dataframe column."""
    if column not in frame.columns:
        return ""

    series = frame[column].dropna()
    if series.empty:
        return ""

    return str(series.iloc[0])


def optional_float(value: Any) -> float | None:
    """Convert optional float-like value."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None