# -*- coding: utf-8 -*-
"""Memory-safe aggregation for System Node Mass Flow Rate.

This module handles System Node Mass Flow Rate separately from the normal
zone/surface/system aggregation path because this variable can be very large
for ApartmentMidRise and because its key_value entries are node names rather
than direct zone names.

Initial supported node naming conventions
-----------------------------------------
A node key is mapped to a source zone when the normalized key starts with:

    <source zone> DIRECT AIR INLET NODE NAME
    <source zone> ZONE EQUIP INLET

Examples:
    CORE_ZN DIRECT AIR INLET NODE NAME -> Core_ZN
    LGSTORE1 DIRECT AIR INLET NODE NAME -> LGstore1
    DINING DIRECT AIR INLET NODE NAME -> Dining
    G N1 APARTMENT ZONE EQUIP INLET -> G N1 Apartment

The suffix list is centralized in node_mapping.py so additional conventions
can be added later, for example ERV SUP FAN OUTLET NODE, RETURN AIR NODE NAME, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scalebridge.data.aggregation.node_mapping import (
    DEFAULT_ZONE_DELIVERY_NODE_SUFFIXES,
    SourceZoneNodeMatcher,
    build_source_zone_to_aggregate_zone,
    extract_aggregate_groups,
)
from scalebridge.data.aggregation.rules import AggregationRuleOutputs


SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME = "System Node Mass Flow Rate"
SYSTEM_NODE_MASS_FLOW_OUTPUT_NAME = "System_Node_Mass_Flow_Rate"

DEFAULT_NODE_SUFFIX_PATTERNS = DEFAULT_ZONE_DELIVERY_NODE_SUFFIXES

REQUIRED_PARQUET_COLUMNS = (
    "timestamp_raw",
    "key_value",
    "variable_name",
    "units",
    "semantic_role",
    "value",
)


@dataclass(frozen=True)
class SystemNodeMassFlowAggregationResult:
    """Aggregated system-node mass-flow output and diagnostics."""

    long_frame: pd.DataFrame
    wide_by_zone: dict[str, pd.DataFrame]
    mapping_frame: pd.DataFrame
    unmapped_nodes_frame: pd.DataFrame
    diagnostics_frame: pd.DataFrame
    rule_summary_frame: pd.DataFrame
    source_key_count: int
    mapped_key_count: int
    unmapped_key_count: int
    mapped_row_count: int
    skipped_row_count: int


def aggregate_system_node_mass_flow_rate_from_parquet(
    *,
    parquet_path: Path,
    plan: dict[str, Any],
    batch_size: int = 250_000,
    node_suffix_patterns: tuple[str, ...] = DEFAULT_NODE_SUFFIX_PATTERNS,
) -> SystemNodeMassFlowAggregationResult:
    """Aggregate System Node Mass Flow Rate from parquet in batches.

    Parameters
    ----------
    parquet_path:
        Canonical timestep_system_node_mass_flow_rate.parquet path.
    plan:
        Aggregation plan containing aggregate_zones/source_zones.
    batch_size:
        Number of rows per pyarrow batch.
    node_suffix_patterns:
        Node naming suffixes used after the source zone name.

    Returns
    -------
    SystemNodeMassFlowAggregationResult
        Aggregated long/wide outputs and diagnostics.
    """
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"System node mass-flow parquet not found: {parquet_path}")

    aggregate_groups = extract_aggregate_groups(plan)
    source_zone_to_aggregate_zone = build_source_zone_to_aggregate_zone(aggregate_groups)

    matcher = SourceZoneNodeMatcher(
        source_zones=tuple(source_zone_to_aggregate_zone.keys()),
        suffix_patterns=node_suffix_patterns,
    )

    parquet_file = pq.ParquetFile(parquet_path)

    available_columns = set(parquet_file.schema.names)
    missing = sorted(set(REQUIRED_PARQUET_COLUMNS).difference(available_columns))
    if missing:
        raise ValueError(
            f"System node mass-flow parquet missing required columns {missing}: {parquet_path}"
        )

    partial_group_frames: list[pd.DataFrame] = []
    key_mapping_cache: dict[str, str | None] = {}

    unique_keys_seen: set[str] = set()
    mapped_keys_seen: set[str] = set()
    unmapped_keys_seen: set[str] = set()

    mapped_row_count = 0
    skipped_row_count = 0
    units = ""
    semantic_role = ""

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=list(REQUIRED_PARQUET_COLUMNS),
    ):
        frame = batch.to_pandas()

        if frame.empty:
            continue

        if not units:
            units = first_non_empty(frame.get("units"))
        if not semantic_role:
            semantic_role = first_non_empty(frame.get("semantic_role"))

        key_values = frame["key_value"].astype(str)
        batch_unique_keys = key_values.dropna().drop_duplicates().tolist()

        for key in batch_unique_keys:
            if key not in key_mapping_cache:
                key_mapping_cache[key] = matcher.match_source_zone(key)
            unique_keys_seen.add(key)
            if key_mapping_cache[key]:
                mapped_keys_seen.add(key)
            else:
                unmapped_keys_seen.add(key)

        frame["source_zone"] = key_values.map(key_mapping_cache)
        mapped = frame[frame["source_zone"].notna()].copy()
        skipped_row_count += int(len(frame) - len(mapped))

        if mapped.empty:
            continue

        mapped["aggregate_zone_id"] = mapped["source_zone"].map(
            source_zone_to_aggregate_zone
        )
        mapped = mapped[mapped["aggregate_zone_id"].notna()].copy()

        if mapped.empty:
            continue

        mapped["value"] = pd.to_numeric(mapped["value"], errors="coerce").fillna(0.0)

        grouped = (
            mapped.groupby(["aggregate_zone_id", "timestamp_raw"], as_index=False)["value"]
            .sum()
        )
        partial_group_frames.append(grouped)
        mapped_row_count += int(len(mapped))

    mapping_rows = []
    for key in sorted(unique_keys_seen):
        source_zone = key_mapping_cache.get(key)
        mapping_rows.append(
            {
                "key_value": key,
                "match_status": "mapped" if source_zone else "unmapped",
                "source_zone": source_zone or "",
                "aggregate_zone_id": (
                    source_zone_to_aggregate_zone.get(source_zone, "")
                    if source_zone
                    else ""
                ),
                "matched_suffix_pattern": (
                    matcher.match_suffix_pattern(key, source_zone)
                    if source_zone
                    else ""
                ),
            }
        )

    mapping_frame = pd.DataFrame(mapping_rows)
    unmapped_nodes_frame = mapping_frame[
        mapping_frame["match_status"] == "unmapped"
    ].copy()

    if partial_group_frames:
        grouped_all = (
            pd.concat(partial_group_frames, ignore_index=True)
            .groupby(["aggregate_zone_id", "timestamp_raw"], as_index=False)["value"]
            .sum()
        )
    else:
        grouped_all = pd.DataFrame(
            columns=["aggregate_zone_id", "timestamp_raw", "value"]
        )

    long_frame = build_mass_flow_long_frame(
        grouped_all=grouped_all,
        units=units or "kg/s",
        semantic_role=semantic_role or "mass_flow",
    )
    wide_by_zone = build_mass_flow_wide_by_zone(grouped_all)

    diagnostics_frame = build_mass_flow_diagnostics_frame(
        aggregate_groups=aggregate_groups,
        source_zone_to_aggregate_zone=source_zone_to_aggregate_zone,
        mapped_keys=mapped_keys_seen,
        unmapped_keys=unmapped_keys_seen,
        matcher=matcher,
    )

    rule_summary_frame = build_mass_flow_rule_summary_frame(
        aggregate_groups=aggregate_groups,
        mapping_frame=mapping_frame,
    )

    return SystemNodeMassFlowAggregationResult(
        long_frame=long_frame,
        wide_by_zone=wide_by_zone,
        mapping_frame=mapping_frame,
        unmapped_nodes_frame=unmapped_nodes_frame,
        diagnostics_frame=diagnostics_frame,
        rule_summary_frame=rule_summary_frame,
        source_key_count=len(unique_keys_seen),
        mapped_key_count=len(mapped_keys_seen),
        unmapped_key_count=len(unmapped_keys_seen),
        mapped_row_count=mapped_row_count,
        skipped_row_count=skipped_row_count,
    )


def merge_system_node_mass_flow_outputs(
    *,
    outputs: AggregationRuleOutputs,
    mass_flow_result: SystemNodeMassFlowAggregationResult,
) -> AggregationRuleOutputs:
    """Merge special mass-flow output into normal aggregation-rule outputs."""
    merged_wide_by_zone: dict[str, pd.DataFrame] = {
        zone_id: frame.copy()
        for zone_id, frame in outputs.wide_by_zone.items()
    }

    for aggregate_zone_id, mass_flow_wide in mass_flow_result.wide_by_zone.items():
        if aggregate_zone_id in merged_wide_by_zone:
            merged_wide_by_zone[aggregate_zone_id] = merged_wide_by_zone[
                aggregate_zone_id
            ].merge(
                mass_flow_wide,
                on="timestamp_raw",
                how="outer",
            )
        else:
            merged_wide_by_zone[aggregate_zone_id] = mass_flow_wide.copy()

    merged_long_frame = concat_dataframes(
        outputs.long_frame,
        mass_flow_result.long_frame,
    )
    merged_diagnostics_frame = concat_dataframes(
        outputs.diagnostics_frame,
        mass_flow_result.diagnostics_frame,
    )
    merged_rule_summary_frame = concat_dataframes(
        outputs.rule_summary_frame,
        mass_flow_result.rule_summary_frame,
    )

    return AggregationRuleOutputs(
        wide_by_zone=merged_wide_by_zone,
        long_frame=merged_long_frame,
        static_equipment_frame=outputs.static_equipment_frame,
        equipment_contribution_frame=outputs.equipment_contribution_frame,
        diagnostics_frame=merged_diagnostics_frame,
        rule_summary_frame=merged_rule_summary_frame,
    )


def build_mass_flow_long_frame(
    *,
    grouped_all: pd.DataFrame,
    units: str,
    semantic_role: str,
) -> pd.DataFrame:
    """Build long-form mass-flow aggregation output."""
    if grouped_all.empty:
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

    long_frame = grouped_all.copy()
    long_frame.insert(2, "output_variable_name", SYSTEM_NODE_MASS_FLOW_OUTPUT_NAME)
    long_frame.insert(3, "source_variable_name", SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME)
    long_frame.insert(4, "rule_family", "SystemNodeMassFlow")
    long_frame.insert(5, "units", units)
    long_frame.insert(6, "semantic_role", semantic_role)
    return long_frame[
        [
            "aggregate_zone_id",
            "timestamp_raw",
            "output_variable_name",
            "source_variable_name",
            "rule_family",
            "units",
            "semantic_role",
            "value",
        ]
    ]


def build_mass_flow_wide_by_zone(
    grouped_all: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build one wide mass-flow dataframe per aggregate zone."""
    wide_by_zone: dict[str, pd.DataFrame] = {}

    if grouped_all.empty:
        return wide_by_zone

    for aggregate_zone_id, current in grouped_all.groupby("aggregate_zone_id"):
        wide = current[["timestamp_raw", "value"]].copy()
        wide = wide.rename(columns={"value": SYSTEM_NODE_MASS_FLOW_OUTPUT_NAME})
        wide_by_zone[str(aggregate_zone_id)] = wide.sort_values("timestamp_raw")

    return wide_by_zone


def build_mass_flow_diagnostics_frame(
    *,
    aggregate_groups: dict[str, tuple[str, ...]],
    source_zone_to_aggregate_zone: dict[str, str],
    mapped_keys: set[str],
    unmapped_keys: set[str],
    matcher: SourceZoneNodeMatcher,
) -> pd.DataFrame:
    """Build diagnostics rows for system-node mass-flow mapping."""
    rows: list[dict[str, Any]] = []

    mapped_source_zones = set()
    for key in mapped_keys:
        source_zone = matcher.match_source_zone(key)
        if source_zone:
            mapped_source_zones.add(source_zone)

    for aggregate_zone_id, source_zones in aggregate_groups.items():
        source_zones_without_nodes = [
            source_zone
            for source_zone in source_zones
            if source_zone not in mapped_source_zones
        ]
        if source_zones_without_nodes:
            rows.append(
                {
                    "severity": "warning",
                    "rule_family": "SystemNodeMassFlow",
                    "aggregate_zone_id": aggregate_zone_id,
                    "source_variable_name": SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME,
                    "output_variable_name": SYSTEM_NODE_MASS_FLOW_OUTPUT_NAME,
                    "message": (
                        "some source zones in this aggregate zone did not match "
                        "known system-node mass-flow naming patterns"
                    ),
                    "source_zones": " | ".join(source_zones_without_nodes),
                    "matched_key_count": len(mapped_keys),
                    "matched_keys": " | ".join(sorted(mapped_keys)[:100]),
                }
            )

    if unmapped_keys:
        rows.append(
            {
                "severity": "info",
                "rule_family": "SystemNodeMassFlow",
                "aggregate_zone_id": "",
                "source_variable_name": SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME,
                "output_variable_name": SYSTEM_NODE_MASS_FLOW_OUTPUT_NAME,
                "message": (
                    "unmapped system-node mass-flow keys were excluded; see "
                    "system_node_mass_flow_unmapped_nodes.csv"
                ),
                "source_zones": "",
                "matched_key_count": len(mapped_keys),
                "matched_keys": " | ".join(sorted(mapped_keys)[:100]),
            }
        )

    return pd.DataFrame(rows)


def build_mass_flow_rule_summary_frame(
    *,
    aggregate_groups: dict[str, tuple[str, ...]],
    mapping_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build rule summary rows for the special mass-flow aggregation."""
    rows: list[dict[str, Any]] = []

    if mapping_frame.empty:
        for aggregate_zone_id in aggregate_groups:
            rows.append(
                {
                    "aggregate_zone_id": aggregate_zone_id,
                    "source_variable_name": SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME,
                    "output_variable_name": SYSTEM_NODE_MASS_FLOW_OUTPUT_NAME,
                    "rule_family": "SystemNodeMassFlow",
                    "status": "warning",
                    "source_key_count": 0,
                    "source_keys": "",
                    "message": "no system-node mass-flow keys mapped",
                }
            )
        return pd.DataFrame(rows)

    mapped = mapping_frame[mapping_frame["match_status"] == "mapped"].copy()

    for aggregate_zone_id in aggregate_groups:
        current = mapped[mapped["aggregate_zone_id"] == aggregate_zone_id]
        rows.append(
            {
                "aggregate_zone_id": aggregate_zone_id,
                "source_variable_name": SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME,
                "output_variable_name": SYSTEM_NODE_MASS_FLOW_OUTPUT_NAME,
                "rule_family": "SystemNodeMassFlow",
                "status": "aggregated" if not current.empty else "warning",
                "source_key_count": int(len(current)),
                "source_keys": " | ".join(current["key_value"].astype(str).head(100)),
                "message": "" if not current.empty else "no mapped source node keys",
            }
        )

    return pd.DataFrame(rows)


def concat_dataframes(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two dataframes while preserving empty-frame behavior."""
    if left is None or left.empty:
        return right.copy()
    if right is None or right.empty:
        return left.copy()
    return pd.concat([left, right], ignore_index=True)


def first_non_empty(series: Any) -> str:
    """Return first non-empty value from a pandas Series-like object."""
    if series is None:
        return ""
    try:
        values = pd.Series(series).dropna().astype(str)
        values = values[values.str.strip() != ""]
        if values.empty:
            return ""
        return str(values.iloc[0])
    except Exception:
        return ""
