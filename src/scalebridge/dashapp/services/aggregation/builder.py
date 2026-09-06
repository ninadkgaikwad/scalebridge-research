"""Thin Dash-facing builder helpers for Phase B Aggregation campaign definitions.

This module translates UI selections into the already-authoritative B1
``AggregationCampaignDefinition``. It does not construct scientific plans or
execute Aggregation.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from scalebridge.data.aggregation.campaign_definition import (
    AggregationCampaignDefinition,
    AggregationPlanRequest,
)
from scalebridge.data.aggregation.models import (
    AggregationRuleSet,
    AggregationStrategy,
    AggregationWeightMode,
)
from scalebridge.data.aggregation.plans import validate_custom_partition_rows

from .definition_store import definition_root, save_definition


def build_plan_requests(
    *,
    strategies: Iterable[str],
    weight_modes: Iterable[str],
    rule_set: str,
    custom_aggregation_id: str | None = None,
) -> tuple[AggregationPlanRequest, ...]:
    """Build a deterministic strategy × weight-mode request matrix."""
    strategy_values = _clean_unique(strategies)
    weight_values = _clean_unique(weight_modes)
    if not strategy_values:
        raise ValueError("Select at least one Aggregation strategy")
    if not weight_values:
        raise ValueError("Select at least one Aggregation weight mode")

    rule = AggregationRuleSet(rule_set)
    custom_id = str(custom_aggregation_id or "").strip()

    requests: list[AggregationPlanRequest] = []
    for strategy_value in strategy_values:
        strategy = AggregationStrategy(strategy_value)
        for weight_value in weight_values:
            kwargs: dict[str, Any] = {}
            if strategy == AggregationStrategy.CUSTOM_GROUPS:
                if not custom_id:
                    raise ValueError(
                        "Custom Aggregation ID is required when custom_groups is selected"
                    )
                kwargs["custom_aggregation_ids"] = (custom_id,)
            requests.append(
                AggregationPlanRequest(
                    strategy=strategy,
                    weight_mode=AggregationWeightMode(weight_value),
                    rule_set=rule,
                    **kwargs,
                )
            )
    return tuple(requests)


def validate_custom_group_rows(
    *,
    rows: Iterable[dict[str, Any]],
    selected_case_rows: Iterable[dict[str, Any]],
    aggregation_id: str,
) -> list[dict[str, str]]:
    """Validate editable custom grouping rows using the scientific partition rule."""
    aggregation_id = str(aggregation_id or "").strip()
    if not aggregation_id:
        raise ValueError("Custom Aggregation ID is required")

    selected = {
        str(row.get("case_id", "")).strip(): row
        for row in selected_case_rows
        if str(row.get("case_id", "")).strip()
    }
    if not selected:
        raise ValueError("Select at least one Generation case")

    normalized_rows: list[dict[str, str]] = []
    grouped: dict[str, list[dict[str, str]]] = {case_id: [] for case_id in selected}

    for row in rows or []:
        case_id = str(row.get("case_id", "")).strip()
        source_zone = str(row.get("source_zone_name", "")).strip()
        aggregate_zone = str(row.get("aggregate_zone_name", "")).strip()
        if not case_id or not source_zone:
            continue
        if case_id not in selected:
            raise ValueError(f"Custom grouping contains unselected case_id={case_id}")
        normalized = {
            "case_id": case_id,
            "aggregation_id": aggregation_id,
            "source_zone_name": source_zone,
            "aggregate_zone_name": aggregate_zone,
        }
        normalized_rows.append(normalized)
        grouped[case_id].append(normalized)

    for case_id, case_row in selected.items():
        if str(case_row.get("zone_inventory_status", "")) != "available":
            raise ValueError(
                f"Thermal-zone inventory is unavailable for selected case {case_id}"
            )
        approved = {
            str(zone).strip().upper()
            for zone in case_row.get("thermal_zone_names", [])
            if str(zone).strip()
        }
        validation = validate_custom_partition_rows(
            aggregation_id=aggregation_id,
            rows=grouped.get(case_id, []),
            approved_source_zone_set=approved,
        )
        if not validation["valid"]:
            raise ValueError(
                f"Invalid custom grouping for case_id={case_id}: {validation}"
            )

    return normalized_rows


def custom_grouping_path(campaign_id: str) -> Path:
    """Return the persistent CSV path adjacent to saved definitions."""
    return definition_root() / "custom_groups" / f"{campaign_id}.csv"


def save_custom_grouping(
    *,
    campaign_id: str,
    rows: Iterable[dict[str, str]],
) -> Path:
    """Persist validated custom grouping rows in the scientific CSV schema."""
    path = custom_grouping_path(campaign_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "aggregation_id",
        "source_zone_name",
        "aggregate_zone_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def relative_custom_grouping_path(campaign_id: str) -> str:
    """Return the definition-relative path consumed by B1/B2."""
    return str(Path("custom_groups") / f"{campaign_id}.csv")


def build_definition(
    *,
    aggregation_campaign_id: str,
    parent_generation_campaign_id: str,
    machine_id: str,
    case_ids: Iterable[str],
    plan_requests: Iterable[AggregationPlanRequest],
    custom_zone_groups_path: str | None,
    case_limit: int | None = None,
    max_variables: int | None = None,
    preview_rows: int = 100,
    write_legacy_pickle: bool = False,
    continue_on_error: bool = True,
    aggregate_zone_name_stem: str = "Aggregated_Zone",
    system_node_name_pattern: str = "DIRECT AIR INLET NODE",
    mlflow_enabled: bool = True,
    mlflow_tracking_uri: str | None = "http://127.0.0.1:5000",
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_strict: bool = False,
) -> AggregationCampaignDefinition:
    """Create the authoritative B1 model from UI-safe primitive values."""
    return AggregationCampaignDefinition(
        aggregation_campaign_id=str(aggregation_campaign_id or "").strip(),
        parent_generation_campaign_id=str(parent_generation_campaign_id or "").strip(),
        machine_id=str(machine_id or "").strip(),
        case_ids=tuple(_clean_unique(case_ids)),
        case_limit=_optional_positive_int(case_limit),
        plan_requests=tuple(plan_requests),
        custom_zone_groups_path=custom_zone_groups_path,
        max_variables=_optional_positive_int(max_variables),
        preview_rows=int(preview_rows or 0),
        write_legacy_pickle=bool(write_legacy_pickle),
        continue_on_error=bool(continue_on_error),
        aggregate_zone_name_stem=str(aggregate_zone_name_stem or "").strip(),
        system_node_name_pattern=str(system_node_name_pattern or "").strip(),
        mlflow_enabled=bool(mlflow_enabled),
        mlflow_tracking_uri=str(mlflow_tracking_uri or "").strip() or None,
        mlflow_experiment_name=str(mlflow_experiment_name or "").strip() or None,
        mlflow_run_name=str(mlflow_run_name or "").strip() or None,
        mlflow_strict=bool(mlflow_strict),
    )


def save_builder_definition(
    *,
    definition: AggregationCampaignDefinition,
    custom_rows: Iterable[dict[str, str]] | None = None,
) -> tuple[Path, Path | None]:
    """Persist custom grouping first, then the validated B1 definition."""
    custom_path: Path | None = None
    if definition.custom_zone_groups_path:
        custom_path = save_custom_grouping(
            campaign_id=definition.aggregation_campaign_id,
            rows=custom_rows or [],
        )
    definition_path = save_definition(definition)
    return definition_path, custom_path


def _clean_unique(values: Iterable[Any] | None) -> list[str]:
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in (values or [])
            if str(value).strip()
        )
    )


def _optional_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("Expected a positive integer")
    return parsed
