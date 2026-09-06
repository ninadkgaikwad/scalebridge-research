"""Artifact-first discovery of completed Phase C runs for Phase D."""
from __future__ import annotations

from typing import Any

from scalebridge.dashapp.services.heat_input.results_data import (
    discover_phase_c_runs,
    load_run_ref_from_key,
)
from scalebridge.data.thermal_modeling.campaign_runner import load_matrix_aggregation_runs

_SUCCESS = {"completed", "completed_with_warnings"}


def completed_phase_c_runs() -> list[dict[str, Any]]:
    return [
        row
        for row in discover_phase_c_runs()
        if str(row.get("status") or "").strip().casefold() in _SUCCESS
    ]


def phase_c_run_options() -> list[dict[str, str]]:
    return [
        {
            "label": (
                f"{row['phase_c_run_id']} | {row['campaign_id']} | "
                f"{row['status']} | matrix {row.get('matrix_run_id') or 'unknown'}"
            ),
            "value": f"{row['campaign_id']}::{row['phase_c_run_id']}",
        }
        for row in completed_phase_c_runs()
    ]


def resolve_phase_c_context(run_key: str) -> dict[str, Any]:
    ref = load_run_ref_from_key(run_key)
    manifest = dict(ref["manifest"])
    matrix_run_id = str(manifest.get("matrix_run_id") or "").strip()
    if not matrix_run_id:
        raise ValueError("Selected Phase C run does not record matrix_run_id")

    campaign_root = ref["campaign_root"]
    items = load_matrix_aggregation_runs(campaign_root, matrix_run_id=matrix_run_id)
    rows = [item.__dict__.copy() for item in items]
    generation_campaign_id = str(manifest.get("campaign_id") or ref["campaign_id"])

    return {
        "parent_generation_campaign_id": generation_campaign_id,
        "phase_c_campaign_id": str(ref["campaign_id"]),
        "phase_c_campaign_run_id": str(ref["phase_c_run_id"]),
        "phase_c_run_key": run_key,
        "campaign_root": str(campaign_root),
        "matrix_run_id": matrix_run_id,
        "phase_c_status": str(manifest.get("status") or ""),
        "phase_c_created_at_utc": str(manifest.get("created_at_utc") or ""),
        "aggregation_rows": rows,
        "aggregation_run_count": len(rows),
        "case_ids": sorted({item.case_id for item in items}, key=str.casefold),
        "aggregation_ids": sorted({item.aggregation_id for item in items}, key=str.casefold),
        "weight_modes": sorted({item.weight_mode for item in items}, key=str.casefold),
        "strategies": sorted({item.plan_strategy for item in items}, key=str.casefold),
        "rule_sets": sorted({item.rule_set for item in items}, key=str.casefold),
        "buildings": sorted({item.building_type for item in items}, key=str.casefold),
        "weather_locations": sorted({item.weather_location for item in items}, key=str.casefold),
    }


def case_options(context: dict[str, Any]) -> list[dict[str, str]]:
    labels: dict[str, str] = {}
    for row in context.get("aggregation_rows") or []:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        building = str(row.get("building_type") or "Unknown building")
        weather = str(row.get("weather_location") or "Unknown weather")
        labels[case_id] = f"{building} | {weather} | {case_id}"
    return [
        {"label": labels[value], "value": value}
        for value in sorted(labels, key=lambda item: labels[item].casefold())
    ]


def aggregation_options(context: dict[str, Any]) -> list[dict[str, str]]:
    details: dict[str, set[str]] = {}
    for row in context.get("aggregation_rows") or []:
        aggregation_id = str(row.get("aggregation_id") or "")
        if not aggregation_id:
            continue
        descriptor = " / ".join(
            part
            for part in (
                str(row.get("plan_strategy") or ""),
                str(row.get("rule_set") or ""),
            )
            if part
        )
        details.setdefault(aggregation_id, set()).add(descriptor)
    options = []
    for aggregation_id in sorted(details, key=str.casefold):
        suffix = "; ".join(sorted(filter(None, details[aggregation_id]), key=str.casefold))
        label = f"{aggregation_id} | {suffix}" if suffix else aggregation_id
        options.append({"label": label, "value": aggregation_id})
    return options


def selected_aggregation_count(
    context: dict[str, Any],
    *,
    case_ids: list[str] | tuple[str, ...] | None = None,
    aggregation_ids: list[str] | tuple[str, ...] | None = None,
    weight_modes: list[str] | tuple[str, ...] | None = None,
    max_aggregation_runs: int | None = None,
) -> int:
    cases = set(case_ids or [])
    aggregations = set(aggregation_ids or [])
    weights = set(weight_modes or [])
    rows = list(context.get("aggregation_rows") or [])
    matched = [
        row
        for row in rows
        if (not cases or row.get("case_id") in cases)
        and (not aggregations or row.get("aggregation_id") in aggregations)
        and (not weights or row.get("weight_mode") in weights)
    ]
    if max_aggregation_runs is not None:
        matched = matched[: max(0, int(max_aggregation_runs))]
    return len(matched)
