# -*- coding: utf-8 -*-
"""Discover exact Stage B aggregate-zone outputs for Stage C."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.discovery import resolve_campaign_root, resolve_repo_root
from scalebridge.data.heat_input_regression.models import AggregationZoneOutputRef


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _optional(path: Path) -> Path | None:
    return path if path.is_file() else None


def infer_aggregation_level(aggregation_id: str) -> str:
    match = re.search(r"(?:^|_)[lL](\d{1,2})(?:_|$)", aggregation_id)
    return f"L{int(match.group(1)):02d}" if match else aggregation_id


def infer_aggregation_family(aggregation_id: str, strategy: str) -> str:
    text = f"{aggregation_id} {strategy}".casefold()
    if "all_to_one" in text or "all_thermal_zones_to_one" in text:
        return "all_to_one"
    if "identity" in text:
        return "identity"
    if "functional" in text:
        return "functional"
    if "intermediate" in text:
        return "intermediate"
    if "spatial" in text:
        return "spatial_detailed"
    return "custom"


def discover_from_matrix_run(*, campaign_root: Path, matrix_run_id: str, case_id: str | None = None, aggregation_id: str | None = None, weight_mode: str | None = None, aggregate_zone_id: str | None = None) -> tuple[list[AggregationZoneOutputRef], list[dict[str, Any]]]:
    csv_path = campaign_root / "aggregation" / "matrix_runs" / matrix_run_id / "aggregation_matrix_outputs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Aggregation matrix outputs not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    refs: list[AggregationZoneOutputRef] = []
    issues: list[dict[str, Any]] = []
    for row in rows:
        if case_id and row.get("case_id") != case_id:
            continue
        if aggregation_id and row.get("aggregation_id") != aggregation_id:
            continue
        if weight_mode and row.get("weight_mode") != weight_mode:
            continue
        run_root_text = str(row.get("run_root", "")).strip()
        if not run_root_text:
            issues.append({**row, "reason": "matrix row missing run_root"})
            continue
        try:
            found, current_issues = discover_from_aggregation_run(
                campaign_root=campaign_root, aggregation_run_root=Path(run_root_text),
                matrix_metadata=row, aggregate_zone_id=aggregate_zone_id,
            )
            refs.extend(found); issues.extend(current_issues)
        except Exception as exc:
            issues.append({**row, "reason": f"{type(exc).__name__}: {exc}"})
    return refs, issues


def discover_from_aggregation_run(*, campaign_root: Path, aggregation_run_root: Path, matrix_metadata: dict[str, Any] | None = None, aggregate_zone_id: str | None = None) -> tuple[list[AggregationZoneOutputRef], list[dict[str, Any]]]:
    run_root = Path(aggregation_run_root).expanduser().resolve()
    manifest_path = run_root / "aggregation_manifest.json"
    plan_path = run_root / "inputs" / "aggregation_plan.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Aggregation manifest not found: {manifest_path}")
    if not plan_path.is_file():
        raise FileNotFoundError(f"Aggregation plan not found: {plan_path}")
    manifest, plan = _json(manifest_path), _json(plan_path)
    meta = matrix_metadata or {}
    case_id = str(manifest.get("case_id") or plan.get("source_case_id") or meta.get("case_id") or "")
    source_run_id = str(manifest.get("source_generation_run_id") or plan.get("source_generation_run_id") or "")
    aggregation_id = str(manifest.get("plan_aggregation_id") or plan.get("aggregation_id") or meta.get("aggregation_id") or "")
    strategy = str(manifest.get("strategy") or plan.get("strategy") or meta.get("plan_strategy") or "")
    source_manifest_path = _optional(run_root / "inputs" / "source_run_manifest.json")
    source_manifest = _json(source_manifest_path) if source_manifest_path else {}
    case_spec = source_manifest.get("case_spec", {}) if isinstance(source_manifest.get("case_spec", {}), dict) else {}
    generation_case_root = campaign_root / "generation" / "cases" / case_id
    generation_run_root = generation_case_root / "runs" / source_run_id
    variable_manifest_path = _optional(generation_run_root / "canonical" / "variable_manifest.json")
    rdd_path = _optional(generation_case_root / "rdd_probe" / "rdd_variable_intersection.json")
    zones_root = run_root / "zones"
    if not zones_root.is_dir():
        raise FileNotFoundError(f"Aggregation zones folder not found: {zones_root}")
    refs, issues = [], []
    aggregate_zone_count = len([p for p in zones_root.iterdir() if p.is_dir()])
    for zone_root in sorted(p for p in zones_root.iterdir() if p.is_dir()):
        if aggregate_zone_id and zone_root.name != aggregate_zone_id:
            continue
        required = {
            "wide_parquet": zone_root / "aggregated_timeseries_wide.parquet",
            "static_equipment": zone_root / "aggregated_static_equipment.csv",
            "equipment_contributions": zone_root / "equipment_contributions.csv",
            "zone_mapping": zone_root / "zone_mapping.csv",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            issues.append({"case_id": case_id, "aggregation_run_id": manifest.get("aggregation_run_id", run_root.name), "aggregate_zone_id": zone_root.name, "reason": "missing required zone files: " + ", ".join(missing)})
            continue
        level = str(meta.get("aggregation_level") or infer_aggregation_level(aggregation_id))
        family = str(meta.get("aggregation_family") or infer_aggregation_family(aggregation_id, strategy))
        refs.append(AggregationZoneOutputRef(
            campaign_id=str(plan.get("campaign_id") or source_manifest.get("campaign_id") or campaign_root.name),
            case_id=case_id, building_type=str(meta.get("building_type") or case_spec.get("building_type") or plan.get("building_type") or ""),
            climate_zone=str(meta.get("climate_zone") or case_spec.get("climate_zone") or plan.get("climate_zone") or ""),
            weather_location=str(meta.get("weather_location") or case_spec.get("weather_location") or case_spec.get("weather_name") or plan.get("weather_location") or ""),
            source_generation_run_id=source_run_id, aggregation_run_id=str(manifest.get("aggregation_run_id") or run_root.name),
            aggregation_id=aggregation_id, aggregation_level=level, aggregation_family=family, strategy=strategy,
            rule_set=str(manifest.get("rule_set") or plan.get("rule_set") or ""), weight_mode=str(manifest.get("weight_mode") or plan.get("weight_mode") or meta.get("weight_mode") or ""),
            aggregate_zone_id=zone_root.name, aggregate_zone_count=aggregate_zone_count, campaign_root=campaign_root, aggregation_run_root=run_root, aggregation_manifest_path=manifest_path, aggregation_plan_path=plan_path, zone_root=zone_root,
            wide_parquet_path=required["wide_parquet"], static_equipment_path=required["static_equipment"], equipment_contributions_path=required["equipment_contributions"], zone_mapping_path=required["zone_mapping"],
            rule_summary_path=_optional(run_root / "diagnostics" / "rule_summary.csv"), rule_diagnostics_path=_optional(run_root / "diagnostics" / "rule_diagnostics.csv"), loaded_variables_path=_optional(run_root / "diagnostics" / "loaded_variables.csv"), schedule_mapping_path=_optional(run_root / "diagnostics" / "schedule_equipment_mapping_used.csv"),
            node_temperature_summary_path=_optional(run_root / "diagnostics" / "system_node_temperature_summary.csv"), node_temperature_mapping_path=_optional(run_root / "diagnostics" / "system_node_temperature_mapping.csv"),
            node_mass_flow_summary_path=_optional(run_root / "diagnostics" / "system_node_mass_flow_summary.csv"), node_mass_flow_mapping_path=_optional(run_root / "diagnostics" / "system_node_mass_flow_mapping.csv"),
            source_run_manifest_path=source_manifest_path, rdd_intersection_path=rdd_path, variable_manifest_path=variable_manifest_path,
        ))
    return refs, issues


def resolve_inputs(*, campaign_id: str, campaign_root: str | None, generated_data_root: str | None) -> Path:
    return resolve_campaign_root(repo_root=resolve_repo_root(), campaign_id=campaign_id, campaign_root=campaign_root, generated_data_root=generated_data_root)
