# -*- coding: utf-8 -*-
"""General campaign orchestration for ScaleBridge Phase B Aggregation.

This module deliberately contains no scientific aggregation rules. It composes
existing plan builders and ``run_aggregation_for_generation_run`` into a
persistent, campaign-definition-driven workflow suitable for both CLI and BGIRS.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.campaign_definition import (
    AggregationCampaignDefinition,
    AggregationPlanRequest,
)
from scalebridge.data.aggregation.discovery import (
    discover_generation_runs,
    load_json,
    resolve_campaign_root,
    resolve_repo_root,
)
from scalebridge.data.aggregation.engine import run_aggregation_for_generation_run
from scalebridge.data.aggregation.models import (
    AggregationStrategy,
    GenerationRunRef,
    SUCCESS_STATUSES,
)
from scalebridge.data.aggregation.plans import (
    build_and_write_plan_for_run,
    group_custom_rows_by_case,
    read_csv,
)
from scalebridge.data.aggregation.writers import write_csv, write_json
from scalebridge.tracking.mlflow.aggregation import (
    maybe_end_mlflow_run,
    maybe_start_mlflow_parent_run,
    mlflow_log_campaign_summary,
    mlflow_log_case_result,
)


@dataclass(frozen=True)
class CampaignPlanRef:
    """Exact plan selected for one general campaign execution item."""

    case_id: str
    source_generation_run_id: str
    plan_path: Path
    plan_build_root: Path
    plan_build_id: str
    aggregation_id: str
    aggregation_level: str
    aggregation_level_index: int
    aggregation_family: str
    plan_strategy: str
    rule_set: str
    weight_mode: str
    aggregate_zone_count: int | None
    source_zone_count: int | None
    building_type: str
    climate_zone: str
    weather_location: str


@dataclass(frozen=True)
class AggregationCampaignRunResult:
    """Summary returned to CLI/Dash after one general Aggregation campaign."""

    aggregation_campaign_id: str
    parent_generation_campaign_id: str
    matrix_run_id: str
    status: str
    summary_dir: Path
    selected_plan_count: int
    successful_plan_count: int
    failed_plan_count: int
    return_code: int


def run_aggregation_campaign(
    *,
    definition: AggregationCampaignDefinition,
    definition_path: Path | None = None,
    dry_run: bool = False,
    matrix_run_id: str | None = None,
) -> AggregationCampaignRunResult:
    """Build exact plans and execute a general Aggregation campaign.

    ``dry_run=True`` still performs upstream discovery and writes a complete plan
    build plus matrix selection summary, but skips scientific aggregation runs.
    This makes it useful for BGIRS preview/validation without duplicating plan
    logic in the app.
    """
    repo_root = resolve_repo_root()
    definition_base = (
        definition_path.expanduser().resolve().parent
        if definition_path is not None
        else repo_root
    )

    campaign_root = _resolve_parent_campaign_root(
        definition=definition,
        repo_root=repo_root,
        definition_base=definition_base,
    )
    cases_root = campaign_root / "generation" / "cases"
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Generation cases folder does not exist: {cases_root}")

    run_refs, missing_generation_rows = discover_generation_runs(
        cases_root=cases_root,
        case_id=None,
        include_failed=False,
    )
    run_refs = _select_generation_runs(run_refs, definition)
    if not run_refs:
        raise ValueError(
            "No successful Generation runs matched this Aggregation campaign. "
            f"parent_campaign={definition.parent_generation_campaign_id}"
        )

    custom_group_rows_by_case = _load_custom_group_rows(
        definition=definition,
        definition_base=definition_base,
    )

    plan_build_id = _build_plan_build_id(definition.aggregation_campaign_id)
    plan_build_root = campaign_root / "aggregation" / "plans" / plan_build_id
    plan_build_root.mkdir(parents=True, exist_ok=True)

    selected_plans, plan_build_payload = _build_campaign_plans(
        definition=definition,
        run_refs=run_refs,
        plan_build_root=plan_build_root,
        custom_group_rows_by_case=custom_group_rows_by_case,
    )
    if not selected_plans:
        raise ValueError("Campaign plan build produced no executable Aggregation plans")

    effective_matrix_run_id = matrix_run_id or _build_matrix_run_id()
    summary_dir = (
        campaign_root
        / "aggregation"
        / "matrix_runs"
        / effective_matrix_run_id
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    selected_plan_rows = [_plan_ref_to_row(item) for item in selected_plans]
    write_csv(summary_dir / "selected_aggregation_plans.csv", selected_plan_rows)
    write_csv(summary_dir / "missing_generation_rows.csv", missing_generation_rows)

    if definition_path is not None and definition_path.is_file():
        shutil.copy2(
            definition_path,
            summary_dir / "aggregation_campaign_definition.json",
        )
    else:
        write_json(
            summary_dir / "aggregation_campaign_definition.json",
            definition.to_dict(),
        )

    print("=" * 100)
    print("SCALEBRIDGE GENERAL AGGREGATION CAMPAIGN RUNNER")
    print("=" * 100)
    print(f"aggregation_campaign_id: {definition.aggregation_campaign_id}")
    print(f"parent_generation_campaign_id: {definition.parent_generation_campaign_id}")
    print(f"machine_id: {definition.machine_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"plan_build_root: {plan_build_root}")
    print(f"matrix_run_id: {effective_matrix_run_id}")
    print(f"selected_generation_case_count: {len(run_refs)}")
    print(f"selected_plan_count: {len(selected_plans)}")
    print(f"dry_run: {dry_run}")
    print(f"write_legacy_pickle: {definition.write_legacy_pickle}")
    print(f"mlflow_enabled: {definition.mlflow_enabled}")
    print()

    _configure_mlflow(definition)
    mlflow_started = False
    if not dry_run:
        try:
            maybe_start_mlflow_parent_run(
                enabled=definition.mlflow_enabled,
                campaign_id=definition.parent_generation_campaign_id,
                experiment_name=definition.mlflow_experiment_name,
                run_name=definition.mlflow_run_name or effective_matrix_run_id,
                params={
                    "aggregation_campaign_id": definition.aggregation_campaign_id,
                    "parent_generation_campaign_id": definition.parent_generation_campaign_id,
                    "machine_id": definition.machine_id,
                    "matrix_run_id": effective_matrix_run_id,
                    "selected_plan_count": len(selected_plans),
                    "strategies": ",".join(definition.requested_strategy_values),
                    "weight_modes": ",".join(definition.requested_weight_mode_values),
                    "write_legacy_pickle": definition.write_legacy_pickle,
                    "continue_on_error": definition.continue_on_error,
                },
            )
            mlflow_started = definition.mlflow_enabled
        except Exception:
            if definition.mlflow_strict:
                raise
            print("WARNING: MLflow unavailable; continuing without MLflow.", flush=True)
            mlflow_started = False

    result_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    run_ref_by_case = {item.case_id: item for item in run_refs}
    started_at = time.perf_counter()

    if dry_run:
        result_rows = [_planned_result_row(item) for item in selected_plans]
    else:
        for index, plan_ref in enumerate(selected_plans, start=1):
            print(
                f"[{index}/{len(selected_plans)}] "
                f"{plan_ref.case_id} | {plan_ref.building_type} | "
                f"{plan_ref.weather_location or plan_ref.climate_zone} | "
                f"{plan_ref.aggregation_id} | {plan_ref.weight_mode}"
            )
            print(f"    plan_path: {plan_ref.plan_path}")

            aggregation_run_id = _build_aggregation_run_id(
                sequence_index=index,
                source_generation_run_id=plan_ref.source_generation_run_id,
                aggregation_id=plan_ref.aggregation_id,
                weight_mode=plan_ref.weight_mode,
            )
            try:
                result = run_aggregation_for_generation_run(
                    campaign_root=campaign_root,
                    run_ref=run_ref_by_case[plan_ref.case_id],
                    plan_path=plan_ref.plan_path,
                    aggregation_run_id=aggregation_run_id,
                    max_variables=definition.max_variables,
                    preview_rows=definition.preview_rows,
                    write_legacy_pickle=definition.write_legacy_pickle,
                )
                row = _completed_result_row(plan_ref, result)
                result_rows.append(row)
                output_rows.append(
                    {
                        "case_id": result.case_id,
                        "source_generation_run_id": result.source_generation_run_id,
                        "aggregation_run_id": result.aggregation_run_id,
                        "building_type": plan_ref.building_type,
                        "climate_zone": plan_ref.climate_zone,
                        "weather_location": plan_ref.weather_location,
                        "aggregation_id": plan_ref.aggregation_id,
                        "aggregation_level": plan_ref.aggregation_level,
                        "aggregation_family": plan_ref.aggregation_family,
                        "weight_mode": plan_ref.weight_mode,
                        "aggregate_zone_count": result.aggregate_zone_count,
                        "run_root": str(result.run_root),
                    }
                )
                mlflow_log_case_result(
                    enabled=mlflow_started,
                    case_result=row,
                    artifact_root=result.run_root / "diagnostics",
                )
            except Exception as exc:
                row = _failed_result_row(
                    plan_ref=plan_ref,
                    aggregation_run_id=aggregation_run_id,
                    exc=exc,
                )
                result_rows.append(row)
                print(f"ERROR aggregating {plan_ref.case_id}: {exc}", flush=True)
                mlflow_log_case_result(
                    enabled=mlflow_started,
                    case_result=row,
                    artifact_root=summary_dir,
                )
                if not definition.continue_on_error:
                    break

    runtime_seconds = time.perf_counter() - started_at
    write_csv(summary_dir / "aggregation_matrix_case_runs.csv", result_rows)
    write_csv(summary_dir / "aggregation_matrix_outputs.csv", output_rows)

    successful_plan_count = sum(
        1
        for row in result_rows
        if str(row.get("status", "")).casefold() in SUCCESS_STATUSES
    )
    failed_plan_count = sum(
        1 for row in result_rows if str(row.get("status", "")).casefold() == "failed"
    )
    planned_plan_count = sum(
        1 for row in result_rows if str(row.get("status", "")).casefold() == "planned"
    )

    if dry_run:
        status = "planned"
        return_code = 0
    elif failed_plan_count:
        status = "completed_with_failures"
        return_code = 1
    elif successful_plan_count == len(selected_plans):
        status = "completed"
        return_code = 0
    else:
        status = "incomplete"
        return_code = 1

    summary = {
        "schema_version": "0.2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation_campaign_id": definition.aggregation_campaign_id,
        "parent_generation_campaign_id": definition.parent_generation_campaign_id,
        # Existing field retained for compatibility with existing Results/MLflow code.
        "campaign_id": definition.parent_generation_campaign_id,
        "machine_id": definition.machine_id,
        "campaign_root": str(campaign_root),
        "matrix_run_id": effective_matrix_run_id,
        "summary_dir": str(summary_dir),
        "plan_build_id": plan_build_id,
        "plan_build_root": str(plan_build_root),
        "dry_run": dry_run,
        "status": status,
        "selected_generation_case_count": len(run_refs),
        "selected_plan_count": len(selected_plans),
        "attempted_plan_count": len(result_rows) if not dry_run else 0,
        "planned_plan_count": planned_plan_count,
        "successful_plan_count": successful_plan_count,
        "failed_plan_count": failed_plan_count,
        "aggregation_ids": sorted({item.aggregation_id for item in selected_plans}),
        "strategies": list(definition.requested_strategy_values),
        "weight_modes": list(definition.requested_weight_mode_values),
        "rule_sets": sorted({item.rule_set for item in selected_plans}),
        "building_types": sorted(
            {item.building_type for item in selected_plans if item.building_type}
        ),
        "climate_zones": sorted(
            {item.climate_zone for item in selected_plans if item.climate_zone}
        ),
        "weather_locations": sorted(
            {item.weather_location for item in selected_plans if item.weather_location}
        ),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "aggregation_campaign_definition": str(
                summary_dir / "aggregation_campaign_definition.json"
            ),
            "selected_aggregation_plans": str(
                summary_dir / "selected_aggregation_plans.csv"
            ),
            "aggregation_matrix_case_runs": str(
                summary_dir / "aggregation_matrix_case_runs.csv"
            ),
            "aggregation_matrix_outputs": str(
                summary_dir / "aggregation_matrix_outputs.csv"
            ),
            "missing_generation_rows": str(
                summary_dir / "missing_generation_rows.csv"
            ),
            "plan_build_summary": str(
                plan_build_root / "aggregation_plan_build_summary.json"
            ),
        },
    }
    write_json(summary_dir / "aggregation_matrix_manifest.json", summary)

    if mlflow_started:
        mlflow_log_campaign_summary(
            enabled=True,
            summary=summary,
            summary_dir=summary_dir,
        )
        maybe_end_mlflow_run(True)

    print()
    print("=" * 100)
    print("GENERAL AGGREGATION CAMPAIGN SUMMARY")
    print("=" * 100)
    print(f"status: {status}")
    print(f"selected_plan_count: {len(selected_plans)}")
    print(f"successful_plan_count: {successful_plan_count}")
    print(f"failed_plan_count: {failed_plan_count}")
    print(f"summary_dir: {summary_dir}")

    return AggregationCampaignRunResult(
        aggregation_campaign_id=definition.aggregation_campaign_id,
        parent_generation_campaign_id=definition.parent_generation_campaign_id,
        matrix_run_id=effective_matrix_run_id,
        status=status,
        summary_dir=summary_dir,
        selected_plan_count=len(selected_plans),
        successful_plan_count=successful_plan_count,
        failed_plan_count=failed_plan_count,
        return_code=return_code,
    )


def _resolve_parent_campaign_root(
    *,
    definition: AggregationCampaignDefinition,
    repo_root: Path,
    definition_base: Path,
) -> Path:
    if definition.parent_generation_campaign_root:
        return _resolve_path(
            definition.parent_generation_campaign_root,
            base=definition_base,
        )

    generated_data_root = definition.generated_data_root
    if generated_data_root:
        generated_data_root = str(
            _resolve_path(generated_data_root, base=definition_base)
        )

    return resolve_campaign_root(
        repo_root=repo_root,
        campaign_id=definition.parent_generation_campaign_id,
        campaign_root=None,
        generated_data_root=generated_data_root,
    )


def _select_generation_runs(
    run_refs: list[GenerationRunRef],
    definition: AggregationCampaignDefinition,
) -> list[GenerationRunRef]:
    selected = list(run_refs)
    if definition.case_ids:
        requested = set(definition.case_ids)
        available = {item.case_id for item in selected}
        missing = sorted(requested.difference(available))
        if missing:
            raise ValueError(
                "Requested case_ids are not available as successful latest Generation "
                f"runs: {missing}"
            )
        selected = [item for item in selected if item.case_id in requested]

    selected.sort(key=lambda item: item.case_id)
    if definition.case_limit is not None:
        selected = selected[: definition.case_limit]
    return selected


def _load_custom_group_rows(
    *,
    definition: AggregationCampaignDefinition,
    definition_base: Path,
) -> dict[str, list[dict[str, Any]]]:
    uses_custom = any(
        request.strategy == AggregationStrategy.CUSTOM_GROUPS
        for request in definition.plan_requests
    )
    if not uses_custom:
        return {}

    assert definition.custom_zone_groups_path is not None
    path = _resolve_path(definition.custom_zone_groups_path, base=definition_base)
    if not path.is_file():
        raise FileNotFoundError(f"Custom grouping CSV does not exist: {path}")
    return group_custom_rows_by_case(read_csv(path))


def _build_campaign_plans(
    *,
    definition: AggregationCampaignDefinition,
    run_refs: list[GenerationRunRef],
    plan_build_root: Path,
    custom_group_rows_by_case: dict[str, list[dict[str, Any]]],
) -> tuple[list[CampaignPlanRef], dict[str, Any]]:
    selected_plans: list[CampaignPlanRef] = []
    all_plan_rows: list[dict[str, Any]] = []
    all_zone_mapping_rows: list[dict[str, Any]] = []
    all_included_rows: list[dict[str, Any]] = []
    all_excluded_rows: list[dict[str, Any]] = []
    all_missing_rows: list[dict[str, Any]] = []

    for request_index, request in enumerate(definition.plan_requests, start=1):
        request_root = plan_build_root / "requests" / f"request_{request_index:03d}"
        request_root.mkdir(parents=True, exist_ok=True)

        for run_ref in run_refs:
            result = build_and_write_plan_for_run(
                run_ref=run_ref,
                output_root=request_root,
                strategy=request.strategy,
                rule_set=request.rule_set,
                weight_mode=request.weight_mode,
                aggregate_zone_name_stem=definition.aggregate_zone_name_stem,
                system_node_name_pattern=definition.system_node_name_pattern,
                custom_group_rows=custom_group_rows_by_case.get(run_ref.case_id, []),
                custom_aggregation_ids=(
                    list(request.custom_aggregation_ids)
                    if request.custom_aggregation_ids
                    else None
                ),
            )
            all_plan_rows.extend(result["plan_rows"])
            all_zone_mapping_rows.extend(result["zone_mapping_rows"])
            all_included_rows.extend(result["included_thermal_zone_rows"])
            all_excluded_rows.extend(result["excluded_zone_rows"])
            all_missing_rows.extend(result["missing_rows"])

            for plan_row in result["plan_rows"]:
                selected_plans.append(
                    _build_plan_ref(
                        plan_row=plan_row,
                        run_ref=run_ref,
                        request=request,
                        request_index=request_index,
                        plan_build_root=plan_build_root,
                    )
                )

    selected_plans.sort(
        key=lambda item: (
            item.aggregation_level_index,
            item.weight_mode,
            item.building_type,
            item.weather_location,
            item.case_id,
            item.aggregation_id,
        )
    )

    write_csv(plan_build_root / "aggregation_plan_index.csv", all_plan_rows)
    write_csv(plan_build_root / "zone_mapping_all_cases.csv", all_zone_mapping_rows)
    write_csv(
        plan_build_root / "included_thermal_zones_all_cases.csv",
        _dedupe_rows(all_included_rows),
    )
    write_csv(
        plan_build_root / "excluded_zones_all_cases.csv",
        _dedupe_rows(all_excluded_rows),
    )
    write_csv(
        plan_build_root / "missing_plan_inputs.csv",
        _dedupe_rows(all_missing_rows),
    )

    summary = {
        "schema_version": "0.2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation_campaign_id": definition.aggregation_campaign_id,
        "parent_generation_campaign_id": definition.parent_generation_campaign_id,
        "campaign_id": definition.parent_generation_campaign_id,
        "plan_build_id": plan_build_root.name,
        "output_root": str(plan_build_root),
        "plan_request_count": len(definition.plan_requests),
        "plan_count": len(selected_plans),
        "zone_mapping_row_count": len(all_zone_mapping_rows),
        "included_thermal_zone_row_count": len(_dedupe_rows(all_included_rows)),
        "excluded_zone_row_count": len(_dedupe_rows(all_excluded_rows)),
        "missing_plan_input_row_count": len(_dedupe_rows(all_missing_rows)),
        "strategies": list(definition.requested_strategy_values),
        "weight_modes": list(definition.requested_weight_mode_values),
    }
    write_json(plan_build_root / "aggregation_plan_build_summary.json", summary)
    return selected_plans, summary


def _build_plan_ref(
    *,
    plan_row: dict[str, Any],
    run_ref: GenerationRunRef,
    request: AggregationPlanRequest,
    request_index: int,
    plan_build_root: Path,
) -> CampaignPlanRef:
    plan_path = Path(str(plan_row["plan_path"])).expanduser().resolve()
    plan_payload = load_json(plan_path)
    case_spec = load_json(run_ref.manifest_path).get("case_spec", {})
    aggregation_id = str(plan_row.get("aggregation_id", "")).strip()

    aggregate_zone_count = _optional_int(plan_row.get("aggregate_zone_count"))
    source_zone_count = _optional_int(plan_row.get("source_zone_count"))

    return CampaignPlanRef(
        case_id=run_ref.case_id,
        source_generation_run_id=run_ref.run_id,
        plan_path=plan_path,
        plan_build_root=plan_build_root,
        plan_build_id=plan_build_root.name,
        aggregation_id=aggregation_id,
        aggregation_level=request.aggregation_level or aggregation_id,
        aggregation_level_index=(
            request.aggregation_level_index
            if request.aggregation_level_index is not None
            else request_index
        ),
        aggregation_family=request.aggregation_family or request.strategy.value,
        plan_strategy=str(plan_payload.get("strategy", request.strategy.value)),
        rule_set=str(plan_payload.get("rule_set", request.rule_set.value)),
        weight_mode=str(plan_payload.get("weight_mode", request.weight_mode.value)),
        aggregate_zone_count=aggregate_zone_count,
        source_zone_count=source_zone_count,
        building_type=str(case_spec.get("building_type", "")),
        climate_zone=str(case_spec.get("climate_zone", "")),
        weather_location=str(
            case_spec.get("weather_location")
            or case_spec.get("weather_name")
            or case_spec.get("city")
            or ""
        ),
    )


def _plan_ref_to_row(plan_ref: CampaignPlanRef) -> dict[str, Any]:
    compression_ratio: float | str = ""
    if plan_ref.aggregate_zone_count and plan_ref.source_zone_count:
        compression_ratio = plan_ref.aggregate_zone_count / plan_ref.source_zone_count

    return {
        "case_id": plan_ref.case_id,
        "source_generation_run_id": plan_ref.source_generation_run_id,
        "building_type": plan_ref.building_type,
        "climate_zone": plan_ref.climate_zone,
        "weather_location": plan_ref.weather_location,
        "aggregation_id": plan_ref.aggregation_id,
        "aggregation_level": plan_ref.aggregation_level,
        "aggregation_level_index": plan_ref.aggregation_level_index,
        "aggregation_family": plan_ref.aggregation_family,
        "weight_mode": plan_ref.weight_mode,
        "plan_strategy": plan_ref.plan_strategy,
        "rule_set": plan_ref.rule_set,
        "aggregate_zone_count": plan_ref.aggregate_zone_count or "",
        "source_zone_count": plan_ref.source_zone_count or "",
        "aggregation_compression_ratio": compression_ratio,
        "plan_build_id": plan_ref.plan_build_id,
        "plan_build_root": str(plan_ref.plan_build_root),
        "plan_path": str(plan_ref.plan_path),
    }


def _planned_result_row(plan_ref: CampaignPlanRef) -> dict[str, Any]:
    return {
        **_plan_ref_to_row(plan_ref),
        "aggregation_run_id": "",
        "status": "planned",
        "run_root": "",
        "loaded_plan_aggregation_id": plan_ref.aggregation_id,
        "loaded_plan_strategy": plan_ref.plan_strategy,
        "loaded_plan_rule_set": plan_ref.rule_set,
        "loaded_plan_weight_mode": plan_ref.weight_mode,
        "loaded_variable_count": "",
        "aggregated_long_rows": "",
        "static_equipment_rows": "",
        "equipment_contribution_rows": "",
        "diagnostic_rows": "",
        "runtime_seconds": "",
        "error_type": "",
        "error_message": "",
        "rule_summary_rows": "",
    }


def _completed_result_row(plan_ref: CampaignPlanRef, result: Any) -> dict[str, Any]:
    row = {
        **_plan_ref_to_row(plan_ref),
        "aggregation_run_id": result.aggregation_run_id,
        "status": result.status,
        "run_root": str(result.run_root),
        "loaded_plan_aggregation_id": result.plan_aggregation_id,
        "loaded_plan_strategy": result.plan_strategy,
        "loaded_plan_rule_set": result.plan_rule_set,
        "loaded_plan_weight_mode": result.plan_weight_mode,
        "loaded_variable_count": result.loaded_variable_count,
        "aggregated_long_rows": result.aggregated_long_rows,
        "static_equipment_rows": result.static_equipment_rows,
        "equipment_contribution_rows": result.equipment_contribution_rows,
        "diagnostic_rows": result.diagnostic_rows,
        "runtime_seconds": result.runtime_seconds,
        "error_type": "",
        "error_message": "",
        "rule_summary_rows": "",
    }
    manifest_path = result.run_root / "aggregation_manifest.json"
    if manifest_path.is_file():
        row["rule_summary_rows"] = load_json(manifest_path).get("rule_summary_rows", "")
    return row


def _failed_result_row(
    *,
    plan_ref: CampaignPlanRef,
    aggregation_run_id: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        **_plan_ref_to_row(plan_ref),
        "aggregation_run_id": aggregation_run_id,
        "status": "failed",
        "run_root": "",
        "loaded_plan_aggregation_id": "",
        "loaded_plan_strategy": "",
        "loaded_plan_rule_set": "",
        "loaded_plan_weight_mode": "",
        "loaded_variable_count": "",
        "aggregated_long_rows": "",
        "static_equipment_rows": "",
        "equipment_contribution_rows": "",
        "diagnostic_rows": "",
        "runtime_seconds": "",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "rule_summary_rows": "",
        "traceback": traceback.format_exc(),
    }


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _configure_mlflow(definition: AggregationCampaignDefinition) -> None:
    if not definition.mlflow_enabled or not definition.mlflow_tracking_uri:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(definition.mlflow_tracking_uri)
    except Exception:
        if definition.mlflow_strict:
            raise


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _build_plan_build_id(aggregation_campaign_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"plan_build_{timestamp}_{_safe_token(aggregation_campaign_id, max_length=40)}"


def _build_matrix_run_id() -> str:
    return f"aggregation_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _build_aggregation_run_id(
    *,
    sequence_index: int,
    source_generation_run_id: str,
    aggregation_id: str,
    weight_mode: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_source = source_generation_run_id.replace("epvwr_", "")[:8]
    return (
        f"aggr_{timestamp}_{sequence_index:04d}_{short_source}_"
        f"{_safe_token(aggregation_id, max_length=36)}_"
        f"{_safe_token(weight_mode, max_length=16)}"
    )


def _safe_token(value: str, *, max_length: int) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in str(value)
    ).strip("_")
    return (safe or "token")[:max_length]
