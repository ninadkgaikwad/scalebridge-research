# -*- coding: utf-8 -*-
"""Production aggregation engine for ScaleBridge EnergyPlus outputs."""

from __future__ import annotations

import argparse
import gc
import traceback
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scalebridge.data.aggregation.discovery import (
    DEFAULT_CAMPAIGN_ID,
    discover_generation_runs,
    load_json,
    resolve_campaign_root,
    resolve_repo_root,
)
from scalebridge.data.aggregation.eio import schedule_equipment_mapping_rows
from scalebridge.data.aggregation.loaders import CanonicalVariableLoader
from scalebridge.data.aggregation.models import GenerationRunRef
from scalebridge.data.aggregation.rules import AggregationRuleOutputs, apply_legacy_v1_rules
from scalebridge.data.aggregation.system_node_mass_flow import (
    SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME,
    aggregate_system_node_mass_flow_rate_from_parquet,
    merge_system_node_mass_flow_outputs,
)
from scalebridge.data.aggregation.system_node_temperature import (
    SYSTEM_NODE_TEMPERATURE_VARIABLE_NAME,
    aggregate_system_node_temperature_from_parquet,
    merge_system_node_temperature_outputs,
)
from scalebridge.data.aggregation.writers import (
    copy_file_if_exists,
    make_safe_name,
    write_csv,
    write_dataframe_csv,
    write_dataframe_parquet,
    write_json,
)

from scalebridge.tracking.mlflow.aggregation import (
    maybe_end_mlflow_run,
    maybe_start_mlflow_parent_run,
    mlflow_log_campaign_summary,
    mlflow_log_case_result,
)


@dataclass(frozen=True)
class AggregationRunResult:
    """Summary of one production aggregation run."""

    case_id: str
    source_generation_run_id: str
    aggregation_run_id: str
    plan_aggregation_id: str
    plan_strategy: str
    plan_rule_set: str
    plan_weight_mode: str
    status: str
    run_root: Path
    aggregate_zone_count: int
    loaded_variable_count: int
    aggregated_long_rows: int
    static_equipment_rows: int
    equipment_contribution_rows: int
    diagnostic_rows: int
    runtime_seconds: float


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for production aggregation."""
    args = parse_args(argv)

    repo_root = resolve_repo_root()
    campaign_root = resolve_campaign_root(
        repo_root=repo_root,
        campaign_id=args.campaign_id,
        campaign_root=args.campaign_root,
        generated_data_root=args.generated_data_root,
    )

    cases_root = campaign_root / "generation" / "cases"
    if not cases_root.is_dir():
        raise SystemExit(f"Generation cases folder does not exist: {cases_root}")

    run_refs, missing_rows = discover_generation_runs(
        cases_root=cases_root,
        case_id=args.case_id,
        include_failed=False,
    )

    if not run_refs:
        raise SystemExit(f"No successful generation runs found under {cases_root}")

    if args.case_limit is not None:
        run_refs = run_refs[: max(0, args.case_limit)]

    if (
        not args.plan_path
        and args.strategy == "custom_groups"
        and not args.aggregation_id
    ):
        raise SystemExit(
            "--aggregation-id is required when discovering custom_groups plans "
            "without an explicit --plan-path."
        )

    output_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    print("=" * 100)
    print("SCALEBRIDGE P1 AGGREGATION RUNNER")
    print("=" * 100)
    print(f"repo_root: {repo_root}")
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"cases_root: {cases_root}")
    print(f"case_count: {len(run_refs)}")
    print(f"strategy: {args.strategy}")
    print(f"aggregation_id: {args.aggregation_id or ''}")
    print(f"rule_set: {args.rule_set}")
    print(f"weight_mode: {args.weight_mode}")
    print(f"write_legacy_pickle: {args.write_legacy_pickle}")
    print()

    mlflow_parent_run = maybe_start_mlflow_parent_run(
        enabled=args.mlflow,
        campaign_id=args.campaign_id,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.mlflow_run_name,
        params={
            "campaign_id": args.campaign_id,
            "strategy": args.strategy,
            "aggregation_id": args.aggregation_id or "",
            "rule_set": args.rule_set,
            "weight_mode": args.weight_mode,
            "case_count": len(run_refs),
            "write_legacy_pickle": args.write_legacy_pickle,
            "continue_on_error": args.continue_on_error,
        },
    )

    try:
        for index, run_ref in enumerate(run_refs, start=1):
            print(f"[{index}/{len(run_refs)}] Aggregating {run_ref.case_id}")

            try:
                plan_path = resolve_plan_path(
                    campaign_root=campaign_root,
                    case_id=run_ref.case_id,
                    explicit_plan_path=args.plan_path,
                    aggregation_id=args.aggregation_id,
                    strategy=args.strategy,
                    rule_set=args.rule_set,
                    weight_mode=args.weight_mode,
                )
                print(f"    plan_path: {plan_path}")

                result = run_aggregation_for_generation_run(
                    campaign_root=campaign_root,
                    run_ref=run_ref,
                    plan_path=plan_path,
                    aggregation_run_id=args.aggregation_run_id,
                    max_variables=args.max_variables,
                    preview_rows=args.preview_rows,
                    write_legacy_pickle=args.write_legacy_pickle,
                )

                row = {
                    "case_id": result.case_id,
                    "source_generation_run_id": result.source_generation_run_id,
                    "aggregation_run_id": result.aggregation_run_id,
                    "plan_aggregation_id": result.plan_aggregation_id,
                    "plan_strategy": result.plan_strategy,
                    "plan_rule_set": result.plan_rule_set,
                    "plan_weight_mode": result.plan_weight_mode,
                    "status": result.status,
                    "run_root": str(result.run_root),
                    "aggregate_zone_count": result.aggregate_zone_count,
                    "loaded_variable_count": result.loaded_variable_count,
                    "aggregated_long_rows": result.aggregated_long_rows,
                    "static_equipment_rows": result.static_equipment_rows,
                    "equipment_contribution_rows": result.equipment_contribution_rows,
                    "diagnostic_rows": result.diagnostic_rows,
                    "runtime_seconds": result.runtime_seconds,
                }

                manifest_path = result.run_root / "aggregation_manifest.json"
                if manifest_path.is_file():
                    manifest_payload = load_json(manifest_path)
                    row["rule_summary_rows"] = manifest_payload.get(
                        "rule_summary_rows", ""
                    )

                result_rows.append(row)

                output_rows.append(
                    {
                        "case_id": result.case_id,
                        "source_generation_run_id": result.source_generation_run_id,
                        "aggregation_run_id": result.aggregation_run_id,
                        "plan_aggregation_id": result.plan_aggregation_id,
                        "plan_strategy": result.plan_strategy,
                        "run_root": str(result.run_root),
                    }
                )

                mlflow_log_case_result(
                    enabled=args.mlflow,
                    case_result=row,
                    artifact_root=result.run_root / "diagnostics",
                )

            except Exception as exc:
                error_row = {
                    "case_id": run_ref.case_id,
                    "source_generation_run_id": run_ref.run_id,
                    "aggregation_run_id": "",
                    "requested_aggregation_id": args.aggregation_id or "",
                    "requested_strategy": args.strategy,
                    "requested_rule_set": args.rule_set,
                    "requested_weight_mode": args.weight_mode,
                    "status": "failed",
                    "run_root": "",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                result_rows.append(error_row)

                print(f"ERROR aggregating {run_ref.case_id}: {exc}")

                if not args.continue_on_error:
                    raise

    finally:
        maybe_end_mlflow_run(args.mlflow)

    summary_root = campaign_root / "aggregation" / "campaign_runs"
    campaign_run_id = build_campaign_aggregation_run_id()
    summary_dir = summary_root / campaign_run_id
    summary_dir.mkdir(parents=True, exist_ok=True)

    write_csv(summary_dir / "aggregation_case_runs.csv", result_rows)
    write_csv(summary_dir / "aggregation_outputs.csv", output_rows)
    write_csv(summary_dir / "discovery_missing_rows.csv", missing_rows)

    summary = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "campaign_aggregation_run_id": campaign_run_id,
        "case_count": len(run_refs),
        "successful_case_count": sum(
            1 for row in result_rows if row["status"] == "completed"
        ),
        "failed_case_count": sum(
            1 for row in result_rows if row["status"] != "completed"
        ),
        "strategy": args.strategy,
        "aggregation_id": args.aggregation_id or "",
        "rule_set": args.rule_set,
        "weight_mode": args.weight_mode,
        "summary_dir": str(summary_dir),
    }

    write_json(summary_dir / "aggregation_campaign_manifest.json", summary)

    mlflow_log_campaign_summary(
        enabled=args.mlflow,
        summary=summary,
        summary_dir=summary_dir,
    )

    print()
    print("=" * 100)
    print("AGGREGATION CAMPAIGN SUMMARY")
    print("=" * 100)
    print(f"case_count: {summary['case_count']}")
    print(f"successful_case_count: {summary['successful_case_count']}")
    print(f"failed_case_count: {summary['failed_case_count']}")
    print(f"summary_dir: {summary_dir}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help=f"Campaign ID. Default: {DEFAULT_CAMPAIGN_ID}",
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help="Explicit campaign root.",
    )
    parser.add_argument(
        "--generated-data-root",
        default=None,
        help="Explicit SCALEBRIDGE_GENERATED_DATA_ROOT.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional case_id filter.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional case limit for campaign-level runs.",
    )
    parser.add_argument(
        "--plan-path",
        default=None,
        help="Explicit aggregation_plan.json path. Best for single-case debugging.",
    )
    parser.add_argument(
        "--aggregation-id",
        default=None,
        help=(
            "Aggregation plan ID to discover under the latest plan_build_* folders. "
            "Useful for custom_groups plans such as rff_user_approved_k2_identity. "
            "Ignored when --plan-path is provided."
        ),
    )
    parser.add_argument(
        "--strategy",
        default="all_thermal_zones_to_one",
        help="Aggregation strategy used to discover latest plan if --plan-path is omitted.",
    )
    parser.add_argument(
        "--rule-set",
        default="legacy_v1",
        help="Aggregation rule set.",
    )
    parser.add_argument(
        "--weight-mode",
        default="equal",
        help="Aggregation weight mode.",
    )
    parser.add_argument(
        "--aggregation-run-id",
        default=None,
        help="Optional aggregation run id. If omitted, generated automatically.",
    )
    parser.add_argument(
        "--max-variables",
        type=int,
        default=None,
        help="Optional max variable count for debugging.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=100,
        help="Number of preview rows to write per zone.",
    )
    parser.add_argument(
        "--write-legacy-pickle",
        action="store_true",
        help="Also write legacy-style Aggregation_Dict_1Zone.pickle.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue campaign aggregation if one case fails.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for aggregation campaign and case runs.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=None,
        help=(
            "MLflow experiment name. Default: "
            "ScaleBridge_Aggregation_<campaign_id>."
        ),
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow parent run name.",
    )

    return parser.parse_args(argv)


def run_aggregation_for_generation_run(
    *,
    campaign_root: Path,
    run_ref: GenerationRunRef,
    plan_path: Path,
    aggregation_run_id: str | None,
    max_variables: int | None,
    preview_rows: int,
    write_legacy_pickle: bool,
) -> AggregationRunResult:
    """Run production aggregation for one generation run."""
    start = time.perf_counter()

    plan = load_json(plan_path)
    plan_aggregation_id = str(plan.get("aggregation_id", "")).strip()
    plan_strategy = str(plan.get("strategy", "")).strip()
    plan_rule_set = str(plan.get("rule_set", "")).strip()
    plan_weight_mode = str(plan.get("weight_mode", "")).strip()

    print(f"    loaded_plan_aggregation_id: {plan_aggregation_id}")
    print(f"    loaded_plan_strategy: {plan_strategy}")
    print(f"    loaded_plan_rule_set: {plan_rule_set}")
    print(f"    loaded_plan_weight_mode: {plan_weight_mode}")

    effective_aggregation_run_id = aggregation_run_id or build_aggregation_run_id(
        source_generation_run_id=run_ref.run_id,
        plan_aggregation_id=plan_aggregation_id,
    )

    run_root = (
        campaign_root
        / "aggregation"
        / "cases"
        / run_ref.case_id
        / "runs"
        / effective_aggregation_run_id
    )
    inputs_root = run_root / "inputs"
    diagnostics_root = run_root / "diagnostics"
    zones_root = run_root / "zones"
    legacy_root = run_root / "legacy"

    inputs_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    zones_root.mkdir(parents=True, exist_ok=True)

    source_manifest = load_json(run_ref.manifest_path)
    eio_payload = load_json(run_ref.run_root / "canonical" / "eio_tables.json")

    zone_mapping_path = plan_path.parent / "zone_mapping.csv"
    excluded_zones_path = plan_path.parent / "excluded_zones.csv"

    zone_mapping_rows = read_csv_dicts(zone_mapping_path)
    excluded_zone_rows = (
        read_csv_dicts(excluded_zones_path)
        if excluded_zones_path.is_file()
        else []
    )

    excluded_zone_names = {
        str(row.get("zone_name", "")).strip()
        for row in excluded_zone_rows
        if str(row.get("zone_name", "")).strip()
    }

    included_zone_names = {
        str(row.get("source_zone", "")).strip()
        for row in zone_mapping_rows
        if str(row.get("source_zone", "")).strip()
    }

    schedule_rows = schedule_equipment_mapping_rows(
        eio_payload=eio_payload,
        included_zone_names=included_zone_names,
    )
    schedule_rows = [
        {
            "case_id": run_ref.case_id,
            "run_id": run_ref.run_id,
            **row,
        }
        for row in schedule_rows
    ]

    loader = CanonicalVariableLoader(run_ref=run_ref)
    records = list(loader.records)
    if max_variables is not None:
        records = records[: max(0, max_variables)]

    outputs = empty_aggregation_rule_outputs()

    loaded_variable_rows: list[dict[str, Any]] = []

    system_node_temperature_rows: list[dict[str, Any]] = []
    system_node_temperature_mapping_frames: list[pd.DataFrame] = []
    system_node_temperature_unmapped_frames: list[pd.DataFrame] = []

    system_node_mass_flow_rows: list[dict[str, Any]] = []
    system_node_mass_flow_mapping_frames: list[pd.DataFrame] = []
    system_node_mass_flow_unmapped_frames: list[pd.DataFrame] = []

    normal_variable_count = 0
    deferred_system_node_temperature_count = 0
    deferred_system_node_mass_flow_count = 0

    for index, record in enumerate(records, start=1):
        if record.variable_name == SYSTEM_NODE_TEMPERATURE_VARIABLE_NAME:
            print(
                f"    [{index}/{len(records)}] Streaming {record.variable_name} "
                "with memory-safe shared node-temperature aggregation"
            )
            deferred_system_node_temperature_count += 1

            loaded_variable_rows.append(
                {
                    "variable_id": record.variable_id,
                    "variable_name": record.variable_name,
                    "load_status": "streamed_memory_safe_node_temperature_aggregation",
                    "row_count": "",
                    "key_value_count": "",
                    "parquet_path": str(record.canonical_parquet_path),
                }
            )

            temperature_result = aggregate_system_node_temperature_from_parquet(
                parquet_path=Path(record.canonical_parquet_path),
                plan=plan,
                zone_mapping_rows=zone_mapping_rows,
                batch_size=250_000,
            )

            outputs = merge_system_node_temperature_outputs(
                outputs=outputs,
                temperature_result=temperature_result,
            )

            system_node_temperature_rows.append(
                {
                    "variable_id": record.variable_id,
                    "variable_name": record.variable_name,
                    "parquet_path": str(record.canonical_parquet_path),
                    "source_key_count": temperature_result.source_key_count,
                    "mapped_key_count": temperature_result.mapped_key_count,
                    "unmapped_key_count": temperature_result.unmapped_key_count,
                    "mapped_row_count": temperature_result.mapped_row_count,
                    "skipped_row_count": temperature_result.skipped_row_count,
                }
            )
            system_node_temperature_mapping_frames.append(
                temperature_result.mapping_frame
            )
            system_node_temperature_unmapped_frames.append(
                temperature_result.unmapped_nodes_frame
            )

            del temperature_result
            gc.collect()
            continue

        if record.variable_name == SYSTEM_NODE_MASS_FLOW_VARIABLE_NAME:
            print(
                f"    [{index}/{len(records)}] Streaming {record.variable_name} "
                "with memory-safe shared node-mass-flow aggregation"
            )
            deferred_system_node_mass_flow_count += 1

            loaded_variable_rows.append(
                {
                    "variable_id": record.variable_id,
                    "variable_name": record.variable_name,
                    "load_status": "streamed_memory_safe_node_flow_aggregation",
                    "row_count": "",
                    "key_value_count": "",
                    "parquet_path": str(record.canonical_parquet_path),
                }
            )

            mass_flow_result = aggregate_system_node_mass_flow_rate_from_parquet(
                parquet_path=Path(record.canonical_parquet_path),
                plan=plan,
                batch_size=250_000,
            )

            outputs = merge_system_node_mass_flow_outputs(
                outputs=outputs,
                mass_flow_result=mass_flow_result,
            )

            system_node_mass_flow_rows.append(
                {
                    "variable_id": record.variable_id,
                    "variable_name": record.variable_name,
                    "parquet_path": str(record.canonical_parquet_path),
                    "source_key_count": mass_flow_result.source_key_count,
                    "mapped_key_count": mass_flow_result.mapped_key_count,
                    "unmapped_key_count": mass_flow_result.unmapped_key_count,
                    "mapped_row_count": mass_flow_result.mapped_row_count,
                    "skipped_row_count": mass_flow_result.skipped_row_count,
                }
            )
            system_node_mass_flow_mapping_frames.append(
                mass_flow_result.mapping_frame
            )
            system_node_mass_flow_unmapped_frames.append(
                mass_flow_result.unmapped_nodes_frame
            )

            del mass_flow_result
            gc.collect()
            continue

        print(
            f"    [{index}/{len(records)}] Loading and aggregating "
            f"{record.variable_name}"
        )

        frame = loader.load_variable_long_by_id(record.variable_id)
        normal_variable_count += 1

        loaded_variable_rows.append(
            {
                "variable_id": record.variable_id,
                "variable_name": record.variable_name,
                "load_status": "loaded_and_aggregated_single_variable",
                "row_count": len(frame),
                "key_value_count": (
                    frame["key_value"].nunique()
                    if "key_value" in frame.columns
                    else ""
                ),
                "parquet_path": str(record.canonical_parquet_path),
            }
        )

        one_variable_outputs = apply_legacy_v1_rules(
            plan=plan,
            variable_frames_by_name={record.variable_name: frame},
            schedule_equipment_rows=schedule_rows,
            zone_mapping_rows=zone_mapping_rows,
            excluded_zone_names=excluded_zone_names,
        )
        outputs = merge_aggregation_rule_outputs(outputs, one_variable_outputs)

        del frame
        del one_variable_outputs
        gc.collect()

    # Inputs/provenance.
    copy_file_if_exists(plan_path, inputs_root / "aggregation_plan.json")
    copy_file_if_exists(zone_mapping_path, inputs_root / "zone_mapping.csv")
    copy_file_if_exists(run_ref.manifest_path, inputs_root / "source_run_manifest.json")

    source_generation_run = {
        "case_id": run_ref.case_id,
        "run_id": run_ref.run_id,
        "status": run_ref.status,
        "run_root": str(run_ref.run_root),
        "manifest_path": str(run_ref.manifest_path),
        "source_manifest": source_manifest,
    }
    write_json(inputs_root / "source_generation_run.json", source_generation_run)

    # Diagnostics.
    write_csv(diagnostics_root / "loaded_variables.csv", loaded_variable_rows)

    write_csv(
        diagnostics_root / "system_node_temperature_summary.csv",
        system_node_temperature_rows,
    )

    if system_node_temperature_mapping_frames:
        write_dataframe_csv(
            diagnostics_root / "system_node_temperature_mapping.csv",
            pd.concat(system_node_temperature_mapping_frames, ignore_index=True),
        )

    if system_node_temperature_unmapped_frames:
        write_dataframe_csv(
            diagnostics_root / "system_node_temperature_unmapped_nodes.csv",
            pd.concat(system_node_temperature_unmapped_frames, ignore_index=True),
        )

    write_csv(
        diagnostics_root / "system_node_mass_flow_summary.csv",
        system_node_mass_flow_rows,
    )

    if system_node_mass_flow_mapping_frames:
        write_dataframe_csv(
            diagnostics_root / "system_node_mass_flow_mapping.csv",
            pd.concat(system_node_mass_flow_mapping_frames, ignore_index=True),
        )

    if system_node_mass_flow_unmapped_frames:
        write_dataframe_csv(
            diagnostics_root / "system_node_mass_flow_unmapped_nodes.csv",
            pd.concat(system_node_mass_flow_unmapped_frames, ignore_index=True),
        )

    write_csv(diagnostics_root / "schedule_equipment_mapping_used.csv", schedule_rows)
    outputs.rule_summary_frame.to_csv(
        diagnostics_root / "rule_summary.csv",
        index=False,
    )
    outputs.diagnostics_frame.to_csv(
        diagnostics_root / "rule_diagnostics.csv",
        index=False,
    )
    outputs.equipment_contribution_frame.to_csv(
        diagnostics_root / "equipment_contributions.csv",
        index=False,
    )

    # Zone outputs.
    for aggregate_zone_id, wide_frame in outputs.wide_by_zone.items():
        zone_dir = zones_root / make_safe_name(aggregate_zone_id)
        zone_dir.mkdir(parents=True, exist_ok=True)

        write_dataframe_parquet(
            zone_dir / "aggregated_timeseries_wide.parquet",
            wide_frame,
        )
        write_dataframe_csv(
            zone_dir / "aggregated_timeseries_wide_preview.csv",
            wide_frame.head(preview_rows),
        )

        zone_long = outputs.long_frame[
            outputs.long_frame["aggregate_zone_id"] == aggregate_zone_id
        ].copy()
        write_dataframe_parquet(
            zone_dir / "aggregated_timeseries_long.parquet",
            zone_long,
        )
        write_dataframe_csv(
            zone_dir / "aggregated_timeseries_long_preview.csv",
            zone_long.head(preview_rows),
        )

        zone_static = outputs.static_equipment_frame[
            outputs.static_equipment_frame["aggregate_zone_id"] == aggregate_zone_id
        ].copy()
        write_dataframe_parquet(
            zone_dir / "aggregated_static_equipment.parquet",
            zone_static,
        )
        write_dataframe_csv(
            zone_dir / "aggregated_static_equipment.csv",
            zone_static,
        )

        zone_contrib = outputs.equipment_contribution_frame[
            outputs.equipment_contribution_frame["aggregate_zone_id"] == aggregate_zone_id
        ].copy()
        write_dataframe_csv(
            zone_dir / "equipment_contributions.csv",
            zone_contrib,
        )
        write_dataframe_parquet(
            zone_dir / "equipment_contributions.parquet",
            zone_contrib,
        )

        copy_file_if_exists(zone_mapping_path, zone_dir / "zone_mapping.csv")

    if write_legacy_pickle:
        legacy_root.mkdir(parents=True, exist_ok=True)
        write_legacy_aggregation_pickle(
            path=legacy_root / "Aggregation_Dict_1Zone.pickle",
            wide_by_zone=outputs.wide_by_zone,
            static_equipment_frame=outputs.static_equipment_frame,
        )

    runtime_seconds = time.perf_counter() - start

    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "case_id": run_ref.case_id,
        "source_generation_run_id": run_ref.run_id,
        "aggregation_run_id": effective_aggregation_run_id,
        "aggregation_run_root": str(run_root),
        "plan_path": str(plan_path),
        "plan_aggregation_id": plan_aggregation_id,
        "strategy": plan_strategy,
        "rule_set": plan_rule_set,
        "weight_mode": plan_weight_mode,
        "aggregate_zone_count": len(outputs.wide_by_zone),
        "loaded_variable_count": normal_variable_count,
        "deferred_system_node_temperature_variable_count": deferred_system_node_temperature_count,
        "deferred_system_node_mass_flow_variable_count": deferred_system_node_mass_flow_count,
        "system_node_temperature_summary": system_node_temperature_rows,
        "system_node_mass_flow_summary": system_node_mass_flow_rows,
        "aggregated_long_rows": int(len(outputs.long_frame)),
        "static_equipment_rows": int(len(outputs.static_equipment_frame)),
        "equipment_contribution_rows": int(len(outputs.equipment_contribution_frame)),
        "diagnostic_rows": int(len(outputs.diagnostics_frame)),
        "rule_summary_rows": int(len(outputs.rule_summary_frame)),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "inputs_root": str(inputs_root),
            "diagnostics_root": str(diagnostics_root),
            "zones_root": str(zones_root),
            "legacy_root": str(legacy_root) if write_legacy_pickle else "",
        },
    }
    write_json(run_root / "aggregation_manifest.json", manifest)

    latest_run = {
        "case_id": run_ref.case_id,
        "aggregation_run_id": effective_aggregation_run_id,
        "plan_aggregation_id": plan_aggregation_id,
        "strategy": plan_strategy,
        "rule_set": plan_rule_set,
        "weight_mode": plan_weight_mode,
        "status": "completed",
        "manifest_path": str(
            (run_root / "aggregation_manifest.json").relative_to(
                campaign_root / "aggregation" / "cases" / run_ref.case_id
            )
        ),
        "created_at_utc": manifest["created_at_utc"],
    }
    write_json(
        campaign_root
        / "aggregation"
        / "cases"
        / run_ref.case_id
        / "latest_run.json",
        latest_run,
    )

    return AggregationRunResult(
        case_id=run_ref.case_id,
        source_generation_run_id=run_ref.run_id,
        aggregation_run_id=effective_aggregation_run_id,
        plan_aggregation_id=plan_aggregation_id,
        plan_strategy=plan_strategy,
        plan_rule_set=plan_rule_set,
        plan_weight_mode=plan_weight_mode,
        status="completed",
        run_root=run_root,
        aggregate_zone_count=len(outputs.wide_by_zone),
        loaded_variable_count=normal_variable_count,
        aggregated_long_rows=int(len(outputs.long_frame)),
        static_equipment_rows=int(len(outputs.static_equipment_frame)),
        equipment_contribution_rows=int(len(outputs.equipment_contribution_frame)),
        diagnostic_rows=int(len(outputs.diagnostics_frame)),
        runtime_seconds=runtime_seconds,
    )



def empty_aggregation_rule_outputs() -> AggregationRuleOutputs:
    """Create an empty AggregationRuleOutputs container.

    This enables one-parquet-at-a-time aggregation. The engine repeatedly merges
    per-variable outputs into this accumulator instead of keeping all loaded
    parquet dataframes in memory at once.
    """
    return AggregationRuleOutputs(
        wide_by_zone={},
        long_frame=pd.DataFrame(),
        static_equipment_frame=pd.DataFrame(),
        equipment_contribution_frame=pd.DataFrame(),
        diagnostics_frame=pd.DataFrame(),
        rule_summary_frame=pd.DataFrame(),
    )


def merge_aggregation_rule_outputs(
    left: AggregationRuleOutputs,
    right: AggregationRuleOutputs,
) -> AggregationRuleOutputs:
    """Merge two AggregationRuleOutputs objects.

    Wide outputs are merged by aggregate zone on timestamp_raw. Long,
    static-equipment, contribution, diagnostic, and rule-summary tables are
    appended row-wise. Empty tables are handled defensively.
    """
    merged_wide_by_zone: dict[str, pd.DataFrame] = {
        zone_id: frame.copy()
        for zone_id, frame in left.wide_by_zone.items()
    }

    for aggregate_zone_id, right_wide in right.wide_by_zone.items():
        if right_wide is None or right_wide.empty:
            continue

        if aggregate_zone_id in merged_wide_by_zone:
            left_wide = merged_wide_by_zone[aggregate_zone_id]
            if (
                "timestamp_raw" in left_wide.columns
                and "timestamp_raw" in right_wide.columns
            ):
                merged_wide_by_zone[aggregate_zone_id] = left_wide.merge(
                    right_wide,
                    on="timestamp_raw",
                    how="outer",
                )
            else:
                merged_wide_by_zone[aggregate_zone_id] = pd.concat(
                    [left_wide, right_wide],
                    axis=1,
                )
        else:
            merged_wide_by_zone[aggregate_zone_id] = right_wide.copy()

    return AggregationRuleOutputs(
        wide_by_zone=merged_wide_by_zone,
        long_frame=concat_dataframes(left.long_frame, right.long_frame),
        static_equipment_frame=concat_dataframes(
            left.static_equipment_frame,
            right.static_equipment_frame,
        ),
        equipment_contribution_frame=concat_dataframes(
            left.equipment_contribution_frame,
            right.equipment_contribution_frame,
        ),
        diagnostics_frame=concat_dataframes(
            left.diagnostics_frame,
            right.diagnostics_frame,
        ),
        rule_summary_frame=concat_dataframes(
            left.rule_summary_frame,
            right.rule_summary_frame,
        ),
    )


def concat_dataframes(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two dataframes while preserving empty-frame behavior."""
    if left is None or left.empty:
        return right.copy() if right is not None else pd.DataFrame()
    if right is None or right.empty:
        return left.copy()
    return pd.concat([left, right], ignore_index=True)


def resolve_plan_path(
    *,
    campaign_root: Path,
    case_id: str,
    explicit_plan_path: str | None,
    aggregation_id: str | None,
    strategy: str,
    rule_set: str,
    weight_mode: str,
) -> Path:
    """Resolve latest matching aggregation plan path.

    Resolution priority:
        1. Explicit --plan-path.
        2. Explicit --aggregation-id.
        3. Legacy deterministic strategy/rule_set/weight_mode aggregation id.
    """
    if explicit_plan_path:
        plan_path = Path(explicit_plan_path).expanduser().resolve()
        if not plan_path.is_file():
            raise FileNotFoundError(f"Explicit aggregation plan not found: {plan_path}")
        return plan_path

    plans_root = campaign_root / "aggregation" / "plans"
    if not plans_root.is_dir():
        raise SystemExit(
            f"Plan root does not exist: {plans_root}. "
            "Run build_p1_aggregation_plan.py first."
        )

    if aggregation_id:
        effective_aggregation_id = aggregation_id
    else:
        effective_aggregation_id = f"{strategy}_{rule_set}_{weight_mode}_v1"

    candidates = sorted(
        plans_root.glob(
            f"plan_build_*"
            f"/{case_id}"
            f"/{effective_aggregation_id}"
            f"/aggregation_plan.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise SystemExit(
            "No matching aggregation_plan.json found. "
            f"plans_root={plans_root}, "
            f"case_id={case_id}, "
            f"aggregation_id={effective_aggregation_id}"
        )

    return candidates[0]


def build_aggregation_run_id(
    *,
    source_generation_run_id: str,
    plan_aggregation_id: str = "",
) -> str:
    """Build aggregation run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_source = source_generation_run_id.replace("epvwr_", "")[:8]

    if plan_aggregation_id:
        safe_plan = make_safe_name(plan_aggregation_id)[:40]
        return f"aggr_{timestamp}_{short_source}_{safe_plan}"

    return f"aggr_{timestamp}_{short_source}"


def build_campaign_aggregation_run_id() -> str:
    """Build campaign-level aggregation run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"aggregation_campaign_{timestamp}"


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows."""
    import csv

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_legacy_aggregation_pickle(
    *,
    path: Path,
    wide_by_zone: dict[str, Any],
    static_equipment_frame: Any,
) -> None:
    """Write a simple legacy-style aggregation pickle.

    This is compatibility-oriented, not the primary data product.
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}

    first_wide = next(iter(wide_by_zone.values()), None)
    if first_wide is not None and "timestamp_raw" in first_wide.columns:
        payload["DateTime_List"] = first_wide["timestamp_raw"].tolist()

    for zone_name, wide_frame in wide_by_zone.items():
        legacy_frame = wide_frame.drop(columns=["timestamp_raw"], errors="ignore")
        payload[zone_name] = legacy_frame

        zone_static = static_equipment_frame[
            static_equipment_frame["aggregate_zone_id"] == zone_name
        ].copy()

        equipment_payload = {}
        for _, row in zone_static.iterrows():
            equipment_payload[str(row["output_variable_name"])] = row["value"]

        payload[f"{zone_name}_Equipment"] = equipment_payload

    with path.open("wb") as stream:
        pickle.dump(payload, stream)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))