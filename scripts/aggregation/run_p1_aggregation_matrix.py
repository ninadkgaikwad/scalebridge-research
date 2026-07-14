# -*- coding: utf-8 -*-
"""Run a ScaleBridge P1 aggregation matrix over exact aggregation plan paths.

This runner is designed for the P1 4-building x 4-climate aggregation study.
It reads aggregation_plan_index.csv files produced by build_p1_aggregation_plan.py,
selects exact aggregation_plan.json paths by aggregation_id / weight_mode / case_id,
and runs each selected plan explicitly.

Why this exists:
    run_p1_aggregation.py is convenient for one aggregation_id, but discovery by
    aggregation_id alone becomes ambiguous once the same aggregation_id exists for
    multiple weight modes. This matrix runner resolves that by selecting exact
    plan_path rows from aggregation_plan_index.csv.

Typical full matrix:
    16 cases x 5 aggregation levels x 3 weight modes = 240 case-plan runs.

Example smoke run:
    python scripts\\aggregation\\run_p1_aggregation_matrix.py `
      --campaign-id p1_compact_4b4c_labpc_1w_v1 `
      --aggregation-id p1_l01_all_to_one `
      --weight-mode equal `
      --continue-on-error `
      --write-legacy-pickle `
      --mlflow `
      --mlflow-experiment-name ScaleBridge_P1_Aggregation_4b4c_1w

Example full run:
    python scripts\\aggregation\\run_p1_aggregation_matrix.py `
      --campaign-id p1_compact_4b4c_labpc_1w_v1 `
      --aggregation-id p1_l01_all_to_one `
      --aggregation-id p1_l02_functional `
      --aggregation-id p1_l03_intermediate `
      --aggregation-id p1_l04_spatial_detailed `
      --aggregation-id p1_l05_identity `
      --weight-mode equal `
      --weight-mode floor_area `
      --weight-mode volume `
      --continue-on-error `
      --write-legacy-pickle `
      --mlflow `
      --mlflow-experiment-name ScaleBridge_P1_Aggregation_4b4c_1w
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.discovery import (
    DEFAULT_CAMPAIGN_ID,
    discover_generation_runs,
    load_json,
    resolve_campaign_root,
    resolve_repo_root,
)
from scalebridge.data.aggregation.engine import run_aggregation_for_generation_run
from scalebridge.data.aggregation.models import GenerationRunRef
from scalebridge.data.aggregation.writers import write_csv, write_json

from scalebridge.tracking.mlflow.aggregation import (
    maybe_end_mlflow_run,
    maybe_start_mlflow_parent_run,
    mlflow_log_campaign_summary,
    mlflow_log_case_result,
)


DEFAULT_AGGREGATION_IDS = [
    "p1_l01_all_to_one",
    "p1_l02_functional",
    "p1_l03_intermediate",
    "p1_l04_spatial_detailed",
    "p1_l05_identity",
]

DEFAULT_WEIGHT_MODES = [
    "equal",
    "floor_area",
    "volume",
]

AGGREGATION_ID_METADATA = {
    "p1_l01_all_to_one": {
        "aggregation_level": "L01",
        "aggregation_level_index": 1,
        "aggregation_family": "all_to_one",
    },
    "p1_l02_functional": {
        "aggregation_level": "L02",
        "aggregation_level_index": 2,
        "aggregation_family": "functional",
    },
    "p1_l03_intermediate": {
        "aggregation_level": "L03",
        "aggregation_level_index": 3,
        "aggregation_family": "intermediate",
    },
    "p1_l04_spatial_detailed": {
        "aggregation_level": "L04",
        "aggregation_level_index": 4,
        "aggregation_family": "spatial_detailed",
    },
    "p1_l05_identity": {
        "aggregation_level": "L05",
        "aggregation_level_index": 5,
        "aggregation_family": "identity",
    },
}


@dataclass(frozen=True)
class MatrixPlanRef:
    """One exact aggregation plan selected for matrix execution."""

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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the aggregation matrix runner."""
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

    run_refs, missing_generation_rows = discover_generation_runs(
        cases_root=cases_root,
        case_id=args.case_id,
        include_failed=False,
    )
    if not run_refs:
        raise SystemExit(f"No successful generation runs found under {cases_root}")

    run_ref_by_case_id = {run_ref.case_id: run_ref for run_ref in run_refs}

    aggregation_ids = args.aggregation_id or list(DEFAULT_AGGREGATION_IDS)
    weight_modes = args.weight_mode or list(DEFAULT_WEIGHT_MODES)

    selected_plans = discover_matrix_plans(
        campaign_root=campaign_root,
        run_ref_by_case_id=run_ref_by_case_id,
        aggregation_ids=aggregation_ids,
        weight_modes=weight_modes,
        plan_build_root=args.plan_build_root,
        latest_per_key=not args.keep_duplicate_plan_builds,
    )

    if args.case_limit is not None:
        selected_plans = selected_plans[: max(0, args.case_limit)]

    if not selected_plans:
        raise SystemExit(
            "No aggregation plans selected. "
            f"campaign_root={campaign_root}, "
            f"aggregation_ids={aggregation_ids}, "
            f"weight_modes={weight_modes}"
        )

    matrix_run_id = args.matrix_run_id or build_matrix_run_id()
    summary_root = campaign_root / "aggregation" / "matrix_runs"
    summary_dir = summary_root / matrix_run_id
    summary_dir.mkdir(parents=True, exist_ok=True)

    selected_plan_rows = [matrix_plan_ref_to_row(plan_ref) for plan_ref in selected_plans]
    write_csv(summary_dir / "selected_aggregation_plans.csv", selected_plan_rows)

    result_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []

    print("=" * 100)
    print("SCALEBRIDGE P1 AGGREGATION MATRIX RUNNER")
    print("=" * 100)
    print(f"repo_root: {repo_root}")
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"cases_root: {cases_root}")
    print(f"matrix_run_id: {matrix_run_id}")
    print(f"summary_dir: {summary_dir}")
    print(f"selected_plan_count: {len(selected_plans)}")
    print(f"aggregation_ids: {aggregation_ids}")
    print(f"weight_modes: {weight_modes}")
    print(f"write_legacy_pickle: {args.write_legacy_pickle}")
    print(f"mlflow: {args.mlflow}")
    print()

    mlflow_parent_run = maybe_start_mlflow_parent_run(
        enabled=args.mlflow,
        campaign_id=args.campaign_id,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.mlflow_run_name or matrix_run_id,
        params={
            "campaign_id": args.campaign_id,
            "matrix_run_id": matrix_run_id,
            "selected_plan_count": len(selected_plans),
            "aggregation_ids": ",".join(aggregation_ids),
            "weight_modes": ",".join(weight_modes),
            "write_legacy_pickle": args.write_legacy_pickle,
            "continue_on_error": args.continue_on_error,
        },
    )

    started_at = time.perf_counter()

    try:
        for index, plan_ref in enumerate(selected_plans, start=1):
            print(
                f"[{index}/{len(selected_plans)}] "
                f"{plan_ref.case_id} | "
                f"{plan_ref.building_type} | "
                f"{plan_ref.weather_location or plan_ref.climate_zone} | "
                f"{plan_ref.aggregation_id} | "
                f"{plan_ref.weight_mode}"
            )
            print(f"    plan_path: {plan_ref.plan_path}")

            run_ref = run_ref_by_case_id[plan_ref.case_id]
            aggregation_run_id = build_matrix_aggregation_run_id(
                sequence_index=index,
                source_generation_run_id=run_ref.run_id,
                aggregation_id=plan_ref.aggregation_id,
                weight_mode=plan_ref.weight_mode,
            )

            try:
                result = run_aggregation_for_generation_run(
                    campaign_root=campaign_root,
                    run_ref=run_ref,
                    plan_path=plan_ref.plan_path,
                    aggregation_run_id=aggregation_run_id,
                    max_variables=args.max_variables,
                    preview_rows=args.preview_rows,
                    write_legacy_pickle=args.write_legacy_pickle,
                )

                row = {
                    **matrix_plan_ref_to_row(plan_ref),
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
                }

                if plan_ref.source_zone_count and plan_ref.aggregate_zone_count:
                    row["aggregation_compression_ratio"] = (
                        plan_ref.aggregate_zone_count / plan_ref.source_zone_count
                    )
                else:
                    row["aggregation_compression_ratio"] = ""

                manifest_path = result.run_root / "aggregation_manifest.json"
                if manifest_path.is_file():
                    manifest_payload = load_json(manifest_path)
                    row["rule_summary_rows"] = manifest_payload.get("rule_summary_rows", "")

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
                    enabled=args.mlflow,
                    case_result=row,
                    artifact_root=result.run_root / "diagnostics",
                )

            except Exception as exc:
                row = {
                    **matrix_plan_ref_to_row(plan_ref),
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
                    "aggregation_compression_ratio": (
                        plan_ref.aggregate_zone_count / plan_ref.source_zone_count
                        if plan_ref.aggregate_zone_count and plan_ref.source_zone_count
                        else ""
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                result_rows.append(row)

                print(f"ERROR aggregating {plan_ref.case_id}: {exc}")

                mlflow_log_case_result(
                    enabled=args.mlflow,
                    case_result=row,
                    artifact_root=summary_dir,
                )

                if not args.continue_on_error:
                    raise

    finally:
        pass

    runtime_seconds = time.perf_counter() - started_at

    write_csv(summary_dir / "aggregation_matrix_case_runs.csv", result_rows)
    write_csv(summary_dir / "aggregation_matrix_outputs.csv", output_rows)
    write_csv(summary_dir / "missing_generation_rows.csv", missing_generation_rows)

    summary = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "matrix_run_id": matrix_run_id,
        "summary_dir": str(summary_dir),
        "selected_plan_count": len(selected_plans),
        "successful_plan_count": sum(
            1 for row in result_rows if row.get("status") == "completed"
        ),
        "failed_plan_count": sum(
            1 for row in result_rows if row.get("status") != "completed"
        ),
        "aggregation_ids": aggregation_ids,
        "weight_modes": weight_modes,
        "building_types": sorted(
            {str(row.get("building_type", "")) for row in result_rows if row.get("building_type")}
        ),
        "climate_zones": sorted(
            {str(row.get("climate_zone", "")) for row in result_rows if row.get("climate_zone")}
        ),
        "weather_locations": sorted(
            {str(row.get("weather_location", "")) for row in result_rows if row.get("weather_location")}
        ),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "selected_aggregation_plans": str(summary_dir / "selected_aggregation_plans.csv"),
            "aggregation_matrix_case_runs": str(summary_dir / "aggregation_matrix_case_runs.csv"),
            "aggregation_matrix_outputs": str(summary_dir / "aggregation_matrix_outputs.csv"),
            "missing_generation_rows": str(summary_dir / "missing_generation_rows.csv"),
        },
    }
    write_json(summary_dir / "aggregation_matrix_manifest.json", summary)

    mlflow_log_campaign_summary(
        enabled=args.mlflow,
        summary=summary,
        summary_dir=summary_dir,
    )
    maybe_end_mlflow_run(args.mlflow)

    print()
    print("=" * 100)
    print("AGGREGATION MATRIX SUMMARY")
    print("=" * 100)
    print(f"selected_plan_count: {summary['selected_plan_count']}")
    print(f"successful_plan_count: {summary['successful_plan_count']}")
    print(f"failed_plan_count: {summary['failed_plan_count']}")
    print(f"summary_dir: {summary_dir}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
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
        help=(
            "Optional limit on selected plan count. "
            "Useful for smoke testing, not a case count."
        ),
    )
    parser.add_argument(
        "--plan-build-root",
        default=None,
        help=(
            "Optional specific plan_build_* root. If omitted, all plan_build_* "
            "folders under <campaign_root>/aggregation/plans are scanned."
        ),
    )
    parser.add_argument(
        "--aggregation-id",
        action="append",
        default=None,
        help=(
            "Aggregation ID to include. Can be repeated. "
            "Default: all P1 levels L01-L05."
        ),
    )
    parser.add_argument(
        "--weight-mode",
        action="append",
        default=None,
        help=(
            "Weight mode to include. Can be repeated. "
            "Default: equal, floor_area, volume."
        ),
    )
    parser.add_argument(
        "--keep-duplicate-plan-builds",
        action="store_true",
        help=(
            "Keep duplicate plans from older plan_build_* folders. "
            "Default behavior keeps only latest per case_id/aggregation_id/weight_mode."
        ),
    )
    parser.add_argument(
        "--matrix-run-id",
        default=None,
        help="Optional matrix run id. If omitted, generated automatically.",
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
        help="Continue matrix aggregation if one case-plan fails.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for aggregation matrix and case-plan runs.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=None,
        help=(
            "MLflow experiment name. Default behavior follows aggregation "
            "tracking helper."
        ),
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow parent run name.",
    )

    return parser.parse_args(argv)


def discover_matrix_plans(
    *,
    campaign_root: Path,
    run_ref_by_case_id: dict[str, GenerationRunRef],
    aggregation_ids: list[str],
    weight_modes: list[str],
    plan_build_root: str | None,
    latest_per_key: bool,
) -> list[MatrixPlanRef]:
    """Discover exact matrix plan paths from aggregation_plan_index.csv files."""
    plans_root = campaign_root / "aggregation" / "plans"
    if plan_build_root:
        roots = [Path(plan_build_root).expanduser().resolve()]
    else:
        roots = sorted(
            [path for path in plans_root.glob("plan_build_*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    if not roots:
        raise SystemExit(
            f"No plan_build_* folders found under {plans_root}. "
            "Run build_p1_aggregation_plan.py first."
        )

    selected_by_key: dict[tuple[str, str, str], MatrixPlanRef] = {}
    selected_with_duplicates: list[MatrixPlanRef] = []

    aggregation_id_set = set(aggregation_ids)
    weight_mode_set = set(weight_modes)

    for root in roots:
        index_path = root / "aggregation_plan_index.csv"
        if not index_path.is_file():
            continue

        for raw_row in read_csv_dicts(index_path):
            case_id = str(raw_row.get("case_id", "")).strip()
            aggregation_id = str(raw_row.get("aggregation_id", "")).strip()
            weight_mode = str(raw_row.get("weight_mode", "")).strip()

            if not case_id or not aggregation_id:
                continue
            if case_id not in run_ref_by_case_id:
                continue
            if aggregation_id not in aggregation_id_set:
                continue
            if weight_mode not in weight_mode_set:
                continue

            plan_path = resolve_index_plan_path(raw_row=raw_row, index_path=index_path)
            if not plan_path.is_file():
                continue

            run_ref = run_ref_by_case_id[case_id]
            source_manifest = load_json(run_ref.manifest_path)
            case_spec = source_manifest.get("case_spec", {})

            plan_payload = load_json(plan_path)
            level_meta = AGGREGATION_ID_METADATA.get(
                aggregation_id,
                {
                    "aggregation_level": aggregation_id,
                    "aggregation_level_index": -1,
                    "aggregation_family": "custom",
                },
            )

            plan_ref = MatrixPlanRef(
                case_id=case_id,
                source_generation_run_id=run_ref.run_id,
                plan_path=plan_path,
                plan_build_root=root,
                plan_build_id=root.name,
                aggregation_id=aggregation_id,
                aggregation_level=str(level_meta["aggregation_level"]),
                aggregation_level_index=int(level_meta["aggregation_level_index"]),
                aggregation_family=str(level_meta["aggregation_family"]),
                plan_strategy=str(
                    raw_row.get("strategy")
                    or plan_payload.get("strategy")
                    or ""
                ),
                rule_set=str(
                    raw_row.get("rule_set")
                    or plan_payload.get("rule_set")
                    or ""
                ),
                weight_mode=str(
                    raw_row.get("weight_mode")
                    or plan_payload.get("weight_mode")
                    or ""
                ),
                aggregate_zone_count=parse_optional_int(
                    raw_row.get("aggregate_zone_count")
                )
                or infer_aggregate_zone_count(plan_payload),
                source_zone_count=parse_optional_int(
                    raw_row.get("source_zone_count")
                )
                or infer_source_zone_count(plan_payload),
                building_type=str(case_spec.get("building_type", "")),
                climate_zone=str(case_spec.get("climate_zone", "")),
                weather_location=str(
                    case_spec.get("weather_location")
                    or case_spec.get("weather_name")
                    or case_spec.get("city")
                    or ""
                ),
            )

            if latest_per_key:
                key = (plan_ref.case_id, plan_ref.aggregation_id, plan_ref.weight_mode)
                if key not in selected_by_key:
                    selected_by_key[key] = plan_ref
            else:
                selected_with_duplicates.append(plan_ref)

    selected = (
        list(selected_by_key.values())
        if latest_per_key
        else selected_with_duplicates
    )

    return sorted(
        selected,
        key=lambda item: (
            item.weight_mode,
            item.aggregation_level_index,
            item.building_type,
            item.weather_location,
            item.case_id,
        ),
    )


def resolve_index_plan_path(*, raw_row: dict[str, Any], index_path: Path) -> Path:
    """Resolve a plan path from a row in aggregation_plan_index.csv."""
    raw_plan_path = str(raw_row.get("plan_path", "")).strip()
    if raw_plan_path:
        path = Path(raw_plan_path)
        if path.is_absolute():
            return path
        return (index_path.parent / path).resolve()

    case_id = str(raw_row.get("case_id", "")).strip()
    aggregation_id = str(raw_row.get("aggregation_id", "")).strip()
    return (index_path.parent / case_id / aggregation_id / "aggregation_plan.json").resolve()


def matrix_plan_ref_to_row(plan_ref: MatrixPlanRef) -> dict[str, Any]:
    """Convert a matrix plan ref into a CSV/MLflow-ready row."""
    aggregate_zone_count = plan_ref.aggregate_zone_count or ""
    source_zone_count = plan_ref.source_zone_count or ""
    compression_ratio: float | str
    if plan_ref.aggregate_zone_count and plan_ref.source_zone_count:
        compression_ratio = plan_ref.aggregate_zone_count / plan_ref.source_zone_count
    else:
        compression_ratio = ""

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
        "aggregate_zone_count": aggregate_zone_count,
        "source_zone_count": source_zone_count,
        "aggregation_compression_ratio": compression_ratio,
        "plan_build_id": plan_ref.plan_build_id,
        "plan_build_root": str(plan_ref.plan_build_root),
        "plan_path": str(plan_ref.plan_path),
    }


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_optional_int(value: Any) -> int | None:
    """Parse optional integer values from CSV cells."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def infer_aggregate_zone_count(plan_payload: dict[str, Any]) -> int | None:
    """Infer aggregate zone count from an aggregation plan payload."""
    aggregate_zones = plan_payload.get("aggregate_zones")
    if isinstance(aggregate_zones, list):
        return len(aggregate_zones)
    return None


def infer_source_zone_count(plan_payload: dict[str, Any]) -> int | None:
    """Infer unique source zone count from an aggregation plan payload."""
    aggregate_zones = plan_payload.get("aggregate_zones")
    if not isinstance(aggregate_zones, list):
        return None

    source_zones: set[str] = set()
    for aggregate_zone in aggregate_zones:
        if not isinstance(aggregate_zone, dict):
            continue
        for zone in aggregate_zone.get("source_zones", []) or []:
            source_zones.add(str(zone).strip())
    return len(source_zones)


def build_matrix_run_id() -> str:
    """Build matrix run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"aggregation_matrix_{timestamp}"


def build_matrix_aggregation_run_id(
    *,
    sequence_index: int,
    source_generation_run_id: str,
    aggregation_id: str,
    weight_mode: str,
) -> str:
    """Build unique aggregation run ID for one case-plan matrix item."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_source = source_generation_run_id.replace("epvwr_", "")[:8]
    safe_aggregation_id = safe_token(aggregation_id, max_length=36)
    safe_weight_mode = safe_token(weight_mode, max_length=16)
    return (
        f"aggr_{timestamp}_{sequence_index:04d}_"
        f"{short_source}_{safe_aggregation_id}_{safe_weight_mode}"
    )


def safe_token(value: str, *, max_length: int) -> str:
    """Make a compact filesystem-safe token."""
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    text = "".join(safe_chars).strip("_")
    return text[:max_length] or "token"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
