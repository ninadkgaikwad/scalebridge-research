# -*- coding: utf-8 -*-
"""Probe legacy_v1 aggregation rules on one ScaleBridge generation run.

This is a development validation script. Production aggregation orchestration
will later live in:

    src/scalebridge/data/aggregation/engine.py
    src/scalebridge/data/aggregation/writers.py

This script intentionally calls modular code from src and writes preview outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.discovery import (
    DEFAULT_CAMPAIGN_ID,
    discover_generation_runs,
    load_json,
    resolve_campaign_root,
    resolve_repo_root,
)
from scalebridge.data.aggregation.eio import schedule_equipment_mapping_rows
from scalebridge.data.aggregation.loaders import CanonicalVariableLoader
from scalebridge.data.aggregation.rules import apply_legacy_v1_rules


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
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

    if len(run_refs) > 1:
        print(f"Found {len(run_refs)} generation runs. Using the first one.")
    run_ref = run_refs[0]

    plan_path = resolve_plan_path(
        campaign_root=campaign_root,
        case_id=run_ref.case_id,
        explicit_plan_path=args.plan_path,
        strategy=args.strategy,
    )
    if not plan_path.is_file():
        raise SystemExit(f"Aggregation plan not found: {plan_path}")

    output_dir = resolve_output_dir(
        campaign_root=campaign_root,
        output_dir=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SCALEBRIDGE AGGREGATION RULE PROBE")
    print("=" * 100)
    print(f"repo_root: {repo_root}")
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"case_id: {run_ref.case_id}")
    print(f"run_id: {run_ref.run_id}")
    print(f"plan_path: {plan_path}")
    print(f"output_dir: {output_dir}")
    print()

    plan = load_json(plan_path)
    eio_payload = load_json(run_ref.run_root / "canonical" / "eio_tables.json")

    zone_mapping_rows = read_csv_dicts(plan_path.parent / "zone_mapping.csv")
    excluded_zone_rows = read_csv_dicts(plan_path.parent / "excluded_zones.csv")
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

    # Add case/run fields for easier debugging.
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

    if args.max_variables is not None:
        records = records[: max(0, args.max_variables)]

    variable_frames_by_name: dict[str, Any] = {}
    load_rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] Loading {record.variable_name}")
        frame = loader.load_variable_long_by_id(record.variable_id)
        variable_frames_by_name[record.variable_name] = frame
        load_rows.append(
            {
                "variable_id": record.variable_id,
                "variable_name": record.variable_name,
                "row_count": len(frame),
                "key_value_count": (
                    frame["key_value"].nunique() if "key_value" in frame.columns else ""
                ),
                "parquet_path": str(record.canonical_parquet_path),
            }
        )

    print()
    print("Applying legacy_v1 rules...")
    outputs = apply_legacy_v1_rules(
        plan=plan,
        variable_frames_by_name=variable_frames_by_name,
        schedule_equipment_rows=schedule_rows,
        zone_mapping_rows=zone_mapping_rows,
        excluded_zone_names=excluded_zone_names,
    )

    write_csv(output_dir / "loaded_variables.csv", load_rows)
    write_csv(output_dir / "schedule_equipment_mapping_used.csv", schedule_rows)

    outputs.rule_summary_frame.to_csv(
        output_dir / "rule_probe_summary.csv",
        index=False,
    )
    outputs.diagnostics_frame.to_csv(
        output_dir / "rule_diagnostics.csv",
        index=False,
    )
    outputs.static_equipment_frame.to_csv(
        output_dir / "static_equipment_preview.csv",
        index=False,
    )

    outputs.equipment_contribution_frame.to_csv(
        output_dir / "equipment_contribution_preview.csv",
        index=False,
    )

    outputs.long_frame.head(args.preview_rows).to_csv(
        output_dir / "aggregated_timeseries_long_preview.csv",
        index=False,
    )

    for aggregate_zone_id, wide_frame in outputs.wide_by_zone.items():
        safe_zone = make_safe_name(aggregate_zone_id)
        wide_frame.head(args.preview_rows).to_csv(
            output_dir / f"{safe_zone}_aggregated_timeseries_wide_preview.csv",
            index=False,
        )

        if args.write_full_parquet:
            wide_frame.to_parquet(
                output_dir / f"{safe_zone}_aggregated_timeseries_wide.parquet",
                index=False,
            )

    if args.write_full_parquet:
        outputs.long_frame.to_parquet(
            output_dir / "aggregated_timeseries_long.parquet",
            index=False,
        )
        outputs.static_equipment_frame.to_parquet(
            output_dir / "static_equipment.parquet",
            index=False,
        )
        outputs.equipment_contribution_frame.to_parquet(
            output_dir / "equipment_contribution.parquet",
            index=False,
        )

    summary = {
        "schema_version": "0.1.0",
        "created_at_local": datetime.now().isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "case_id": run_ref.case_id,
        "run_id": run_ref.run_id,
        "plan_path": str(plan_path),
        "output_dir": str(output_dir),
        "loaded_variable_count": len(variable_frames_by_name),
        "aggregated_zone_count": len(outputs.wide_by_zone),
        "aggregated_long_rows": int(len(outputs.long_frame)),
        "static_equipment_rows": int(len(outputs.static_equipment_frame)),
        "equipment_contribution_rows": int(len(outputs.equipment_contribution_frame)),
        "diagnostic_rows": int(len(outputs.diagnostics_frame)),
        "rule_summary_rows": int(len(outputs.rule_summary_frame)),
        "write_full_parquet": args.write_full_parquet,
    }
    write_json(output_dir / "rule_probe_summary.json", summary)

    print()
    print("=" * 100)
    print("RULE PROBE SUMMARY")
    print("=" * 100)
    print(f"loaded_variable_count: {summary['loaded_variable_count']}")
    print(f"aggregated_zone_count: {summary['aggregated_zone_count']}")
    print(f"aggregated_long_rows: {summary['aggregated_long_rows']}")
    print(f"static_equipment_rows: {summary['static_equipment_rows']}")
    print(f"diagnostic_rows: {summary['diagnostic_rows']}")
    print(f"rule_summary_rows: {summary['rule_summary_rows']}")
    print()
    print(f"Wrote probe outputs to: {output_dir}")

    if missing_rows:
        write_csv(output_dir / "discovery_missing_rows.csv", missing_rows)

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
        "--plan-path",
        default=None,
        help="Explicit aggregation_plan.json path.",
    )
    parser.add_argument(
        "--strategy",
        default="all_thermal_zones_to_one",
        help="Plan strategy to auto-discover if --plan-path is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output folder. Default: "
            "<campaign_root>/aggregation/probes/rule_probe_<timestamp>."
        ),
    )
    parser.add_argument(
        "--max-variables",
        type=int,
        default=None,
        help="Optional max number of variables to load for quick debugging.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=100,
        help="Rows to write in preview CSVs.",
    )
    parser.add_argument(
        "--write-full-parquet",
        action="store_true",
        help="Also write full aggregated Parquet outputs from the probe.",
    )

    return parser.parse_args(argv)


def resolve_plan_path(
    *,
    campaign_root: Path,
    case_id: str,
    explicit_plan_path: str | None,
    strategy: str,
) -> Path:
    """Resolve aggregation_plan.json path."""
    if explicit_plan_path:
        return Path(explicit_plan_path).expanduser().resolve()

    plans_root = campaign_root / "aggregation" / "plans"
    if not plans_root.is_dir():
        raise SystemExit(
            f"Plan root does not exist: {plans_root}. "
            "Run build_p1_aggregation_plan.py first."
        )

    candidates = sorted(
        plans_root.glob(
            f"plan_build_*"
            f"/{case_id}"
            f"/{strategy}_legacy_v1_equal_v1"
            f"/aggregation_plan.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise SystemExit(
            "No matching aggregation_plan.json found. "
            f"plans_root={plans_root}, case_id={case_id}, strategy={strategy}"
        )

    return candidates[0]


def resolve_output_dir(
    *,
    campaign_root: Path,
    output_dir: str | None,
) -> Path:
    """Resolve output directory."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return campaign_root / "aggregation" / "probes" / f"rule_probe_{timestamp}"


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows as dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    if not fieldnames:
        fieldnames = ["note"]
        rows = [{"note": "no rows"}]

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_safe_name(value: str) -> str:
    """Return a filesystem-safe name."""
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return safe.strip("_") or "unnamed"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))