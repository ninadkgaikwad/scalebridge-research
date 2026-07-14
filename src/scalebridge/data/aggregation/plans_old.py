# -*- coding: utf-8 -*-
"""Aggregation plan builders for ScaleBridge."""

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
from scalebridge.data.aggregation.eio import zone_information_rows
from scalebridge.data.aggregation.models import (
    AggregateZoneGroup,
    AggregationPlan,
    AggregationRuleSet,
    AggregationStrategy,
    AggregationWeightMode,
    GenerationRunRef,
)


DEFAULT_AGGREGATE_ZONE_NAME_STEM = "Aggregated_Zone"
DEFAULT_SYSTEM_NODE_NAME_PATTERN = "DIRECT AIR INLET NODE"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for building aggregation plans."""
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
        raise SystemExit(
            "No successful generation runs found. "
            f"cases_root={cases_root}, case_id={args.case_id}"
        )

    output_root = resolve_output_root(
        campaign_root=campaign_root,
        output_root=args.output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SCALEBRIDGE AGGREGATION PLAN BUILDER")
    print("=" * 100)
    print(f"repo_root: {repo_root}")
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"cases_root: {cases_root}")
    print(f"output_root: {output_root}")
    print(f"strategy: {args.strategy}")
    print(f"rule_set: {args.rule_set}")
    print(f"weight_mode: {args.weight_mode}")
    print()

    plan_rows: list[dict[str, Any]] = []
    all_zone_mapping_rows: list[dict[str, Any]] = []
    all_included_rows: list[dict[str, Any]] = []
    all_excluded_rows: list[dict[str, Any]] = []
    all_missing_rows: list[dict[str, Any]] = list(missing_rows)

    for index, run_ref in enumerate(run_refs, start=1):
        print(f"[{index}/{len(run_refs)}] Building plan for {run_ref.case_id}")

        result = build_and_write_plan_for_run(
            run_ref=run_ref,
            output_root=output_root,
            strategy=AggregationStrategy(args.strategy),
            rule_set=AggregationRuleSet(args.rule_set),
            weight_mode=AggregationWeightMode(args.weight_mode),
            aggregate_zone_name_stem=args.aggregate_zone_name_stem,
            system_node_name_pattern=args.system_node_name_pattern,
        )

        plan_rows.append(result["plan_row"])
        all_zone_mapping_rows.extend(result["zone_mapping_rows"])
        all_included_rows.extend(result["included_thermal_zone_rows"])
        all_excluded_rows.extend(result["excluded_zone_rows"])
        all_missing_rows.extend(result["missing_rows"])

    write_csv(output_root / "aggregation_plan_index.csv", plan_rows)
    write_csv(output_root / "zone_mapping_all_cases.csv", all_zone_mapping_rows)
    write_csv(output_root / "included_thermal_zones_all_cases.csv", all_included_rows)
    write_csv(output_root / "excluded_zones_all_cases.csv", all_excluded_rows)
    write_csv(output_root / "missing_plan_inputs.csv", all_missing_rows)

    summary = {
        "schema_version": "0.1.0",
        "created_at_local": datetime.now().isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "output_root": str(output_root),
        "strategy": args.strategy,
        "rule_set": args.rule_set,
        "weight_mode": args.weight_mode,
        "plan_count": len(plan_rows),
        "zone_mapping_row_count": len(all_zone_mapping_rows),
        "included_thermal_zone_row_count": len(all_included_rows),
        "excluded_zone_row_count": len(all_excluded_rows),
        "missing_plan_input_row_count": len(all_missing_rows),
    }
    write_json(output_root / "aggregation_plan_build_summary.json", summary)

    print()
    print("=" * 100)
    print("PLAN BUILD SUMMARY")
    print("=" * 100)
    print(f"plan_count: {summary['plan_count']}")
    print(f"zone_mapping_row_count: {summary['zone_mapping_row_count']}")
    print(f"included_thermal_zone_row_count: {summary['included_thermal_zone_row_count']}")
    print(f"excluded_zone_row_count: {summary['excluded_zone_row_count']}")
    print(f"missing_plan_input_row_count: {summary['missing_plan_input_row_count']}")
    print()
    print(f"Wrote plan outputs to: {output_root}")

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
        "--output-root",
        default=None,
        help=(
            "Output root for plans. Default: "
            "<campaign_root>/aggregation/plans/<timestamp>."
        ),
    )
    parser.add_argument(
        "--strategy",
        default=AggregationStrategy.ALL_THERMAL_ZONES_TO_ONE.value,
        choices=[item.value for item in AggregationStrategy],
    )
    parser.add_argument(
        "--rule-set",
        default=AggregationRuleSet.LEGACY_V1.value,
        choices=[item.value for item in AggregationRuleSet],
    )
    parser.add_argument(
        "--weight-mode",
        default=AggregationWeightMode.EQUAL.value,
        choices=[item.value for item in AggregationWeightMode],
    )
    parser.add_argument(
        "--aggregate-zone-name-stem",
        default=DEFAULT_AGGREGATE_ZONE_NAME_STEM,
    )
    parser.add_argument(
        "--system-node-name-pattern",
        default=DEFAULT_SYSTEM_NODE_NAME_PATTERN,
    )

    return parser.parse_args(argv)


def resolve_output_root(
    *,
    campaign_root: Path,
    output_root: str | None,
) -> Path:
    """Resolve output root for aggregation plans."""
    if output_root:
        return Path(output_root).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return campaign_root / "aggregation" / "plans" / f"plan_build_{timestamp}"


def build_and_write_plan_for_run(
    *,
    run_ref: GenerationRunRef,
    output_root: Path,
    strategy: AggregationStrategy,
    rule_set: AggregationRuleSet,
    weight_mode: AggregationWeightMode,
    aggregate_zone_name_stem: str,
    system_node_name_pattern: str,
) -> dict[str, Any]:
    """Build and write one aggregation plan for one generation run."""
    manifest = load_json(run_ref.manifest_path)
    case_spec = manifest.get("case_spec", {})
    canonical_root = run_ref.run_root / "canonical"
    eio_tables_path = canonical_root / "eio_tables.json"

    missing_rows: list[dict[str, Any]] = []
    if not eio_tables_path.is_file():
        missing_rows.append(
            {
                "case_id": run_ref.case_id,
                "run_id": run_ref.run_id,
                "missing_file": str(eio_tables_path),
                "reason": "canonical eio_tables.json missing",
            }
        )
        eio_payload: dict[str, Any] = {}
    else:
        eio_payload = load_json(eio_tables_path)

    zone_rows = zone_information_rows(eio_payload=eio_payload)
    included_rows = [
        row for row in zone_rows if row.get("included_thermal_zone") == "true"
    ]
    excluded_rows = [
        row for row in zone_rows if row.get("included_thermal_zone") == "false"
    ]

    source_zones = tuple(str(row["zone_name"]) for row in included_rows)

    if strategy == AggregationStrategy.ALL_THERMAL_ZONES_TO_ONE:
        aggregate_zones = (
            AggregateZoneGroup(
                aggregate_zone_id=f"{aggregate_zone_name_stem}_1",
                source_zones=source_zones,
            ),
        )
    elif strategy == AggregationStrategy.IDENTITY:
        aggregate_zones = tuple(
            AggregateZoneGroup(
                aggregate_zone_id=f"{aggregate_zone_name_stem}_{index}",
                source_zones=(zone_name,),
            )
            for index, zone_name in enumerate(source_zones, start=1)
        )
    else:
        raise NotImplementedError(
            "custom_groups will be implemented after validating the automatic "
            "all_thermal_zones_to_one and identity plan paths."
        )

    aggregation_id = build_aggregation_id(
        strategy=strategy,
        rule_set=rule_set,
        weight_mode=weight_mode,
    )

    plan = AggregationPlan(
        schema_version="0.1.0",
        aggregation_id=aggregation_id,
        strategy=strategy,
        rule_set=rule_set,
        weight_mode=weight_mode,
        aggregate_zone_name_stem=aggregate_zone_name_stem,
        system_node_name_pattern=system_node_name_pattern,
        source_case_id=run_ref.case_id,
        source_generation_run_id=run_ref.run_id,
        campaign_id=str(manifest.get("campaign_id", "")),
        building_type=str(case_spec.get("building_type", "")),
        weather_location=str(case_spec.get("weather_location", "")),
        climate_zone=str(case_spec.get("climate_zone", "")),
        thermal_zone_filter={
            "source_table": "Zone Information",
            "include_when": {
                "Part of Total Building Area": "Yes",
            },
        },
        aggregate_zones=aggregate_zones,
    )

    case_plan_root = output_root / run_ref.case_id / aggregation_id
    case_plan_root.mkdir(parents=True, exist_ok=True)

    write_json(case_plan_root / "aggregation_plan.json", plan.to_dict())

    zone_mapping_rows = build_zone_mapping_rows(
        run_ref=run_ref,
        manifest=manifest,
        plan=plan,
        included_rows=included_rows,
    )
    included_output_rows = [
        {**base_case_fields(run_ref=run_ref, manifest=manifest), **row}
        for row in included_rows
    ]
    excluded_output_rows = [
        {**base_case_fields(run_ref=run_ref, manifest=manifest), **row}
        for row in excluded_rows
    ]

    write_csv(case_plan_root / "zone_mapping.csv", zone_mapping_rows)
    write_csv(case_plan_root / "included_thermal_zones.csv", included_output_rows)
    write_csv(case_plan_root / "excluded_zones.csv", excluded_output_rows)
    write_csv(case_plan_root / "missing_plan_inputs.csv", missing_rows)

    plan_row = {
        **base_case_fields(run_ref=run_ref, manifest=manifest),
        "aggregation_id": aggregation_id,
        "strategy": strategy.value,
        "rule_set": rule_set.value,
        "weight_mode": weight_mode.value,
        "aggregate_zone_count": len(plan.aggregate_zones),
        "source_zone_count": len(source_zones),
        "plan_path": str(case_plan_root / "aggregation_plan.json"),
        "case_plan_root": str(case_plan_root),
    }

    return {
        "plan": plan,
        "plan_row": plan_row,
        "zone_mapping_rows": zone_mapping_rows,
        "included_thermal_zone_rows": included_output_rows,
        "excluded_zone_rows": excluded_output_rows,
        "missing_rows": missing_rows,
    }


def build_aggregation_id(
    *,
    strategy: AggregationStrategy,
    rule_set: AggregationRuleSet,
    weight_mode: AggregationWeightMode,
) -> str:
    """Build a readable deterministic aggregation ID."""
    if strategy == AggregationStrategy.ALL_THERMAL_ZONES_TO_ONE:
        strategy_label = "all_thermal_zones_to_one"
    else:
        strategy_label = strategy.value

    return f"{strategy_label}_{rule_set.value}_{weight_mode.value}_v1"


def build_zone_mapping_rows(
    *,
    run_ref: GenerationRunRef,
    manifest: dict[str, Any],
    plan: AggregationPlan,
    included_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build source-zone to aggregate-zone mapping rows."""
    base = base_case_fields(run_ref=run_ref, manifest=manifest)
    zone_metadata_by_name = {
        str(row.get("zone_name", "")): row for row in included_rows
    }

    rows: list[dict[str, Any]] = []
    for group in plan.aggregate_zones:
        for source_zone in group.source_zones:
            zone_meta = zone_metadata_by_name.get(source_zone, {})
            rows.append(
                {
                    **base,
                    "aggregation_id": plan.aggregation_id,
                    "strategy": plan.strategy.value,
                    "rule_set": plan.rule_set.value,
                    "weight_mode": plan.weight_mode.value,
                    "aggregate_zone_id": group.aggregate_zone_id,
                    "source_zone": source_zone,
                    "floor_area_m2": zone_meta.get("floor_area_m2", ""),
                    "volume_m3": zone_meta.get("volume_m3", ""),
                    "part_of_total_building_area": zone_meta.get(
                        "part_of_total_building_area", ""
                    ),
                }
            )

    return rows


def base_case_fields(
    *,
    run_ref: GenerationRunRef,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return common case/run fields."""
    case_spec = manifest.get("case_spec", {})
    return {
        "case_id": run_ref.case_id,
        "run_id": run_ref.run_id,
        "campaign_id": manifest.get("campaign_id", ""),
        "building_type": case_spec.get("building_type", ""),
        "weather_location": case_spec.get("weather_location", ""),
        "climate_zone": case_spec.get("climate_zone", ""),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))