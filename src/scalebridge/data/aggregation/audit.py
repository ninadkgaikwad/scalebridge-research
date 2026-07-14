# -*- coding: utf-8 -*-
"""Aggregation input audit for ScaleBridge EnergyPlus generation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.discovery import (
    DEFAULT_CAMPAIGN_ID,
    discover_generation_runs,
    load_json,
    load_rdd_variable_intersection,
    resolve_campaign_root,
    resolve_repo_root,
)
from scalebridge.data.aggregation.eio import (
    schedule_equipment_mapping_rows,
    zone_information_rows,
)
from scalebridge.data.aggregation.models import GenerationRunRef, SUCCESS_STATUSES


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for aggregation input audit."""
    args = parse_args(argv)

    repo_root = resolve_repo_root()
    campaign_root = resolve_campaign_root(
        repo_root=repo_root,
        campaign_id=args.campaign_id,
        campaign_root=args.campaign_root,
        generated_data_root=args.generated_data_root,
    )

    generation_root = campaign_root / "generation"
    cases_root = generation_root / "cases"

    if not cases_root.is_dir():
        raise SystemExit(f"Generation cases folder does not exist: {cases_root}")

    output_dir = resolve_output_dir(
        campaign_root=campaign_root,
        output_dir=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SCALEBRIDGE AGGREGATION INPUT AUDIT")
    print("=" * 100)
    print(f"repo_root: {repo_root}")
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"cases_root: {cases_root}")
    print(f"output_dir: {output_dir}")
    print(f"inspect_parquet: {args.inspect_parquet}")
    print()

    run_refs, missing_rows = discover_generation_runs(
        cases_root=cases_root,
        case_id=args.case_id,
        include_failed=args.include_failed,
    )

    all_generation_rows: list[dict[str, Any]] = []
    all_available_variable_rows: list[dict[str, Any]] = []
    all_variable_key_count_rows: list[dict[str, Any]] = []
    all_zone_information_rows: list[dict[str, Any]] = []
    all_included_zone_rows: list[dict[str, Any]] = []
    all_excluded_zone_rows: list[dict[str, Any]] = []
    all_schedule_mapping_rows: list[dict[str, Any]] = []
    all_rdd_variable_rows: list[dict[str, Any]] = []

    for index, run_ref in enumerate(run_refs, start=1):
        print(f"[{index}/{len(run_refs)}] Auditing {run_ref.case_id} / {run_ref.run_id}")

        run_audit = audit_one_generation_run(
            run_ref=run_ref,
            inspect_parquet=args.inspect_parquet,
            max_parquet_files=args.max_parquet_files,
        )

        all_generation_rows.append(run_audit["generation_run"])
        all_available_variable_rows.extend(run_audit["available_variables"])
        all_variable_key_count_rows.extend(run_audit["variable_key_counts"])
        all_zone_information_rows.extend(run_audit["zone_information"])
        all_included_zone_rows.extend(run_audit["included_thermal_zones"])
        all_excluded_zone_rows.extend(run_audit["excluded_zones"])
        all_schedule_mapping_rows.extend(run_audit["schedule_equipment_mapping"])
        all_rdd_variable_rows.extend(run_audit["rdd_variable_intersection"])
        missing_rows.extend(run_audit["missing_expected_files"])

    write_csv(output_dir / "generation_runs.csv", all_generation_rows)
    write_csv(output_dir / "available_variables.csv", all_available_variable_rows)
    write_csv(output_dir / "variable_key_counts.csv", all_variable_key_count_rows)
    write_csv(output_dir / "zone_information.csv", all_zone_information_rows)
    write_csv(output_dir / "included_thermal_zones.csv", all_included_zone_rows)
    write_csv(output_dir / "excluded_zones.csv", all_excluded_zone_rows)
    write_csv(output_dir / "schedule_equipment_mapping.csv", all_schedule_mapping_rows)
    write_csv(output_dir / "rdd_variable_intersection.csv", all_rdd_variable_rows)
    write_csv(output_dir / "missing_expected_files.csv", missing_rows)

    summary = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "generation_cases_root": str(cases_root),
        "output_dir": str(output_dir),
        "inspect_parquet": args.inspect_parquet,
        "case_filter": args.case_id,
        "generation_run_count": len(run_refs),
        "available_variable_count": len(all_available_variable_rows),
        "variable_key_count_rows": len(all_variable_key_count_rows),
        "zone_information_rows": len(all_zone_information_rows),
        "included_thermal_zone_rows": len(all_included_zone_rows),
        "excluded_zone_rows": len(all_excluded_zone_rows),
        "schedule_equipment_mapping_rows": len(all_schedule_mapping_rows),
        "rdd_variable_intersection_rows": len(all_rdd_variable_rows),
        "missing_expected_file_rows": len(missing_rows),
        "success_statuses": sorted(SUCCESS_STATUSES),
        "thermal_zone_rule": {
            "source_table": "Zone Information",
            "include_when": {
                "Part of Total Building Area": "Yes",
            },
        },
        "optional_artifacts": {
            "rdd_variable_intersection": (
                "generation/cases/<case_id>/rdd_probe/"
                "rdd_variable_intersection.json"
            )
        },
    }
    write_json(output_dir / "aggregation_input_audit.json", summary)

    print()
    print("=" * 100)
    print("AUDIT SUMMARY")
    print("=" * 100)
    print(f"generation_run_count: {summary['generation_run_count']}")
    print(f"available_variable_count: {summary['available_variable_count']}")
    print(f"included_thermal_zone_rows: {summary['included_thermal_zone_rows']}")
    print(f"excluded_zone_rows: {summary['excluded_zone_rows']}")
    print(f"schedule_equipment_mapping_rows: {summary['schedule_equipment_mapping_rows']}")
    print(f"rdd_variable_intersection_rows: {summary['rdd_variable_intersection_rows']}")
    print(f"missing_expected_file_rows: {summary['missing_expected_file_rows']}")
    print()
    print(f"Wrote audit files to: {output_dir}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help=f"Campaign ID under Data/ScaleBridge/campaigns. Default: {DEFAULT_CAMPAIGN_ID}",
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help="Explicit campaign root. Overrides --generated-data-root and --campaign-id.",
    )
    parser.add_argument(
        "--generated-data-root",
        default=None,
        help=(
            "Explicit ScaleBridge generated data root. If omitted, uses "
            "<repo_root>/../../Data/ScaleBridge."
        ),
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional case_id filter.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output folder for audit files. Default: "
            "<campaign_root>/aggregation/audits/<timestamp>."
        ),
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed/latest non-success runs in the audit.",
    )
    parser.add_argument(
        "--inspect-parquet",
        action="store_true",
        help=(
            "Read Parquet files to compute actual unique key_value counts. "
            "This can be slow for full annual outputs."
        ),
    )
    parser.add_argument(
        "--max-parquet-files",
        type=int,
        default=None,
        help="Optional max number of parquet files to inspect per generation run.",
    )

    return parser.parse_args(argv)


def resolve_output_dir(
    *,
    campaign_root: Path,
    output_dir: str | None,
) -> Path:
    """Resolve audit output directory."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return campaign_root / "aggregation" / "audits" / f"input_audit_{timestamp}"


def audit_one_generation_run(
    *,
    run_ref: GenerationRunRef,
    inspect_parquet: bool,
    max_parquet_files: int | None,
) -> dict[str, Any]:
    """Audit one generation run root."""
    run_root = run_ref.run_root
    canonical_root = run_root / "canonical"

    manifest = load_json(run_ref.manifest_path)
    case_spec = manifest.get("case_spec", {})

    metadata_path = canonical_root / "metadata.json"
    variable_manifest_csv_path = canonical_root / "variable_manifest.csv"
    variable_manifest_json_path = canonical_root / "variable_manifest.json"
    eio_tables_path = canonical_root / "eio_tables.json"

    missing_rows: list[dict[str, Any]] = []
    for expected_path, reason in (
        (metadata_path, "canonical metadata missing"),
        (variable_manifest_csv_path, "variable manifest csv missing"),
        (variable_manifest_json_path, "variable manifest json missing"),
        (eio_tables_path, "eio tables json missing"),
    ):
        if not expected_path.is_file():
            missing_rows.append(
                {
                    "case_id": run_ref.case_id,
                    "run_id": run_ref.run_id,
                    "missing_file": str(expected_path),
                    "reason": reason,
                }
            )

    metadata = load_json(metadata_path) if metadata_path.is_file() else {}
    variable_rows = read_variable_manifest_rows(
        csv_path=variable_manifest_csv_path,
        json_path=variable_manifest_json_path,
    )
    eio_payload = load_json(eio_tables_path) if eio_tables_path.is_file() else {}
    rdd_intersection = load_rdd_variable_intersection(run_ref.case_root)

    generation_row = {
        **base_case_fields(run_ref=run_ref, manifest=manifest),
        "status": run_ref.status,
        "timestep_minutes": case_spec.get("timestep_minutes", ""),
        "generation_mode": metadata.get("generation_mode", ""),
        "parent_output_variable_count": metadata.get("parent_output_variable_count", ""),
        "selected_output_variable_count": metadata.get("selected_output_variable_count", ""),
        "produced_signal_count": metadata.get("produced_signal_count", ""),
        "canonical_variable_parquet_count": metadata.get("canonical_variable_parquet_count", ""),
        "rdd_probe_status": rdd_intersection.status,
        "rdd_requested_variable_count": rdd_intersection.requested_variable_count,
        "rdd_available_variable_count": rdd_intersection.rdd_available_variable_count,
        "rdd_unavailable_variable_count": rdd_intersection.rdd_unavailable_variable_count,
        "rdd_variable_intersection_path": (
            str(rdd_intersection.path) if rdd_intersection.path else ""
        ),
        "run_root": str(run_root),
        "manifest_path": str(run_ref.manifest_path),
    }

    available_variable_rows = build_available_variable_rows(
        run_ref=run_ref,
        manifest=manifest,
        metadata=metadata,
        variable_rows=variable_rows,
        rdd_intersection=rdd_intersection,
    )

    variable_key_count_rows = build_variable_key_count_rows(
        run_ref=run_ref,
        run_root=run_root,
        variable_rows=variable_rows,
        inspect_parquet=inspect_parquet,
        max_parquet_files=max_parquet_files,
    )

    zone_rows_raw = zone_information_rows(eio_payload=eio_payload)
    zone_rows = [
        {**base_case_fields(run_ref=run_ref, manifest=manifest), **row}
        for row in zone_rows_raw
    ]
    included_zone_rows = [
        row for row in zone_rows if row.get("included_thermal_zone") == "true"
    ]
    excluded_zone_rows = [
        row for row in zone_rows if row.get("included_thermal_zone") == "false"
    ]

    schedule_rows_raw = schedule_equipment_mapping_rows(
        eio_payload=eio_payload,
        included_zone_names={row["zone_name"] for row in included_zone_rows},
    )
    schedule_rows = [
        {**base_case_fields(run_ref=run_ref, manifest=manifest), **row}
        for row in schedule_rows_raw
    ]

    rdd_rows = build_rdd_variable_rows(
        run_ref=run_ref,
        manifest=manifest,
        rdd_intersection=rdd_intersection,
    )

    return {
        "generation_run": generation_row,
        "available_variables": available_variable_rows,
        "variable_key_counts": variable_key_count_rows,
        "zone_information": zone_rows,
        "included_thermal_zones": included_zone_rows,
        "excluded_zones": excluded_zone_rows,
        "schedule_equipment_mapping": schedule_rows,
        "rdd_variable_intersection": rdd_rows,
        "missing_expected_files": missing_rows,
    }


def read_variable_manifest_rows(
    *,
    csv_path: Path,
    json_path: Path,
) -> list[dict[str, Any]]:
    """Read variable manifest rows from CSV first, then JSON fallback."""
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    if json_path.is_file():
        payload = load_json(json_path)

        # Current variable-wise manifest usually stores a top-level artifacts list.
        artifacts = payload.get("artifacts", [])
        if isinstance(artifacts, list):
            return [dict(item) for item in artifacts if isinstance(item, dict)]

        # Defensive fallback if future metadata stores rows under variables.
        variables = payload.get("variables", [])
        if isinstance(variables, list):
            return [dict(item) for item in variables if isinstance(item, dict)]

    return []


def build_available_variable_rows(
    *,
    run_ref: GenerationRunRef,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    variable_rows: list[dict[str, Any]],
    rdd_intersection: Any,
) -> list[dict[str, Any]]:
    """Build available generated variable inventory rows."""
    case_spec = manifest.get("case_spec", {})
    base = base_case_fields(run_ref=run_ref, manifest=manifest)

    rows: list[dict[str, Any]] = []
    for row in variable_rows:
        variable_name = str(row.get("variable_name", ""))
        semantic_role = find_semantic_role(case_spec, variable_name)

        if rdd_intersection.status == "present":
            if variable_name in rdd_intersection.available_set:
                rdd_status = "available"
            elif variable_name in rdd_intersection.unavailable_set:
                rdd_status = "unavailable"
            else:
                rdd_status = "not_listed"
        else:
            rdd_status = "unknown_missing_rdd_probe"

        rows.append(
            {
                **base,
                "generation_mode": metadata.get("generation_mode", ""),
                "variable_id": row.get("variable_id", ""),
                "variable_name": variable_name,
                "semantic_role": semantic_role,
                "reporting_frequency": row.get("reporting_frequency", ""),
                "row_count": row.get("row_count", ""),
                "column_count": row.get("column_count", ""),
                "raw_csv_deleted": row.get("raw_csv_deleted", ""),
                "canonical_parquet_path": row.get("canonical_parquet_path", ""),
                "legacy_pickle_path": row.get("legacy_pickle_path", ""),
                "rdd_variable_status": rdd_status,
            }
        )

    return rows


def build_rdd_variable_rows(
    *,
    run_ref: GenerationRunRef,
    manifest: dict[str, Any],
    rdd_intersection: Any,
) -> list[dict[str, Any]]:
    """Build rows describing optional RDD variable availability."""
    base = base_case_fields(run_ref=run_ref, manifest=manifest)

    if rdd_intersection.status != "present":
        return [
            {
                **base,
                "rdd_probe_status": rdd_intersection.status,
                "variable_name": "",
                "rdd_variable_status": "",
                "rdd_variable_intersection_path": "",
            }
        ]

    rows: list[dict[str, Any]] = []

    for variable_name in rdd_intersection.available_variables:
        rows.append(
            {
                **base,
                "rdd_probe_status": "present",
                "variable_name": variable_name,
                "rdd_variable_status": "available",
                "rdd_variable_intersection_path": str(rdd_intersection.path),
            }
        )

    for variable_name in rdd_intersection.unavailable_variables:
        rows.append(
            {
                **base,
                "rdd_probe_status": "present",
                "variable_name": variable_name,
                "rdd_variable_status": "unavailable",
                "rdd_variable_intersection_path": str(rdd_intersection.path),
            }
        )

    return rows


def build_variable_key_count_rows(
    *,
    run_ref: GenerationRunRef,
    run_root: Path,
    variable_rows: list[dict[str, Any]],
    inspect_parquet: bool,
    max_parquet_files: int | None,
) -> list[dict[str, Any]]:
    """Build variable key-count rows using manifest and optional Parquet inspection."""
    rows: list[dict[str, Any]] = []
    base = {
        "case_id": run_ref.case_id,
        "run_id": run_ref.run_id,
    }

    selected_rows = variable_rows
    if max_parquet_files is not None:
        selected_rows = selected_rows[: max(0, max_parquet_files)]

    for row in selected_rows:
        variable_id = str(row.get("variable_id", ""))
        variable_name = str(row.get("variable_name", ""))
        manifest_column_count = row.get("column_count", "")
        manifest_row_count = row.get("row_count", "")
        parquet_path = resolve_variable_parquet_path(
            run_root=run_root,
            manifest_path_value=str(row.get("canonical_parquet_path", "")),
            variable_id=variable_id,
        )

        actual_key_count = ""
        first_key_values = ""
        parquet_status = "not_inspected"

        if inspect_parquet:
            actual_key_count, first_key_values, parquet_status = inspect_parquet_keys(
                parquet_path
            )

        rows.append(
            {
                **base,
                "variable_id": variable_id,
                "variable_name": variable_name,
                "manifest_row_count": manifest_row_count,
                "manifest_column_count": manifest_column_count,
                "actual_key_value_count": actual_key_count,
                "first_key_values": first_key_values,
                "parquet_status": parquet_status,
                "parquet_path": str(parquet_path),
            }
        )

    return rows


def resolve_variable_parquet_path(
    *,
    run_root: Path,
    manifest_path_value: str,
    variable_id: str,
) -> Path:
    """Resolve variable parquet path portably."""
    value = manifest_path_value.strip()

    if value:
        candidate = Path(value)
        if candidate.is_absolute() and candidate.is_file():
            return candidate

        name_candidate = run_root / "canonical" / "variables" / candidate.name
        if name_candidate.is_file():
            return name_candidate

    return run_root / "canonical" / "variables" / f"{variable_id}.parquet"


def inspect_parquet_keys(parquet_path: Path) -> tuple[str, str, str]:
    """Inspect key_value counts from a Parquet file."""
    if not parquet_path.is_file():
        return "", "", "missing"

    try:
        import pandas as pd
    except Exception as exc:
        return "", "", f"pandas_import_failed:{type(exc).__name__}"

    try:
        frame = pd.read_parquet(parquet_path, columns=["key_value"])
    except Exception as exc:
        return "", "", f"read_failed:{type(exc).__name__}"

    try:
        unique_values = sorted(str(item) for item in frame["key_value"].dropna().unique())
    except Exception as exc:
        return "", "", f"count_failed:{type(exc).__name__}"

    first_values = " | ".join(unique_values[:25])
    return str(len(unique_values)), first_values, "inspected"


def base_case_fields(
    *,
    run_ref: GenerationRunRef,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return common case/run fields for CSV rows."""
    case_spec = manifest.get("case_spec", {})
    return {
        "case_id": run_ref.case_id,
        "run_id": run_ref.run_id,
        "campaign_id": manifest.get("campaign_id", ""),
        "building_type": case_spec.get("building_type", ""),
        "weather_location": case_spec.get("weather_location", ""),
        "climate_zone": case_spec.get("climate_zone", ""),
    }


def find_semantic_role(case_spec: dict[str, Any], variable_name: str) -> str:
    """Find semantic role from embedded case_spec output variable requests."""
    for request in case_spec.get("output_variables", []):
        if str(request.get("variable_name", "")).casefold() == variable_name.casefold():
            return str(request.get("semantic_role", "") or "")
    return ""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, even when rows are empty."""
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