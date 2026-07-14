# -*- coding: utf-8 -*-
"""Probe the ScaleBridge aggregation canonical variable loader.

This is a development/validation wrapper. Loader logic lives in:

    src/scalebridge/data/aggregation/loaders.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.discovery import (
    DEFAULT_CAMPAIGN_ID,
    discover_generation_runs,
    resolve_campaign_root,
    resolve_repo_root,
)
from scalebridge.data.aggregation.loaders import CanonicalVariableLoader


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
        raise SystemExit(
            "No successful generation runs found. "
            f"cases_root={cases_root}, case_id={args.case_id}, "
            f"missing_rows={missing_rows}"
        )

    if len(run_refs) > 1 and not args.allow_multiple:
        print(
            f"Found {len(run_refs)} generation runs. "
            "Using the first one. Pass --allow-multiple to probe all.",
        )
        run_refs = run_refs[:1]

    output_dir = resolve_output_dir(
        campaign_root=campaign_root,
        output_dir=args.output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SCALEBRIDGE AGGREGATION VARIABLE LOADER PROBE")
    print("=" * 100)
    print(f"repo_root: {repo_root}")
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"cases_root: {cases_root}")
    print(f"output_dir: {output_dir}")
    print(f"variable_id: {args.variable_id}")
    print(f"variable_name: {args.variable_name}")
    print()

    probe_rows: list[dict[str, Any]] = []
    available_rows: list[dict[str, Any]] = []

    for run_ref in run_refs:
        loader = CanonicalVariableLoader(run_ref=run_ref)

        for record in loader.records:
            available_rows.append(
                {
                    "case_id": run_ref.case_id,
                    "run_id": run_ref.run_id,
                    **record.to_dict(),
                }
            )

        if args.list_only:
            continue

        frame = load_requested_variable(
            loader=loader,
            variable_id=args.variable_id,
            variable_name=args.variable_name,
        )

        record = (
            loader.get_record_by_id(args.variable_id)
            if args.variable_id
            else loader.get_record_by_name(args.variable_name)
        )

        row = summarize_loaded_frame(
            case_id=run_ref.case_id,
            run_id=run_ref.run_id,
            variable_id=record.variable_id,
            variable_name=record.variable_name,
            frame=frame,
        )
        probe_rows.append(row)

        print(f"case_id: {run_ref.case_id}")
        print(f"run_id: {run_ref.run_id}")
        print(f"variable_id: {record.variable_id}")
        print(f"variable_name: {record.variable_name}")
        print(f"rows: {row['row_count']}")
        print(f"columns: {row['column_count']}")
        print(f"unique key_value count: {row['unique_key_value_count']}")
        print(f"first key_values: {row['first_key_values']}")
        print(f"first timestamp_raw: {row['first_timestamp_raw']}")
        print(f"last timestamp_raw: {row['last_timestamp_raw']}")
        print()

        if args.write_preview:
            preview_path = (
                output_dir
                / f"{run_ref.case_id}_{record.variable_id}_head{args.preview_rows}.csv"
            )
            frame.head(args.preview_rows).to_csv(preview_path, index=False)
            print(f"Wrote preview: {preview_path}")

    write_csv(output_dir / "available_variables_from_loader.csv", available_rows)
    write_csv(output_dir / "variable_loader_probe_summary.csv", probe_rows)

    summary = {
        "schema_version": "0.1.0",
        "created_at_local": datetime.now().isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "case_filter": args.case_id,
        "output_dir": str(output_dir),
        "run_count": len(run_refs),
        "available_variable_rows": len(available_rows),
        "probe_rows": len(probe_rows),
        "variable_id": args.variable_id,
        "variable_name": args.variable_name,
        "list_only": args.list_only,
    }
    write_json(output_dir / "variable_loader_probe_summary.json", summary)

    print("=" * 100)
    print("LOADER PROBE COMPLETE")
    print("=" * 100)
    print(f"available_variable_rows: {len(available_rows)}")
    print(f"probe_rows: {len(probe_rows)}")
    print(f"Wrote outputs to: {output_dir}")

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
        "--output-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "<campaign_root>/aggregation/probes/variable_loader_<timestamp>."
        ),
    )
    parser.add_argument(
        "--variable-id",
        default=None,
        help="Variable ID to load, e.g. timestep_schedule_value.",
    )
    parser.add_argument(
        "--variable-name",
        default=None,
        help='Variable name to load, e.g. "Schedule Value".',
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available variables; do not load a Parquet file.",
    )
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Probe all discovered successful generation runs.",
    )
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Write a CSV preview of the loaded variable dataframe.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Rows to write when --write-preview is enabled.",
    )

    args = parser.parse_args(argv)

    if not args.list_only and not args.variable_id and not args.variable_name:
        parser.error("Provide --variable-id, --variable-name, or --list-only.")

    if args.variable_id and args.variable_name:
        parser.error("Use either --variable-id or --variable-name, not both.")

    return args


def resolve_output_dir(
    *,
    campaign_root: Path,
    output_dir: str | None,
) -> Path:
    """Resolve output directory."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return campaign_root / "aggregation" / "probes" / f"variable_loader_{timestamp}"


def load_requested_variable(
    *,
    loader: CanonicalVariableLoader,
    variable_id: str | None,
    variable_name: str | None,
):
    """Load requested variable by ID or name."""
    if variable_id:
        return loader.load_variable_long_by_id(variable_id)
    if variable_name:
        return loader.load_variable_long_by_name(variable_name)

    raise ValueError("variable_id or variable_name is required")


def summarize_loaded_frame(
    *,
    case_id: str,
    run_id: str,
    variable_id: str,
    variable_name: str,
    frame,
) -> dict[str, Any]:
    """Summarize one loaded canonical variable dataframe."""
    key_values = sorted(str(item) for item in frame["key_value"].dropna().unique())

    if "timestamp_raw" in frame.columns and len(frame) > 0:
        first_timestamp = str(frame["timestamp_raw"].iloc[0])
        last_timestamp = str(frame["timestamp_raw"].iloc[-1])
    else:
        first_timestamp = ""
        last_timestamp = ""

    return {
        "case_id": case_id,
        "run_id": run_id,
        "variable_id": variable_id,
        "variable_name": variable_name,
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "columns": " | ".join(str(item) for item in frame.columns),
        "unique_key_value_count": len(key_values),
        "first_key_values": " | ".join(key_values[:25]),
        "first_timestamp_raw": first_timestamp,
        "last_timestamp_raw": last_timestamp,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV."""
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

    import csv

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))