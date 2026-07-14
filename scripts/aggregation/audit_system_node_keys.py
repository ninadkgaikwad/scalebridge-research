# -*- coding: utf-8 -*-
"""Audit System Node Temperature and Mass Flow Rate key conventions.

This script scans canonical variable-wise parquet files for the P1 compact
campaign and summarizes the `key_value` naming conventions used by:

    - System Node Temperature
    - System Node Mass Flow Rate

It is intentionally read-only. It reads only lightweight columns from each
matching parquet file and writes CSV/JSON diagnostics that can be inspected
before updating the shared node-to-zone mapping architecture.

Typical use from repo root:

    python scripts\aggregation\audit_system_node_keys.py `
      --campaign-id p1_compact_4b4c_labpc_1w_v1

Optional single-case use:

    python scripts\aggregation\audit_system_node_keys.py `
      --campaign-id p1_compact_4b4c_labpc_1w_v1 `
      --case-id epcase_4abb05aa9c546958efafe7c4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "pyarrow is required for this audit script. "
        f"Original import error: {exc}"
    )


DEFAULT_CAMPAIGN_ID = "p1_compact_4b4c_labpc_1w_v1"

TARGET_VARIABLE_NAMES = (
    "System Node Temperature",
    "System Node Mass Flow Rate",
)

KEY_SUFFIX_CANDIDATES = (
    "DIRECT AIR INLET NODE NAME",
    "DIRECT AIR INLET NODE",
    "ZONE EQUIP INLET",
    "ERV SUP FAN OUTLET NODE",
    "RETURN AIR NODE NAME",
    "RETURN AIR NODE",
    "EXHAUST FAN NODE",
    "OUTLET NODE",
    "INLET NODE",
    "AIR NODE",
)


@dataclass(frozen=True)
class VariableRecord:
    """Reference to one canonical parquet variable file."""

    case_id: str
    run_id: str
    campaign_id: str
    building_type: str
    weather_location: str
    climate_zone: str
    variable_id: str
    variable_name: str
    canonical_parquet_path: Path


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

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else campaign_root
        / "aggregation"
        / "diagnostics"
        / f"system_node_key_audit_{timestamp_id()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("SCALEBRIDGE SYSTEM NODE KEY AUDIT")
    print("=" * 100)
    print(f"repo_root: {repo_root}")
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"cases_root: {cases_root}")
    print(f"output_root: {output_root}")
    print(f"target_variables: {list(TARGET_VARIABLE_NAMES)}")
    print()

    records, missing_rows = discover_target_variable_records(
        campaign_root=campaign_root,
        cases_root=cases_root,
        case_id=args.case_id,
    )

    if args.case_limit is not None:
        keep_cases = []
        seen_cases = set()
        for record in records:
            if record.case_id not in seen_cases:
                seen_cases.add(record.case_id)
                keep_cases.append(record.case_id)
            if len(keep_cases) >= max(0, args.case_limit):
                break
        keep_case_set = set(keep_cases)
        records = [record for record in records if record.case_id in keep_case_set]

    records = sorted(
        records,
        key=lambda r: (
            r.building_type,
            r.weather_location,
            r.case_id,
            r.variable_name,
        ),
    )

    print(f"matching_variable_file_count: {len(records)}")
    print(f"missing_target_rows: {len(missing_rows)}")
    print()

    key_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    case_summary_rows: list[dict[str, Any]] = []
    variable_summary_rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        print(
            f"[{index}/{len(records)}] "
            f"{record.case_id} | {record.building_type} | "
            f"{record.weather_location} | {record.variable_name}"
        )

        audit = audit_one_variable_file(record=record)

        key_rows.extend(audit["key_rows"])
        pattern_rows.extend(audit["pattern_rows"])
        case_summary_rows.append(audit["case_summary_row"])
        variable_summary_rows.append(audit["variable_summary_row"])

    building_pattern_rows = build_building_pattern_summary(pattern_rows)
    building_key_rows = build_building_key_summary(key_rows)

    write_csv(output_root / "system_node_key_values.csv", key_rows)
    write_csv(output_root / "system_node_key_patterns.csv", pattern_rows)
    write_csv(output_root / "system_node_case_summary.csv", case_summary_rows)
    write_csv(output_root / "system_node_variable_summary.csv", variable_summary_rows)
    write_csv(output_root / "system_node_building_pattern_summary.csv", building_pattern_rows)
    write_csv(output_root / "system_node_building_key_summary.csv", building_key_rows)
    write_csv(output_root / "system_node_missing_target_variables.csv", missing_rows)

    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "cases_root": str(cases_root),
        "output_root": str(output_root),
        "target_variable_names": list(TARGET_VARIABLE_NAMES),
        "key_suffix_candidates": list(KEY_SUFFIX_CANDIDATES),
        "matching_variable_file_count": len(records),
        "missing_target_row_count": len(missing_rows),
        "outputs": {
            "key_values": str(output_root / "system_node_key_values.csv"),
            "key_patterns": str(output_root / "system_node_key_patterns.csv"),
            "case_summary": str(output_root / "system_node_case_summary.csv"),
            "variable_summary": str(output_root / "system_node_variable_summary.csv"),
            "building_pattern_summary": str(output_root / "system_node_building_pattern_summary.csv"),
            "building_key_summary": str(output_root / "system_node_building_key_summary.csv"),
            "missing_target_variables": str(output_root / "system_node_missing_target_variables.csv"),
        },
    }
    write_json(output_root / "system_node_key_audit_manifest.json", manifest)

    print()
    print("=" * 100)
    print("SYSTEM NODE KEY AUDIT SUMMARY")
    print("=" * 100)
    print(f"matching_variable_file_count: {len(records)}")
    print(f"missing_target_row_count: {len(missing_rows)}")
    print(f"key_value_row_count: {len(key_rows)}")
    print(f"pattern_row_count: {len(pattern_rows)}")
    print(f"output_root: {output_root}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help=f"Campaign ID. Default: {DEFAULT_CAMPAIGN_ID}",
    )
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--output-root", default=None)

    return parser.parse_args(argv)


def resolve_repo_root() -> Path:
    """Resolve repository root from current working directory."""
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src" / "scalebridge").is_dir():
            return candidate
    return cwd


def resolve_campaign_root(
    *,
    repo_root: Path,
    campaign_id: str,
    campaign_root: str | None,
    generated_data_root: str | None,
) -> Path:
    """Resolve campaign root using explicit path, env var, or repo-relative default."""
    if campaign_root:
        return Path(campaign_root).expanduser().resolve()

    if generated_data_root:
        root = Path(generated_data_root).expanduser().resolve()
    else:
        import os

        env_value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT", "").strip()
        if env_value:
            root = Path(env_value).expanduser().resolve()
        else:
            root = (repo_root / ".." / ".." / "Data" / "ScaleBridge").resolve()

    return root / "campaigns" / campaign_id


def discover_target_variable_records(
    *,
    campaign_root: Path,
    cases_root: Path,
    case_id: str | None,
) -> tuple[list[VariableRecord], list[dict[str, Any]]]:
    """Discover temperature and mass-flow parquet files from latest generation runs."""
    records: list[VariableRecord] = []
    missing_rows: list[dict[str, Any]] = []

    case_dirs = sorted(
        path
        for path in cases_root.iterdir()
        if path.is_dir() and (case_id is None or path.name == case_id)
    )

    for case_dir in case_dirs:
        latest_path = case_dir / "latest_run.json"
        if not latest_path.is_file():
            missing_rows.append(
                {
                    "case_id": case_dir.name,
                    "missing_item": "latest_run.json",
                    "path": str(latest_path),
                }
            )
            continue

        latest = load_json(latest_path)
        run_id = str(latest.get("run_id", "")).strip()
        if not run_id:
            missing_rows.append(
                {
                    "case_id": case_dir.name,
                    "missing_item": "run_id_in_latest_run_json",
                    "path": str(latest_path),
                }
            )
            continue

        run_root = case_dir / "runs" / run_id
        manifest_path = run_root / "canonical" / "variable_manifest.json"
        if not manifest_path.is_file():
            missing_rows.append(
                {
                    "case_id": case_dir.name,
                    "run_id": run_id,
                    "missing_item": "canonical_variable_manifest_json",
                    "path": str(manifest_path),
                }
            )
            continue

        manifest = load_json(manifest_path)
        variable_rows = variable_rows_from_manifest(manifest)

        found_names = set()
        for row in variable_rows:
            variable_name = str(row.get("variable_name", "")).strip()
            if variable_name not in TARGET_VARIABLE_NAMES:
                continue

            found_names.add(variable_name)

            variable_id = str(row.get("variable_id", "")).strip()
            parquet_path_raw = (
                row.get("canonical_parquet_path")
                or row.get("parquet_path")
                or row.get("path")
                or ""
            )
            parquet_path = resolve_parquet_path(
                run_root=run_root,
                variable_id=variable_id,
                parquet_path_raw=str(parquet_path_raw),
            )

            if not parquet_path.is_file():
                missing_rows.append(
                    {
                        "case_id": case_dir.name,
                        "run_id": run_id,
                        "variable_name": variable_name,
                        "variable_id": variable_id,
                        "missing_item": "canonical_parquet_file",
                        "path": str(parquet_path),
                    }
                )
                continue

            records.append(
                VariableRecord(
                    case_id=case_dir.name,
                    run_id=run_id,
                    campaign_id=str(latest.get("campaign_id", "")),
                    building_type=str(latest.get("building_type", "")),
                    weather_location=str(
                        latest.get("weather_location")
                        or latest.get("weather_city")
                        or ""
                    ),
                    climate_zone=str(latest.get("climate_zone", "")),
                    variable_id=variable_id,
                    variable_name=variable_name,
                    canonical_parquet_path=parquet_path,
                )
            )

        for target_name in TARGET_VARIABLE_NAMES:
            if target_name not in found_names:
                missing_rows.append(
                    {
                        "case_id": case_dir.name,
                        "run_id": run_id,
                        "variable_name": target_name,
                        "missing_item": "target_variable_not_in_manifest",
                        "path": str(manifest_path),
                    }
                )

    return records, missing_rows


def variable_rows_from_manifest(manifest: Any) -> list[dict[str, Any]]:
    """Extract variable rows from known manifest shapes."""
    if isinstance(manifest, list):
        return [row for row in manifest if isinstance(row, dict)]

    if not isinstance(manifest, dict):
        return []

    for key in ("variables", "variable_rows", "generated_variables", "records", "items"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    dict_rows = [
        value
        for value in manifest.values()
        if isinstance(value, dict) and "variable_name" in value
    ]
    if dict_rows:
        return dict_rows

    return []


def resolve_parquet_path(
    *,
    run_root: Path,
    variable_id: str,
    parquet_path_raw: str,
) -> Path:
    """Resolve parquet path robustly across absolute/relative manifest values."""
    raw = parquet_path_raw.strip()
    if raw:
        raw_path = Path(raw)
        if raw_path.is_absolute():
            return raw_path
        candidate = run_root / raw_path
        if candidate.is_file():
            return candidate

    return run_root / "canonical" / "variables" / f"{variable_id}.parquet"


def audit_one_variable_file(*, record: VariableRecord) -> dict[str, Any]:
    """Audit one canonical system-node parquet file."""
    parquet_path = record.canonical_parquet_path

    metadata = pq.ParquetFile(parquet_path).metadata
    parquet_row_count = metadata.num_rows
    parquet_size_bytes = parquet_path.stat().st_size

    dataset = ds.dataset(parquet_path, format="parquet")
    key_table = dataset.to_table(columns=["key_value"])
    key_series = key_table.column("key_value").to_pandas()
    key_counts = key_series.value_counts(dropna=False).to_dict()
    unique_keys = sorted(str(key) for key in key_counts.keys())

    suffix_counter = Counter()
    token_counter = Counter()
    key_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []

    for key_value in unique_keys:
        normalized_key = normalize_identifier(key_value)
        matched_suffix = classify_key_suffix(normalized_key)
        inferred_zone_prefix = infer_zone_prefix(normalized_key, matched_suffix)

        suffix_counter[matched_suffix] += 1

        for token in normalized_key.split():
            token_counter[token] += 1

        key_rows.append(
            {
                **record_base_row(record),
                "parquet_path": str(parquet_path),
                "parquet_size_bytes": parquet_size_bytes,
                "parquet_row_count": parquet_row_count,
                "key_value": key_value,
                "normalized_key_value": normalized_key,
                "matched_suffix_pattern": matched_suffix,
                "inferred_zone_prefix": inferred_zone_prefix,
                "key_row_count": int(key_counts.get(key_value, 0)),
            }
        )

    for suffix, count in sorted(suffix_counter.items(), key=lambda item: (-item[1], item[0])):
        pattern_rows.append(
            {
                **record_base_row(record),
                "matched_suffix_pattern": suffix,
                "unique_key_count": count,
                "fraction_unique_keys": round(count / max(1, len(unique_keys)), 6),
            }
        )

    case_summary_row = {
        **record_base_row(record),
        "parquet_path": str(parquet_path),
        "parquet_size_bytes": parquet_size_bytes,
        "parquet_row_count": parquet_row_count,
        "unique_key_count": len(unique_keys),
        "matched_suffix_patterns": " | ".join(
            f"{suffix}:{count}"
            for suffix, count in sorted(
                suffix_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }

    variable_summary_row = {
        **record_base_row(record),
        "parquet_size_mb": round(parquet_size_bytes / 1024 / 1024, 3),
        "parquet_row_count": parquet_row_count,
        "unique_key_count": len(unique_keys),
        "top_tokens": " | ".join(
            f"{token}:{count}" for token, count in token_counter.most_common(30)
        ),
    }

    return {
        "key_rows": key_rows,
        "pattern_rows": pattern_rows,
        "case_summary_row": case_summary_row,
        "variable_summary_row": variable_summary_row,
    }


def record_base_row(record: VariableRecord) -> dict[str, Any]:
    """Common metadata row."""
    return {
        "campaign_id": record.campaign_id,
        "case_id": record.case_id,
        "run_id": record.run_id,
        "building_type": record.building_type,
        "weather_location": record.weather_location,
        "climate_zone": record.climate_zone,
        "variable_id": record.variable_id,
        "variable_name": record.variable_name,
    }


def normalize_identifier(value: str) -> str:
    """Normalize EnergyPlus key identifiers for matching."""
    text = str(value or "").upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_key_suffix(normalized_key: str) -> str:
    """Classify a key by known suffix patterns."""
    for suffix in KEY_SUFFIX_CANDIDATES:
        normalized_suffix = normalize_identifier(suffix)
        if normalized_key.endswith(normalized_suffix):
            return normalized_suffix
    return "UNCLASSIFIED"


def infer_zone_prefix(normalized_key: str, matched_suffix: str) -> str:
    """Infer the zone prefix before the matched suffix."""
    if not normalized_key:
        return ""

    if matched_suffix and matched_suffix != "UNCLASSIFIED":
        prefix = normalized_key[: -len(matched_suffix)].strip()
        return prefix

    return ""


def build_building_pattern_summary(
    pattern_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize suffix patterns by building and variable."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in pattern_rows:
        key = (
            str(row.get("building_type", "")),
            str(row.get("variable_name", "")),
            str(row.get("matched_suffix_pattern", "")),
        )
        if key not in grouped:
            grouped[key] = {
                "building_type": key[0],
                "variable_name": key[1],
                "matched_suffix_pattern": key[2],
                "weather_locations": set(),
                "climate_zones": set(),
                "case_ids": set(),
                "unique_key_count_sum": 0,
            }

        out = grouped[key]
        out["weather_locations"].add(str(row.get("weather_location", "")))
        out["climate_zones"].add(str(row.get("climate_zone", "")))
        out["case_ids"].add(str(row.get("case_id", "")))
        out["unique_key_count_sum"] += int(row.get("unique_key_count", 0) or 0)

    output = []
    for out in grouped.values():
        output.append(
            {
                "building_type": out["building_type"],
                "variable_name": out["variable_name"],
                "matched_suffix_pattern": out["matched_suffix_pattern"],
                "case_count": len(out["case_ids"]),
                "weather_locations": " | ".join(sorted(out["weather_locations"])),
                "climate_zones": " | ".join(sorted(out["climate_zones"])),
                "unique_key_count_sum": out["unique_key_count_sum"],
            }
        )

    return sorted(
        output,
        key=lambda row: (
            row["building_type"],
            row["variable_name"],
            -int(row["unique_key_count_sum"]),
            row["matched_suffix_pattern"],
        ),
    )


def build_building_key_summary(
    key_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize unique key names by building and variable."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in key_rows:
        key = (
            str(row.get("building_type", "")),
            str(row.get("variable_name", "")),
            str(row.get("key_value", "")),
        )
        if key not in grouped:
            grouped[key] = {
                "building_type": key[0],
                "variable_name": key[1],
                "key_value": key[2],
                "normalized_key_value": row.get("normalized_key_value", ""),
                "matched_suffix_pattern": row.get("matched_suffix_pattern", ""),
                "inferred_zone_prefix": row.get("inferred_zone_prefix", ""),
                "weather_locations": set(),
                "case_ids": set(),
                "key_row_count_sum": 0,
            }

        out = grouped[key]
        out["weather_locations"].add(str(row.get("weather_location", "")))
        out["case_ids"].add(str(row.get("case_id", "")))
        out["key_row_count_sum"] += int(row.get("key_row_count", 0) or 0)

    output = []
    for out in grouped.values():
        output.append(
            {
                "building_type": out["building_type"],
                "variable_name": out["variable_name"],
                "key_value": out["key_value"],
                "normalized_key_value": out["normalized_key_value"],
                "matched_suffix_pattern": out["matched_suffix_pattern"],
                "inferred_zone_prefix": out["inferred_zone_prefix"],
                "case_count": len(out["case_ids"]),
                "weather_locations": " | ".join(sorted(out["weather_locations"])),
                "key_row_count_sum": out["key_row_count_sum"],
            }
        )

    return sorted(
        output,
        key=lambda row: (
            row["building_type"],
            row["variable_name"],
            row["matched_suffix_pattern"],
            row["key_value"],
        ),
    )


def load_json(path: Path) -> Any:
    """Load JSON."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV from dictionaries."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fieldnames:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def timestamp_id() -> str:
    """Timestamp suitable for file/folder names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
