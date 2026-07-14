# -*- coding: utf-8 -*-
"""Audit System Node Temperature and Mass Flow Rate key conventions.

Version 2 is more robust than the first script:
  - It can discover variables from variable_manifest.csv first.
  - It falls back to recursive JSON manifest scanning.
  - It falls back again to canonical/variables/*.parquet filename/name scanning.
  - It accepts normalized variable names, e.g. timestep_system_node_temperature.

Typical use from repo root:

    python scripts\aggregation\audit_system_node_keys.py `
      --campaign-id p1_compact_4b4c_labpc_1w_v1

The script is read-only and writes diagnostics under:
    <campaign_root>/aggregation/diagnostics/system_node_key_audit_<timestamp>
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

TARGET_VARIABLE_SLUGS = {
    "system_node_temperature": "System Node Temperature",
    "timestep_system_node_temperature": "System Node Temperature",
    "system_node_mass_flow_rate": "System Node Mass Flow Rate",
    "timestep_system_node_mass_flow_rate": "System Node Mass Flow Rate",
}

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
    discovery_method: str


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

    records, missing_rows, discovery_rows = discover_target_variable_records(
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
            f"{record.weather_location} | {record.variable_name} | "
            f"{record.discovery_method}"
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
    write_csv(output_root / "system_node_discovery_debug.csv", discovery_rows)

    manifest = {
        "schema_version": "0.2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "cases_root": str(cases_root),
        "output_root": str(output_root),
        "target_variable_names": list(TARGET_VARIABLE_NAMES),
        "target_variable_slugs": TARGET_VARIABLE_SLUGS,
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
            "discovery_debug": str(output_root / "system_node_discovery_debug.csv"),
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

    if len(records) == 0:
        print()
        print("No records were discovered. Inspect:")
        print(f"  {output_root / 'system_node_discovery_debug.csv'}")
        print(f"  {output_root / 'system_node_missing_target_variables.csv'}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
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
    cases_root: Path,
    case_id: str | None,
) -> tuple[list[VariableRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover target parquet files from CSV, JSON, or filename fallbacks."""
    records: list[VariableRecord] = []
    missing_rows: list[dict[str, Any]] = []
    discovery_rows: list[dict[str, Any]] = []

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
        canonical_root = run_root / "canonical"
        variables_root = canonical_root / "variables"

        base = {
            "case_id": case_dir.name,
            "run_id": run_id,
            "campaign_id": str(latest.get("campaign_id", "")),
            "building_type": str(latest.get("building_type", "")),
            "weather_location": str(
                latest.get("weather_location")
                or latest.get("weather_city")
                or ""
            ),
            "climate_zone": str(latest.get("climate_zone", "")),
        }

        case_records: list[VariableRecord] = []
        candidate_rows: list[dict[str, Any]] = []

        csv_path = canonical_root / "variable_manifest.csv"
        if csv_path.is_file():
            csv_rows = load_csv_dicts(csv_path)
            candidate_rows.extend(
                {
                    **row,
                    "_discovery_source": "variable_manifest.csv",
                    "_manifest_path": str(csv_path),
                }
                for row in csv_rows
            )
            discovery_rows.append(
                {
                    **base,
                    "discovery_source": "variable_manifest.csv",
                    "path": str(csv_path),
                    "candidate_row_count": len(csv_rows),
                }
            )

        json_path = canonical_root / "variable_manifest.json"
        if json_path.is_file():
            manifest = load_json(json_path)
            json_rows = list(recursive_dict_rows(manifest))
            candidate_rows.extend(
                {
                    **row,
                    "_discovery_source": "variable_manifest.json_recursive",
                    "_manifest_path": str(json_path),
                }
                for row in json_rows
            )
            discovery_rows.append(
                {
                    **base,
                    "discovery_source": "variable_manifest.json_recursive",
                    "path": str(json_path),
                    "candidate_row_count": len(json_rows),
                }
            )

        for row in candidate_rows:
            match_name = match_target_variable_row(row)
            if match_name is None:
                continue

            record = variable_record_from_row(
                base=base,
                run_root=run_root,
                variables_root=variables_root,
                row=row,
                match_name=match_name,
            )
            if record is not None and record.canonical_parquet_path.is_file():
                case_records.append(record)

        # Fallback: direct filename scan.
        if variables_root.is_dir():
            for parquet_path in sorted(variables_root.glob("*.parquet")):
                slug = normalize_slug(parquet_path.stem)
                match_name = TARGET_VARIABLE_SLUGS.get(slug)
                if match_name is None:
                    continue

                case_records.append(
                    VariableRecord(
                        **base,
                        variable_id=parquet_path.stem,
                        variable_name=match_name,
                        canonical_parquet_path=parquet_path,
                        discovery_method="filename_fallback",
                    )
                )

            discovery_rows.append(
                {
                    **base,
                    "discovery_source": "filename_fallback",
                    "path": str(variables_root),
                    "candidate_row_count": len(list(variables_root.glob("*.parquet"))) if variables_root.is_dir() else 0,
                }
            )

        # Deduplicate by variable_name + path.
        dedup: dict[tuple[str, str], VariableRecord] = {}
        for record in case_records:
            dedup[(record.variable_name, str(record.canonical_parquet_path))] = record
        case_records = list(dedup.values())
        records.extend(case_records)

        found_names = {record.variable_name for record in case_records}
        for target_name in TARGET_VARIABLE_NAMES:
            if target_name not in found_names:
                missing_rows.append(
                    {
                        **base,
                        "variable_name": target_name,
                        "missing_item": "target_variable_not_discovered",
                        "canonical_root": str(canonical_root),
                        "variables_root": str(variables_root),
                    }
                )

    return records, missing_rows, discovery_rows


def variable_record_from_row(
    *,
    base: dict[str, Any],
    run_root: Path,
    variables_root: Path,
    row: dict[str, Any],
    match_name: str,
) -> VariableRecord | None:
    """Build a VariableRecord from one candidate manifest row."""
    variable_id = first_nonempty(
        row,
        (
            "variable_id",
            "id",
            "canonical_variable_id",
            "output_variable_id",
            "file_stem",
        ),
    )
    variable_id = str(variable_id or "").strip()

    parquet_raw = first_nonempty(
        row,
        (
            "canonical_parquet_path",
            "parquet_path",
            "path",
            "file_path",
            "output_path",
            "canonical_path",
        ),
    )
    parquet_path = resolve_parquet_path(
        run_root=run_root,
        variables_root=variables_root,
        variable_id=variable_id,
        parquet_path_raw=str(parquet_raw or ""),
        match_name=match_name,
    )

    if not variable_id:
        variable_id = parquet_path.stem if parquet_path else normalize_slug(match_name)

    return VariableRecord(
        **base,
        variable_id=variable_id,
        variable_name=match_name,
        canonical_parquet_path=parquet_path,
        discovery_method=str(row.get("_discovery_source", "manifest_row")),
    )


def match_target_variable_row(row: dict[str, Any]) -> str | None:
    """Return canonical target variable name if a row appears to describe it."""
    candidate_values = []
    for key in (
        "variable_name",
        "name",
        "requested_variable_name",
        "output_variable_name",
        "energyplus_variable_name",
        "semantic_name",
        "canonical_name",
        "variable_id",
        "id",
        "canonical_variable_id",
        "path",
        "parquet_path",
        "canonical_parquet_path",
        "file_path",
    ):
        value = row.get(key)
        if value is not None:
            candidate_values.append(str(value))

    for value in candidate_values:
        normalized_name = normalize_text_name(value)
        for target in TARGET_VARIABLE_NAMES:
            if normalized_name == normalize_text_name(target):
                return target

        slug = normalize_slug(Path(value).stem if value.lower().endswith(".parquet") else value)
        if slug in TARGET_VARIABLE_SLUGS:
            return TARGET_VARIABLE_SLUGS[slug]

        # Useful for absolute paths that include timestep_system_node_temperature.parquet.
        for known_slug, target in TARGET_VARIABLE_SLUGS.items():
            if known_slug in normalize_slug(value):
                return target

    return None


def resolve_parquet_path(
    *,
    run_root: Path,
    variables_root: Path,
    variable_id: str,
    parquet_path_raw: str,
    match_name: str,
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
        candidate = variables_root / raw_path
        if candidate.is_file():
            return candidate

    if variable_id:
        candidate = variables_root / f"{variable_id}.parquet"
        if candidate.is_file():
            return candidate

    # Filename fallback candidates.
    for slug, target in TARGET_VARIABLE_SLUGS.items():
        if target == match_name:
            candidate = variables_root / f"{slug}.parquet"
            if candidate.is_file():
                return candidate

    # Last-resort nonexisting path for diagnostics.
    return variables_root / f"{normalize_slug(match_name)}.parquet"


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
        "discovery_method": record.discovery_method,
    }


def normalize_identifier(value: str) -> str:
    """Normalize EnergyPlus key identifiers for matching."""
    text = str(value or "").upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text_name(value: str) -> str:
    """Normalize a human-readable variable name."""
    return normalize_identifier(value)


def normalize_slug(value: str) -> str:
    """Normalize a value into a variable/file slug."""
    text = str(value or "").strip()
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"\.parquet$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
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
        return normalized_key[: -len(matched_suffix)].strip()

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


def recursive_dict_rows(obj: Any) -> list[dict[str, Any]]:
    """Return every dictionary contained in a JSON-like object."""
    rows: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            rows.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(obj)
    return rows


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value for possible keys."""
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def load_json(path: Path) -> Any:
    """Load JSON."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_csv_dicts(path: Path) -> list[dict[str, Any]]:
    """Load CSV dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


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
