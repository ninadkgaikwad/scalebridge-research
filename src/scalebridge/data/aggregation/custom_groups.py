"""
User-approved custom grouping utilities for ScaleBridge aggregation.

Increment 5 purpose:
    - Approve one or more grouping suggestions by suggestion_id.
    - Support approving all generated suggestions in one run.
    - Validate approved or manually edited custom groupings.
    - Support multiple aggregation styles per case in one CSV.
    - Do not build formal aggregation plans yet.
    - Do not run aggregation.

Design rules:
    - Every aggregation_id is independently validated.
    - Every aggregation_id must be a complete partition of approved zones.
    - Every approved zone appears exactly once per aggregation_id.
    - No excluded/non-approved zones are allowed.
    - Default max aggregate zones equals approved zone count unless overridden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class CustomGroupingValidationResult:
    custom_grouping_path: Path
    approved_zone_features_path: Path
    valid: bool
    aggregation_count: int
    row_count: int
    approved_zones: list[str]
    errors: list[str]
    warnings: list[str]
    aggregation_summaries: list[dict[str, Any]]


def timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_zone_name(value: Any) -> str:
    return str(value).strip().upper()


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def get_approved_zones_from_features(path: str | Path) -> list[str]:
    rows = read_csv_dicts(path)
    zones: list[str] = []
    seen: set[str] = set()

    for row in rows:
        if not truthy(row.get("include_flag", "true")):
            continue

        zone = normalize_zone_name(row.get("zone_name"))
        if zone and zone not in seen:
            zones.append(zone)
            seen.add(zone)

    return zones


CUSTOM_GROUP_FIELDNAMES = [
    "case_id",
    "aggregation_id",
    "source_suggestion_id",
    "candidate_family",
    "n_approved_zones",
    "n_aggregate_zones",
    "effective_max_aggregate_zones",
    "aggregate_zone_name",
    "source_zone_name",
    "source_zone_index",
    "grouping_family",
    "grouping_reason",
    "approval_status",
    "approved_by",
    "approval_notes",
]


def normalize_custom_group_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Normalize rows from either suggested_groupings.csv or approved custom CSV.

    Required logical fields:
        case_id
        aggregation_id or suggestion_id
        aggregate_zone_name
        source_zone_name

    This allows manually edited custom files to be simpler while still producing
    the full approved_custom_groups.csv schema downstream.
    """
    normalized: list[dict[str, Any]] = []

    for row in rows:
        suggestion_id = str(row.get("suggestion_id", row.get("source_suggestion_id", ""))).strip()
        aggregation_id = str(row.get("aggregation_id", "")).strip() or suggestion_id

        item = {
            "case_id": str(row.get("case_id", "")).strip(),
            "aggregation_id": aggregation_id,
            "source_suggestion_id": suggestion_id,
            "candidate_family": str(row.get("candidate_family", "")).strip(),
            "n_approved_zones": str(row.get("n_approved_zones", "")).strip(),
            "n_aggregate_zones": str(row.get("n_aggregate_zones", "")).strip(),
            "effective_max_aggregate_zones": str(
                row.get("effective_max_aggregate_zones", "")
            ).strip(),
            "aggregate_zone_name": str(row.get("aggregate_zone_name", "")).strip(),
            "source_zone_name": normalize_zone_name(row.get("source_zone_name")),
            "source_zone_index": str(row.get("source_zone_index", "")).strip(),
            "grouping_family": str(
                row.get("grouping_family", row.get("candidate_family", ""))
            ).strip(),
            "grouping_reason": str(
                row.get("grouping_reason", row.get("reason", ""))
            ).strip(),
            "approval_status": str(row.get("approval_status", "approved")).strip()
            or "approved",
            "approved_by": str(row.get("approved_by", "user")).strip() or "user",
            "approval_notes": str(row.get("approval_notes", "")).strip(),
        }

        normalized.append(item)

    return normalized


def approve_suggestion_rows(
    *,
    suggested_groupings_path: str | Path,
    suggestion_id: str,
    aggregation_id: str | None = None,
    approved_by: str = "user",
    approval_notes: str = "",
) -> list[dict[str, Any]]:
    rows = read_csv_dicts(suggested_groupings_path)
    selected = [
        row
        for row in rows
        if str(row.get("suggestion_id", "")).strip() == suggestion_id
    ]

    if not selected:
        raise ValueError(
            f"No rows found for suggestion_id={suggestion_id} in {suggested_groupings_path}"
        )

    normalized = normalize_custom_group_rows(selected)
    final_aggregation_id = aggregation_id or suggestion_id

    approved_rows: list[dict[str, Any]] = []
    for row in normalized:
        updated = dict(row)
        updated["aggregation_id"] = final_aggregation_id
        updated["approval_status"] = "approved"
        updated["approved_by"] = approved_by
        updated["approval_notes"] = approval_notes
        approved_rows.append(updated)

    return approved_rows


def unique_suggestion_ids_from_file(suggested_groupings_path: str | Path) -> list[str]:
    rows = read_csv_dicts(suggested_groupings_path)
    seen: set[str] = set()
    suggestion_ids: list[str] = []

    for row in rows:
        suggestion_id = str(row.get("suggestion_id", "")).strip()
        if suggestion_id and suggestion_id not in seen:
            suggestion_ids.append(suggestion_id)
            seen.add(suggestion_id)

    return suggestion_ids


def suffix_from_suggestion_id(suggestion_id: str) -> str:
    """
    Build a readable suffix from common suggestion IDs.

    Examples:
        epcase_abc_k1_all_to_one -> k1_all_to_one
        epcase_abc_k2_identity   -> k2_identity
    """
    parts = str(suggestion_id).split("_")
    for idx, part in enumerate(parts):
        if part.startswith("k") and len(part) >= 2 and part[1:].isdigit():
            return "_".join(parts[idx:])
    return suggestion_id


def approve_multiple_suggestion_rows(
    *,
    suggested_groupings_path: str | Path,
    suggestion_ids: list[str] | None = None,
    approve_all: bool = False,
    aggregation_id: str | None = None,
    aggregation_id_prefix: str | None = None,
    approved_by: str = "user",
    approval_notes: str = "",
) -> list[dict[str, Any]]:
    """
    Approve multiple grouping suggestions into one approved_custom_groups.csv.

    Rules:
        - approve_all=True selects all suggestion_id blocks from suggested_groupings.csv.
        - Otherwise suggestion_ids must be provided.
        - aggregation_id is allowed only when approving exactly one suggestion.
        - aggregation_id_prefix is used to generate readable IDs for multiple styles.
    """
    if approve_all:
        selected_ids = unique_suggestion_ids_from_file(suggested_groupings_path)
    else:
        selected_ids = list(suggestion_ids or [])

    if not selected_ids:
        raise ValueError("No suggestion IDs selected for approval.")

    if aggregation_id and len(selected_ids) != 1:
        raise ValueError(
            "aggregation_id can only be used when approving exactly one suggestion."
        )

    approved_rows: list[dict[str, Any]] = []

    for suggestion_id in selected_ids:
        if aggregation_id:
            final_aggregation_id = aggregation_id
        elif aggregation_id_prefix:
            final_aggregation_id = (
                f"{aggregation_id_prefix}_{suffix_from_suggestion_id(suggestion_id)}"
            )
        else:
            final_aggregation_id = suggestion_id

        rows_for_suggestion = approve_suggestion_rows(
            suggested_groupings_path=suggested_groupings_path,
            suggestion_id=suggestion_id,
            aggregation_id=final_aggregation_id,
            approved_by=approved_by,
            approval_notes=approval_notes,
        )

        approved_rows.extend(rows_for_suggestion)

    return approved_rows


def group_rows_by_aggregation(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        aggregation_id = str(row.get("aggregation_id", "")).strip()
        grouped.setdefault(aggregation_id, []).append(row)
    return grouped


def validate_custom_group_rows(
    *,
    rows: list[dict[str, Any]],
    approved_zones: list[str],
    max_aggregate_zones: int | None = None,
) -> tuple[bool, list[str], list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    summaries: list[dict[str, Any]] = []

    approved_set = set(approved_zones)
    effective_max = min(
        max_aggregate_zones if max_aggregate_zones is not None else len(approved_zones),
        len(approved_zones),
    )

    if effective_max < 1:
        errors.append(f"effective_max_aggregate_zones must be >= 1, got {effective_max}")

    grouped = group_rows_by_aggregation(rows)

    if not grouped:
        errors.append("No aggregation_id groups found in custom grouping rows.")

    for aggregation_id, agg_rows in grouped.items():
        if not aggregation_id:
            errors.append("Found empty aggregation_id.")
            continue

        used_zones = [
            normalize_zone_name(row.get("source_zone_name"))
            for row in agg_rows
        ]
        used_set = set(used_zones)

        duplicate_zones = sorted(
            {zone for zone in used_zones if used_zones.count(zone) > 1}
        )
        missing_zones = sorted(approved_set - used_set)
        extra_zones = sorted(used_set - approved_set)

        aggregate_zone_names = [
            str(row.get("aggregate_zone_name", "")).strip()
            for row in agg_rows
        ]
        aggregate_zone_set = {name for name in aggregate_zone_names if name}
        empty_aggregate_zone_rows = [
            idx
            for idx, name in enumerate(aggregate_zone_names, start=1)
            if not name
        ]

        n_aggregate_zones = len(aggregate_zone_set)

        if duplicate_zones:
            errors.append(
                f"{aggregation_id}: duplicate source zones: {duplicate_zones}"
            )
        if missing_zones:
            errors.append(
                f"{aggregation_id}: missing approved zones: {missing_zones}"
            )
        if extra_zones:
            errors.append(
                f"{aggregation_id}: non-approved/extra source zones: {extra_zones}"
            )
        if empty_aggregate_zone_rows:
            errors.append(
                f"{aggregation_id}: empty aggregate_zone_name rows: "
                f"{empty_aggregate_zone_rows}"
            )
        if n_aggregate_zones < 1:
            errors.append(f"{aggregation_id}: no aggregate zones found.")
        if n_aggregate_zones > effective_max:
            errors.append(
                f"{aggregation_id}: n_aggregate_zones={n_aggregate_zones} exceeds "
                f"effective_max_aggregate_zones={effective_max}."
            )

        summaries.append(
            {
                "aggregation_id": aggregation_id,
                "row_count": len(agg_rows),
                "approved_zone_count": len(approved_zones),
                "used_zone_count": len(used_set),
                "n_aggregate_zones": n_aggregate_zones,
                "effective_max_aggregate_zones": effective_max,
                "missing_zones": missing_zones,
                "extra_zones": extra_zones,
                "duplicate_zones": duplicate_zones,
                "aggregate_zones": sorted(aggregate_zone_set),
                "valid": (
                    not duplicate_zones
                    and not missing_zones
                    and not extra_zones
                    and not empty_aggregate_zone_rows
                    and n_aggregate_zones >= 1
                    and n_aggregate_zones <= effective_max
                ),
            }
        )

    return len(errors) == 0, errors, warnings, summaries


def validate_custom_grouping_file(
    *,
    custom_grouping_path: str | Path,
    approved_zone_features_path: str | Path,
    max_aggregate_zones: int | None = None,
) -> CustomGroupingValidationResult:
    custom_path = Path(custom_grouping_path).expanduser().resolve()
    features_path = Path(approved_zone_features_path).expanduser().resolve()

    approved_zones = get_approved_zones_from_features(features_path)
    raw_rows = read_csv_dicts(custom_path)
    rows = normalize_custom_group_rows(raw_rows)

    valid, errors, warnings, summaries = validate_custom_group_rows(
        rows=rows,
        approved_zones=approved_zones,
        max_aggregate_zones=max_aggregate_zones,
    )

    return CustomGroupingValidationResult(
        custom_grouping_path=custom_path,
        approved_zone_features_path=features_path,
        valid=valid,
        aggregation_count=len(group_rows_by_aggregation(rows)),
        row_count=len(rows),
        approved_zones=approved_zones,
        errors=errors,
        warnings=warnings,
        aggregation_summaries=summaries,
    )


def write_approved_grouping_outputs(
    *,
    approved_rows: list[dict[str, Any]],
    approved_zone_features_path: str | Path,
    output_root: str | Path,
    max_aggregate_zones: int | None = None,
    notes_title: str = "Approved Custom Groupings",
) -> dict[str, Any]:
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    approved_csv = out_root / "approved_custom_groups.csv"
    write_csv(approved_csv, approved_rows, CUSTOM_GROUP_FIELDNAMES)

    validation = validate_custom_grouping_file(
        custom_grouping_path=approved_csv,
        approved_zone_features_path=approved_zone_features_path,
        max_aggregate_zones=max_aggregate_zones,
    )

    payload = {
        "custom_grouping_path": str(approved_csv),
        "approved_zone_features_path": str(Path(approved_zone_features_path).resolve()),
        "valid": validation.valid,
        "aggregation_count": validation.aggregation_count,
        "row_count": validation.row_count,
        "approved_zones": validation.approved_zones,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "aggregation_summaries": validation.aggregation_summaries,
        "uses_raw_idf": False,
        "uses_opyplus": False,
    }

    write_json(out_root / "approved_custom_groups.json", payload)

    lines = [
        f"# {notes_title}",
        "",
        f"- Valid: `{validation.valid}`",
        f"- Aggregation count: `{validation.aggregation_count}`",
        f"- Row count: `{validation.row_count}`",
        f"- Approved zones: `{', '.join(validation.approved_zones)}`",
        "",
        "## Aggregations",
        "",
    ]

    for summary in validation.aggregation_summaries:
        lines.append(f"### `{summary['aggregation_id']}`")
        lines.append(f"- Valid: `{summary['valid']}`")
        lines.append(f"- Aggregate zone count: `{summary['n_aggregate_zones']}`")
        lines.append(f"- Missing zones: `{summary['missing_zones']}`")
        lines.append(f"- Extra zones: `{summary['extra_zones']}`")
        lines.append(f"- Duplicate zones: `{summary['duplicate_zones']}`")
        lines.append(f"- Aggregate zones: `{summary['aggregate_zones']}`")
        lines.append("")

    if validation.errors:
        lines.append("## Errors")
        lines.append("")
        for error in validation.errors:
            lines.append(f"- {error}")
        lines.append("")

    write_text(out_root / "user_grouping_notes.md", "\n".join(lines))

    if not validation.valid:
        raise ValueError(
            "Approved grouping failed validation. "
            f"See {out_root / 'approved_custom_groups.json'}"
        )

    return payload


def write_validation_outputs(
    *,
    validation: CustomGroupingValidationResult,
    output_root: str | Path,
) -> dict[str, Any]:
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "custom_grouping_path": str(validation.custom_grouping_path),
        "approved_zone_features_path": str(validation.approved_zone_features_path),
        "valid": validation.valid,
        "aggregation_count": validation.aggregation_count,
        "row_count": validation.row_count,
        "approved_zones": validation.approved_zones,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "aggregation_summaries": validation.aggregation_summaries,
        "uses_raw_idf": False,
        "uses_opyplus": False,
    }

    write_json(out_root / "custom_grouping_validation.json", payload)

    fieldnames = [
        "aggregation_id",
        "valid",
        "row_count",
        "approved_zone_count",
        "used_zone_count",
        "n_aggregate_zones",
        "effective_max_aggregate_zones",
        "missing_zones",
        "extra_zones",
        "duplicate_zones",
        "aggregate_zones",
    ]

    csv_rows = []
    for summary in validation.aggregation_summaries:
        csv_rows.append(
            {
                **summary,
                "missing_zones": "|".join(summary["missing_zones"]),
                "extra_zones": "|".join(summary["extra_zones"]),
                "duplicate_zones": "|".join(summary["duplicate_zones"]),
                "aggregate_zones": "|".join(summary["aggregate_zones"]),
            }
        )

    write_csv(out_root / "custom_grouping_validation.csv", csv_rows, fieldnames)

    return payload