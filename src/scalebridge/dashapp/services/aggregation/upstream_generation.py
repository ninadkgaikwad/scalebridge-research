"""Read-only upstream Generation discovery for BGIRS Phase B Aggregation.

B5 intentionally does not create Aggregation plans or execute Aggregation.
It exposes the authoritative latest successful Generation runs, their lineage,
scientific case metadata, and optional thermal-zone inventory needed by the
future Aggregation Campaign Builder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from scalebridge.data.aggregation.discovery import (
    discover_generation_runs,
    load_json,
)
from scalebridge.data.aggregation.models import SUCCESS_STATUSES
from scalebridge.data.aggregation.zone_inventory import build_zone_inventory_from_eio
from scalebridge.dashapp.adapters.generation.filesystem import (
    discover_generation_campaign_ids,
)
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


def campaigns_root() -> Path:
    """Return the configured ScaleBridge campaign root."""
    return resolve_generated_data_root() / "campaigns"


def parent_campaign_options() -> list[dict[str, str]]:
    """Return Generation campaigns suitable for a parent-campaign dropdown."""
    return [
        {"label": campaign_id, "value": campaign_id}
        for campaign_id in discover_generation_campaign_ids(campaigns_root())
    ]


def discover_parent_campaigns() -> list[dict[str, Any]]:
    """Return compact metadata for all campaigns containing Generation cases.

    A malformed case never prevents discovery of other campaigns/cases.
    ``eligible_case_count`` counts only latest runs accepted by the authoritative
    Aggregation ``SUCCESS_STATUSES`` contract.
    """
    rows: list[dict[str, Any]] = []
    root = campaigns_root()

    for campaign_id in discover_generation_campaign_ids(root):
        result = discover_generation_cases(
            campaign_id,
            include_zone_inventory=False,
        )
        case_rows = result["cases"]
        issue_rows = result["issues"]
        rows.append(
            {
                "campaign_id": campaign_id,
                "parent_generation_campaign_id": campaign_id,
                "eligible_case_count": len(case_rows),
                "discovery_issue_count": len(issue_rows),
                "building_types": _unique_values(case_rows, "building_type"),
                "weather_locations": _unique_values(case_rows, "weather_location"),
                "climate_zones": _unique_values(case_rows, "climate_zone"),
                "statuses": _unique_values(case_rows, "status"),
                "path": str(root / campaign_id),
            }
        )

    return rows


def discover_generation_cases(
    campaign_id: str,
    *,
    include_zone_inventory: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Discover latest eligible Generation runs for one parent campaign.

    Returns
    -------
    dict
        ``cases`` contains selectable successful latest Generation runs.
        ``issues`` contains malformed/missing-input discovery diagnostics.

    Notes
    -----
    - Eligibility is delegated to ``discover_generation_runs`` and therefore
      uses the same success contract as the scientific Aggregation runner.
    - Zone inventory is parsed read-only from the selected run's own
      ``canonical/eio_tables.json``. No Phase B artifact is written.
    - A missing/unreadable EIO inventory does not hide an otherwise eligible
      Generation run; it is surfaced through ``zone_inventory_status`` and an
      issue row so the Campaign Builder can explain why zone-aware operations
      are unavailable.
    """
    campaign_id = str(campaign_id or "").strip()
    if not campaign_id:
        return {
            "cases": [],
            "issues": [
                {
                    "campaign_id": "",
                    "case_id": "",
                    "run_id": "",
                    "code": "missing_campaign_id",
                    "message": "A parent Generation campaign_id is required",
                    "path": "",
                }
            ],
        }

    campaign_root = campaigns_root() / campaign_id
    cases_root = campaign_root / "generation" / "cases"

    if not cases_root.is_dir():
        return {
            "cases": [],
            "issues": [
                {
                    "campaign_id": campaign_id,
                    "case_id": "",
                    "run_id": "",
                    "code": "generation_cases_root_not_found",
                    "message": "Generation cases directory does not exist",
                    "path": str(cases_root),
                }
            ],
        }

    try:
        run_refs, missing_rows = discover_generation_runs(
            cases_root=cases_root,
            case_id=None,
            include_failed=False,
        )
    except Exception as exc:
        return {
            "cases": [],
            "issues": [
                {
                    "campaign_id": campaign_id,
                    "case_id": "",
                    "run_id": "",
                    "code": "generation_discovery_failed",
                    "message": str(exc),
                    "path": str(cases_root),
                }
            ],
        }

    issues = [
        {
            "campaign_id": campaign_id,
            "case_id": str(row.get("case_id", "")),
            "run_id": str(row.get("run_id", "")),
            "code": "generation_input_unavailable",
            "message": str(row.get("reason", "")),
            "path": str(row.get("missing_file", "")),
        }
        for row in missing_rows
    ]

    cases: list[dict[str, Any]] = []
    for run_ref in run_refs:
        try:
            manifest = load_json(run_ref.manifest_path)
        except Exception as exc:
            issues.append(
                {
                    "campaign_id": campaign_id,
                    "case_id": run_ref.case_id,
                    "run_id": run_ref.run_id,
                    "code": "generation_manifest_unreadable",
                    "message": str(exc),
                    "path": str(run_ref.manifest_path),
                }
            )
            continue

        case_spec = _mapping(manifest.get("case_spec"))
        execution = _mapping(manifest.get("execution"))
        validation = _mapping(manifest.get("validation"))
        tags = _mapping(case_spec.get("tags"))

        status = str(manifest.get("status") or run_ref.status or "").strip().casefold()
        # The discovery primitive already applies this contract; retain an
        # explicit guard against an inconsistent/corrupt manifest payload.
        if status not in SUCCESS_STATUSES:
            issues.append(
                {
                    "campaign_id": campaign_id,
                    "case_id": run_ref.case_id,
                    "run_id": run_ref.run_id,
                    "code": "generation_manifest_status_not_eligible",
                    "message": f"Generation manifest status is {status!r}",
                    "path": str(run_ref.manifest_path),
                }
            )
            continue

        row: dict[str, Any] = {
            "campaign_id": campaign_id,
            "parent_generation_campaign_id": campaign_id,
            "case_id": run_ref.case_id,
            "run_id": run_ref.run_id,
            "status": status,
            "building_type": _first_text(
                case_spec.get("building_type"),
                tags.get("source_idf_name"),
            ),
            "prototype_standard": _text(case_spec.get("prototype_standard")),
            "prototype_year": _text(case_spec.get("prototype_year")),
            "weather_location": _first_text(
                case_spec.get("weather_location"),
                tags.get("source_weather_name"),
            ),
            "climate_zone": _text(case_spec.get("climate_zone")),
            "machine_id": _text(execution.get("machine_id")),
            "hostname": _text(execution.get("hostname")),
            "warning_count": _optional_int(validation.get("warnings")),
            "severe_error_count": _optional_int(validation.get("severe_errors")),
            "fatal_error_count": _optional_int(validation.get("fatal_errors")),
            "requested_signal_count": _optional_int(validation.get("requested_signals")),
            "produced_signal_count": _optional_int(validation.get("produced_signals")),
            "case_root": str(run_ref.case_root),
            "run_root": str(run_ref.run_root),
            "manifest_path": str(run_ref.manifest_path),
            "zone_inventory_status": "not_requested",
            "thermal_zone_count": None,
            "excluded_zone_count": None,
            "thermal_zone_names": [],
            "eio_tables_path": "",
        }

        if include_zone_inventory:
            _attach_zone_inventory(
                row=row,
                issues=issues,
                campaign_id=campaign_id,
                run_ref=run_ref,
            )

        cases.append(row)

    cases.sort(
        key=lambda row: (
            str(row.get("building_type", "")).casefold(),
            str(row.get("weather_location", "")).casefold(),
            str(row.get("case_id", "")).casefold(),
        )
    )
    return {"cases": cases, "issues": issues}


def filter_generation_cases(
    rows: Iterable[dict[str, Any]] | None,
    *,
    building_types: Iterable[str] | None = None,
    weather_locations: Iterable[str] | None = None,
    climate_zones: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter already-discovered Generation case rows without filesystem I/O."""
    selected = list(rows or [])
    filters = (
        ("building_type", building_types),
        ("weather_location", weather_locations),
        ("climate_zone", climate_zones),
        ("case_id", case_ids),
        ("run_id", run_ids),
        ("status", statuses),
    )

    for key, values in filters:
        normalized = {
            str(value)
            for value in (values or [])
            if str(value).strip()
        }
        if normalized:
            selected = [
                row for row in selected
                if str(row.get(key, "")) in normalized
            ]

    return selected


def selection_facets(rows: Iterable[dict[str, Any]] | None) -> dict[str, list[str]]:
    """Return sorted dropdown facets for a discovered/filtered case collection."""
    selected = list(rows or [])
    return {
        "building_types": _unique_values(selected, "building_type"),
        "weather_locations": _unique_values(selected, "weather_location"),
        "climate_zones": _unique_values(selected, "climate_zone"),
        "case_ids": _unique_values(selected, "case_id"),
        "run_ids": _unique_values(selected, "run_id"),
        "statuses": _unique_values(selected, "status"),
    }


def _attach_zone_inventory(
    *,
    row: dict[str, Any],
    issues: list[dict[str, Any]],
    campaign_id: str,
    run_ref: Any,
) -> None:
    eio_path = run_ref.run_root / "canonical" / "eio_tables.json"
    row["eio_tables_path"] = str(eio_path)

    if not eio_path.is_file():
        row["zone_inventory_status"] = "missing"
        issues.append(
            {
                "campaign_id": campaign_id,
                "case_id": run_ref.case_id,
                "run_id": run_ref.run_id,
                "code": "zone_inventory_source_missing",
                "message": "Selected Generation run has no canonical/eio_tables.json",
                "path": str(eio_path),
            }
        )
        return

    try:
        inventory = build_zone_inventory_from_eio(
            case_id=run_ref.case_id,
            eio_tables_path=eio_path,
        )
    except Exception as exc:
        row["zone_inventory_status"] = "error"
        issues.append(
            {
                "campaign_id": campaign_id,
                "case_id": run_ref.case_id,
                "run_id": run_ref.run_id,
                "code": "zone_inventory_unreadable",
                "message": str(exc),
                "path": str(eio_path),
            }
        )
        return

    row["zone_inventory_status"] = "available"
    row["thermal_zone_count"] = inventory.approved_zone_count
    row["excluded_zone_count"] = inventory.excluded_zone_count
    row["thermal_zone_names"] = sorted(
        (item.zone_name for item in inventory.approved_rows),
        key=str.casefold,
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique_values(rows: Iterable[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {
            str(row.get(key, "")).strip()
            for row in rows
            if str(row.get(key, "")).strip()
        },
        key=str.casefold,
    )
