"""Read-only Phase B Aggregation discovery for the Phase C Campaign Builder.

Phase C must be able to consume both modern Phase-B campaign definitions and
historical/legacy Aggregation matrices that predate the definition store.  The
Phase-B Results service already owns the authoritative artifact-level discovery
of every matrix under ``campaigns/*/aggregation/matrix_runs``; this module
reuses that discovery rather than maintaining a second filesystem scanner.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from scalebridge.dashapp.services.aggregation.definition_store import (
    load_definition as load_aggregation_definition,
)
from scalebridge.dashapp.services.aggregation.results_data import (
    discover_matrix_runs as discover_all_matrix_runs,
)
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


def _discovered_rows(aggregation_campaign_id: str | None = None) -> list[dict[str, Any]]:
    rows = discover_all_matrix_runs()
    if aggregation_campaign_id is None:
        return rows
    wanted = str(aggregation_campaign_id).strip()
    return [row for row in rows if str(row.get("aggregation_campaign_id")) == wanted]


def parent_aggregation_options() -> list[dict[str, str]]:
    """Return every artifact-discoverable Aggregation campaign.

    Modern matrices are grouped by their stored ``aggregation_campaign_id``.
    Legacy matrices are grouped by Phase B's established
    ``legacy::<parent_generation_campaign_id>`` discovery identifier.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for row in _discovered_rows():
        campaign_id = str(row.get("aggregation_campaign_id") or "").strip()
        parent_id = str(row.get("parent_generation_campaign_id") or "").strip()
        if not campaign_id or not parent_id:
            continue
        bucket = grouped.setdefault(
            campaign_id,
            {
                "parent": parent_id,
                "matrix_count": 0,
                "successful": 0,
            },
        )
        bucket["matrix_count"] += 1
        bucket["successful"] += int(row.get("successful_plan_count") or 0)

    options: list[dict[str, str]] = []
    for campaign_id, info in grouped.items():
        parent_id = str(info["parent"])
        if campaign_id.startswith("legacy::"):
            label = (
                f"{parent_id} | legacy Aggregation outputs | "
                f"{info['matrix_count']} matrix run(s)"
            )
        else:
            label = (
                f"{campaign_id} | Generation={parent_id} | "
                f"{info['matrix_count']} matrix run(s)"
            )
        options.append({"label": label, "value": campaign_id})

    return sorted(options, key=lambda row: row["label"].casefold())


def _try_definition(aggregation_campaign_id: str):
    if str(aggregation_campaign_id).startswith("legacy::"):
        return None
    try:
        return load_aggregation_definition(aggregation_campaign_id)
    except Exception:
        return None


def resolve_parent_context(aggregation_campaign_id: str) -> dict[str, Any]:
    """Resolve Generation lineage from a definition or directly from matrix artifacts."""
    aggregation_campaign_id = str(aggregation_campaign_id or "").strip()
    if not aggregation_campaign_id:
        raise ValueError("Aggregation campaign ID is required")

    discovered = _discovered_rows(aggregation_campaign_id)
    definition = _try_definition(aggregation_campaign_id)

    if definition is not None:
        generation_id = str(definition.parent_generation_campaign_id)
        if definition.parent_generation_campaign_root:
            campaign_root = Path(
                definition.parent_generation_campaign_root
            ).expanduser().resolve()
            generated_root = campaign_root.parent.parent
            root_method = "aggregation_definition.parent_generation_campaign_root"
        elif definition.generated_data_root:
            generated_root = Path(definition.generated_data_root).expanduser().resolve()
            campaign_root = generated_root / "campaigns" / generation_id
            root_method = "aggregation_definition.generated_data_root"
        else:
            generated_root = resolve_generated_data_root().expanduser().resolve()
            campaign_root = generated_root / "campaigns" / generation_id
            root_method = "live_settings.generated_data_root"
        artifact_parents = {
            str(row.get("parent_generation_campaign_id") or "").strip()
            for row in discovered
            if str(row.get("parent_generation_campaign_id") or "").strip()
        }
        if artifact_parents and artifact_parents != {generation_id}:
            raise ValueError(
                "Aggregation definition lineage disagrees with discovered matrix "
                f"artifacts: definition={generation_id}, artifacts={sorted(artifact_parents)}"
            )
        machine_id = str(definition.machine_id)
        strategies = list(definition.requested_strategy_values)
        weights = list(definition.requested_weight_mode_values)
    else:
        if not discovered:
            raise FileNotFoundError(
                f"No Aggregation matrix artifacts found for {aggregation_campaign_id!r}"
            )
        generation_ids = {
            str(row.get("parent_generation_campaign_id") or "").strip()
            for row in discovered
            if str(row.get("parent_generation_campaign_id") or "").strip()
        }
        if len(generation_ids) != 1:
            raise ValueError(
                "Artifact-discovered Aggregation campaign resolves to multiple "
                f"Generation parents: {sorted(generation_ids)}"
            )
        generation_id = next(iter(generation_ids))
        generated_root = resolve_generated_data_root().expanduser().resolve()
        campaign_root = generated_root / "campaigns" / generation_id
        root_method = "phase_b_matrix_artifact_discovery"
        machine_id = "artifact-discovered"
        strategies = []
        weights = []

    return {
        "parent_aggregation_campaign_id": aggregation_campaign_id,
        "parent_generation_campaign_id": generation_id,
        "aggregation_machine_id": machine_id,
        "generated_data_root": str(generated_root),
        "campaign_root": str(campaign_root),
        "campaign_root_exists": campaign_root.is_dir(),
        "matrix_runs_root": str(campaign_root / "aggregation" / "matrix_runs"),
        "root_resolution_method": root_method,
        "requested_strategies": strategies,
        "requested_weight_modes": weights,
        "definition_available": definition is not None,
        "legacy_artifact_only": aggregation_campaign_id.startswith("legacy::"),
    }


def discover_matrix_runs(aggregation_campaign_id: str) -> list[dict[str, Any]]:
    """Return all readable matrices for one modern or legacy Aggregation campaign."""
    rows: list[dict[str, Any]] = []
    for discovered in _discovered_rows(aggregation_campaign_id):
        manifest_path = Path(str(discovered.get("manifest_path") or ""))
        manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
        selected = _as_int(discovered.get("selected_plan_count"))
        successful = _as_int(discovered.get("successful_plan_count"))
        failed = _as_int(discovered.get("failed_plan_count"))
        if successful <= 0:
            readiness = "unusable"
        elif failed > 0:
            readiness = "partial"
        else:
            readiness = "ready"

        campaign_id = str(discovered.get("aggregation_campaign_id") or "")
        rows.append(
            {
                **discovered,
                "matrix_run_id": str(
                    discovered.get("matrix_run_id")
                    or manifest.get("matrix_run_id")
                    or manifest_path.parent.name
                ),
                "aggregation_campaign_id": campaign_id,
                "ownership_status": (
                    "legacy_unscoped"
                    if campaign_id.startswith("legacy::")
                    else "scoped"
                ),
                "readiness": readiness,
                "selected_plan_count": selected,
                "successful_plan_count": successful,
                "failed_plan_count": failed,
                "building_types": list(manifest.get("building_types") or []),
                "weather_locations": list(manifest.get("weather_locations") or []),
                "climate_zones": list(manifest.get("climate_zones") or []),
                "aggregation_ids": list(manifest.get("aggregation_ids") or []),
                "weight_modes": list(manifest.get("weight_modes") or []),
                "created_at_utc": str(
                    discovered.get("created_at_utc")
                    or manifest.get("created_at_utc")
                    or ""
                ),
                "path": str(discovered.get("matrix_root") or manifest_path.parent),
                "manifest_path": str(manifest_path),
            }
        )
    return rows


def matrix_run_options(aggregation_campaign_id: str) -> list[dict[str, str]]:
    """Return matrix dropdown options with readiness made explicit."""
    options: list[dict[str, str]] = []
    for row in discover_matrix_runs(aggregation_campaign_id):
        label = (
            f"{row['matrix_run_id']} | {row['readiness']} | "
            f"{row['successful_plan_count']}/{row['selected_plan_count']} successful"
        )
        options.append({"label": label, "value": str(row["matrix_run_id"])})
    return options


def matrix_summary(
    aggregation_campaign_id: str,
    matrix_run_id: str,
) -> dict[str, Any]:
    """Load compact matrix metadata and selectable campaign-scope facets."""
    matrix_run_id = str(matrix_run_id or "").strip()
    matching = [
        row
        for row in discover_matrix_runs(aggregation_campaign_id)
        if row["matrix_run_id"] == matrix_run_id
    ]
    if not matching:
        raise FileNotFoundError(
            f"Matrix run {matrix_run_id!r} is not available for "
            f"Aggregation campaign {aggregation_campaign_id!r}"
        )
    row = dict(matching[0])
    matrix_root = Path(row["path"])

    case_rows = _read_csv(matrix_root / "aggregation_matrix_case_runs.csv")
    successful_rows = [
        item
        for item in case_rows
        if str(item.get("status") or "").strip().casefold()
        in {"completed", "completed_with_warnings"}
    ]

    scope_rows = [_normalize_scope_row(item) for item in successful_rows]
    row.update(
        {
            "case_ids": _unique(successful_rows, "case_id"),
            "case_options": _case_options(scope_rows),
            # Backward-compatible artifact identifiers used by the scientific
            # runner.  The UI no longer presents these as the modeler-facing
            # Aggregation selector because they mix strategy/weight/custom IDs.
            "aggregation_ids": _unique(successful_rows, "aggregation_id")
            or list(row.get("aggregation_ids") or []),
            "weight_modes": _scope_values(scope_rows, "weight_mode")
            or list(row.get("weight_modes") or []),
            "strategies": _scope_values(scope_rows, "strategy"),
            "custom_grouping_ids": _scope_values(scope_rows, "custom_grouping_id"),
            "rule_sets": _scope_values(scope_rows, "rule_set"),
            "scope_rows": scope_rows,
            "successful_case_plan_rows": len(successful_rows),
            "recorded_case_plan_rows": len(case_rows),
        }
    )
    return row



def scope_options(summary: dict[str, Any], key: str) -> list[dict[str, str]]:
    """Return modeler-facing options for one normalized Phase-B scope dimension."""
    if key == "case_id":
        return list(summary.get("case_options") or [])
    values = _scope_values(list(summary.get("scope_rows") or []), key)
    return [{"label": value, "value": value} for value in values]


def resolve_scope_selection(
    summary: dict[str, Any],
    *,
    case_id: str | None = None,
    strategy: str | None = None,
    custom_grouping_id: str | None = None,
    weight_mode: str | None = None,
    rule_set: str | None = None,
) -> dict[str, Any]:
    """Compile clear UI lineage selectors to the existing runner scope fields.

    The authoritative Phase C runner still consumes ``case_id``,
    ``aggregation_id`` and ``weight_mode``.  This helper preserves that contract
    while preventing the UI from conflating a custom grouping ID with a complete
    aggregation plan identifier.
    """
    rows = list(summary.get("scope_rows") or [])
    if not rows:
        return {
            "case_id": case_id or None,
            "aggregation_id": None,
            "weight_mode": weight_mode or None,
        }

    chosen = {
        "case_id": str(case_id or "").strip(),
        "strategy": str(strategy or "").strip(),
        "custom_grouping_id": str(custom_grouping_id or "").strip(),
        "weight_mode": str(weight_mode or "").strip(),
        "rule_set": str(rule_set or "").strip(),
    }

    if chosen["custom_grouping_id"] and chosen["strategy"] != "custom_groups":
        raise ValueError(
            "Custom Grouping ID is only applicable when Aggregation Strategy is custom_groups"
        )

    filtered = rows
    for key, value in chosen.items():
        if value:
            filtered = [row for row in filtered if str(row.get(key) or "") == value]

    if not filtered:
        raise ValueError("No successful Phase B case-plan row matches the selected scope")

    aggregation_id = None
    plan_selector_used = any(
        chosen[key]
        for key in ("strategy", "custom_grouping_id", "rule_set")
    )
    if plan_selector_used:
        aggregation_ids = sorted(
            {str(row.get("aggregation_id") or "") for row in filtered if row.get("aggregation_id")}
        )
        if len(aggregation_ids) != 1:
            raise ValueError(
                "The selected Aggregation lineage still matches multiple Phase B plan IDs. "
                "Choose a Weight Mode (and Custom Grouping ID when applicable) to make "
                "the selection unambiguous, or clear Aggregation Strategy to include all plans."
            )
        aggregation_id = aggregation_ids[0]

    return {
        "case_id": chosen["case_id"] or None,
        "aggregation_id": aggregation_id,
        "weight_mode": chosen["weight_mode"] or None,
    }


def _normalize_scope_row(row: dict[str, Any]) -> dict[str, str]:
    aggregation_id = str(row.get("aggregation_id") or "").strip()
    strategy = str(
        row.get("plan_strategy")
        or row.get("loaded_plan_strategy")
        or row.get("aggregation_family")
        or ""
    ).strip()
    if strategy not in {"all_thermal_zones_to_one", "custom_groups", "identity"}:
        lowered = aggregation_id.casefold()
        if lowered.startswith("identity"):
            strategy = "identity"
        elif lowered.startswith("all_thermal_zones_to_one") or lowered.startswith("all_to_one"):
            strategy = "all_thermal_zones_to_one"
        elif aggregation_id:
            strategy = "custom_groups"

    custom_grouping_id = aggregation_id if strategy == "custom_groups" else ""
    return {
        "case_id": str(row.get("case_id") or "").strip(),
        "building_type": str(row.get("building_type") or "").strip(),
        "weather_location": str(row.get("weather_location") or "").strip(),
        "aggregation_id": aggregation_id,
        "strategy": strategy,
        "custom_grouping_id": custom_grouping_id,
        "weight_mode": str(
            row.get("weight_mode") or row.get("loaded_plan_weight_mode") or ""
        ).strip(),
        "rule_set": str(
            row.get("rule_set") or row.get("loaded_plan_rule_set") or ""
        ).strip(),
    }


def _case_options(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    labels: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        building = str(row.get("building_type") or "") or "Unknown building"
        weather = str(row.get("weather_location") or "") or "Unknown weather"
        labels[case_id] = f"{building} | {weather} | {case_id}"
    return [
        {"label": labels[case_id], "value": case_id}
        for case_id in sorted(labels, key=lambda value: labels[value].casefold())
    ]


def _scope_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {
            str(row.get(key) or "").strip()
            for row in rows
            if str(row.get(key) or "").strip()
        },
        key=str.casefold,
    )

def validate_matrix_selection(
    aggregation_campaign_id: str,
    matrix_run_id: str,
) -> dict[str, Any]:
    """Validate that the selected matrix has at least one successful plan."""
    summary = matrix_summary(aggregation_campaign_id, matrix_run_id)
    if int(summary.get("successful_plan_count") or 0) <= 0:
        raise ValueError(
            f"Matrix run {matrix_run_id} contains no successful Aggregation plans"
        )
    return summary


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _unique(rows: list[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {
            str(row.get(key) or "").strip()
            for row in rows
            if str(row.get(key) or "").strip()
        }
    )


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
