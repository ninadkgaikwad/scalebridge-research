# -*- coding: utf-8 -*-
"""Phase D D5 aggregation-lineage and all-to-one counterpart resolution."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class PhaseDLineageError(RuntimeError):
    """Raised when D5 authoritative lineage cannot be resolved safely."""


@dataclass(frozen=True)
class AggregateZoneMembership:
    aggregate_zone_id: str
    source_zone_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_zone_id": self.aggregate_zone_id,
            "source_zone_ids": list(self.source_zone_ids),
            "source_zone_count": len(self.source_zone_ids),
        }


@dataclass(frozen=True)
class AggregationLineage:
    campaign_id: str
    case_id: str
    source_generation_run_id: str
    aggregation_matrix_run_id: str
    aggregation_run_id: str
    aggregation_id: str
    building_type: str
    climate_zone: str
    weather_location: str
    strategy: str
    rule_set: str
    weight_mode: str
    schema_version: str
    created_at_utc: str | None
    aggregate_zones: tuple[AggregateZoneMembership, ...]
    plan_path: Path
    manifest_path: Path
    zone_mapping_path: Path
    matrix_record_index: int

    @property
    def aggregate_zone_count(self) -> int:
        return len(self.aggregate_zones)

    @property
    def source_zone_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    source_zone
                    for aggregate_zone in self.aggregate_zones
                    for source_zone in aggregate_zone.source_zone_ids
                },
                key=str.casefold,
            )
        )

    @property
    def source_zone_count(self) -> int:
        return len(self.source_zone_ids)

    @property
    def is_single_zone_full_coverage(self) -> bool:
        """True when the realized aggregation has one zone covering all sources.

        This is intentionally structural and independent of aggregation_id,
        user-defined style/family labels, or grouping strategy names.
        """
        if self.aggregate_zone_count != 1:
            return False
        return set(self.aggregate_zones[0].source_zone_ids) == set(self.source_zone_ids)

    @property
    def is_all_to_one(self) -> bool:
        """Backward-compatible alias for structural one-zone full coverage."""
        return self.is_single_zone_full_coverage

    def compatibility_signature(self) -> dict[str, Any]:
        """Data-affecting fields that must match except grouping/style itself.

        Aggregation ID/name, level/family, aggregate-zone names, and grouping
        strategy are deliberately excluded.  Dep2 only needs a compatible
        one-zone realization over the same source-zone population.
        """
        plan = _load_json(self.plan_path)
        return {
            "campaign_id": self.campaign_id,
            "case_id": self.case_id,
            "source_generation_run_id": self.source_generation_run_id,
            "building_type": self.building_type,
            "climate_zone": self.climate_zone,
            "weather_location": self.weather_location,
            "rule_set": self.rule_set,
            "weight_mode": self.weight_mode,
            "schema_version": self.schema_version,
            "system_node_name_pattern": plan.get("system_node_name_pattern"),
            "thermal_zone_filter": plan.get("thermal_zone_filter"),
            "source_zone_ids": list(self.source_zone_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "case_id": self.case_id,
            "source_generation_run_id": self.source_generation_run_id,
            "aggregation_matrix_run_id": self.aggregation_matrix_run_id,
            "aggregation_run_id": self.aggregation_run_id,
            "aggregation_id": self.aggregation_id,
            "building_type": self.building_type,
            "climate_zone": self.climate_zone,
            "weather_location": self.weather_location,
            "strategy": self.strategy,
            "rule_set": self.rule_set,
            "weight_mode": self.weight_mode,
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "aggregate_zone_count": self.aggregate_zone_count,
            "source_zone_count": self.source_zone_count,
            "source_zone_ids": list(self.source_zone_ids),
            "aggregate_zones": [item.to_dict() for item in self.aggregate_zones],
            "is_single_zone_full_coverage": self.is_single_zone_full_coverage,
            "is_all_to_one": self.is_all_to_one,
            "compatibility_signature": self.compatibility_signature(),
            "matrix_record_index": self.matrix_record_index,
            "plan_path": str(self.plan_path),
            "manifest_path": str(self.manifest_path),
            "zone_mapping_path": str(self.zone_mapping_path),
        }


@dataclass(frozen=True)
class PhaseCUsability:
    phase_c_campaign_run_id: str
    phase_c_inference_run_id: str
    aggregation_run_id: str
    aggregation_id: str
    weight_mode: str
    expected_zone_ids: tuple[str, ...]
    usable_zone_ids: tuple[str, ...]
    missing_zone_ids: tuple[str, ...]
    usable: bool
    inference_root: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_c_campaign_run_id": self.phase_c_campaign_run_id,
            "phase_c_inference_run_id": self.phase_c_inference_run_id,
            "aggregation_run_id": self.aggregation_run_id,
            "aggregation_id": self.aggregation_id,
            "weight_mode": self.weight_mode,
            "expected_zone_ids": list(self.expected_zone_ids),
            "usable_zone_ids": list(self.usable_zone_ids),
            "missing_zone_ids": list(self.missing_zone_ids),
            "usable": self.usable,
            "inference_root": str(self.inference_root),
        }


@dataclass(frozen=True)
class CounterpartResolution:
    status: str
    dependent_2_available: bool
    selected_aggregation_run_id: str | None
    compatible_candidate_run_ids: tuple[str, ...]
    all_to_one_candidate_run_ids: tuple[str, ...]
    mismatch_details: tuple[dict[str, Any], ...]
    selection_policy: str | None
    current_lineage: AggregationLineage
    selected_lineage: AggregationLineage | None
    phase_c_usability: PhaseCUsability | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dependent_2_available": self.dependent_2_available,
            "selected_aggregation_run_id": self.selected_aggregation_run_id,
            "compatible_candidate_run_ids": list(self.compatible_candidate_run_ids),
            "all_to_one_candidate_run_ids": list(self.all_to_one_candidate_run_ids),
            "mismatch_details": list(self.mismatch_details),
            "selection_policy": self.selection_policy,
            "current_lineage": self.current_lineage.to_dict(),
            "selected_lineage": (
                self.selected_lineage.to_dict() if self.selected_lineage else None
            ),
            "phase_c_usability": (
                self.phase_c_usability.to_dict() if self.phase_c_usability else None
            ),
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PhaseDLineageError(f"Required JSON not found: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise PhaseDLineageError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PhaseDLineageError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _first_nonempty(mapping: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _matrix_rows(campaign_root: Path, matrix_run_id: str) -> list[dict[str, str]]:
    path = (
        campaign_root
        / "aggregation"
        / "matrix_runs"
        / matrix_run_id
        / "aggregation_matrix_case_runs.csv"
    )
    rows = _read_csv(path)
    if not rows:
        raise PhaseDLineageError(f"Aggregation matrix has no case-run rows: {path}")
    return rows


def _resolve_run_root(
    campaign_root: Path,
    case_id: str,
    aggregation_run_id: str,
) -> Path:
    direct = (
        campaign_root
        / "aggregation"
        / "cases"
        / case_id
        / "runs"
        / aggregation_run_id
    )
    if direct.is_dir():
        return direct.resolve()
    raise PhaseDLineageError(
        f"Aggregation run root not found for case={case_id}, run={aggregation_run_id}: "
        f"{direct}"
    )


def load_aggregation_lineage(
    *,
    campaign_root: Path,
    matrix_run_id: str,
    aggregation_run_id: str,
) -> AggregationLineage:
    """Load one complete authoritative Phase B aggregation lineage."""

    campaign_root = Path(campaign_root).expanduser().resolve()
    rows = _matrix_rows(campaign_root, matrix_run_id)
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if _first_nonempty(row, ("aggregation_run_id", "run_id"))
        == aggregation_run_id
    ]
    if not matches:
        raise PhaseDLineageError(
            f"Aggregation run '{aggregation_run_id}' is not present in matrix "
            f"'{matrix_run_id}'"
        )
    if len(matches) > 1:
        raise PhaseDLineageError(
            f"Aggregation run '{aggregation_run_id}' appears multiple times in matrix"
        )

    matrix_index, row = matches[0]
    case_id = _first_nonempty(row, ("case_id",))
    run_root = _resolve_run_root(campaign_root, case_id, aggregation_run_id)
    manifest_path = run_root / "aggregation_manifest.json"
    plan_path = run_root / "inputs" / "aggregation_plan.json"
    zone_mapping_path = run_root / "inputs" / "zone_mapping.csv"

    manifest = _load_json(manifest_path)
    plan = _load_json(plan_path)
    zone_rows = _read_csv(zone_mapping_path)

    aggregate_zone_order: list[str] = []
    source_by_aggregate: dict[str, list[str]] = {}
    for zone_row in zone_rows:
        aggregate_zone_id = _first_nonempty(zone_row, ("aggregate_zone_id",))
        source_zone = _first_nonempty(zone_row, ("source_zone", "source_zone_id"))
        if not aggregate_zone_id or not source_zone:
            raise PhaseDLineageError(
                f"Invalid zone mapping row in {zone_mapping_path}: {zone_row}"
            )
        if aggregate_zone_id not in source_by_aggregate:
            aggregate_zone_order.append(aggregate_zone_id)
            source_by_aggregate[aggregate_zone_id] = []
        if source_zone not in source_by_aggregate[aggregate_zone_id]:
            source_by_aggregate[aggregate_zone_id].append(source_zone)

    memberships = tuple(
        AggregateZoneMembership(
            aggregate_zone_id=zone_id,
            source_zone_ids=tuple(source_by_aggregate[zone_id]),
        )
        for zone_id in aggregate_zone_order
    )

    return AggregationLineage(
        campaign_id=_first_nonempty(plan, ("campaign_id",)) or campaign_root.name,
        case_id=_first_nonempty(manifest, ("case_id",))
        or _first_nonempty(plan, ("source_case_id",))
        or case_id,
        source_generation_run_id=_first_nonempty(
            manifest, ("source_generation_run_id",)
        )
        or _first_nonempty(plan, ("source_generation_run_id",))
        or _first_nonempty(row, ("source_generation_run_id",)),
        aggregation_matrix_run_id=matrix_run_id,
        aggregation_run_id=aggregation_run_id,
        aggregation_id=_first_nonempty(
            manifest, ("aggregation_id", "plan_aggregation_id")
        )
        or _first_nonempty(plan, ("aggregation_id",))
        or _first_nonempty(row, ("aggregation_id",)),
        building_type=_first_nonempty(plan, ("building_type",))
        or _first_nonempty(row, ("building_type",)),
        climate_zone=_first_nonempty(plan, ("climate_zone",))
        or _first_nonempty(row, ("climate_zone",)),
        weather_location=_first_nonempty(plan, ("weather_location",))
        or _first_nonempty(row, ("weather_location",)),
        strategy=_first_nonempty(manifest, ("strategy",))
        or _first_nonempty(plan, ("strategy",))
        or _first_nonempty(row, ("loaded_plan_strategy", "plan_strategy")),
        rule_set=_first_nonempty(manifest, ("rule_set",))
        or _first_nonempty(plan, ("rule_set",))
        or _first_nonempty(row, ("loaded_plan_rule_set", "rule_set")),
        weight_mode=_first_nonempty(manifest, ("weight_mode",))
        or _first_nonempty(plan, ("weight_mode",))
        or _first_nonempty(row, ("weight_mode",)),
        schema_version=_first_nonempty(manifest, ("schema_version",))
        or _first_nonempty(plan, ("schema_version",)),
        created_at_utc=_first_nonempty(manifest, ("created_at_utc",)) or None,
        aggregate_zones=memberships,
        plan_path=plan_path.resolve(),
        manifest_path=manifest_path.resolve(),
        zone_mapping_path=zone_mapping_path.resolve(),
        matrix_record_index=matrix_index,
    )


def _diff_signatures(
    current: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    keys = sorted(set(current) | set(candidate))
    return {
        key: {"current": current.get(key), "candidate": candidate.get(key)}
        for key in keys
        if current.get(key) != candidate.get(key)
    }


def _phase_c_child_id(payload: Any, candidate_keys: tuple[str, ...]) -> str:
    if isinstance(payload, dict):
        for key in candidate_keys:
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        for value in payload.values():
            found = _phase_c_child_id(value, candidate_keys)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _phase_c_child_id(value, candidate_keys)
            if found:
                return found
    return ""


def _find_cli_option(payload: dict[str, Any], option: str) -> str:
    values: list[str] = []
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        command = result.get("command", [])
        if not isinstance(command, list):
            continue
        for index, token in enumerate(command[:-1]):
            if token == option:
                value = str(command[index + 1]).strip()
                if value and value not in values:
                    values.append(value)
    if len(values) > 1:
        raise PhaseDLineageError(
            f"Phase C campaign contains multiple values for {option}: {values}"
        )
    return values[0] if values else ""


def resolve_phase_c_usability(
    *,
    campaign_root: Path,
    phase_c_campaign_run_id: str,
    lineage: AggregationLineage,
) -> PhaseCUsability:
    """Verify usable C8 inference lineage for every expected aggregate zone."""

    campaign_root = Path(campaign_root).expanduser().resolve()
    campaign_manifest_path = (
        campaign_root
        / "heat_input_regression"
        / "campaign_runs"
        / phase_c_campaign_run_id
        / "phase_c_campaign_run_manifest.json"
    )
    manifest = _load_json(campaign_manifest_path)

    manifest_campaign_id = _first_nonempty(manifest, ("campaign_id",))
    manifest_matrix_id = _first_nonempty(manifest, ("matrix_run_id",))
    if manifest_campaign_id and manifest_campaign_id != lineage.campaign_id:
        raise PhaseDLineageError(
            f"Phase C campaign mismatch: {manifest_campaign_id} != {lineage.campaign_id}"
        )
    if manifest_matrix_id and manifest_matrix_id != lineage.aggregation_matrix_run_id:
        raise PhaseDLineageError(
            f"Phase C matrix mismatch: {manifest_matrix_id} != "
            f"{lineage.aggregation_matrix_run_id}"
        )

    inference_run_id = _phase_c_child_id(
        manifest, ("inference_run_id", "heat_input_inference_run_id")
    ) or _find_cli_option(manifest, "--inference-run-id")
    if not inference_run_id:
        raise PhaseDLineageError("Phase C campaign does not identify inference_run_id")

    inference_root = (
        campaign_root
        / "heat_input_regression"
        / "inference_runs"
        / inference_run_id
    )
    if not inference_root.is_dir():
        raise PhaseDLineageError(f"Phase C inference root not found: {inference_root}")

    zone_base = (
        inference_root
        / "cases"
        / lineage.case_id
        / lineage.aggregation_id
        / lineage.weight_mode
    )

    expected = tuple(item.aggregate_zone_id for item in lineage.aggregate_zones)
    usable: list[str] = []
    missing: list[str] = []

    for zone_id in expected:
        zone_root = zone_base / zone_id
        manifest_path = zone_root / "annual_component_predictions_manifest.json"
        if manifest_path.is_file():
            payload = _load_json(manifest_path)
            status = str(payload.get("status", "")).strip().lower()
            if status in {"", "completed", "passed", "success", "finished"}:
                usable.append(zone_id)
                continue
        missing.append(zone_id)

    return PhaseCUsability(
        phase_c_campaign_run_id=phase_c_campaign_run_id,
        phase_c_inference_run_id=inference_run_id,
        aggregation_run_id=lineage.aggregation_run_id,
        aggregation_id=lineage.aggregation_id,
        weight_mode=lineage.weight_mode,
        expected_zone_ids=expected,
        usable_zone_ids=tuple(usable),
        missing_zone_ids=tuple(missing),
        usable=not missing,
        inference_root=inference_root.resolve(),
    )


def resolve_all_to_one_counterpart(
    *,
    campaign_root: Path,
    matrix_run_id: str,
    aggregation_run_id: str,
    phase_c_campaign_run_id: str,
) -> CounterpartResolution:
    """Resolve the D5 Dependent-2 structural one-zone counterpart deterministically."""

    campaign_root = Path(campaign_root).expanduser().resolve()
    current = load_aggregation_lineage(
        campaign_root=campaign_root,
        matrix_run_id=matrix_run_id,
        aggregation_run_id=aggregation_run_id,
    )

    rows = _matrix_rows(campaign_root, matrix_run_id)
    candidates: list[AggregationLineage] = []
    for row in rows:
        candidate_run_id = _first_nonempty(row, ("aggregation_run_id", "run_id"))
        if not candidate_run_id:
            continue
        candidate = load_aggregation_lineage(
            campaign_root=campaign_root,
            matrix_run_id=matrix_run_id,
            aggregation_run_id=candidate_run_id,
        )
        if candidate.is_single_zone_full_coverage:
            candidates.append(candidate)

    candidate_ids = tuple(item.aggregation_run_id for item in candidates)

    if current.is_single_zone_full_coverage:
        phase_c = resolve_phase_c_usability(
            campaign_root=campaign_root,
            phase_c_campaign_run_id=phase_c_campaign_run_id,
            lineage=current,
        )
        return CounterpartResolution(
            status="matched_self",
            dependent_2_available=phase_c.usable,
            selected_aggregation_run_id=current.aggregation_run_id,
            compatible_candidate_run_ids=(current.aggregation_run_id,),
            all_to_one_candidate_run_ids=candidate_ids,
            mismatch_details=(),
            selection_policy="self",
            current_lineage=current,
            selected_lineage=current,
            phase_c_usability=phase_c,
        )

    compatible: list[AggregationLineage] = []
    mismatches: list[dict[str, Any]] = []
    current_signature = current.compatibility_signature()

    for candidate in candidates:
        diff = _diff_signatures(
            current_signature,
            candidate.compatibility_signature(),
        )
        if not diff:
            compatible.append(candidate)
        else:
            mismatches.append(
                {
                    "aggregation_run_id": candidate.aggregation_run_id,
                    "mismatches": diff,
                }
            )

    if not candidates:
        return CounterpartResolution(
            status="unavailable_no_counterpart",
            dependent_2_available=False,
            selected_aggregation_run_id=None,
            compatible_candidate_run_ids=(),
            all_to_one_candidate_run_ids=(),
            mismatch_details=(),
            selection_policy=None,
            current_lineage=current,
            selected_lineage=None,
            phase_c_usability=None,
        )

    if not compatible:
        return CounterpartResolution(
            status="invalid_configuration_mismatch",
            dependent_2_available=False,
            selected_aggregation_run_id=None,
            compatible_candidate_run_ids=(),
            all_to_one_candidate_run_ids=candidate_ids,
            mismatch_details=tuple(mismatches),
            selection_policy=None,
            current_lineage=current,
            selected_lineage=None,
            phase_c_usability=None,
        )

    compatible.sort(
        key=lambda item: (
            item.matrix_record_index,
            item.created_at_utc or "",
            item.aggregation_run_id,
        )
    )
    selected = compatible[0]
    status = (
        "matched_exact"
        if len(compatible) == 1
        else "ambiguous_multiple_counterparts"
    )
    phase_c = resolve_phase_c_usability(
        campaign_root=campaign_root,
        phase_c_campaign_run_id=phase_c_campaign_run_id,
        lineage=selected,
    )

    return CounterpartResolution(
        status=status,
        dependent_2_available=phase_c.usable,
        selected_aggregation_run_id=selected.aggregation_run_id,
        compatible_candidate_run_ids=tuple(
            item.aggregation_run_id for item in compatible
        ),
        all_to_one_candidate_run_ids=candidate_ids,
        mismatch_details=tuple(mismatches),
        selection_policy=(
            "first_deterministic_candidate"
            if len(compatible) > 1
            else "exact_single_candidate"
        ),
        current_lineage=current,
        selected_lineage=selected,
        phase_c_usability=phase_c,
    )
