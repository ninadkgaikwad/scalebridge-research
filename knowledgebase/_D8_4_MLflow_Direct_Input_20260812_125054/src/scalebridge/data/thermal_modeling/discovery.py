# -*- coding: utf-8 -*-
"""Deterministic upstream artifact discovery for Phase D D2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .source_refs import (
    AggregationRunRef,
    AggregationZoneRef,
    PhaseCChildRunRefs,
    PhaseCZoneRef,
    PhaseDDiscoveryResult,
)


class PhaseDDiscoveryError(RuntimeError):
    """Raised when authoritative upstream lineage cannot be resolved."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise PhaseDDiscoveryError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _first_nonempty(mapping: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise PhaseDDiscoveryError(f"{label} not found: {path}")
    return path.resolve()


def _optional_file(path: Path) -> Path | None:
    return path.resolve() if path.is_file() else None


def _required_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise PhaseDDiscoveryError(f"{label} not found: {path}")
    return path.resolve()


def _find_exact_directory(root: Path, directory_name: str) -> Path:
    direct = root / directory_name
    if direct.is_dir():
        return direct.resolve()
    matches = sorted(
        path.resolve()
        for path in root.rglob(directory_name)
        if path.is_dir() and path.name == directory_name
    )
    if not matches:
        raise PhaseDDiscoveryError(
            f"Directory '{directory_name}' not found under {root}"
        )
    if len(matches) > 1:
        raise PhaseDDiscoveryError(
            f"Directory '{directory_name}' is ambiguous under {root}: {matches}"
        )
    return matches[0]


def resolve_aggregation_run(
    *,
    campaign_root: Path,
    matrix_run_id: str,
    aggregation_run_id: str,
) -> AggregationRunRef:
    """Resolve one Phase B run using the authoritative matrix output table."""

    campaign_root = Path(campaign_root).expanduser().resolve()
    matrix_csv = (
        campaign_root
        / "aggregation"
        / "matrix_runs"
        / matrix_run_id
        / "aggregation_matrix_outputs.csv"
    )
    rows = _read_csv(matrix_csv)
    matches = [
        row
        for row in rows
        if _first_nonempty(
            row,
            ("aggregation_run_id", "run_id"),
        ) == aggregation_run_id
        or Path(str(row.get("run_root", "")).strip()).name == aggregation_run_id
    ]
    if not matches:
        raise PhaseDDiscoveryError(
            f"Aggregation run '{aggregation_run_id}' is not present in {matrix_csv}"
        )
    if len(matches) > 1:
        unique_roots = {str(row.get("run_root", "")).strip() for row in matches}
        if len(unique_roots) > 1:
            raise PhaseDDiscoveryError(
                f"Aggregation run '{aggregation_run_id}' has multiple matrix rows"
            )
    row = matches[0]

    run_root_text = str(row.get("run_root", "")).strip()
    if run_root_text and Path(run_root_text).is_dir():
        run_root = Path(run_root_text).resolve()
    else:
        run_root = _find_exact_directory(
            campaign_root / "aggregation", aggregation_run_id
        )

    manifest_path = _required_file(
        run_root / "aggregation_manifest.json", "aggregation manifest"
    )
    plan_path = _required_file(
        run_root / "inputs" / "aggregation_plan.json", "aggregation plan"
    )
    zone_mapping_path = _required_file(
        run_root / "inputs" / "zone_mapping.csv", "run zone mapping"
    )
    manifest = _load_json(manifest_path)
    plan = _load_json(plan_path)

    return AggregationRunRef(
        campaign_id=campaign_root.name,
        case_id=_first_nonempty(
            manifest, ("case_id", "source_case_id")
        ) or _first_nonempty(row, ("case_id",)),
        matrix_run_id=matrix_run_id,
        aggregation_run_id=aggregation_run_id,
        aggregation_id=_first_nonempty(
            manifest, ("aggregation_id",)
        ) or _first_nonempty(plan, ("aggregation_id", "plan_id")) or _first_nonempty(
            row, ("aggregation_id",)
        ),
        weight_mode=_first_nonempty(
            manifest, ("weight_mode",)
        ) or _first_nonempty(plan, ("weight_mode",)) or _first_nonempty(
            row, ("weight_mode",)
        ),
        strategy=_first_nonempty(
            manifest, ("strategy", "aggregation_strategy")
        ) or _first_nonempty(plan, ("strategy", "aggregation_strategy")),
        run_root=run_root,
        manifest_path=manifest_path,
        plan_path=plan_path,
        zone_mapping_path=zone_mapping_path,
    )


def resolve_aggregation_zone(
    *,
    aggregation_run: AggregationRunRef,
    aggregate_zone_id: str,
) -> AggregationZoneRef:
    """Resolve one aggregate-zone output from a selected Phase B run."""

    zone_root = _required_dir(
        aggregation_run.run_root / "zones" / aggregate_zone_id,
        "aggregation zone root",
    )
    wide_parquet = _required_file(
        zone_root / "aggregated_timeseries_wide.parquet",
        "Phase B wide Parquet",
    )
    return AggregationZoneRef(
        aggregation_run=aggregation_run,
        aggregate_zone_id=aggregate_zone_id,
        zone_root=zone_root,
        wide_parquet_path=wide_parquet,
        wide_preview_path=_optional_file(
            zone_root / "aggregated_timeseries_wide_preview.csv"
        ),
        long_parquet_path=_optional_file(
            zone_root / "aggregated_timeseries_long.parquet"
        ),
        zone_mapping_path=_optional_file(zone_root / "zone_mapping.csv"),
        static_equipment_path=_optional_file(
            zone_root / "aggregated_static_equipment.csv"
        ),
    )


_CHILD_KEYS: dict[str, tuple[str, ...]] = {
    "audit_run_id": ("audit_run_id", "heat_input_audit_run_id"),
    "feature_run_id": ("feature_run_id", "heat_input_feature_run_id"),
    "split_run_id": ("split_run_id", "heat_input_split_run_id"),
    "dataset_run_id": ("dataset_run_id", "heat_input_dataset_run_id"),
    "training_run_id": ("training_run_id", "heat_input_training_run_id"),
    "evaluation_run_id": ("evaluation_run_id", "heat_input_evaluation_run_id"),
    "inference_run_id": ("inference_run_id", "heat_input_inference_run_id"),
    "mlflow_registration_run_id": (
        "mlflow_registration_run_id",
        "phase_c_mlflow_registration_run_id",
    ),
}


def _recursive_find_value(payload: Any, candidate_keys: tuple[str, ...]) -> str:
    if isinstance(payload, dict):
        direct = _first_nonempty(payload, candidate_keys)
        if direct:
            return direct
        for value in payload.values():
            found = _recursive_find_value(value, candidate_keys)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _recursive_find_value(value, candidate_keys)
            if found:
                return found
    return ""


def _find_cli_option(payload: dict[str, Any], option_name: str) -> str:
    """Find one CLI option value in Phase C campaign result commands."""

    values: set[str] = set()

    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue

        command = result.get("command", [])
        if not isinstance(command, list):
            continue

        for index, token in enumerate(command[:-1]):
            if token == option_name:
                value = str(command[index + 1]).strip()
                if value:
                    values.add(value)

    if not values:
        return ""

    if len(values) > 1:
        raise PhaseDDiscoveryError(
            f"Phase C campaign manifest contains multiple values for "
            f"{option_name}: {sorted(values)}"
        )

    return next(iter(values))


def resolve_phase_c_child_runs(
    *,
    campaign_root: Path,
    phase_c_campaign_run_id: str,
) -> PhaseCChildRunRefs:
    """Resolve authoritative Phase C child run IDs from its campaign manifest."""

    campaign_root = Path(campaign_root).expanduser().resolve()
    run_root = _required_dir(
        campaign_root
        / "heat_input_regression"
        / "campaign_runs"
        / phase_c_campaign_run_id,
        "Phase C campaign run root",
    )
    plan_path = _required_file(
        run_root / "phase_c_campaign_plan.json", "Phase C campaign plan"
    )
    manifest_path = _required_file(
        run_root / "phase_c_campaign_run_manifest.json",
        "Phase C campaign manifest",
    )
    payload = _load_json(manifest_path)

    cli_options = {
        "audit_run_id": "--audit-run-id",
        "feature_run_id": "--feature-run-id",
        "split_run_id": "--split-run-id",
        "dataset_run_id": "--dataset-run-id",
        "training_run_id": "--training-run-id",
        "evaluation_run_id": "--evaluation-run-id",
        "inference_run_id": "--inference-run-id",
        "mlflow_registration_run_id": "--phase-c-run-id",
    }

    values: dict[str, str | None] = {}

    for output_name, candidate_keys in _CHILD_KEYS.items():
        # Support both possible Phase C campaign-manifest layouts:
        #
        # 1. explicit/nested run-ID fields;
        # 2. authoritative command arrays containing --*-run-id arguments.
        value = _recursive_find_value(payload, candidate_keys)

        if not value:
            value = _find_cli_option(
                payload,
                cli_options[output_name],
            )

        # C9 uses the Phase C campaign run ID as its registration-run identity.
        if (
            output_name == "mlflow_registration_run_id"
            and not value
        ):
            value = _first_nonempty(
                payload,
                ("phase_c_run_id",),
            )

        if output_name != "mlflow_registration_run_id" and not value:
            raise PhaseDDiscoveryError(
                f"Phase C campaign manifest does not resolve {output_name}: "
                f"{manifest_path}"
            )

        values[output_name] = value or None

    return PhaseCChildRunRefs(
        campaign_run_id=phase_c_campaign_run_id,
        campaign_run_root=run_root,
        campaign_plan_path=plan_path,
        campaign_manifest_path=manifest_path,
        audit_run_id=str(values["audit_run_id"]),
        feature_run_id=str(values["feature_run_id"]),
        split_run_id=str(values["split_run_id"]),
        dataset_run_id=str(values["dataset_run_id"]),
        training_run_id=str(values["training_run_id"]),
        evaluation_run_id=str(values["evaluation_run_id"]),
        inference_run_id=str(values["inference_run_id"]),
        mlflow_registration_run_id=(
            str(values["mlflow_registration_run_id"])
            if values["mlflow_registration_run_id"] else None
        ),
    )


def _resolve_zone_leaf(
    *,
    run_root: Path,
    case_id: str,
    aggregate_zone_id: str,
    aggregation_run_id: str | None = None,
    aggregation_id: str | None = None,
    weight_mode: str | None = None,
) -> Path:
    """Resolve one Phase C zone across authoritative upstream layouts.

    Audit and split runs use:
        cases/<case_id>/<aggregation_run_id>/<zone>

    Inference runs use:
        cases/<case_id>/<aggregation_id>/<weight_mode>/<zone>
    """

    preferred_paths: list[Path] = []

    if aggregation_run_id:
        preferred_paths.append(
            run_root
            / "cases"
            / case_id
            / aggregation_run_id
            / aggregate_zone_id
        )

    if aggregation_id and weight_mode:
        preferred_paths.append(
            run_root
            / "cases"
            / case_id
            / aggregation_id
            / weight_mode
            / aggregate_zone_id
        )

    for preferred in preferred_paths:
        if preferred.is_dir():
            return preferred.resolve()

    matches: list[Path] = []

    for candidate in run_root.rglob(aggregate_zone_id):
        if not candidate.is_dir():
            continue

        if candidate.name != aggregate_zone_id:
            continue

        if case_id not in candidate.parts:
            continue

        run_layout_match = (
            aggregation_run_id is not None
            and aggregation_run_id in candidate.parts
        )

        inference_layout_match = (
            aggregation_id is not None
            and weight_mode is not None
            and aggregation_id in candidate.parts
            and weight_mode in candidate.parts
        )

        if run_layout_match or inference_layout_match:
            matches.append(candidate.resolve())

    unique_matches = sorted(set(matches))

    if not unique_matches:
        raise PhaseDDiscoveryError(
            "Phase C zone directory not found for "
            f"case={case_id}, "
            f"aggregation_run_id={aggregation_run_id}, "
            f"aggregation_id={aggregation_id}, "
            f"weight={weight_mode}, "
            f"zone={aggregate_zone_id}, "
            f"root={run_root}"
        )

    if len(unique_matches) > 1:
        raise PhaseDDiscoveryError(
            f"Phase C zone directory is ambiguous: {unique_matches}"
        )

    return unique_matches[0]


def _first_existing(root: Path, candidates: tuple[str, ...], *, required: bool) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path.resolve()
    if required:
        raise PhaseDDiscoveryError(
            f"None of the required files exist under {root}: {candidates}"
        )
    return None


def resolve_phase_c_zone(
    *,
    campaign_root: Path,
    child_runs: PhaseCChildRunRefs,
    case_id: str,
    aggregation_run_id: str,
    aggregation_id: str,
    weight_mode: str,
    aggregate_zone_id: str,
) -> PhaseCZoneRef:
    """Resolve applicability, inference, and split artifacts for one zone."""

    heat_root = Path(campaign_root).resolve() / "heat_input_regression"
    audit_root = _required_dir(
        heat_root / "audit_runs" / child_runs.audit_run_id,
        "Phase C audit run root",
    )
    inference_root = _required_dir(
        heat_root / "inference_runs" / child_runs.inference_run_id,
        "Phase C inference run root",
    )
    split_root = _required_dir(
        heat_root / "split_runs" / child_runs.split_run_id,
        "Phase C split run root",
    )

    audit_zone_root = _resolve_zone_leaf(
        run_root=audit_root,
        case_id=case_id,
        aggregation_run_id=aggregation_run_id,
        aggregation_id=aggregation_id,
        weight_mode=weight_mode,
        aggregate_zone_id=aggregate_zone_id,
    )
    inference_zone_root = _resolve_zone_leaf(
        run_root=inference_root,
        case_id=case_id,
        aggregation_id=aggregation_id,
        weight_mode=weight_mode,
        aggregate_zone_id=aggregate_zone_id,
    )

    split_zone_root = _resolve_zone_leaf(
        run_root=split_root,
        case_id=case_id,
        aggregation_run_id=aggregation_run_id,
        aggregation_id=aggregation_id,
        weight_mode=weight_mode,
        aggregate_zone_id=aggregate_zone_id,
    )

    predictions_parquet = _first_existing(
        inference_zone_root,
        (
            "annual_component_predictions.parquet",
            "predictions.parquet",
            "full_year_predictions.parquet",
            "component_predictions.parquet",
        ),
        required=True,
    )
    assert predictions_parquet is not None

    return PhaseCZoneRef(
        case_id=case_id,
        aggregation_id=aggregation_id,
        weight_mode=weight_mode,
        aggregate_zone_id=aggregate_zone_id,
        applicable_models_path=_required_file(
            audit_zone_root / "applicable_models.csv",
            "Phase C applicable-model catalog",
        ),
        unavailable_models_path=_required_file(
            audit_zone_root / "unavailable_models.csv",
            "Phase C unavailable-model catalog",
        ),
        signal_catalog_path=_required_file(
            audit_zone_root / "heat_input_signal_catalog.csv",
            "Phase C signal catalog",
        ),
        inference_zone_root=inference_zone_root,
        predictions_parquet_path=predictions_parquet,
        predictions_preview_path=_first_existing(
            inference_zone_root,
            (
                "annual_component_predictions_preview.csv",
                "predictions_preview.csv",
                "full_year_predictions_preview.csv",
                "component_predictions_preview.csv",
            ),
            required=False,
        ),
        component_prediction_summary_path=_optional_file(
            inference_zone_root / "component_prediction_summary.csv"
        ),
        timestamp_component_availability_path=_optional_file(
            inference_zone_root / "timestamp_component_availability.csv"
        ),
        split_zone_root=split_zone_root,
        split_assignments_parquet_path=(
            _first_existing(
                split_zone_root,
                (
                    "split_assignments.parquet",
                    "timestamp_split_assignments.parquet",
                ),
                required=False,
            )
            if split_zone_root else None
        ),
        split_assignments_preview_path=(
            _first_existing(
                split_zone_root,
                (
                    "split_assignments_preview.csv",
                    "timestamp_split_assignments_preview.csv",
                ),
                required=False,
            )
            if split_zone_root else None
        ),
        metadata={
            "audit_zone_root": str(audit_zone_root),
            "audit_run_id": child_runs.audit_run_id,
            "split_run_id": child_runs.split_run_id,
            "inference_run_id": child_runs.inference_run_id,
        },
    )


def discover_phase_d_sources(
    *,
    campaign_root: Path,
    matrix_run_id: str,
    aggregation_run_id: str,
    phase_c_campaign_run_id: str,
    aggregate_zone_id: str,
) -> PhaseDDiscoveryResult:
    """Resolve all D2 sources required for one Phase D aggregate-zone product."""

    campaign_root = Path(campaign_root).expanduser().resolve()
    aggregation_run = resolve_aggregation_run(
        campaign_root=campaign_root,
        matrix_run_id=matrix_run_id,
        aggregation_run_id=aggregation_run_id,
    )
    aggregation_zone = resolve_aggregation_zone(
        aggregation_run=aggregation_run,
        aggregate_zone_id=aggregate_zone_id,
    )
    phase_c_runs = resolve_phase_c_child_runs(
        campaign_root=campaign_root,
        phase_c_campaign_run_id=phase_c_campaign_run_id,
    )
    phase_c_zone = resolve_phase_c_zone(
        campaign_root=campaign_root,
        child_runs=phase_c_runs,
        case_id=aggregation_run.case_id,
        aggregation_run_id=aggregation_run.aggregation_run_id,
        aggregation_id=aggregation_run.aggregation_id,
        weight_mode=aggregation_run.weight_mode,
        aggregate_zone_id=aggregate_zone_id,
    )
    return PhaseDDiscoveryResult(
        campaign_root=campaign_root,
        aggregation_zone=aggregation_zone,
        phase_c_runs=phase_c_runs,
        phase_c_zone=phase_c_zone,
    )
