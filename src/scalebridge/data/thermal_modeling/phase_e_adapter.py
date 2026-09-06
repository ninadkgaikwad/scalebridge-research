# -*- coding: utf-8 -*-
"""Adapter from authoritative Phase D final manifests to E0-2 contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import ModelingSilo, PhaseDMode
from .phase_e_contracts import (
    AggregationLineageBinding,
    PhaseEContractError,
    PhaseEDataContract,
    PhaseESignalBinding,
    PhaseESignalRole,
    SpatialDependencyContract,
    TemporalOwnershipContract,
)
from .silo_contracts import PhysicalRole, TemporalConfig


_REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "silo",
    "mode",
    "temporal_config",
    "heat_representation",
    "final_columns",
    "current_zone_ids",
    "provenance",
}


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _role_from_final_column(item: Mapping[str, Any]) -> PhaseESignalRole:
    try:
        physical_role = PhysicalRole(str(item["physical_role"]))
    except (KeyError, ValueError) as exc:
        raise PhaseEContractError(
            f"Unsupported/missing Phase D physical_role in final column: {item!r}"
        ) from exc

    temporal_role = str(item.get("temporal_role", ""))

    if physical_role is PhysicalRole.STATE:
        if temporal_role != "model_input":
            raise PhaseEContractError(
                "Phase D state columns consumed by E0-2 must be model_input columns"
            )
        return PhaseESignalRole.OBSERVED_STATE
    if physical_role is PhysicalRole.CONTROL_INPUT:
        return PhaseESignalRole.CONTROL_INPUT
    if physical_role is PhysicalRole.DISTURBANCE:
        return PhaseESignalRole.DISTURBANCE
    if physical_role is PhysicalRole.TARGET:
        return PhaseESignalRole.TARGET
    if physical_role is PhysicalRole.METADATA:
        return PhaseESignalRole.METADATA

    raise PhaseEContractError(
        f"Unhandled Phase D physical role: {physical_role.value!r}"
    )


def _binding(item: Mapping[str, Any]) -> PhaseESignalBinding:
    return PhaseESignalBinding(
        column_name=str(item["name"]),
        base_signal=str(item["base_signal"]),
        role=_role_from_final_column(item),
        aggregate_zone_id=(
            None
            if item.get("aggregate_zone_id") is None
            else str(item["aggregate_zone_id"])
        ),
        temporal_role=str(item["temporal_role"]),
        offset_steps=(
            None
            if item.get("offset_steps") is None
            else int(item["offset_steps"])
        ),
        units=None if item.get("units") is None else str(item["units"]),
    )


def _lineage(provenance: Mapping[str, Any]) -> AggregationLineageBinding:
    known = {
        "campaign_id",
        "case_id",
        "aggregation_matrix_run_id",
        "aggregation_run_id",
        "aggregation_id",
        "weight_mode",
        "phase_c_campaign_run_id",
        "dependent_2_match_status",
        "dependent_2_source_aggregation_run_id",
    }
    return AggregationLineageBinding(
        campaign_id=provenance.get("campaign_id"),
        case_id=provenance.get("case_id"),
        aggregation_matrix_run_id=provenance.get("aggregation_matrix_run_id"),
        aggregation_run_id=provenance.get("aggregation_run_id"),
        aggregation_id=provenance.get("aggregation_id"),
        weight_mode=provenance.get("weight_mode"),
        phase_c_campaign_run_id=provenance.get("phase_c_campaign_run_id"),
        dependent_2_match_status=provenance.get("dependent_2_match_status"),
        dependent_2_source_aggregation_run_id=provenance.get(
            "dependent_2_source_aggregation_run_id"
        ),
        extra={
            str(key): value
            for key, value in provenance.items()
            if key not in known
        },
    )


def build_phase_e_data_contract(
    manifest: Mapping[str, Any],
) -> PhaseEDataContract:
    """Build a method-neutral E0-2 contract from a final Phase D manifest.

    The adapter consumes manifest semantics only.  It never infers campaign,
    zone, signal, partition, or spatial identity from filesystem names.
    """

    missing = _REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise PhaseEContractError(
            f"Phase D manifest is missing required E0-2 fields: {sorted(missing)}"
        )

    try:
        silo = ModelingSilo(str(manifest["silo"]))
        mode = PhaseDMode(str(manifest["mode"]))
    except ValueError as exc:
        raise PhaseEContractError(
            "Phase D manifest contains unsupported silo or spatial mode"
        ) from exc

    temporal_payload = manifest["temporal_config"]
    if not isinstance(temporal_payload, Mapping):
        raise PhaseEContractError("temporal_config must be a mapping")

    temporal_validation = TemporalConfig(
        silo=silo,
        input_lag=int(temporal_payload["input_lag"]),
        target_horizon=int(temporal_payload["target_horizon"]),
        policy_name=str(temporal_payload["policy_name"]),
        policy_parameters=dict(temporal_payload.get("policy_parameters", {})),
        policy_realization_id=temporal_payload.get("policy_realization_id"),
    )
    policy = temporal_validation.policy

    temporal = TemporalOwnershipContract(
        silo=silo,
        input_lag=temporal_validation.input_lag,
        target_horizon=temporal_validation.target_horizon,
        policy_name=temporal_validation.policy_name,
        policy_parameters=dict(temporal_validation.policy_parameters),
        primary_partitions=tuple(item.value for item in policy.partitions),
        allowed_partition_values=tuple(
            item.value for item in policy.allowed_partition_values
        ),
    )

    current_zone_ids = tuple(str(value) for value in manifest["current_zone_ids"])
    independent_zone_id = manifest.get("independent_zone_id")

    if mode is PhaseDMode.INDEPENDENT:
        if independent_zone_id is None:
            raise PhaseEContractError(
                "Independent Phase D manifest requires independent_zone_id"
            )
        modeled_zone_ids = (str(independent_zone_id),)
    else:
        if independent_zone_id is not None:
            raise PhaseEContractError(
                "Dependent Phase D manifest cannot define independent_zone_id"
            )
        modeled_zone_ids = current_zone_ids

    final_columns = manifest["final_columns"]
    if not isinstance(final_columns, list):
        raise PhaseEContractError("final_columns must be a list")

    inputs: list[PhaseESignalBinding] = []
    targets: list[PhaseESignalBinding] = []
    metadata: list[str] = []

    for item in final_columns:
        if not isinstance(item, Mapping):
            raise PhaseEContractError("Each final_columns entry must be a mapping")
        role = _role_from_final_column(item)
        if role is PhaseESignalRole.METADATA:
            metadata.append(str(item["name"]))
            continue
        binding = _binding(item)
        if role is PhaseESignalRole.TARGET:
            targets.append(binding)
        else:
            inputs.append(binding)

    lag0_disturbance_sources = tuple(
        dict.fromkeys(
            item.aggregate_zone_id
            for item in inputs
            if item.role is PhaseESignalRole.DISTURBANCE
            and item.offset_steps == 0
            and item.aggregate_zone_id is not None
        )
    )

    dep2_source = manifest.get("dependent_2_source_zone_id")
    spatial = SpatialDependencyContract(
        mode=mode,
        modeled_zone_ids=modeled_zone_ids,
        independent_zone_id=(
            str(independent_zone_id) if independent_zone_id is not None else None
        ),
        dependent_2_source_zone_id=(
            str(dep2_source) if dep2_source is not None else None
        ),
        disturbance_source_zone_ids=lag0_disturbance_sources,
    )

    manifest_hash = _canonical_json_sha256(manifest)
    return PhaseEDataContract(
        contract_id=f"phasee0_{manifest_hash[:16]}",
        source_manifest_sha256=manifest_hash,
        phase_d_schema_version=str(manifest["schema_version"]),
        phase_d_d7_schema_version=(
            None
            if manifest.get("d7_schema_version") is None
            else str(manifest["d7_schema_version"])
        ),
        spatial=spatial,
        temporal=temporal,
        heat_representation=dict(manifest["heat_representation"]),
        input_bindings=tuple(inputs),
        target_bindings=tuple(targets),
        metadata_columns=tuple(metadata),
        provenance=_lineage(dict(manifest["provenance"])),
        row_count=(
            None if manifest.get("row_count") is None else int(manifest["row_count"])
        ),
        included_row_count=(
            None
            if manifest.get("included_row_count") is None
            else int(manifest["included_row_count"])
        ),
        partition_counts={
            str(key): int(value)
            for key, value in dict(manifest.get("partition_counts", {})).items()
        },
    )


def load_phase_e_data_contract(
    manifest_path: str | Path,
) -> PhaseEDataContract:
    """Load one Phase D manifest without inferring meaning from its path."""

    path = Path(manifest_path)
    if not path.is_file():
        raise PhaseEContractError(f"Phase D manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseEContractError(
            f"Unable to read Phase D manifest: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PhaseEContractError("Phase D manifest root must be a JSON object")
    return build_phase_e_data_contract(payload)


def validate_materialized_columns(
    contract: PhaseEDataContract,
    columns: Iterable[str],
) -> None:
    """Verify the materialized dataset exposes every manifest-bound column."""

    actual = {str(item) for item in columns}
    missing = set(contract.required_materialized_columns()) - actual
    if missing:
        raise PhaseEContractError(
            f"Materialized Phase D dataset is missing contract columns: {sorted(missing)}"
        )


def validate_partition_values(
    contract: PhaseEDataContract,
    partitions: Iterable[str],
) -> None:
    """Validate outer partition labels without constructing new Phase E splits."""

    values = {str(value) for value in partitions}
    allowed = set(contract.temporal.allowed_partition_values)
    unknown = values - allowed
    if unknown:
        raise PhaseEContractError(
            f"Unknown Phase D partition values for this policy: {sorted(unknown)}"
        )
