from __future__ import annotations

"""Phase-D TRAIN-only enforcement for E0-8."""

from typing import Any, Mapping, Sequence

from .contracts import HPOContractError, HPODataSelection


def create_train_only_selection(
    phase_d_contract: Any,
    *,
    selection_policy: str,
    selection_payload: Mapping[str, Any],
    source_partitions: Sequence[str] = ("train",),
) -> HPODataSelection:
    """Build a deterministic HPO selection tied to one E0-2 Phase-D contract."""
    selection = HPODataSelection.create(
        phase_d_contract_id=str(phase_d_contract.contract_id),
        phase_d_source_manifest_sha256=str(phase_d_contract.source_manifest_sha256),
        source_partitions=source_partitions,
        selection_policy=selection_policy,
        selection_payload=selection_payload,
    )
    validate_train_only_selection(phase_d_contract, selection)
    return selection


def validate_train_only_selection(
    phase_d_contract: Any,
    selection: HPODataSelection,
) -> None:
    """Reject any outer Phase-D partition outside the tuning-authorized set."""
    if selection.phase_d_contract_id != str(phase_d_contract.contract_id):
        raise HPOContractError("HPO data selection belongs to a different Phase-D contract")
    if selection.phase_d_source_manifest_sha256 != str(
        phase_d_contract.source_manifest_sha256
    ).lower():
        raise HPOContractError("HPO data selection source-manifest hash does not match Phase-D contract")

    # The E0-2 contract remains the authoritative leakage guard.
    try:
        phase_d_contract.temporal.assert_hyperparameter_tuning_partitions(
            selection.source_partitions
        )
    except Exception as exc:
        raise HPOContractError(str(exc)) from exc

    allowed = {
        str(value).lower()
        for value in phase_d_contract.temporal.hyperparameter_tuning_source_partitions
    }
    seen = {str(value).lower() for value in selection.source_partitions}
    if not seen or not seen.issubset(allowed):
        raise HPOContractError(
            "E0-8 HPO source partitions must be a non-empty subset of Phase-D tuning-authorized TRAIN"
        )
    # E0-8 v1 ratifies the stronger contract: only literal Phase-D TRAIN is eligible.
    if seen != {"train"}:
        raise HPOContractError(
            f"E0-8 v1 requires Phase-D TRAIN as the sole HPO outer source; got {sorted(seen)}"
        )
