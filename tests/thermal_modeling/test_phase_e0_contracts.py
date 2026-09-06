# -*- coding: utf-8 -*-
from __future__ import annotations

import copy

import pytest

from scalebridge.data.thermal_modeling.constants import PhaseDMode
from scalebridge.data.thermal_modeling.phase_e_adapter import (
    build_phase_e_data_contract,
    validate_materialized_columns,
    validate_partition_values,
)
from scalebridge.data.thermal_modeling.phase_e_contracts import (
    PhaseEContractError,
    PhaseESignalRole,
    get_scientific_signal_spec,
)


def _col(
    name: str,
    physical_role: str,
    temporal_role: str,
    base_signal: str,
    zone: str | None,
    offset: int | None,
    units: str | None,
) -> dict[str, object]:
    return {
        "name": name,
        "physical_role": physical_role,
        "temporal_role": temporal_role,
        "aggregate_zone_id": zone,
        "base_signal": base_signal,
        "offset_steps": offset,
        "units": units,
    }


def _metadata_columns() -> list[dict[str, object]]:
    return [
        _col("timestamp", "metadata", "anchor_timestamp", "timestamp", None, None, None),
        _col("included", "metadata", "selection", "included", None, None, None),
        _col("partition", "metadata", "partition", "partition", None, None, None),
        _col("window_id", "metadata", "selection_window", "window_id", None, None, None),
        _col("season", "metadata", "season", "season", None, None, None),
    ]


def _ind_manifest(*, qac: bool = True, solar: bool = True) -> dict[str, object]:
    cols = _metadata_columns()
    cols += [
        _col(
            "outdoor_temperature__lag_0",
            "disturbance",
            "model_input",
            "outdoor_temperature",
            None,
            0,
            "degC",
        ),
        _col(
            "Dining__zone_temperature__lag_0",
            "state",
            "model_input",
            "zone_temperature",
            "Dining",
            0,
            "degC",
        ),
    ]
    if qac:
        cols.append(
            _col(
                "Dining__qac__lag_0",
                "control_input",
                "model_input",
                "qac",
                "Dining",
                0,
                "W",
            )
        )
    if solar:
        cols.extend(
            [
                _col(
                    "Dining__qsol1__lag_0",
                    "disturbance",
                    "model_input",
                    "qsol1",
                    "Dining",
                    0,
                    "W",
                ),
                _col(
                    "Dining__qsol2__lag_0",
                    "disturbance",
                    "model_input",
                    "qsol2",
                    "Dining",
                    0,
                    "W",
                ),
            ]
        )
    cols.extend(
        [
            _col(
                "Dining__zic__lag_0",
                "disturbance",
                "model_input",
                "zic",
                "Dining",
                0,
                "W",
            ),
            _col(
                "Dining__zir__lag_0",
                "disturbance",
                "model_input",
                "zir",
                "Dining",
                0,
                "W",
            ),
            _col(
                "Dining__zone_temperature__target_1",
                "target",
                "prediction_target",
                "zone_temperature",
                "Dining",
                1,
                "degC",
            ),
        ]
    )
    return {
        "schema_version": "phase_d_d6_silo_contract_v1",
        "d7_schema_version": "phase_d_d7_final_dataset_v1",
        "silo": "ml_sciml",
        "mode": "independent",
        "independent_zone_id": "Dining",
        "current_zone_ids": ["Dining", "Kitchen"],
        "dependent_2_source_zone_id": None,
        "heat_representation": {
            "representation": "grouped_qzic_qzir",
            "include_visible_lighting_in_qzir": True,
            "folder_name": "grp_vrin",
        },
        "temporal_config": {
            "silo": "ml_sciml",
            "input_lag": 1,
            "target_horizon": 1,
            "policy_name": "monthly_distributed_holdout",
            "policy_realization_id": None,
            "policy_parameters": {
                "train_fraction": 0.70,
                "test_fraction": 0.15,
                "validation_fraction": 0.15,
            },
        },
        "final_columns": cols,
        "provenance": {
            "campaign_id": "campaign",
            "case_id": "case",
            "aggregation_matrix_run_id": "matrix",
            "aggregation_run_id": "run",
            "aggregation_id": "identity",
            "weight_mode": "equal",
            "phase_c_campaign_run_id": "phase_c",
        },
        "row_count": 100,
        "included_row_count": 95,
        "partition_counts": {
            "train": 65,
            "validation": 15,
            "test": 15,
            "excluded": 5,
        },
    }


def _dep1_manifest() -> dict[str, object]:
    manifest = _ind_manifest()
    manifest["mode"] = "dependent1"
    manifest["independent_zone_id"] = None
    manifest["current_zone_ids"] = ["Dining", "Kitchen"]
    cols = _metadata_columns() + [
        _col("outdoor_temperature__lag_0", "disturbance", "model_input", "outdoor_temperature", None, 0, "degC"),
        _col("Dining__zone_temperature__lag_0", "state", "model_input", "zone_temperature", "Dining", 0, "degC"),
        _col("Dining__qac__lag_0", "control_input", "model_input", "qac", "Dining", 0, "W"),
        _col("Dining__qsol1__lag_0", "disturbance", "model_input", "qsol1", "Dining", 0, "W"),
        _col("Dining__zic__lag_0", "disturbance", "model_input", "zic", "Dining", 0, "W"),
        _col("Dining__zir__lag_0", "disturbance", "model_input", "zir", "Dining", 0, "W"),
        _col("Kitchen__zone_temperature__lag_0", "state", "model_input", "zone_temperature", "Kitchen", 0, "degC"),
        # Kitchen is structurally uncontrolled and has fewer disturbances.
        _col("Kitchen__zic__lag_0", "disturbance", "model_input", "zic", "Kitchen", 0, "W"),
        _col("Kitchen__zir__lag_0", "disturbance", "model_input", "zir", "Kitchen", 0, "W"),
        _col("Dining__zone_temperature__target_1", "target", "prediction_target", "zone_temperature", "Dining", 1, "degC"),
        _col("Kitchen__zone_temperature__target_1", "target", "prediction_target", "zone_temperature", "Kitchen", 1, "degC"),
    ]
    manifest["final_columns"] = cols
    return manifest


def _dep2_manifest() -> dict[str, object]:
    manifest = _ind_manifest()
    manifest["mode"] = "dependent2"
    manifest["independent_zone_id"] = None
    manifest["current_zone_ids"] = ["Dining", "Kitchen"]
    manifest["dependent_2_source_zone_id"] = "RestaurantFastFood_All"
    manifest["provenance"]["dependent_2_match_status"] = "matched"
    manifest["provenance"]["dependent_2_source_aggregation_run_id"] = "all_to_one_run"
    manifest["final_columns"] = _metadata_columns() + [
        _col("outdoor_temperature__lag_0", "disturbance", "model_input", "outdoor_temperature", None, 0, "degC"),
        _col("Dining__zone_temperature__lag_0", "state", "model_input", "zone_temperature", "Dining", 0, "degC"),
        _col("Dining__qac__lag_0", "control_input", "model_input", "qac", "Dining", 0, "W"),
        _col("Kitchen__zone_temperature__lag_0", "state", "model_input", "zone_temperature", "Kitchen", 0, "degC"),
        _col("Kitchen__qac__lag_0", "control_input", "model_input", "qac", "Kitchen", 0, "W"),
        _col("RestaurantFastFood_All__qsol1__lag_0", "disturbance", "model_input", "qsol1", "RestaurantFastFood_All", 0, "W"),
        _col("RestaurantFastFood_All__qsol2__lag_0", "disturbance", "model_input", "qsol2", "RestaurantFastFood_All", 0, "W"),
        _col("RestaurantFastFood_All__zic__lag_0", "disturbance", "model_input", "zic", "RestaurantFastFood_All", 0, "W"),
        _col("RestaurantFastFood_All__zir__lag_0", "disturbance", "model_input", "zir", "RestaurantFastFood_All", 0, "W"),
        _col("Dining__zone_temperature__target_1", "target", "prediction_target", "zone_temperature", "Dining", 1, "degC"),
        _col("Kitchen__zone_temperature__target_1", "target", "prediction_target", "zone_temperature", "Kitchen", 1, "degC"),
    ]
    return manifest


def test_qac_and_phvac_have_locked_distinct_scientific_semantics() -> None:
    qac = get_scientific_signal_spec("qac")
    phvac = get_scientific_signal_spec("phvac")

    assert qac.domain.value == "thermal_power"
    assert qac.role is PhaseESignalRole.CONTROL_INPUT
    assert qac.thermal_balance_input is True
    assert qac.sign_convention == "positive_heating_negative_cooling"

    assert phvac.domain.value == "electrical_power"
    assert phvac.role is PhaseESignalRole.AUXILIARY_OUTPUT
    assert phvac.thermal_balance_input is False


def test_independent_contract_uses_manifest_identity_not_all_current_zones() -> None:
    contract = build_phase_e_data_contract(_ind_manifest())

    assert contract.spatial.mode is PhaseDMode.INDEPENDENT
    assert contract.modeled_zone_ids == ("Dining",)
    assert contract.spatial.independent_zone_id == "Dining"


def test_exact_binding_resolution_replaces_substring_guessing() -> None:
    contract = build_phase_e_data_contract(_ind_manifest())

    qac = contract.require_input("qac", aggregate_zone_id="Dining")
    assert qac.column_name == "Dining__qac__lag_0"

    with pytest.raises(PhaseEContractError):
        contract.require_input("qac", aggregate_zone_id="Kitchen")


def test_structurally_uncontrolled_zone_does_not_get_zero_qac_fabricated() -> None:
    contract = build_phase_e_data_contract(_ind_manifest(qac=False))

    assert contract.find_input("qac", aggregate_zone_id="Dining") is None
    with pytest.raises(PhaseEContractError, match="must not infer"):
        contract.require_input("qac", aggregate_zone_id="Dining")


def test_dependent1_supports_zone_specific_feature_availability() -> None:
    contract = build_phase_e_data_contract(_dep1_manifest())

    assert set(contract.modeled_zone_ids) == {"Dining", "Kitchen"}
    assert "qsol1" in contract.available_lag0_signals("Dining")
    assert "qsol1" not in contract.available_lag0_signals("Kitchen")
    assert "qac" in contract.available_lag0_signals("Dining")
    assert "qac" not in contract.available_lag0_signals("Kitchen")


def test_dependent2_preserves_current_state_control_and_aggregate_disturbance_source() -> None:
    contract = build_phase_e_data_contract(_dep2_manifest())

    assert contract.spatial.mode is PhaseDMode.DEPENDENT2
    assert contract.modeled_zone_ids == ("Dining", "Kitchen")
    assert (
        contract.spatial.dependent_2_source_zone_id
        == "RestaurantFastFood_All"
    )
    assert contract.spatial.disturbance_source_zone_ids == (
        "RestaurantFastFood_All",
    )
    assert contract.require_input("qac", aggregate_zone_id="Dining")
    assert contract.require_input("zic", aggregate_zone_id="RestaurantFastFood_All")
    assert contract.find_input("zic", aggregate_zone_id="Dining") is None


def test_latent_states_are_not_fabricated_from_phase_d_targets() -> None:
    contract = build_phase_e_data_contract(_ind_manifest())
    assert contract.latent_state_bindings == ()
    assert contract.require_target(
        "zone_temperature",
        aggregate_zone_id="Dining",
        horizon=1,
    ).role is PhaseESignalRole.TARGET


def test_phase_d_owns_outer_partitions_and_hpo_source_is_training_only() -> None:
    contract = build_phase_e_data_contract(_ind_manifest())

    assert contract.temporal.outer_partition_owner == "phase_d"
    assert contract.temporal.hyperparameter_tuning_source_partitions == ("train",)
    contract.temporal.assert_hyperparameter_tuning_partitions(["train", "train"])

    for forbidden in ("validation", "test", "excluded"):
        with pytest.raises(PhaseEContractError):
            contract.temporal.assert_hyperparameter_tuning_partitions(
                ["train", forbidden]
            )


def test_partition_validator_accepts_only_phase_d_policy_values() -> None:
    contract = build_phase_e_data_contract(_ind_manifest())

    validate_partition_values(
        contract,
        ["train", "validation", "test", "excluded"],
    )
    with pytest.raises(PhaseEContractError):
        validate_partition_values(contract, ["train", "phase_e_custom_split"])


def test_materialized_column_validator_uses_exact_manifest_columns() -> None:
    contract = build_phase_e_data_contract(_ind_manifest())
    columns = list(contract.required_materialized_columns())

    validate_materialized_columns(contract, columns)
    columns.remove("Dining__zic__lag_0")
    with pytest.raises(PhaseEContractError, match="Dining__zic__lag_0"):
        validate_materialized_columns(contract, columns)


def test_phvac_is_forbidden_as_thermal_model_input() -> None:
    manifest = _ind_manifest()
    manifest["final_columns"].insert(
        -1,
        _col(
            "Dining__phvac__lag_0",
            "disturbance",
            "model_input",
            "phvac",
            "Dining",
            0,
            "W",
        ),
    )
    with pytest.raises(PhaseEContractError, match="PHVAC"):
        build_phase_e_data_contract(manifest)


def test_qac_role_cannot_be_reinterpreted_as_disturbance() -> None:
    manifest = _ind_manifest()
    for item in manifest["final_columns"]:
        if item["base_signal"] == "qac":
            item["physical_role"] = "disturbance"
            break
    with pytest.raises(PhaseEContractError, match="canonical role"):
        build_phase_e_data_contract(manifest)


def test_source_manifest_hash_and_contract_id_are_deterministic() -> None:
    a = build_phase_e_data_contract(_ind_manifest())
    b = build_phase_e_data_contract(copy.deepcopy(_ind_manifest()))

    assert a.source_manifest_sha256 == b.source_manifest_sha256
    assert a.contract_id == b.contract_id


def test_provenance_is_carried_from_manifest_without_filesystem_inference() -> None:
    manifest = _ind_manifest()
    manifest["provenance"]["custom_note"] = "kept"
    contract = build_phase_e_data_contract(manifest)

    assert contract.provenance.campaign_id == "campaign"
    assert contract.provenance.aggregation_run_id == "run"
    assert contract.provenance.extra == {"custom_note": "kept"}
    assert contract.provenance.to_dict()["identity_inferred_from_filesystem"] is False


def test_contract_serialization_records_e0_boundaries() -> None:
    payload = build_phase_e_data_contract(_dep2_manifest()).to_dict()

    assert payload["schema_version"] == "phase_e0_e02_contract_v1"
    assert payload["spatial"]["physical_coupling_defined_here"] is False
    assert payload["temporal"]["phase_e_may_redefine_outer_partitions"] is False
    assert payload["latent_state_source"] == "method_or_topology_contract_not_phase_d"
