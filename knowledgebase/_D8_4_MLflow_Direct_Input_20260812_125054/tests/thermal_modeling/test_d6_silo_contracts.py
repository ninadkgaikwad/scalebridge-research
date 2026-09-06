# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from scalebridge.data.thermal_modeling.constants import ModelingSilo, PhaseDMode
from scalebridge.data.thermal_modeling.silo_contracts import (
    COMMON_POLICY_COLUMNS,
    CONTROL_SIGNAL,
    D6ContractError,
    HeatInputRepresentation,
    HeatRepresentationConfig,
    ML_SCIML_POLICY_CATALOG,
    OPT_BAYES_POLICY_CATALOG,
    Partition,
    PhysicalRole,
    Season,
    SiloProductContract,
    TemporalConfig,
    ZoneSignalAvailability,
    get_policy_contract,
    validate_partition_record,
)


DINING = ZoneSignalAvailability(
    aggregate_zone_id="Dining",
    available_disturbances=(
        "qsol1",
        "qsol2",
        "qzic_p",
        "qzic_l",
        "qzic_ee",
        "qzir_p",
        "qzir_l",
        "qzivr_l",
    ),
)

KITCHEN = ZoneSignalAvailability(
    aggregate_zone_id="Kitchen",
    available_disturbances=(
        "qsol1",
        "qzic_p",
        "qzic_ee",
        "qzir_p",
        "qzir_ee",
        "qzivr_l",
    ),
)

ALL_ZONE = ZoneSignalAvailability(
    aggregate_zone_id="RestaurantFastFood_All",
    available_disturbances=(
        "qsol1",
        "qsol2",
        "qzic_p",
        "qzic_l",
        "qzic_ee",
        "qzir_p",
        "qzir_l",
        "qzir_ee",
        "qzivr_l",
    ),
)


def ml_temporal(lag: int = 12, horizon: int = 6) -> TemporalConfig:
    return TemporalConfig(
        silo=ModelingSilo.ML_SCIML,
        input_lag=lag,
        target_horizon=horizon,
        policy_name="monthly_distributed_holdout",
    )


def opt_temporal() -> TemporalConfig:
    return TemporalConfig(
        silo=ModelingSilo.OPT_BAYES,
        input_lag=1,
        target_horizon=1,
        policy_name="seasonal_distributed",
    )


def grouped(include_visible: bool = True) -> HeatRepresentationConfig:
    return HeatRepresentationConfig(
        representation=HeatInputRepresentation.GROUPED,
        include_visible_lighting_in_qzir=include_visible,
    )


def test_policy_catalogs_have_locked_defaults() -> None:
    assert "monthly_distributed_holdout" in ML_SCIML_POLICY_CATALOG
    assert "seasonal_distributed" in OPT_BAYES_POLICY_CATALOG

    ml = get_policy_contract(
        ModelingSilo.ML_SCIML, "monthly_distributed_holdout"
    )
    assert ml.uses_full_year is True
    assert ml.partitions == (
        Partition.TRAIN,
        Partition.VALIDATION,
        Partition.TEST,
    )

    opt = get_policy_contract(
        ModelingSilo.OPT_BAYES, "seasonal_distributed"
    )
    assert opt.uses_full_year is False
    assert opt.partitions == (Partition.TRAIN, Partition.TEST)
    assert opt.requires_window_id_for_included is True
    assert opt.requires_season_for_included is True


def test_opt_bayes_requires_lag_1_horizon_1() -> None:
    with pytest.raises(D6ContractError):
        TemporalConfig(
            silo=ModelingSilo.OPT_BAYES,
            input_lag=2,
            target_horizon=1,
            policy_name="seasonal_distributed",
        )
    with pytest.raises(D6ContractError):
        TemporalConfig(
            silo=ModelingSilo.OPT_BAYES,
            input_lag=1,
            target_horizon=2,
            policy_name="seasonal_distributed",
        )


def test_ml_sciml_supports_variable_lag_and_horizon() -> None:
    config = ml_temporal(lag=24, horizon=12)
    assert config.lag_horizon_folder == "l24_h12"


def test_policy_realization_id_prevents_folder_collision() -> None:
    config = TemporalConfig(
        silo=ModelingSilo.OPT_BAYES,
        input_lag=1,
        target_horizon=1,
        policy_name="custom_datetime_ranges",
        policy_realization_id="paper_window_set_a",
    )
    assert (
        config.policy_folder
        == "cdr_rpaper_window_set_a"
    )


def test_common_policy_columns_are_locked() -> None:
    assert COMMON_POLICY_COLUMNS == (
        "timestamp",
        "included",
        "partition",
        "window_id",
        "season",
    )


def test_grouped_representation_keeps_solar_and_groups_internal_heat() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.INDEPENDENT,
        temporal=ml_temporal(lag=1, horizon=1),
        heat=grouped(True),
        current_zones=(DINING,),
    )
    base = contract.base_columns(independent_zone_id="Dining")
    names = [item.name for item in base]
    assert names == [
        "outdoor_temperature",
        "Dining__zone_temperature",
        "Dining__qac",
        "Dining__qsol1",
        "Dining__qsol2",
        "Dining__zic",
        "Dining__zir",
    ]


def test_grouped_visible_lighting_can_remain_separate() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.INDEPENDENT,
        temporal=ml_temporal(lag=1, horizon=1),
        heat=grouped(False),
        current_zones=(DINING,),
    )
    names = [
        item.name
        for item in contract.base_columns(independent_zone_id="Dining")
    ]
    assert "Dining__zir" in names
    assert "Dining__qzivr_l" in names


def test_component_representation_preserves_available_components_only() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.INDEPENDENT,
        temporal=ml_temporal(lag=1, horizon=1),
        heat=HeatRepresentationConfig(
            HeatInputRepresentation.COMPONENTS,
            include_visible_lighting_in_qzir=True,
        ),
        current_zones=(KITCHEN,),
    )
    names = [
        item.name
        for item in contract.base_columns(independent_zone_id="Kitchen")
    ]
    assert "Kitchen__qzic_p" in names
    assert "Kitchen__qzir_ee" in names
    assert "Kitchen__qzic_l" not in names
    assert "Kitchen__zic" not in names
    assert "Kitchen__zir" not in names


def test_zone_temperature_is_state_qac_is_control_everything_else_disturbance() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.INDEPENDENT,
        temporal=ml_temporal(lag=1, horizon=1),
        heat=grouped(),
        current_zones=(DINING,),
    )
    base = contract.base_columns(independent_zone_id="Dining")
    role_by_signal = {item.base_signal: item.physical_role for item in base}
    assert role_by_signal["zone_temperature"] is PhysicalRole.STATE
    assert role_by_signal[CONTROL_SIGNAL] is PhysicalRole.CONTROL_INPUT
    assert role_by_signal["outdoor_temperature"] is PhysicalRole.DISTURBANCE
    assert role_by_signal["qsol1"] is PhysicalRole.DISTURBANCE
    assert role_by_signal["zic"] is PhysicalRole.DISTURBANCE
    assert role_by_signal["zir"] is PhysicalRole.DISTURBANCE


def test_dependent1_is_wide_with_zone_qualified_columns_and_outdoor_once() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.DEPENDENT1,
        temporal=ml_temporal(lag=1, horizon=1),
        heat=grouped(),
        current_zones=(DINING, KITCHEN),
    )
    names = [item.name for item in contract.base_columns()]
    assert names.count("outdoor_temperature") == 1
    assert "Dining__zone_temperature" in names
    assert "Kitchen__zone_temperature" in names
    assert "Dining__qac" in names
    assert "Kitchen__qac" in names
    assert "Dining__zic" in names
    assert "Kitchen__zic" in names


def test_dependent2_uses_current_states_controls_and_all_to_one_disturbances() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.DEPENDENT2,
        temporal=ml_temporal(lag=1, horizon=1),
        heat=grouped(),
        current_zones=(DINING, KITCHEN),
        dependent_2_source_zone=ALL_ZONE,
    )
    names = [item.name for item in contract.base_columns()]
    assert "Dining__zone_temperature" in names
    assert "Kitchen__zone_temperature" in names
    assert "Dining__qac" in names
    assert "Kitchen__qac" in names
    assert "RestaurantFastFood_All__zic" in names
    assert "RestaurantFastFood_All__zir" in names
    assert "Dining__zic" not in names
    assert "Kitchen__zic" not in names
    assert names.count("outdoor_temperature") == 1


def test_dependent2_without_counterpart_is_not_a_valid_contract() -> None:
    with pytest.raises(D6ContractError):
        SiloProductContract(
            silo=ModelingSilo.ML_SCIML,
            mode=PhaseDMode.DEPENDENT2,
            temporal=ml_temporal(),
            heat=grouped(),
            current_zones=(DINING, KITCHEN),
        )


def test_qac_is_required_for_every_current_zone() -> None:
    no_qac = ZoneSignalAvailability(
        aggregate_zone_id="Broken",
        available_disturbances=("qsol1",),
        qac_available=False,
    )
    with pytest.raises(D6ContractError):
        SiloProductContract(
            silo=ModelingSilo.ML_SCIML,
            mode=PhaseDMode.INDEPENDENT,
            temporal=ml_temporal(),
            heat=grouped(),
            current_zones=(no_qac,),
        )


def test_ml_temporal_expansion_is_stable_and_targets_only_states() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.INDEPENDENT,
        temporal=ml_temporal(lag=3, horizon=2),
        heat=grouped(),
        current_zones=(DINING,),
    )
    columns = contract.final_columns(independent_zone_id="Dining")
    names = [item.name for item in columns]

    assert "Dining__zone_temperature__lag_0" in names
    assert "Dining__zone_temperature__lag_2" in names
    assert "Dining__qac__lag_2" in names
    assert "outdoor_temperature__lag_2" in names
    assert "Dining__zone_temperature__target_1" in names
    assert "Dining__zone_temperature__target_2" in names
    assert "Dining__qac__target_1" not in names
    assert "Dining__zic__target_1" not in names


def test_opt_bayes_uses_same_temporal_column_notation() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.OPT_BAYES,
        mode=PhaseDMode.INDEPENDENT,
        temporal=opt_temporal(),
        heat=grouped(),
        current_zones=(DINING,),
    )
    names = [
        item.name
        for item in contract.final_columns(independent_zone_id="Dining")
    ]
    assert "Dining__zone_temperature__lag_0" in names
    assert "Dining__zone_temperature__target_1" in names
    assert "Dining__zone_temperature__target_2" not in names


def test_ml_output_path_has_zone_heat_lag_horizon_and_policy() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.INDEPENDENT,
        temporal=ml_temporal(lag=12, horizon=6),
        heat=grouped(True),
        current_zones=(DINING,),
    )
    path = contract.relative_output_dir(independent_zone_id="Dining")
    assert path.as_posix() == (
        "ind/Dining/"
        "grp_vrin/"
        "l12_h6/mdh"
    )


def test_opt_output_path_has_policy_nesting_too() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.OPT_BAYES,
        mode=PhaseDMode.DEPENDENT1,
        temporal=opt_temporal(),
        heat=grouped(True),
        current_zones=(DINING, KITCHEN),
    )
    path = contract.relative_output_dir()
    assert path.as_posix() == (
        "dep1/grp_vrin/"
        "l1_h1/sd"
    )


def test_compact_folder_tokens_preserve_semantics() -> None:
    ml = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.DEPENDENT1,
        temporal=ml_temporal(),
        heat=grouped(),
        current_zones=(DINING, KITCHEN),
    )
    opt = SiloProductContract(
        silo=ModelingSilo.OPT_BAYES,
        mode=PhaseDMode.DEPENDENT2,
        temporal=opt_temporal(),
        heat=grouped(),
        current_zones=(DINING, KITCHEN),
        dependent_2_source_zone=ALL_ZONE,
    )

    assert ml.silo_folder_name == "ml"
    assert ml.product_folder_name == "dep1"
    assert ml.heat.folder_name == "grp_vrin"
    assert ml.temporal.lag_horizon_folder == "l12_h6"
    assert ml.temporal.policy_folder == "mdh"

    assert opt.silo_folder_name == "ob"
    assert opt.product_folder_name == "dep2"
    assert opt.temporal.lag_horizon_folder == "l1_h1"
    assert opt.temporal.policy_folder == "sd"


def test_each_realization_has_exactly_one_parquet_and_manifest() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.ML_SCIML,
        mode=PhaseDMode.DEPENDENT1,
        temporal=ml_temporal(),
        heat=grouped(),
        current_zones=(DINING, KITCHEN),
    )
    data, manifest = contract.expected_files()
    assert data.name == "data.parquet"
    assert manifest.name == "manifest.json"
    assert "train.parquet" not in str(data)
    assert "validation.parquet" not in str(data)
    assert "test.parquet" not in str(data)


def test_manifest_contract_marks_split_files_forbidden() -> None:
    contract = SiloProductContract(
        silo=ModelingSilo.OPT_BAYES,
        mode=PhaseDMode.DEPENDENT1,
        temporal=opt_temporal(),
        heat=grouped(),
        current_zones=(DINING, KITCHEN),
    )
    manifest = contract.to_manifest_contract()
    assert manifest["storage_contract"]["one_parquet_per_realization"] is True
    assert manifest["storage_contract"]["split_files_forbidden"] is True


def test_ml_partition_validation() -> None:
    policy = get_policy_contract(
        ModelingSilo.ML_SCIML, "monthly_distributed_holdout"
    )
    validate_partition_record(
        policy,
        included=True,
        partition=Partition.TRAIN,
        window_id=None,
        season=None,
    )
    validate_partition_record(
        policy,
        included=False,
        partition=Partition.EXCLUDED,
        window_id=None,
        season=None,
    )
    with pytest.raises(D6ContractError):
        validate_partition_record(
            policy,
            included=True,
            partition=Partition.EXCLUDED,
            window_id=None,
            season=None,
        )


def test_opt_selected_rows_require_window_and_season() -> None:
    policy = get_policy_contract(
        ModelingSilo.OPT_BAYES, "seasonal_distributed"
    )
    validate_partition_record(
        policy,
        included=True,
        partition=Partition.TRAIN,
        window_id="winter_train_01",
        season=Season.WINTER,
    )
    with pytest.raises(D6ContractError):
        validate_partition_record(
            policy,
            included=True,
            partition=Partition.TRAIN,
            window_id=None,
            season=Season.WINTER,
        )
    with pytest.raises(D6ContractError):
        validate_partition_record(
            policy,
            included=True,
            partition=Partition.TEST,
            window_id="winter_test_01",
            season=None,
        )


def test_opt_nonselected_rows_are_explicitly_excluded() -> None:
    policy = get_policy_contract(
        ModelingSilo.OPT_BAYES, "seasonal_distributed"
    )
    validate_partition_record(
        policy,
        included=False,
        partition=Partition.EXCLUDED,
        window_id=None,
        season=None,
    )
    with pytest.raises(D6ContractError):
        validate_partition_record(
            policy,
            included=False,
            partition=Partition.TRAIN,
            window_id=None,
            season=None,
        )
