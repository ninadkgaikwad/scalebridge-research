# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scalebridge.data.thermal_modeling.builders import (
    availability_from_canonical_table,
    build_final_dataset,
    build_physical_table,
    expand_temporal_dataset,
    write_final_dataset,
)
from scalebridge.data.thermal_modeling.constants import ModelingSilo, PhaseDMode
from scalebridge.data.thermal_modeling.policies import (
    assign_chronological_holdout,
    assign_contiguous_identification,
    assign_custom_datetime_ranges,
    assign_monthly_distributed_holdout,
    assign_seasonal_block_holdout,
    assign_seasonal_distributed,
    assign_seasonal_holdout,
    normalize_policy_name,
)
from scalebridge.data.thermal_modeling.silo_contracts import (
    D6ContractError,
    HeatInputRepresentation,
    HeatRepresentationConfig,
    SiloProductContract,
    TemporalConfig,
)


def canonical(zone_shift: float = 0.0, periods: int = 12000) -> pd.DataFrame:
    ts = pd.date_range(
        "2001-01-01 00:05:00",
        periods=periods,
        freq="5min",
    )
    x = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "zone_temperature": 20.0 + zone_shift + 0.001 * x,
            "outdoor_temperature": 5.0 + 0.002 * x,
            "qac": 100.0 + x,
            "qsol1": 10.0 + 0.1 * x,
            "qsol2": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzic_p": 1.0 + 0.01 * x,
            "qzic_l": 2.0 + 0.01 * x,
            "qzic_ee": 3.0 + 0.01 * x,
            "qzic_ge": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzic_oe": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzic_hwe": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzic_se": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzir_p": 4.0 + 0.01 * x,
            "qzir_l": 5.0 + 0.01 * x,
            "qzir_ee": 6.0 + 0.01 * x,
            "qzir_ge": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzir_oe": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzir_hwe": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzir_se": pd.Series([pd.NA] * periods, dtype="Float64"),
            "qzivr_l": 7.0 + 0.01 * x,
            "zic": 6.0 + 0.03 * x,
            "zir": 22.0 + 0.04 * x,
        }
    )


def heat() -> HeatRepresentationConfig:
    return HeatRepresentationConfig(
        HeatInputRepresentation.GROUPED,
        include_visible_lighting_in_qzir=True,
    )


def availability(zone_id: str, table: pd.DataFrame):
    return availability_from_canonical_table(zone_id, table)


def ml_contract(mode, zones, dep2=None, lag=3, horizon=2):
    return SiloProductContract(
        ModelingSilo.ML_SCIML,
        mode,
        TemporalConfig(
            ModelingSilo.ML_SCIML,
            lag,
            horizon,
            "monthly_distributed_holdout",
            {
                "train_fraction": 0.70,
                "test_fraction": 0.15,
                "validation_fraction": 0.15,
            },
        ),
        heat(),
        zones,
        dep2,
    )


def ob_contract(mode, zones, dep2=None, offset=0):
    return SiloProductContract(
        ModelingSilo.OPT_BAYES,
        mode,
        TemporalConfig(
            ModelingSilo.OPT_BAYES,
            1,
            1,
            "seasonal_distributed",
            {
                "season_offset_days": offset,
                "train_days": 21,
                "test_days": 7,
            },
        ),
        heat(),
        zones,
        dep2,
    )


def test_mdh_is_train_test_validation_contiguous_per_month() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2001-03-01 00:00", freq="5min"))
    frame, diag = assign_monthly_distributed_holdout(ts)
    assert diag.parameters["partition_order"] == ["train", "test", "validation"]
    january = frame[(frame["timestamp"] - pd.Timedelta(nanoseconds=1)).dt.month == 1]
    transitions = january["partition"].ne(january["partition"].shift()).sum()
    assert transitions == 3
    assert january["partition"].iloc[0] == "train"
    assert "test" in set(january["partition"])
    assert january["partition"].iloc[-1] == "validation"


def test_mdh_default_fractions_are_70_15_15() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2001-02-01 00:00", freq="5min"))
    frame, _ = assign_monthly_distributed_holdout(ts)
    counts = frame["partition"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.70, abs=0.001)
    assert counts["test"] == pytest.approx(0.15, abs=0.001)
    assert counts["validation"] == pytest.approx(0.15, abs=0.001)


def test_sd_default_selects_21_train_then_7_test_days_per_season() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2002-01-01 00:00", freq="5min"))
    frame, diag = assign_seasonal_distributed(ts)
    assert diag.parameters["season_offset_days"] == 0
    for season in ("winter", "spring", "summer", "fall"):
        selected = frame[(frame["season"] == season) & frame["included"]]
        assert set(selected["partition"]) == {"train", "test"}
        assert selected["window_id"].notna().all()


def test_sd_global_offset_moves_all_season_windows() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2002-01-01 00:00", freq="5min"))
    shifted, diag = assign_seasonal_distributed(ts, season_offset_days=7)
    assert diag.parameters["season_offset_days"] == 7

    # Non-wrapping seasons start exactly seven days later.
    expected_starts = {
        "spring": pd.Timestamp("2001-03-08 00:05"),
        "summer": pd.Timestamp("2001-06-08 00:05"),
        "fall": pd.Timestamp("2001-09-08 00:05"),
    }
    for season, expected in expected_starts.items():
        first = shifted[
            (shifted["season"] == season)
            & (shifted["partition"] == "train")
        ]["timestamp"].iloc[0]
        assert first == expected

    # Winter uses the cyclic meteorological coordinate Dec -> Jan -> Feb.
    # The train block therefore begins on Dec 8 for a seven-day offset.
    winter_december_train = shifted[
        (shifted["season"] == "winter")
        & (shifted["partition"] == "train")
        & (shifted["timestamp"].dt.month == 12)
    ]
    assert winter_december_train["timestamp"].iloc[0] == pd.Timestamp(
        "2001-12-08 00:05"
    )


def test_sd_rejects_window_larger_than_shortest_season() -> None:
    ts = pd.Series(pd.date_range("2001-01-01", periods=10, freq="5min"))
    with pytest.raises(D6ContractError):
        assign_seasonal_distributed(
            ts,
            season_offset_days=70,
            train_days=21,
            test_days=7,
        )


def test_availability_omits_complete_zero_or_not_applicable_nulls() -> None:
    table = canonical(periods=100)
    info = availability("Dining", table)
    assert "qsol1" in info.available_disturbances
    assert "qsol2" not in info.available_disturbances
    assert "qzic_p" in info.available_disturbances
    assert "qzic_ge" not in info.available_disturbances


def test_independent_physical_table_uses_zone_qualified_names() -> None:
    dining = canonical(periods=100)
    d = availability("Dining", dining)
    contract = ml_contract(PhaseDMode.INDEPENDENT, (d,))
    out = build_physical_table(
        contract,
        {"Dining": dining},
        independent_zone_id="Dining",
    )
    assert "Dining__zone_temperature" in out
    assert "Dining__qac" in out
    assert "Dining__zic" in out
    assert list(out).count("outdoor_temperature") == 1


def test_dep1_concatenates_current_zone_state_control_disturbances() -> None:
    dining = canonical(0.0, 100)
    kitchen = canonical(1.0, 100)
    d = availability("Dining", dining)
    k = availability("Kitchen", kitchen)
    contract = ml_contract(PhaseDMode.DEPENDENT1, (d, k))
    out = build_physical_table(contract, {"Dining": dining, "Kitchen": kitchen})
    assert "Dining__zone_temperature" in out
    assert "Kitchen__zone_temperature" in out
    assert "Dining__zic" in out
    assert "Kitchen__zic" in out
    assert list(out).count("outdoor_temperature") == 1


def test_dep2_uses_current_state_control_but_all_to_one_disturbances() -> None:
    dining = canonical(0.0, 100)
    kitchen = canonical(1.0, 100)
    all_zone = canonical(2.0, 100)
    d = availability("Dining", dining)
    k = availability("Kitchen", kitchen)
    a = availability("RestaurantFastFood_All", all_zone)
    contract = ml_contract(PhaseDMode.DEPENDENT2, (d, k), a)
    out = build_physical_table(
        contract,
        {"Dining": dining, "Kitchen": kitchen},
        dependent_2_source_table=all_zone,
    )
    assert "Dining__zone_temperature" in out
    assert "Kitchen__qac" in out
    assert "RestaurantFastFood_All__zic" in out
    assert "Dining__zic" not in out
    assert "Kitchen__zic" not in out


def test_dep1_rejects_outdoor_temperature_disagreement() -> None:
    dining = canonical(0.0, 100)
    kitchen = canonical(1.0, 100)
    kitchen["outdoor_temperature"] += 1.0
    d = availability("Dining", dining)
    k = availability("Kitchen", kitchen)
    with pytest.raises(D6ContractError):
        build_physical_table(
            ml_contract(PhaseDMode.DEPENDENT1, (d, k)),
            {"Dining": dining, "Kitchen": kitchen},
        )


def test_ml_temporal_expansion_excludes_partition_boundary_leakage() -> None:
    dining = canonical(periods=12000)
    d = availability("Dining", dining)
    contract = ml_contract(
        PhaseDMode.INDEPENDENT,
        (d,),
        lag=12,
        horizon=6,
    )
    result = build_final_dataset(
        contract,
        {"Dining": dining},
        independent_zone_id="Dining",
    )
    assert len(result.table) == len(dining)
    assert "Dining__zone_temperature__lag_11" in result.table
    assert "Dining__zone_temperature__target_6" in result.table
    assert (result.table["partition"] == "excluded").any()
    assert not result.table.loc[result.table["included"], contract.final_columns(
        independent_zone_id="Dining"
    )[5].name].isna().any()


def test_opt_bayes_retains_full_axis_but_only_selected_subset_is_included() -> None:
    dining = canonical(periods=105120)
    d = availability("Dining", dining)
    contract = ob_contract(PhaseDMode.INDEPENDENT, (d,))
    result = build_final_dataset(
        contract,
        {"Dining": dining},
        independent_zone_id="Dining",
    )
    assert len(result.table) == 105120
    assert result.table["included"].sum() < len(result.table)
    assert set(result.table["partition"]) == {"train", "test", "excluded"}
    assert result.table.loc[result.table["included"], "window_id"].notna().all()


def test_temporal_expansion_preserves_locked_column_order() -> None:
    dining = canonical(periods=200)
    d = availability("Dining", dining)
    contract = ml_contract(PhaseDMode.INDEPENDENT, (d,))
    physical = build_physical_table(
        contract,
        {"Dining": dining},
        independent_zone_id="Dining",
    )

    expanded, _ = expand_temporal_dataset(
        physical,
        contract,
        independent_zone_id="Dining",
    )

    expected = [
        item.name
        for item in contract.final_columns(independent_zone_id="Dining")
    ]
    assert list(expanded.columns) == expected


def test_temporal_expansion_emits_no_dataframe_fragmentation_warning() -> None:
    dining = canonical(0.0, 500)
    kitchen = canonical(1.0, 500)
    d = availability("Dining", dining)
    k = availability("Kitchen", kitchen)
    contract = ml_contract(PhaseDMode.DEPENDENT1, (d, k))
    physical = build_physical_table(
        contract,
        {"Dining": dining, "Kitchen": kitchen},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expanded, _ = expand_temporal_dataset(physical, contract)

    fragmentation = [
        item for item in caught
        if "highly fragmented" in str(item.message).lower()
    ]
    assert fragmentation == []
    assert len(expanded) == len(physical)


def test_writer_creates_one_parquet_and_manifest_only(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dining = canonical(periods=12000)
    d = availability("Dining", dining)
    contract = ml_contract(PhaseDMode.INDEPENDENT, (d,), lag=1, horizon=1)
    result = build_final_dataset(
        contract,
        {"Dining": dining},
        independent_zone_id="Dining",
    )
    data, manifest = write_final_dataset(
        result,
        silo_root=tmp_path,
        contract=contract,
        independent_zone_id="Dining",
    )
    assert data.is_file()
    assert manifest.is_file()
    assert data.as_posix().endswith(
        "ml/ind/Dining/grp_vrin/l1_h1/mdh/data.parquet"
    )
    files = [p.name for p in data.parent.iterdir() if p.is_file()]
    assert sorted(files) == ["data.parquet", "manifest.json"]



def test_policy_aliases_cover_complete_catalog() -> None:
    assert normalize_policy_name("mdh") == "monthly_distributed_holdout"
    assert normalize_policy_name("ch") == "chronological_holdout"
    assert normalize_policy_name("sh") == "seasonal_holdout"
    assert normalize_policy_name("sd") == "seasonal_distributed"
    assert normalize_policy_name("sbh") == "seasonal_block_holdout"
    assert normalize_policy_name("ci") == "contiguous_identification"
    assert normalize_policy_name("cdr") == "custom_datetime_ranges"


def test_ch_partitions_whole_axis_once_train_test_validation() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", periods=1000, freq="5min"))
    frame, diag = assign_chronological_holdout(ts)
    assert diag.parameters["partition_order"] == ["train", "test", "validation"]
    assert frame["partition"].iloc[0] == "train"
    assert frame["partition"].iloc[-1] == "validation"
    transitions = frame["partition"].ne(frame["partition"].shift()).sum()
    assert transitions == 3
    counts = frame["partition"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.70, abs=0.002)
    assert counts["test"] == pytest.approx(0.15, abs=0.002)
    assert counts["validation"] == pytest.approx(0.15, abs=0.002)


def test_sh_assigns_complete_meteorological_seasons() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2002-01-01 00:00", freq="5min"))
    frame, diag = assign_seasonal_holdout(ts)
    assert diag.parameters["train_seasons"] == ["winter", "spring"]
    expected = {
        "winter": "train",
        "spring": "train",
        "summer": "test",
        "fall": "validation",
    }
    for season, partition in expected.items():
        rows = frame[frame["season"] == season]
        assert set(rows["partition"]) == {partition}
        assert rows["included"].all()


def test_sbh_assigns_whole_season_train_test_blocks() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2002-01-01 00:00", freq="5min"))
    frame, diag = assign_seasonal_block_holdout(
        ts,
        train_seasons=("winter", "spring"),
        test_seasons=("summer",),
    )
    assert diag.parameters["train_seasons"] == ["winter", "spring"]
    assert set(frame.loc[frame["season"] == "winter", "partition"]) == {"train"}
    assert set(frame.loc[frame["season"] == "spring", "partition"]) == {"train"}
    assert set(frame.loc[frame["season"] == "summer", "partition"]) == {"test"}
    assert set(frame.loc[frame["season"] == "fall", "partition"]) == {"excluded"}


def test_ci_selects_one_contiguous_21_day_train_7_day_test_window() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2001-03-01 00:00", freq="5min"))
    frame, diag = assign_contiguous_identification(
        ts,
        start_datetime="2001-01-05 00:05",
        train_days=21,
        test_days=7,
    )
    assert (frame["partition"] == "train").sum() == 21 * 288
    assert (frame["partition"] == "test").sum() == 7 * 288
    assert diag.parameters["start_datetime"].startswith("2001-01-05T00:05")
    assert set(frame.loc[frame["included"], "window_id"]) == {"ci_train_01", "ci_test_01"}


def test_cdr_supports_multiple_explicit_nonoverlapping_ranges() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2001-03-01 00:00", freq="5min"))
    train = [
        "2001-01-01T00:05:00/2001-01-08T00:05:00",
        "2001-02-01T00:05:00/2001-02-08T00:05:00",
    ]
    test = [
        "2001-01-08T00:05:00/2001-01-10T00:05:00",
        "2001-02-08T00:05:00/2001-02-10T00:05:00",
    ]
    frame, diag = assign_custom_datetime_ranges(ts, train_ranges=train, test_ranges=test)
    assert (frame["partition"] == "train").sum() == 14 * 288
    assert (frame["partition"] == "test").sum() == 4 * 288
    assert diag.parameters["range_semantics"] == "half_open_start_inclusive_end_exclusive"
    assert set(frame.loc[frame["included"], "window_id"]) == {
        "cdr_train_01", "cdr_train_02", "cdr_test_01", "cdr_test_02"
    }


def test_cdr_rejects_train_test_overlap() -> None:
    ts = pd.Series(pd.date_range("2001-01-01 00:05", "2001-02-01 00:00", freq="5min"))
    with pytest.raises(D6ContractError, match="cannot overlap"):
        assign_custom_datetime_ranges(
            ts,
            train_ranges=["2001-01-01T00:05:00/2001-01-10T00:05:00"],
            test_ranges=["2001-01-09T00:05:00/2001-01-12T00:05:00"],
        )


def test_availability_allows_structurally_unavailable_qac_without_fabrication() -> None:
    table = canonical(periods=100)
    table["qac"] = pd.Series([pd.NA] * len(table), dtype="Float64")
    avail = availability_from_canonical_table(
        "Corridor",
        table,
        qac_available=False,
    )
    assert avail.qac_available is False


def test_physical_table_omits_structurally_unavailable_qac() -> None:
    table = canonical(periods=100)
    table["qac"] = pd.Series([pd.NA] * len(table), dtype="Float64")
    avail = availability_from_canonical_table(
        "Corridor",
        table,
        qac_available=False,
    )
    contract = ml_contract(PhaseDMode.INDEPENDENT, (avail,), lag=1, horizon=1)
    physical = build_physical_table(
        contract,
        {"Corridor": table},
        independent_zone_id="Corridor",
    )
    assert "Corridor__zone_temperature" in physical.columns
    assert "Corridor__qac" not in physical.columns


def test_availability_still_rejects_missing_qac_when_declared_available() -> None:
    table = canonical(periods=100)
    table.loc[10, "qac"] = np.nan
    with pytest.raises(D6ContractError, match="missing QAC control values"):
        availability_from_canonical_table(
            "Conditioned",
            table,
            qac_available=True,
        )
