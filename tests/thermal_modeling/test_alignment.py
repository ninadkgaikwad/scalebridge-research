import pandas as pd
import pytest

from scalebridge.data.thermal_modeling.alignment import (
    PhaseDAlignmentError,
    TimestampNormalizationConfig,
    align_sources,
    clean_phase_b,
    parse_energyplus_timestamp,
    rewrite_placeholder_year,
)


def test_energyplus_24_hour_rollover():
    assert parse_energyplus_timestamp(
        "12/31  24:00:00", 2001
    ) == pd.Timestamp("2002-01-01 00:00:00")


def test_placeholder_year_rewrite_preserves_annual_rollover():
    source = pd.Series(
        ["2001-01-01 00:05:00", "2002-01-01 00:00:00"]
    )
    output = rewrite_placeholder_year(source, 2013)
    assert list(output) == [
        pd.Timestamp("2013-01-01 00:05:00"),
        pd.Timestamp("2014-01-01 00:00:00"),
    ]


def test_null_duplicate_remnant_is_removed_and_sources_align():
    phase_b = pd.DataFrame(
        {
            "timestamp_raw": [
                "01/01  00:05:00",
                "01/01  00:05:00",
                "01/01  00:10:00",
            ],
            "Zone_Air_Temperature_": [20.0, None, 21.0],
            "Site_Outdoor_Air_Drybulb_Temperature_": [5.0, None, 6.0],
        }
    )
    phase_c = pd.DataFrame(
        {
            "timestamp": [
                "2001-01-01 00:05:00",
                "2001-01-01 00:10:00",
            ],
            "predicted_QAC": [1.0, 2.0],
        }
    )
    splits = pd.DataFrame(
        {
            "timestamp": [
                "2001-01-01 00:05:00",
                "2001-01-01 00:10:00",
            ],
            "split": ["train", "test"],
        }
    )
    aligned, diagnostics = align_sources(
        phase_b,
        phase_c,
        splits,
        TimestampNormalizationConfig(2013),
    )
    assert len(aligned) == 2
    assert diagnostics.phase_b_simple_null_remnant_groups == 1
    assert diagnostics.phase_b_duplicate_rows_removed == 1
    assert aligned.timestamp.iloc[0] == pd.Timestamp("2013-01-01 00:05:00")


def test_complementary_duplicate_rows_are_coalesced():
    phase_b = pd.DataFrame(
        {
            "timestamp_raw": [
                "03/01  23:25:00",
                "03/01  23:25:00",
            ],
            "Zone_Air_Temperature_": [19.99865, None],
            "Site_Outdoor_Air_Drybulb_Temperature_": [None, 1.35],
        }
    )
    cleaned, diagnostics = clean_phase_b(
        phase_b,
        TimestampNormalizationConfig(),
    )
    assert len(cleaned) == 1
    assert diagnostics["complementary_duplicate_groups_merged"] == 1
    assert diagnostics["conflicting_duplicate_groups"] == 0
    assert cleaned.loc[0, "Zone_Air_Temperature_"] == pytest.approx(19.99865)
    assert cleaned.loc[0, "Site_Outdoor_Air_Drybulb_Temperature_"] == pytest.approx(1.35)


def test_identical_duplicate_rows_are_collapsed():
    phase_b = pd.DataFrame(
        {
            "timestamp_raw": ["01/01  00:05:00"] * 2,
            "Zone_Air_Temperature_": [20.0, 20.0],
            "Site_Outdoor_Air_Drybulb_Temperature_": [5.0, 5.0],
        }
    )
    cleaned, diagnostics = clean_phase_b(
        phase_b,
        TimestampNormalizationConfig(),
    )
    assert len(cleaned) == 1
    assert diagnostics["identical_duplicate_groups_collapsed"] == 1


def test_conflicting_nonnull_duplicates_fail():
    phase_b = pd.DataFrame(
        {
            "timestamp_raw": ["01/01  00:05:00"] * 2,
            "Zone_Air_Temperature_": [20.0, 21.0],
            "Site_Outdoor_Air_Drybulb_Temperature_": [5.0, 5.0],
        }
    )
    with pytest.raises(PhaseDAlignmentError, match="conflicting"):
        clean_phase_b(phase_b, TimestampNormalizationConfig())


def test_complementary_group_missing_required_value_fails():
    phase_b = pd.DataFrame(
        {
            "timestamp_raw": ["01/01  00:05:00"] * 2,
            "Zone_Air_Temperature_": [20.0, None],
            "Site_Outdoor_Air_Drybulb_Temperature_": [None, None],
        }
    )
    with pytest.raises(PhaseDAlignmentError, match="incomplete"):
        clean_phase_b(phase_b, TimestampNormalizationConfig())


def test_leap_year_rejected_for_current_source_shape():
    with pytest.raises(ValueError, match="non-leap"):
        TimestampNormalizationConfig(2024)
