"""Tests for the locked 64-case P1 EnergyPlus campaign."""

from __future__ import annotations

import json
from pathlib import Path

from scalebridge.integration.energyplus.p1 import (
    P1_BUILDING_TYPES,
    P1_CLIMATES,
    P1_REQUIRED_VARIABLE_NAMES,
    build_p1_case_specs,
    p1_output_variables,
    write_p1_campaign_manifest,
)
from scalebridge.integration.energyplus.prototypes import (
    COMMERCIAL_TMY3_BY_LOCATION,
)


def _write_p1_source_library(root: Path) -> None:
    """Create all prototype and EPW files required by the P1 builder."""
    prototype_root = root / "Commercial_Prototypes" / "ASHRAE" / "90_1_2013"
    weather_root = root / "TMY3_WeatherFiles_Commercial"
    prototype_root.mkdir(parents=True)
    weather_root.mkdir(parents=True)

    for location in P1_CLIMATES:
        epw_name = COMMERCIAL_TMY3_BY_LOCATION[location]
        (weather_root / epw_name).write_text(
            f"LOCATION,{location}\n",
            encoding="utf-8",
        )
        for building_type in P1_BUILDING_TYPES:
            idf_path = (
                prototype_root
                / f"ASHRAE901_{building_type}_STD2013_{location}.idf"
            )
            idf_path.write_text(
                f"! WeatherFile: {epw_name}\nVersion,9.0;\n",
                encoding="utf-8",
            )


def test_p1_required_variable_registry_is_locked() -> None:
    """The factory must include every supplied legacy P1 variable."""
    requests = p1_output_variables()

    assert len(P1_REQUIRED_VARIABLE_NAMES) == 35
    assert len(requests) == 35
    assert all(request.required for request in requests)
    assert all(request.reporting_frequency == "timestep" for request in requests)
    assert "Zone Lights Convective Heating Rate" in {
        request.variable_name for request in requests
    }


def test_build_p1_case_specs_creates_64_annual_cases(tmp_path: Path) -> None:
    """All 16 buildings and four locked climates must form annual cases."""
    _write_p1_source_library(tmp_path)

    cases = build_p1_case_specs(external_data_root=tmp_path)

    assert len(cases) == 64
    assert {(case.building_type, case.weather_location) for case in cases} == {
        (building, location)
        for building in P1_BUILDING_TYPES
        for location in P1_CLIMATES
    }
    assert all(case.timestep_minutes == 5 for case in cases)
    assert all(case.run_period.calendar_year == 2013 for case in cases)
    assert all(case.run_period.start_month == 1 for case in cases)
    assert all(case.run_period.end_month == 12 for case in cases)
    assert all(len(case.output_variables) == 35 for case in cases)


def test_write_p1_campaign_manifest_persists_json_and_csv(
    tmp_path: Path,
) -> None:
    """Campaign construction must be inspectable before distributed runs."""
    source_root = tmp_path / "Data"
    _write_p1_source_library(source_root)
    cases = build_p1_case_specs(external_data_root=source_root)

    result = write_p1_campaign_manifest(
        cases,
        generated_data_root=tmp_path / "generated",
    )

    assert result.case_count == 64
    assert result.json_path.is_file()
    assert result.csv_path.is_file()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["case_count"] == 64
    assert payload["cases"][0]["output_variable_count"] == 35
