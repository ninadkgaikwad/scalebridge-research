"""Shared test fixtures for EnergyPlus manifest contracts."""

from __future__ import annotations

import pytest

from scalebridge.integration.energyplus import (
    CaseSpec,
    OutputVariableRequest,
    RunPeriod,
)


@pytest.fixture
def case_spec() -> CaseSpec:
    """Return a representative five-minute P1 EnergyPlus case."""
    return CaseSpec(
        case_name="small_office_seattle",
        building_type="OfficeSmall",
        prototype_standard="ASHRAE 90.1",
        prototype_year="2013",
        weather_location="Seattle",
        idf_path="inputs/model.idf",
        epw_path="inputs/weather.epw",
        idf_sha256="a" * 64,
        epw_sha256="b" * 64,
        run_period=RunPeriod(
            start_month=1,
            start_day=1,
            end_month=12,
            end_day=31,
            calendar_year=2013,
        ),
        timestep_minutes=5,
        output_variables=(
            OutputVariableRequest(variable_name="Zone Air Temperature"),
            OutputVariableRequest(
                variable_name="Site Outdoor Air Drybulb Temperature",
                semantic_role="outdoor_temperature",
            ),
        ),
        tags={"paper": "P1"},
    )
