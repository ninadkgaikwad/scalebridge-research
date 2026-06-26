"""Tests for backend-independent EnergyPlus IDF preparation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scalebridge.integration.energyplus import (
    CaseSpec,
    IdfPreparationError,
    IdfPreparer,
    OutputVariableRequest,
    RunPeriod,
    ScheduleOperation,
)


class FakeRecord(dict[str, Any]):
    """Mutable in-memory IDF record used by the fake backend."""


class FakeIdfBackend:
    """In-memory implementation of the IDF backend protocol for unit tests."""

    def __init__(self, source_model: dict[str, list[FakeRecord]]) -> None:
        self.source_model = source_model
        self.active_model: dict[str, list[FakeRecord]] = {}
        self.saved_model: dict[str, list[FakeRecord]] | None = None

    def load(self, source_path: Path) -> dict[str, list[FakeRecord]]:
        """Return an isolated model copy so source records remain unchanged."""
        self.active_model = deepcopy(self.source_model)
        return self.active_model

    def save(
        self,
        model: dict[str, list[FakeRecord]],
        destination_path: Path,
    ) -> None:
        """Capture the prepared model and emulate an IDF file write."""
        self.saved_model = deepcopy(model)
        destination_path.write_text("prepared IDF\n", encoding="utf-8")

    def list_records(
        self,
        model: dict[str, list[FakeRecord]],
        object_type: str,
    ) -> list[FakeRecord]:
        """Return records for one object type."""
        return list(model.setdefault(object_type, []))

    def find_named_record(
        self,
        model: dict[str, list[FakeRecord]],
        object_type: str,
        record_name: str,
    ) -> FakeRecord | None:
        """Find one record by case-insensitive name."""
        matches = [
            record
            for record in model.setdefault(object_type, [])
            if str(record.get("name", "")).casefold() == record_name.casefold()
        ]
        if len(matches) > 1:
            raise RuntimeError("duplicate fake records")
        return matches[0] if matches else None

    def add_record(
        self,
        model: dict[str, list[FakeRecord]],
        object_type: str,
        fields: Mapping[str, Any],
    ) -> FakeRecord:
        """Append one record to the requested object table."""
        record = FakeRecord(fields)
        model.setdefault(object_type, []).append(record)
        return record

    def update_record(
        self,
        record: FakeRecord,
        fields: Mapping[str, Any],
    ) -> None:
        """Update one fake record."""
        record.update(fields)

    def delete_record(self, record: FakeRecord) -> None:
        """Remove a record from whichever active table contains it."""
        for records in self.active_model.values():
            if record in records:
                records.remove(record)
                return
        raise RuntimeError("record was not present in active fake model")


@pytest.fixture
def source_model() -> dict[str, list[FakeRecord]]:
    """Return a representative legacy IDF object collection."""
    return {
        "RunPeriod": [
            FakeRecord(
                begin_month=1,
                begin_day_of_month=1,
                begin_year="",
                day_of_week_for_start_day="Sunday",
                end_month=12,
                end_day_of_month=31,
                end_year="",
            )
        ],
        "TimeStep": [FakeRecord(number_of_timesteps_per_hour=4)],
        "Schedule:Compact": [
            FakeRecord(
                name="BLDG_LIGHT_SCH",
                schedule_type_limits_name="Fraction",
                field_1="Through: 12/31",
                field_2="For: AllDays",
                field_3="Until: 24:00,0.5",
            ),
            FakeRecord(name="DELETE_ME", field_1="Through: 12/31"),
        ],
        "Output:Variable": [
            FakeRecord(
                key_value="*",
                variable_name="Legacy Variable",
                reporting_frequency="hourly",
            )
        ],
        "Output:VariableDictionary": [
            FakeRecord(key_field="regular", sort_option="Name")
        ],
    }


@pytest.fixture
def case_spec(tmp_path: Path) -> CaseSpec:
    """Return a case containing each supported schedule operation."""
    source_idf = tmp_path / "source.idf"
    source_idf.write_text("source IDF\n", encoding="utf-8")

    return CaseSpec(
        case_name="idf_preparation_test",
        idf_path=source_idf,
        epw_path=tmp_path / "weather.epw",
        idf_sha256="a" * 64,
        epw_sha256="b" * 64,
        run_period=RunPeriod(
            start_month=3,
            start_day=2,
            end_month=11,
            end_day=30,
            calendar_year=2013,
        ),
        timestep_minutes=5,
        output_variables=(
            OutputVariableRequest(variable_name="Zone Air Temperature"),
            OutputVariableRequest(
                variable_name="Site Outdoor Air Drybulb Temperature",
                key_value="Environment",
                reporting_frequency="hourly",
            ),
        ),
        schedule_operations=(
            ScheduleOperation(
                operation="replace_fields",
                schedule_name="BLDG_LIGHT_SCH",
                fields={"field_3": "Until: 24:00,0.7"},
            ),
            ScheduleOperation(
                operation="rename",
                schedule_name="BLDG_LIGHT_SCH",
                fields={"name": "BLDG_LIGHT_SCH_UPDATED"},
            ),
            ScheduleOperation(operation="delete", schedule_name="DELETE_ME"),
            ScheduleOperation(
                operation="add",
                schedule_name="NEW_EQUIPMENT_SCH",
                fields={
                    "schedule_type_limits_name": "Fraction",
                    "field_1": "Through: 12/31",
                    "field_2": "For: AllDays",
                    "field_3": "Until: 24:00,0.4",
                },
            ),
        ),
    )


def test_preparer_applies_complete_case_configuration(
    tmp_path: Path,
    source_model: dict[str, list[FakeRecord]],
    case_spec: CaseSpec,
) -> None:
    """Preparation must apply run, schedule, and all output requirements."""
    backend = FakeIdfBackend(source_model)
    destination = tmp_path / "prepared.idf"

    result = IdfPreparer(backend).prepare(case_spec, destination)

    assert backend.saved_model is not None
    prepared = backend.saved_model
    assert prepared["RunPeriod"][0]["begin_month"] == 3
    assert prepared["RunPeriod"][0]["begin_day_of_month"] == 2
    assert prepared["RunPeriod"][0]["begin_year"] == 2013
    assert prepared["RunPeriod"][0]["day_of_week_for_start_day"] == "Saturday"
    assert prepared["RunPeriod"][0]["end_month"] == 11
    assert prepared["RunPeriod"][0]["end_day_of_month"] == 30
    assert prepared["RunPeriod"][0]["end_year"] == 2013
    assert prepared["TimeStep"][0]["number_of_timesteps_per_hour"] == 12

    schedules = {record["name"]: record for record in prepared["Schedule:Compact"]}
    assert "DELETE_ME" not in schedules
    assert schedules["BLDG_LIGHT_SCH_UPDATED"]["field_3"] == "Until: 24:00,0.7"
    assert schedules["NEW_EQUIPMENT_SCH"]["field_3"] == "Until: 24:00,0.4"

    assert [record["variable_name"] for record in prepared["Output:Variable"]] == [
        "Zone Air Temperature",
        "Site Outdoor Air Drybulb Temperature",
    ]
    assert prepared["Output:Variable"][1]["key_value"] == "Environment"
    assert prepared["Output:Variable"][1]["reporting_frequency"] == "hourly"
    assert len(prepared["Output:VariableDictionary"]) == 1

    assert result.prepared_idf_path == destination.resolve()
    assert result.output_variable_count == 2
    assert result.schedule_operation_count == 4
    assert result.timestep_per_hour == 12


def test_preparer_does_not_mutate_source_model(
    tmp_path: Path,
    source_model: dict[str, list[FakeRecord]],
    case_spec: CaseSpec,
) -> None:
    """Preparing an IDF must leave the source model representation unchanged."""
    original = deepcopy(source_model)
    backend = FakeIdfBackend(source_model)

    IdfPreparer(backend).prepare(case_spec, tmp_path / "prepared.idf")

    assert source_model == original


def test_preparer_can_disable_variable_dictionary(
    tmp_path: Path,
    source_model: dict[str, list[FakeRecord]],
    case_spec: CaseSpec,
) -> None:
    """A case may explicitly remove variable-dictionary output."""
    backend = FakeIdfBackend(source_model)
    changed_case = case_spec.model_copy(update={"request_variable_dictionary": False})

    result = IdfPreparer(backend).prepare(changed_case, tmp_path / "prepared.idf")

    assert backend.saved_model is not None
    assert backend.saved_model["Output:VariableDictionary"] == []
    assert not result.variable_dictionary_requested


def test_preparer_rejects_source_as_destination(case_spec: CaseSpec) -> None:
    """The preparer must never overwrite the original source IDF."""
    backend = FakeIdfBackend({})

    with pytest.raises(IdfPreparationError, match="must differ"):
        IdfPreparer(backend).prepare(case_spec, case_spec.idf_path)


def test_preparer_requires_existing_schedule(
    tmp_path: Path,
    source_model: dict[str, list[FakeRecord]],
    case_spec: CaseSpec,
) -> None:
    """Update, rename, and delete operations must target existing schedules."""
    backend = FakeIdfBackend(source_model)
    changed_case = case_spec.model_copy(
        update={
            "schedule_operations": (
                ScheduleOperation(
                    operation="replace_fields",
                    schedule_name="MISSING_SCHEDULE",
                    fields={"field_1": "Through: 12/31"},
                ),
            )
        }
    )

    with pytest.raises(IdfPreparationError, match="was not found"):
        IdfPreparer(backend).prepare(changed_case, tmp_path / "prepared.idf")


def test_preparer_requires_single_run_period(
    tmp_path: Path,
    source_model: dict[str, list[FakeRecord]],
    case_spec: CaseSpec,
) -> None:
    """Ambiguous source run-period configuration must fail explicitly."""
    source_model["RunPeriod"].append(FakeRecord(begin_month=2))
    backend = FakeIdfBackend(source_model)

    with pytest.raises(IdfPreparationError, match="exactly one RunPeriod"):
        IdfPreparer(backend).prepare(case_spec, tmp_path / "prepared.idf")
