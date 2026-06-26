"""Tests for canonical ESO time-series and EIO metadata extraction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scalebridge.integration.energyplus import (
    CanonicalExtractionError,
    EnergyPlusOutputExtractor,
)
from scalebridge.integration.energyplus.outputs.eio import parse_eio


class FakeStandardOutput:
    """Small opyplus StandardOutput stand-in with two requested variables."""

    def __init__(self, *, omit_temperature: bool = False) -> None:
        self.start_year: int | None = None
        self._variables = [
            SimpleNamespace(
                ref="CORE_ZN,Zone Air Temperature",
                key_value="CORE_ZN",
                name="Zone Air Temperature",
                unit="C",
                frequency="timestep",
            ),
            SimpleNamespace(
                ref="PERIMETER_ZN,Zone Air Temperature",
                key_value="PERIMETER_ZN",
                name="Zone Air Temperature",
                unit="C",
                frequency="timestep",
            ),
            SimpleNamespace(
                ref="ENVIRONMENT,Site Outdoor Air Drybulb Temperature",
                key_value="ENVIRONMENT",
                name="Site Outdoor Air Drybulb Temperature",
                unit="C",
                frequency="timestep",
            ),
        ]
        columns = {
            "ENVIRONMENT,Site Outdoor Air Drybulb Temperature": [4.0, 5.0]
        }
        if not omit_temperature:
            columns.update(
                {
                    "CORE_ZN,Zone Air Temperature": [21.0, 21.5],
                    "PERIMETER_ZN,Zone Air Temperature": [20.0, 20.5],
                }
            )
        self._frame = pd.DataFrame(
            columns,
            index=pd.to_datetime(["2013-01-01 00:00", "2013-01-01 00:05"]),
        )

    def create_datetime_index(self, start_year: int) -> None:
        """Capture the requested calendar year."""
        self.start_year = start_year

    def get_environments(self) -> dict[str, object]:
        """Return design-day and run-period environments in simulation order."""
        return {"winter design day": object(), "run period 1": object()}

    def get_variables(self) -> dict[str, list[object]]:
        """Return the fake ESO variable catalog."""
        return {"timestep": self._variables}

    def get_data(
        self,
        environment: str,
        *,
        frequency: str,
    ) -> pd.DataFrame:
        """Return the fake run-period timestep data."""
        assert environment == "run period 1"
        assert frequency == "timestep"
        return self._frame


def _write_raw_outputs(root: Path) -> None:
    """Create minimal ESO and representative EIO files."""
    root.mkdir(parents=True)
    (root / "eplusout.eso").write_text("fake eso\n", encoding="utf-8")
    (root / "eplusout.eio").write_text(
        "Program Version,EnergyPlus, Version 9.0.1\n"
        "! <Environment>,Environment Name,Environment Type,Start Date,End Date\n"
        "Environment,RUNPERIOD 1,WeatherFileRunPeriod,01/01/2013,01/02/2013\n"
        "! <Zone Information>,Zone Name,Floor Area {m2}\n"
        "<Zone Information>,CORE_ZN,100.0\n"
        "<Zone Information>,PERIMETER_ZN,75.0\n",
        encoding="utf-8",
    )


def test_extractor_writes_requested_variables_in_long_form(
    tmp_path: Path,
    case_spec,
) -> None:
    """Wildcard requests must retain each matching zone as a distinct signal."""
    raw_root = tmp_path / "raw"
    canonical_root = tmp_path / "canonical"
    _write_raw_outputs(raw_root)
    written_frames: dict[str, pd.DataFrame] = {}

    def capture_parquet(frame: pd.DataFrame, destination: Path) -> None:
        """Capture canonical data while creating a representative artifact."""
        written_frames[destination.name] = frame.copy()
        destination.write_bytes(b"parquet-placeholder")

    fake_output = FakeStandardOutput()
    result = EnergyPlusOutputExtractor(
        eso_loader=lambda path: fake_output,
        parquet_writer=capture_parquet,
    ).extract(
        case_spec=case_spec,
        simulation_directory=raw_root,
        canonical_directory=canonical_root,
    )

    frame = written_frames["timeseries_timestep.parquet"]
    assert tuple(frame.columns) == (
        "timestamp",
        "environment",
        "reporting_frequency",
        "key_value",
        "variable_name",
        "units",
        "semantic_role",
        "value",
    )
    assert result.produced_signal_count == 3
    assert result.row_count == 6
    assert result.timestep_count == 2
    assert set(frame["key_value"]) == {
        "CORE_ZN",
        "PERIMETER_ZN",
        "ENVIRONMENT",
    }
    assert fake_output.start_year == 2013

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["parquet_files"] == {
        "timestep": "timeseries_timestep.parquet"
    }


def test_extractor_rejects_missing_required_request(
    tmp_path: Path,
    case_spec,
) -> None:
    """A completed simulation is invalid when a required signal is absent."""
    raw_root = tmp_path / "raw"
    _write_raw_outputs(raw_root)

    extractor = EnergyPlusOutputExtractor(
        eso_loader=lambda path: FakeStandardOutput(omit_temperature=True),
        parquet_writer=lambda frame, path: None,
    )

    with pytest.raises(CanonicalExtractionError, match="Zone Air Temperature"):
        extractor.extract(
            case_spec=case_spec,
            simulation_directory=raw_root,
            canonical_directory=tmp_path / "canonical",
        )


def test_parse_eio_preserves_category_columns_and_rows(tmp_path: Path) -> None:
    """The one-pass EIO parser must preserve table shape and quoted values."""
    path = tmp_path / "eplusout.eio"
    path.write_text(
        "! <Zone Information>,Zone Name,Description\n"
        'Zone Information,CORE_ZN,"Core, occupied"\n',
        encoding="utf-8",
    )

    tables = parse_eio(path)

    assert tables["Zone Information"].columns == ("Zone Name", "Description")
    assert tables["Zone Information"].rows == (("CORE_ZN", "Core, occupied"),)
    assert list(tables["Zone Information"].to_dataframe().columns) == [
        "Zone Name",
        "Description",
    ]


def test_parse_eio_accepts_bracketed_data_category(tmp_path: Path) -> None:
    """Bracketed category tokens remain supported for legacy EIO variants."""
    path = tmp_path / "eplusout.eio"
    path.write_text(
        "! <Zone Information>,Zone Name\n"
        "<Zone Information>,CORE_ZN\n",
        encoding="utf-8",
    )

    tables = parse_eio(path)

    assert tables["Zone Information"].rows == (("CORE_ZN",),)


def test_parse_eio_normalizes_short_and_empty_overflow_rows(
    tmp_path: Path,
) -> None:
    """Rows must match declared widths after safe padding and trimming."""
    path = tmp_path / "eplusout.eio"
    path.write_text(
        "! <Example>,First,Second\n"
        "Example,one\n"
        "Example,two,three,\n",
        encoding="utf-8",
    )

    table = parse_eio(path)["Example"]

    assert table.columns == ("First", "Second")
    assert table.rows == (("one", ""), ("two", "three"))


def test_parse_eio_preserves_nonempty_undeclared_fields(tmp_path: Path) -> None:
    """Non-empty overflow values must receive explicit generated columns."""
    path = tmp_path / "eplusout.eio"
    path.write_text(
        "! <Example>,First\n"
        "Example,one,important\n",
        encoding="utf-8",
    )

    table = parse_eio(path)["Example"]

    assert table.columns == ("First", "Undeclared Field 1")
    assert table.rows == (("one", "important"),)


def test_extractor_rejects_eio_calendar_year_mismatch(
    tmp_path: Path,
    case_spec,
) -> None:
    """Reported EIO years must agree with the deterministic case year."""
    raw_root = tmp_path / "raw"
    _write_raw_outputs(raw_root)
    eio_path = raw_root / "eplusout.eio"
    eio_path.write_text(
        eio_path.read_text(encoding="utf-8").replace("2013", "2017"),
        encoding="utf-8",
    )

    extractor = EnergyPlusOutputExtractor(
        eso_loader=lambda path: FakeStandardOutput(),
        parquet_writer=lambda frame, path: path.write_bytes(b"placeholder"),
    )

    with pytest.raises(CanonicalExtractionError, match="does not match"):
        extractor.extract(
            case_spec=case_spec,
            simulation_directory=raw_root,
            canonical_directory=tmp_path / "canonical",
        )
