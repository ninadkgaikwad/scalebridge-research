"""Tests for PNNL commercial prototype and weather inventory generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scalebridge.integration.energyplus.prototypes as prototype_module
from scalebridge.integration.energyplus.prototypes import (
    PrototypeInventoryError,
    build_and_write_pnnl_inventory,
    extract_idf_metadata,
    resolve_external_data_root,
    scan_pnnl_commercial_prototypes,
)


def _write_prototype(
    root: Path,
    *,
    building_type: str = "OfficeSmall",
    location: str = "Seattle",
    version: str = "9.0",
    weather_filename: str = "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
) -> Path:
    """Create one minimal PNNL-style IDF fixture."""
    prototype_root = root / "Commercial_Prototypes" / "ASHRAE" / "90_1_2013"
    prototype_root.mkdir(parents=True, exist_ok=True)
    idf_path = (
        prototype_root
        / f"ASHRAE901_{building_type}_STD2013_{location}.idf"
    )
    idf_path.write_text(
        f"! WeatherFile: {weather_filename}\n\nVersion,{version};\n",
        encoding="utf-8",
    )
    return idf_path


def _write_seattle_weather(root: Path) -> Path:
    """Create the explicitly registered Seattle EPW fixture."""
    weather_root = root / "TMY3_WeatherFiles_Commercial"
    weather_root.mkdir(parents=True, exist_ok=True)
    epw_path = weather_root / "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw"
    epw_path.write_text("LOCATION,Seattle\n", encoding="utf-8")
    return epw_path


def test_extract_idf_metadata_reads_version_and_weather(tmp_path: Path) -> None:
    """Metadata extraction must read the native IDF schema and PNNL comment."""
    idf_path = _write_prototype(tmp_path)

    version, weather = extract_idf_metadata(idf_path)

    assert version == "9.0"
    assert weather == "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw"


def test_inventory_marks_complete_case_eligible(tmp_path: Path) -> None:
    """A valid 9.0 prototype with matching registered EPW must be eligible."""
    _write_prototype(tmp_path)
    _write_seattle_weather(tmp_path)

    records = scan_pnnl_commercial_prototypes(external_data_root=tmp_path)

    assert len(records) == 1
    assert records[0].status == "eligible"
    assert records[0].issues == ()
    assert records[0].idf_sha256
    assert records[0].epw_sha256


def test_inventory_marks_unregistered_location_missing_weather(tmp_path: Path) -> None:
    """International locations without registry entries must not be eligible."""
    _write_prototype(
        tmp_path,
        location="Dubai",
        weather_filename="ARE_Dubai.epw",
    )
    (tmp_path / "TMY3_WeatherFiles_Commercial").mkdir(parents=True)

    records = scan_pnnl_commercial_prototypes(external_data_root=tmp_path)

    assert records[0].status == "missing_weather"
    assert "no commercial EPW registry entry for Dubai" in records[0].issues


def test_inventory_marks_version_mismatch(tmp_path: Path) -> None:
    """Non-native prototype versions must be excluded from a 9.0 campaign."""
    _write_prototype(tmp_path, version="9.1")
    _write_seattle_weather(tmp_path)

    records = scan_pnnl_commercial_prototypes(external_data_root=tmp_path)

    assert records[0].status == "version_mismatch"


def test_inventory_marks_weather_reference_mismatch(tmp_path: Path) -> None:
    """An IDF comment inconsistent with the explicit registry must be flagged."""
    _write_prototype(tmp_path, weather_filename="wrong.epw")
    _write_seattle_weather(tmp_path)

    records = scan_pnnl_commercial_prototypes(external_data_root=tmp_path)

    assert records[0].status == "weather_reference_mismatch"


def test_build_inventory_writes_csv_and_json(tmp_path: Path) -> None:
    """Inventory artifacts must include records and status summaries."""
    external_root = tmp_path / "external"
    generated_root = tmp_path / "generated"
    _write_prototype(external_root)
    _write_seattle_weather(external_root)

    result = build_and_write_pnnl_inventory(
        external_data_root=external_root,
        generated_data_root=generated_root,
    )

    assert result.record_count == 1
    assert result.status_counts == {"eligible": 1}
    assert result.csv_path.is_file()
    assert result.json_path.is_file()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 1
    assert payload["records"][0]["status"] == "eligible"


def test_inventory_requires_configured_external_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing explicit, environment, and discovered roots must raise."""
    monkeypatch.delenv("SCALEBRIDGE_EXTERNAL_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        prototype_module,
        "_discover_repository_relative_data_root",
        lambda: None,
    )

    with pytest.raises(PrototypeInventoryError, match="is not set"):
        scan_pnnl_commercial_prototypes()


def test_external_root_uses_repository_relative_data_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The shared repository-relative Data directory is the final fallback."""
    monkeypatch.delenv("SCALEBRIDGE_EXTERNAL_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        prototype_module,
        "_discover_repository_relative_data_root",
        lambda: tmp_path,
    )

    assert resolve_external_data_root() == tmp_path.resolve()
