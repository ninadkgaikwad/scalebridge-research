from __future__ import annotations

import json
from pathlib import Path

import pytest

from scalebridge.dashapp.services.aggregation import upstream_generation as ug


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _eio_payload(*, included=("ZONE A", "ZONE B"), excluded=("PLENUM",)) -> dict:
    rows = []
    for name in included:
        rows.append([name, "0", "0", "0", "0", "Yes", "3", "100", "40"])
    for name in excluded:
        rows.append([name, "0", "0", "0", "0", "No", "2", "25", "10"])
    return {
        "tables": {
            "Zone Information": {
                "columns": [
                    "Zone Name",
                    "North Axis {deg}",
                    "Origin X-Coordinate {m}",
                    "Origin Y-Coordinate {m}",
                    "Origin Z-Coordinate {m}",
                    "Part of Total Building Area",
                    "Ceiling Height {m}",
                    "Volume {m3}",
                    "Floor Area {m2}",
                ],
                "rows": rows,
            }
        }
    }


def _make_case(
    root: Path,
    *,
    campaign_id: str,
    case_id: str,
    run_id: str,
    status: str,
    building_type: str = "OfficeSmall",
    weather_location: str = "Seattle",
    climate_zone: str = "4C",
    with_eio: bool = True,
) -> Path:
    case_root = root / "campaigns" / campaign_id / "generation" / "cases" / case_id
    run_root = case_root / "runs" / run_id
    manifest_rel = f"runs/{run_id}/run_manifest.json"

    _write_json(
        case_root / "latest_run.json",
        {
            "case_id": case_id,
            "run_id": run_id,
            "status": status,
            "manifest_path": manifest_rel,
        },
    )
    _write_json(
        run_root / "run_manifest.json",
        {
            "case_id": case_id,
            "run_id": run_id,
            "campaign_id": campaign_id,
            "status": status,
            "case_spec": {
                "building_type": building_type,
                "prototype_standard": "ASHRAE",
                "prototype_year": "2013",
                "weather_location": weather_location,
                "climate_zone": climate_zone,
                "tags": {},
            },
            "execution": {
                "machine_id": "labpc",
                "hostname": "TESTHOST",
            },
            "validation": {
                "warnings": 2 if status == "completed_with_warnings" else 0,
                "severe_errors": 0,
                "fatal_errors": 0,
                "requested_signals": 30,
                "produced_signals": 29,
            },
        },
    )
    if with_eio:
        _write_json(run_root / "canonical" / "eio_tables.json", _eio_payload())
    return case_root


@pytest.fixture
def generated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ug, "resolve_generated_data_root", lambda: tmp_path)
    return tmp_path


def test_parent_campaign_options_mirror_generation_campaign_discovery(
    generated_root: Path,
) -> None:
    _make_case(
        generated_root,
        campaign_id="campaign_b",
        case_id="case_b",
        run_id="run_b",
        status="completed",
    )
    _make_case(
        generated_root,
        campaign_id="campaign_a",
        case_id="case_a",
        run_id="run_a",
        status="completed",
    )
    (generated_root / "campaigns" / "not_generation").mkdir(parents=True)

    assert ug.parent_campaign_options() == [
        {"label": "campaign_a", "value": "campaign_a"},
        {"label": "campaign_b", "value": "campaign_b"},
    ]


def test_discovery_accepts_completed_and_completed_with_warnings_only(
    generated_root: Path,
) -> None:
    for case_id, status in (
        ("case_completed", "completed"),
        ("case_warnings", "completed_with_warnings"),
        ("case_failed", "failed"),
    ):
        _make_case(
            generated_root,
            campaign_id="parent",
            case_id=case_id,
            run_id=f"run_{case_id}",
            status=status,
        )

    result = ug.discover_generation_cases("parent", include_zone_inventory=False)

    assert {row["case_id"] for row in result["cases"]} == {
        "case_completed",
        "case_warnings",
    }
    assert {row["status"] for row in result["cases"]} == {
        "completed",
        "completed_with_warnings",
    }


def test_discovery_extracts_generation_lineage_metadata_and_zone_inventory(
    generated_root: Path,
) -> None:
    _make_case(
        generated_root,
        campaign_id="parent",
        case_id="case_1",
        run_id="run_1",
        status="completed_with_warnings",
        building_type="RestaurantFastFood",
        weather_location="Buffalo",
        climate_zone="5A",
    )

    result = ug.discover_generation_cases("parent")
    assert result["issues"] == []
    assert len(result["cases"]) == 1

    row = result["cases"][0]
    assert row["parent_generation_campaign_id"] == "parent"
    assert row["case_id"] == "case_1"
    assert row["run_id"] == "run_1"
    assert row["status"] == "completed_with_warnings"
    assert row["building_type"] == "RestaurantFastFood"
    assert row["weather_location"] == "Buffalo"
    assert row["climate_zone"] == "5A"
    assert row["machine_id"] == "labpc"
    assert row["warning_count"] == 2
    assert row["zone_inventory_status"] == "available"
    assert row["thermal_zone_count"] == 2
    assert row["excluded_zone_count"] == 1
    assert row["thermal_zone_names"] == ["ZONE A", "ZONE B"]
    assert row["eio_tables_path"].endswith("canonical/eio_tables.json") or row[
        "eio_tables_path"
    ].endswith(r"canonical\eio_tables.json")


def test_missing_zone_inventory_does_not_hide_eligible_generation_run(
    generated_root: Path,
) -> None:
    _make_case(
        generated_root,
        campaign_id="parent",
        case_id="case_1",
        run_id="run_1",
        status="completed",
        with_eio=False,
    )

    result = ug.discover_generation_cases("parent")
    assert len(result["cases"]) == 1
    assert result["cases"][0]["zone_inventory_status"] == "missing"
    assert result["cases"][0]["thermal_zone_count"] is None
    assert any(issue["code"] == "zone_inventory_source_missing" for issue in result["issues"])


def test_missing_latest_run_is_reported_without_breaking_other_cases(
    generated_root: Path,
) -> None:
    _make_case(
        generated_root,
        campaign_id="parent",
        case_id="case_good",
        run_id="run_good",
        status="completed",
    )
    bad = (
        generated_root
        / "campaigns"
        / "parent"
        / "generation"
        / "cases"
        / "case_missing_latest"
    )
    bad.mkdir(parents=True)

    result = ug.discover_generation_cases("parent", include_zone_inventory=False)

    assert [row["case_id"] for row in result["cases"]] == ["case_good"]
    assert any(
        issue["case_id"] == "case_missing_latest"
        and issue["code"] == "generation_input_unavailable"
        for issue in result["issues"]
    )


def test_parent_campaign_summary_and_facets_are_filterable(
    generated_root: Path,
) -> None:
    _make_case(
        generated_root,
        campaign_id="parent",
        case_id="case_a",
        run_id="run_a",
        status="completed",
        building_type="OfficeSmall",
        weather_location="Seattle",
        climate_zone="4C",
    )
    _make_case(
        generated_root,
        campaign_id="parent",
        case_id="case_b",
        run_id="run_b",
        status="completed_with_warnings",
        building_type="RestaurantFastFood",
        weather_location="Buffalo",
        climate_zone="5A",
    )

    campaigns = ug.discover_parent_campaigns()
    assert len(campaigns) == 1
    summary = campaigns[0]
    assert summary["eligible_case_count"] == 2
    assert summary["building_types"] == ["OfficeSmall", "RestaurantFastFood"]
    assert summary["weather_locations"] == ["Buffalo", "Seattle"]
    assert summary["climate_zones"] == ["4C", "5A"]

    discovered = ug.discover_generation_cases(
        "parent",
        include_zone_inventory=False,
    )["cases"]
    selected = ug.filter_generation_cases(
        discovered,
        building_types=["OfficeSmall"],
        climate_zones=["4C"],
    )
    assert [row["case_id"] for row in selected] == ["case_a"]

    facets = ug.selection_facets(discovered)
    assert facets["case_ids"] == ["case_a", "case_b"]
    assert facets["statuses"] == ["completed", "completed_with_warnings"]


def test_missing_parent_campaign_is_graceful(generated_root: Path) -> None:
    result = ug.discover_generation_cases("does_not_exist")
    assert result["cases"] == []
    assert result["issues"][0]["code"] == "generation_cases_root_not_found"
