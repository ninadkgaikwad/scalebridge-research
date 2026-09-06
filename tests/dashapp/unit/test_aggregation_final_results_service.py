from __future__ import annotations

import csv
import json
from pathlib import Path

from scalebridge.dashapp.services.aggregation import results_data


def test_rule_catalog_preserves_variable_to_output_column_mapping(tmp_path):
    path = tmp_path / "rule_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "aggregate_zone_id",
                "source_variable_name",
                "output_variable_name",
                "rule_family",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "aggregate_zone_id": "Zone1",
                "source_variable_name": "Schedule Value",
                "output_variable_name": "Schedule_Value_People",
                "rule_family": "Schedule",
                "status": "aggregated",
            }
        )
        writer.writerow(
            {
                "aggregate_zone_id": "Zone1",
                "source_variable_name": "Schedule Value",
                "output_variable_name": "Schedule_Value_Lights",
                "rule_family": "Schedule",
                "status": "aggregated",
            }
        )
        writer.writerow(
            {
                "aggregate_zone_id": "Zone1",
                "source_variable_name": "Schedule Value",
                "output_variable_name": "Schedule_Value_GasEquipment",
                "rule_family": "Schedule",
                "status": "warning",
            }
        )

    values = results_data._rule_catalog(str(path.resolve()), path.stat().st_mtime_ns)
    assert ("Zone1", "Schedule Value", "Schedule_Value_People") in values
    assert ("Zone1", "Schedule Value", "Schedule_Value_Lights") in values
    assert ("Zone1", "Schedule Value", "Schedule_Value_GasEquipment") not in values


def test_variable_column_options_are_filtered_by_selected_variable():
    catalog = [
        {
            "aggregate_zone": "Zone1",
            "variable": "Schedule Value",
            "variable_column": "Schedule_Value_People",
        },
        {
            "aggregate_zone": "Zone1",
            "variable": "Schedule Value",
            "variable_column": "Schedule_Value_Lights",
        },
        {
            "aggregate_zone": "Zone1",
            "variable": "Zone Air Temperature",
            "variable_column": "Zone_Air_Temperature_",
        },
    ]
    options = results_data.variable_column_options(
        catalog, variables=["Schedule Value"]
    )
    assert [item["value"] for item in options] == [
        "Schedule_Value_Lights",
        "Schedule_Value_People",
    ]


def test_result_index_reconstructs_local_run_root(monkeypatch, tmp_path):
    campaign_root = tmp_path / "campaigns" / "gen_demo"
    matrix_root = campaign_root / "aggregation" / "matrix_runs" / "matrix_001"
    matrix_root.mkdir(parents=True)
    (matrix_root / "aggregation_matrix_manifest.json").write_text(
        json.dumps(
            {
                "aggregation_campaign_id": "agg_demo",
                "parent_generation_campaign_id": "gen_demo",
                "matrix_run_id": "matrix_001",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    with (matrix_root / "aggregation_matrix_case_runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "aggregation_run_id",
                "status",
                "plan_strategy",
                "weight_mode",
                "rule_set",
                "building_type",
                "weather_location",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case1",
                "aggregation_run_id": "aggr1",
                "status": "completed",
                "plan_strategy": "identity",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
            }
        )
    run_root = (
        campaign_root
        / "aggregation"
        / "cases"
        / "case1"
        / "runs"
        / "aggr1"
    )
    (run_root / "inputs").mkdir(parents=True)
    (run_root / "inputs" / "source_run_manifest.json").write_text(
        json.dumps(
            {
                "case_spec": {
                    "run_period": {"calendar_year": 2013}
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    rows = results_data.result_index("agg_demo")
    assert len(rows) == 1
    assert rows[0]["strategy"] == "identity"
    assert rows[0]["run_root_local"] == str(run_root)
    assert rows[0]["calendar_year"] == 2013


def test_filter_result_index_uses_strict_intersection_and_empty_means_empty():
    rows = [
        {
            "aggregation_campaign_id": "agg1",
            "building_type": "OfficeSmall",
            "weather_location": "Seattle",
            "climate_zone": "4C",
            "strategy": "identity",
            "weight_mode": "equal",
            "rule_set": "legacy_v1",
            "run_token": "agg1::m1::r1",
        },
        {
            "aggregation_campaign_id": "agg1",
            "building_type": "RestaurantFastFood",
            "weather_location": "Tampa",
            "climate_zone": "2A",
            "strategy": "all_thermal_zones_to_one",
            "weight_mode": "floor_area",
            "rule_set": "legacy_v1",
            "run_token": "agg1::m1::r2",
        },
    ]

    selected = results_data.filter_result_index(
        rows,
        aggregation_campaign_ids=["agg1"],
        building_types=["OfficeSmall"],
        weather_locations=["Seattle"],
        climate_zones=["4C"],
        strategies=["identity"],
        weight_modes=["equal"],
        rule_sets=["legacy_v1"],
        run_tokens=["agg1::m1::r1"],
    )
    assert len(selected) == 1
    assert selected[0]["building_type"] == "OfficeSmall"

    assert results_data.filter_result_index(rows, building_types=[]) == []


def test_run_options_include_campaign_building_weather_climate_and_run():
    rows = [
        {
            "aggregation_campaign_id": "agg1",
            "building_type": "OfficeSmall",
            "weather_location": "Seattle",
            "climate_zone": "4C",
            "strategy": "identity",
            "weight_mode": "equal",
            "aggregation_run_id": "aggr1",
            "run_token": "agg1::m1::aggr1",
        }
    ]
    option = results_data.run_options(rows)[0]
    assert option["value"] == "agg1::m1::aggr1"
    assert "agg1" in option["label"]
    assert "OfficeSmall" in option["label"]
    assert "Seattle" in option["label"]
    assert "4C" in option["label"]
    assert "aggr1" in option["label"]


def test_find_run_retains_legacy_single_campaign_summary_api(monkeypatch):
    monkeypatch.setattr(
        results_data,
        "discover_matrix_runs",
        lambda: [
            {
                "aggregation_campaign_id": "agg_demo",
                "matrix_run_id": "matrix_001",
            }
        ],
    )
    row = results_data._find_run("agg_demo", "matrix_001")
    assert row["aggregation_campaign_id"] == "agg_demo"


def test_selected_export_accepts_multi_campaign_argument_name(monkeypatch):
    import inspect

    signature = inspect.signature(results_data.build_selected_data_export)
    assert "aggregation_campaign_ids" in signature.parameters
    assert "aggregation_campaign_id" not in signature.parameters
