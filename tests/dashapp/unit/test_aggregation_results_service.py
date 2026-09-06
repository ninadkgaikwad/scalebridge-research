from __future__ import annotations

import csv
import json
from pathlib import Path

from scalebridge.dashapp.services.aggregation import results_data


def _write_fixture(root: Path):
    run_root = (
        root
        / "campaigns"
        / "gen_demo"
        / "aggregation"
        / "matrix_runs"
        / "aggregation_matrix_001"
    )
    run_root.mkdir(parents=True)
    (run_root / "aggregation_matrix_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "aggregation_campaign_id": "agg_demo",
                "parent_generation_campaign_id": "gen_demo",
                "campaign_id": "gen_demo",
                "matrix_run_id": "aggregation_matrix_001",
                "status": "completed",
                "selected_generation_case_count": 1,
                "selected_plan_count": 1,
                "successful_plan_count": 1,
                "failed_plan_count": 0,
                "runtime_seconds": 12.5,
                "plan_build_id": "plan_001",
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    with (run_root / "aggregation_matrix_case_runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "building_type",
                "weather_location",
                "climate_zone",
                "aggregation_id",
                "plan_strategy",
                "weight_mode",
                "rule_set",
                "source_zone_count",
                "aggregate_zone_count",
                "aggregation_compression_ratio",
                "aggregation_run_id",
                "status",
                "loaded_variable_count",
                "runtime_seconds",
                "error_type",
                "error_message",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case1",
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
                "climate_zone": "4C",
                "aggregation_id": "identity_equal",
                "plan_strategy": "identity",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
                "source_zone_count": "6",
                "aggregate_zone_count": "6",
                "aggregation_compression_ratio": "1.0",
                "aggregation_run_id": "aggr_001",
                "status": "completed",
                "loaded_variable_count": "27",
                "runtime_seconds": "2.0",
                "error_type": "",
                "error_message": "",
            }
        )
    for name in ("aggregation_matrix_outputs.csv", "selected_aggregation_plans.csv"):
        (run_root / name).write_text("case_id\ncase1\n", encoding="utf-8")
    return run_root


def test_discovery_uses_true_aggregation_campaign_id(monkeypatch, tmp_path):
    _write_fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)

    rows = results_data.discover_matrix_runs()
    assert len(rows) == 1
    assert rows[0]["aggregation_campaign_id"] == "agg_demo"
    assert rows[0]["parent_generation_campaign_id"] == "gen_demo"


def test_load_matrix_result_is_read_only_summary_loader(monkeypatch, tmp_path):
    run_root = _write_fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)

    result = results_data.load_matrix_result("agg_demo", "aggregation_matrix_001")
    assert result["manifest"]["status"] == "completed"
    assert len(result["case_runs"]) == 1
    assert result["case_runs"][0]["plan_strategy"] == "identity"
    assert result["missing_generation_rows"] == []
    assert result["artifact_paths"]["matrix_manifest"] == str(
        run_root / "aggregation_matrix_manifest.json"
    )


def test_legacy_manifest_falls_back_to_parent_campaign_namespace(monkeypatch, tmp_path):
    run_root = _write_fixture(tmp_path)
    manifest_path = run_root / "aggregation_matrix_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("aggregation_campaign_id")
    payload.pop("parent_generation_campaign_id")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)

    rows = results_data.discover_matrix_runs()
    assert rows[0]["aggregation_campaign_id"] == "legacy::gen_demo"
    assert rows[0]["parent_generation_campaign_id"] == "gen_demo"
