from __future__ import annotations

import csv
import json
from pathlib import Path

from scalebridge.data.aggregation.validation import (
    AggregationValidationPolicy,
    render_validation_text,
    validate_aggregation_matrix,
    write_validation_reports,
    _read_csv_raw,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["placeholder"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)




def _write_empty_parquet(path: Path) -> None:
    """Write a valid zero-row Parquet when PyArrow is available.

    When PyArrow is unavailable, leave a placeholder file so the production
    validator exercises its documented metadata-unavailable warning path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        path.write_bytes(b"placeholder")
        return

    table = pa.table({"timestamp_raw": pa.array([], type=pa.string())})
    pq.write_table(table, path)


def _build_fixture(tmp_path: Path, *, dry_run: bool = False, rule_warning: bool = True) -> tuple[Path, Path]:
    campaign_root = tmp_path / "data" / "campaigns" / "parent_campaign"
    aggregation_root = campaign_root / "aggregation"
    matrix_dir = aggregation_root / "matrix_runs" / "aggregation_matrix_20260813_010101"
    plan_build = aggregation_root / "plans" / "plan_build_20260813_010000_bgirs_test"
    plan_dir = plan_build / "requests" / "request_001" / "case_1" / "identity_legacy_v1_equal_v1"
    plan_path = plan_dir / "aggregation_plan.json"

    generation_root = campaign_root / "generation" / "cases" / "case_1" / "runs" / "gen_1"
    generation_manifest = generation_root / "manifest.json"
    _write_json(generation_manifest, {"status": "completed", "case_spec": {}})

    definition = {
        "schema_version": "0.1.0",
        "aggregation_campaign_id": "bgirs_test_campaign",
        "parent_generation_campaign_id": "parent_campaign",
        "machine_id": "test-machine",
        "case_ids": [],
        "case_limit": None,
        "plan_requests": [
            {
                "strategy": "identity",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
                "custom_aggregation_ids": [],
                "aggregation_level": "identity",
                "aggregation_level_index": 1,
                "aggregation_family": "identity",
            }
        ],
        "custom_zone_groups_path": None,
        "aggregate_zone_name_stem": "Aggregated_Zone",
        "system_node_name_pattern": "DIRECT AIR INLET NODE",
        "parent_generation_campaign_root": None,
        "generated_data_root": None,
        "max_variables": None,
        "preview_rows": 100,
        "write_legacy_pickle": False,
        "continue_on_error": True,
        "mlflow_enabled": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": None,
        "mlflow_run_name": None,
        "mlflow_strict": False,
    }
    _write_json(matrix_dir / "aggregation_campaign_definition.json", definition)

    plan = {
        "schema_version": "0.1.0",
        "aggregation_id": "identity_legacy_v1_equal_v1",
        "strategy": "identity",
        "rule_set": "legacy_v1",
        "weight_mode": "equal",
        "aggregate_zone_name_stem": "Aggregated_Zone",
        "system_node_name_pattern": "DIRECT AIR INLET NODE",
        "source_case_id": "case_1",
        "source_generation_run_id": "gen_1",
        "campaign_id": "parent_campaign",
        "building_type": "TestBuilding",
        "weather_location": "TestWeather",
        "climate_zone": "TestClimate",
        "thermal_zone_filter": {},
        "aggregate_zones": [
            {"aggregate_zone_id": "Aggregated_Zone_1", "source_zones": ["ZONE_A"]},
            {"aggregate_zone_id": "Aggregated_Zone_2", "source_zones": ["ZONE_B"]},
        ],
    }
    _write_json(plan_path, plan)
    mapping_rows = [
        {"case_id": "case_1", "run_id": "gen_1", "campaign_id": "parent_campaign", "aggregation_id": "identity_legacy_v1_equal_v1", "strategy": "identity", "rule_set": "legacy_v1", "weight_mode": "equal", "aggregate_zone_id": "Aggregated_Zone_1", "source_zone": "ZONE_A"},
        {"case_id": "case_1", "run_id": "gen_1", "campaign_id": "parent_campaign", "aggregation_id": "identity_legacy_v1_equal_v1", "strategy": "identity", "rule_set": "legacy_v1", "weight_mode": "equal", "aggregate_zone_id": "Aggregated_Zone_2", "source_zone": "ZONE_B"},
    ]
    _write_csv(plan_dir / "zone_mapping.csv", mapping_rows)
    _write_csv(plan_dir / "missing_plan_inputs.csv", [])

    selected_row = {
        "case_id": "case_1",
        "source_generation_run_id": "gen_1",
        "building_type": "TestBuilding",
        "climate_zone": "TestClimate",
        "weather_location": "TestWeather",
        "aggregation_id": "identity_legacy_v1_equal_v1",
        "aggregation_level": "identity",
        "aggregation_level_index": "1",
        "aggregation_family": "identity",
        "weight_mode": "equal",
        "plan_strategy": "identity",
        "rule_set": "legacy_v1",
        "aggregate_zone_count": "2",
        "source_zone_count": "2",
        "aggregation_compression_ratio": "1.0",
        "plan_build_id": plan_build.name,
        "plan_build_root": str(plan_build),
        "plan_path": str(plan_path),
    }
    _write_csv(matrix_dir / "selected_aggregation_plans.csv", [selected_row])
    _write_csv(plan_build / "aggregation_plan_index.csv", [selected_row])
    _write_csv(plan_build / "missing_plan_inputs.csv", [])
    _write_json(
        plan_build / "aggregation_plan_build_summary.json",
        {
            "schema_version": "0.2.0",
            "aggregation_campaign_id": "bgirs_test_campaign",
            "parent_generation_campaign_id": "parent_campaign",
            "plan_build_id": plan_build.name,
            "plan_count": 1,
            "missing_plan_input_row_count": 0,
        },
    )
    _write_csv(matrix_dir / "missing_generation_rows.csv", [])

    if dry_run:
        case_row = {**selected_row, "aggregation_run_id": "", "status": "planned", "run_root": "", "loaded_plan_aggregation_id": "identity_legacy_v1_equal_v1", "loaded_plan_strategy": "identity", "loaded_plan_rule_set": "legacy_v1", "loaded_plan_weight_mode": "equal", "loaded_variable_count": "", "aggregated_long_rows": "", "static_equipment_rows": "", "equipment_contribution_rows": "", "diagnostic_rows": "", "runtime_seconds": "", "error_type": "", "error_message": "", "rule_summary_rows": ""}
        _write_csv(matrix_dir / "aggregation_matrix_case_runs.csv", [case_row])
        _write_csv(matrix_dir / "aggregation_matrix_outputs.csv", [])
        successful = 0
        planned = 1
        status = "planned"
    else:
        run_root = aggregation_root / "cases" / "case_1" / "runs" / "aggr_1"
        case_row = {**selected_row, "aggregation_run_id": "aggr_1", "status": "completed", "run_root": str(run_root), "loaded_plan_aggregation_id": "identity_legacy_v1_equal_v1", "loaded_plan_strategy": "identity", "loaded_plan_rule_set": "legacy_v1", "loaded_plan_weight_mode": "equal", "loaded_variable_count": "27", "aggregated_long_rows": "0", "static_equipment_rows": "0", "equipment_contribution_rows": "0", "diagnostic_rows": "1", "runtime_seconds": "1.2", "error_type": "", "error_message": "", "rule_summary_rows": "1"}
        _write_csv(matrix_dir / "aggregation_matrix_case_runs.csv", [case_row])
        _write_csv(matrix_dir / "aggregation_matrix_outputs.csv", [{"case_id": "case_1", "source_generation_run_id": "gen_1", "aggregation_run_id": "aggr_1", "aggregation_id": "identity_legacy_v1_equal_v1", "run_root": str(run_root)}])

        _write_json(
            run_root / "aggregation_manifest.json",
            {
                "schema_version": "0.1.0",
                "status": "completed",
                "case_id": "case_1",
                "source_generation_run_id": "gen_1",
                "aggregation_run_id": "aggr_1",
                "aggregation_run_root": str(run_root),
                "plan_path": str(plan_path),
                "plan_aggregation_id": "identity_legacy_v1_equal_v1",
                "strategy": "identity",
                "rule_set": "legacy_v1",
                "weight_mode": "equal",
                "aggregate_zone_count": 2,
                "loaded_variable_count": 27,
                "aggregated_long_rows": 0,
                "static_equipment_rows": 0,
                "equipment_contribution_rows": 0,
                "diagnostic_rows": 1,
                "rule_summary_rows": 1,
            },
        )
        inputs = run_root / "inputs"
        _write_json(inputs / "aggregation_plan.json", plan)
        _write_csv(inputs / "zone_mapping.csv", mapping_rows)
        _write_json(inputs / "source_run_manifest.json", {"status": "completed"})
        _write_json(inputs / "source_generation_run.json", {"case_id": "case_1", "run_id": "gen_1", "status": "completed", "run_root": str(generation_root), "manifest_path": str(generation_manifest), "source_manifest": {}})

        diagnostics = run_root / "diagnostics"
        _write_csv(diagnostics / "loaded_variables.csv", [{"variable_id": "v1", "variable_name": "X", "load_status": "loaded", "row_count": "1", "key_value_count": "1", "parquet_path": "x"}])
        _write_csv(diagnostics / "rule_summary.csv", [{"status": "warning" if rule_warning else "aggregated", "aggregate_zone_id": "Aggregated_Zone_1"}])
        _write_csv(diagnostics / "rule_diagnostics.csv", [{"severity": "warning", "message": "test"}])
        _write_csv(diagnostics / "schedule_equipment_mapping_used.csv", [])
        _write_csv(diagnostics / "equipment_contributions.csv", [])
        shared = [{"variable_id": "v", "variable_name": "System Node", "parquet_path": "x", "source_key_count": "2", "mapped_key_count": "1", "unmapped_key_count": "1", "mapped_row_count": "10", "skipped_row_count": "10"}]
        _write_csv(diagnostics / "system_node_temperature_summary.csv", shared)
        _write_csv(diagnostics / "system_node_mass_flow_summary.csv", shared)

        for zone in ("Aggregated_Zone_1", "Aggregated_Zone_2"):
            zone_dir = run_root / "zones" / zone
            zone_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "aggregated_timeseries_wide.parquet",
                "aggregated_timeseries_long.parquet",
                "aggregated_static_equipment.parquet",
                "equipment_contributions.parquet",
            ):
                _write_empty_parquet(zone_dir / name)
            _write_csv(zone_dir / "aggregated_timeseries_wide_preview.csv", [])
            _write_csv(zone_dir / "aggregated_timeseries_long_preview.csv", [])
            _write_csv(zone_dir / "aggregated_static_equipment.csv", [])
            _write_csv(zone_dir / "equipment_contributions.csv", [])
            _write_csv(zone_dir / "zone_mapping.csv", mapping_rows)

        successful = 1
        planned = 0
        status = "completed"

    _write_json(
        matrix_dir / "aggregation_matrix_manifest.json",
        {
            "schema_version": "0.2.0",
            "aggregation_campaign_id": "bgirs_test_campaign",
            "parent_generation_campaign_id": "parent_campaign",
            "campaign_id": "parent_campaign",
            "machine_id": "test-machine",
            "campaign_root": str(campaign_root),
            "matrix_run_id": matrix_dir.name,
            "summary_dir": str(matrix_dir),
            "plan_build_id": plan_build.name,
            "plan_build_root": str(plan_build),
            "dry_run": dry_run,
            "status": status,
            "selected_plan_count": 1,
            "attempted_plan_count": 0 if dry_run else 1,
            "planned_plan_count": planned,
            "successful_plan_count": successful,
            "failed_plan_count": 0,
        },
    )
    return campaign_root, matrix_dir


def test_completed_generic_matrix_passes_with_scientific_warnings(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    assert report.status == "PASS"
    assert report.error_count == 0
    assert report.warning_count >= 3  # rule warning + two shared-node unmapped notices (+ parquet availability)
    assert report.checked_run_count == 1
    assert report.checked_zone_count == 2
    assert report.rule_warning_row_count == 1
    assert report.unmapped_system_node_count == 2


def test_dry_run_matrix_is_valid_planned_contract(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path, dry_run=True)
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    assert report.status == "PASS"
    assert report.dry_run is True
    assert report.checked_run_count == 0


def test_missing_required_matrix_file_fails(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    (matrix_dir / "selected_aggregation_plans.csv").unlink()
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    assert report.status == "FAIL"
    assert any(issue.code == "matrix_file_missing" for issue in report.issues)


def test_zone_mapping_plan_mismatch_fails(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    selected = next(csv.DictReader((matrix_dir / "selected_aggregation_plans.csv").open(encoding="utf-8")))
    mapping = Path(selected["plan_path"]).parent / "zone_mapping.csv"
    _write_csv(mapping, [{"aggregate_zone_id": "Aggregated_Zone_1", "source_zone": "ZONE_A"}])
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    assert report.status == "FAIL"
    assert any(issue.code == "zone_mapping_plan_mismatch" for issue in report.issues)


def test_definition_lineage_mismatch_fails(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    path = matrix_dir / "aggregation_campaign_definition.json"
    payload = json.loads(path.read_text())
    payload["parent_generation_campaign_id"] = "different_parent"
    _write_json(path, payload)
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    assert report.status == "FAIL"
    assert any(issue.code == "definition_parent_id_mismatch" for issue in report.issues)


def test_source_generation_provenance_mismatch_fails(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    row = next(csv.DictReader((matrix_dir / "aggregation_matrix_case_runs.csv").open(encoding="utf-8")))
    path = Path(row["run_root"]) / "inputs" / "source_generation_run.json"
    payload = json.loads(path.read_text())
    payload["run_id"] = "wrong_gen"
    _write_json(path, payload)
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    assert report.status == "FAIL"
    assert any(issue.code == "source_generation_run_mismatch" for issue in report.issues)


def test_shared_node_accounting_mismatch_fails(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    row = next(csv.DictReader((matrix_dir / "aggregation_matrix_case_runs.csv").open(encoding="utf-8")))
    path = Path(row["run_root"]) / "diagnostics" / "system_node_temperature_summary.csv"
    _write_csv(path, [{"source_key_count": "2", "mapped_key_count": "2", "unmapped_key_count": "1", "mapped_row_count": "10", "skipped_row_count": "0"}])
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    assert report.status == "FAIL"
    assert any(issue.code == "shared_node_key_accounting_mismatch" for issue in report.issues)


def test_warning_notices_can_be_suppressed_without_changing_pass(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    report = validate_aggregation_matrix(
        matrix_run_dir=matrix_dir,
        campaign_root=campaign_root,
        policy=AggregationValidationPolicy(
            warn_on_rule_warnings=False,
            warn_on_unmapped_system_nodes=False,
        ),
    )
    assert report.status == "PASS"
    assert report.rule_warning_row_count == 1
    assert report.unmapped_system_node_count == 2
    assert not any(issue.code in {"rule_warning_rows", "unmapped_system_nodes"} for issue in report.issues)


def test_report_writers_emit_json_and_text(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path, dry_run=True)
    report = validate_aggregation_matrix(matrix_run_dir=matrix_dir, campaign_root=campaign_root)
    json_path, text_path = write_validation_reports(
        report,
        json_path=tmp_path / "report.json",
        text_path=tmp_path / "report.txt",
    )
    payload = json.loads(json_path.read_text())
    assert payload["status"] == "PASS"
    assert "SCALEBRIDGE GENERIC AGGREGATION VALIDATION REPORT" in text_path.read_text()
    assert "status: PASS" in render_validation_text(report)


def test_read_csv_raw_treats_scalebridge_no_rows_sentinel_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "missing_generation_rows.csv"
    path.write_text("note\nno rows\n", encoding="utf-8")

    assert _read_csv_raw(path) == []


def test_read_csv_raw_does_not_hide_real_note_rows(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic.csv"
    path.write_text("note\nreal scientific note\n", encoding="utf-8")

    rows = _read_csv_raw(path)

    assert rows == [{"note": "real scientific note"}]


def test_completed_fixture_passes_when_empty_csvs_use_scalebridge_sentinel(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)

    sentinel = "note\nno rows\n"
    (matrix_dir / "missing_generation_rows.csv").write_text(sentinel, encoding="utf-8")

    plan_build_summaries = list(campaign_root.rglob("aggregation_plan_build_summary.json"))
    assert plan_build_summaries
    plan_build_root = plan_build_summaries[0].parent
    (plan_build_root / "missing_plan_inputs.csv").write_text(sentinel, encoding="utf-8")

    for plan_path in campaign_root.rglob("aggregation_plan.json"):
        sibling = plan_path.parent / "missing_plan_inputs.csv"
        if sibling.exists():
            sibling.write_text(sentinel, encoding="utf-8")

    report = validate_aggregation_matrix(
        matrix_run_dir=matrix_dir,
        campaign_root=campaign_root,
    )

    assert report.status == "PASS"
    assert report.error_count == 0


def test_source_generation_completed_with_warnings_is_accepted_success(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    row = _read_csv_raw(matrix_dir / "aggregation_matrix_case_runs.csv")[0]
    path = Path(row["run_root"]) / "inputs" / "source_generation_run.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "completed_with_warnings"
    _write_json(path, payload)

    report = validate_aggregation_matrix(
        matrix_run_dir=matrix_dir,
        campaign_root=campaign_root,
    )

    assert report.status == "PASS"
    assert report.error_count == 0
    assert any(
        issue.code == "source_generation_completed_with_warnings"
        and issue.severity == "warning"
        for issue in report.issues
    )


def test_source_generation_failed_status_remains_validation_error(tmp_path: Path) -> None:
    campaign_root, matrix_dir = _build_fixture(tmp_path)
    row = _read_csv_raw(matrix_dir / "aggregation_matrix_case_runs.csv")[0]
    path = Path(row["run_root"]) / "inputs" / "source_generation_run.json"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    _write_json(path, payload)

    report = validate_aggregation_matrix(
        matrix_run_dir=matrix_dir,
        campaign_root=campaign_root,
    )

    assert report.status == "FAIL"
    assert any(
        issue.code == "source_generation_not_successful"
        and issue.severity == "error"
        for issue in report.issues
    )
