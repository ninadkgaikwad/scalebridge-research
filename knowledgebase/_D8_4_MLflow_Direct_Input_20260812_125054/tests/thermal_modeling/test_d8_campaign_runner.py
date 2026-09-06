from __future__ import annotations

import csv
import json
from pathlib import Path

from scalebridge.data.thermal_modeling.campaign_runner import (
    MatrixAggregationRun,
    aggregation_output_root,
    is_completed_aggregation_output,
    load_matrix_aggregation_runs,
    resolve_latest_successful_matrix_run_id,
    resolve_latest_successful_phase_c_run_id,
    build_single_run_command,
    collect_dataset_registry,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_d8_matrix_loader_preserves_authoritative_nomenclature(tmp_path: Path) -> None:
    root=tmp_path/"campaign"
    matrix="aggregation_matrix_test"
    path=root/"aggregation"/"matrix_runs"/matrix/"aggregation_matrix_case_runs.csv"
    path.parent.mkdir(parents=True)
    fields=[
        "case_id","aggregation_run_id","aggregation_id","aggregation_level",
        "aggregation_level_index","aggregation_family","weight_mode","plan_strategy",
        "rule_set","building_type","climate_zone","weather_location",
        "aggregate_zone_count","source_zone_count","status",
    ]
    rows=[
        ["c1","r1","user_whole_building","whole","-1","custom","equal",
         "custom_groups","legacy_v1","RestaurantFastFood","5A","Buffalo","1","2","completed"],
        ["c1","r2","research_scheme_alpha","alpha","-1","custom","floor_area",
         "custom_groups","legacy_v1","RestaurantFastFood","5A","Buffalo","2","2","completed"],
        ["c1","r3","research_scheme_beta","beta","-1","custom","volume",
         "custom_groups","legacy_v1","RestaurantFastFood","5A","Buffalo","2","2","completed"],
    ]
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(fields); w.writerows(rows)

    items=load_matrix_aggregation_runs(root,matrix_run_id=matrix)
    assert [x.aggregation_id for x in items]==[
        "user_whole_building","research_scheme_alpha","research_scheme_beta"
    ]
    assert [x.weight_mode for x in items]==["equal","floor_area","volume"]
    assert items[1].aggregation_family=="custom"
    assert items[1].plan_strategy=="custom_groups"


def test_d8_latest_resolvers_require_success_and_matching_matrix(tmp_path: Path) -> None:
    root=tmp_path/"campaign"
    good="aggregation_matrix_good"
    bad="aggregation_matrix_bad"
    write_json(root/"aggregation"/"matrix_runs"/bad/"aggregation_matrix_manifest.json",
               {"matrix_run_id":bad,"selected_plan_count":2,"successful_plan_count":1,"failed_plan_count":1})
    write_json(root/"aggregation"/"matrix_runs"/good/"aggregation_matrix_manifest.json",
               {"matrix_run_id":good,"selected_plan_count":2,"successful_plan_count":2,"failed_plan_count":0})
    assert resolve_latest_successful_matrix_run_id(root)==good

    write_json(root/"heat_input_regression"/"campaign_runs"/"pc_bad"/"phase_c_campaign_run_manifest.json",
               {"phase_c_run_id":"pc_bad","status":"completed","matrix_run_id":bad})
    write_json(root/"heat_input_regression"/"campaign_runs"/"pc_good"/"phase_c_campaign_run_manifest.json",
               {"phase_c_run_id":"pc_good","status":"completed","matrix_run_id":good})
    assert resolve_latest_successful_phase_c_run_id(root,matrix_run_id=good)=="pc_good"


def test_d8_completion_requires_declared_parquets_and_manifests(tmp_path: Path) -> None:
    run=tmp_path/"phase_d"/"cases"/"c"/"aggregation_runs"/"r"
    write_json(run/"aggregation_manifest.json",{"status":"completed","final_dataset_count":2})
    for name in ("ml/dep1/x","ob/dep1/x"):
        d=run/"silos"/name
        d.mkdir(parents=True)
        (d/"data.parquet").write_bytes(b"x")
        write_json(d/"manifest.json",{})
    assert is_completed_aggregation_output(run)
    (run/"silos"/"ob/dep1/x/manifest.json").unlink()
    assert not is_completed_aggregation_output(run)


def test_d8_output_root_matches_locked_phase_d_hierarchy(tmp_path: Path) -> None:
    p=aggregation_output_root(tmp_path,case_id="epcase_1",aggregation_run_id="aggr_1")
    assert p==tmp_path/"phase_d"/"cases"/"epcase_1"/"aggregation_runs"/"aggr_1"



def test_d8_command_forwards_multiple_policy_selections_and_parameters(tmp_path: Path) -> None:
    item = MatrixAggregationRun(
        case_id="case_1",
        aggregation_run_id="aggr_1",
        aggregation_id="custom_scheme",
        aggregation_level="custom",
        aggregation_level_index=0,
        aggregation_family="custom",
        weight_mode="equal",
        plan_strategy="custom_groups",
        rule_set="legacy_v1",
        building_type="RestaurantFastFood",
        climate_zone="5A",
        weather_location="Buffalo",
        aggregate_zone_count=2,
        source_zone_count=2,
        matrix_record_order=1,
    )
    cmd = build_single_run_command(
        repo_root=tmp_path,
        campaign_root=tmp_path / "campaign",
        output_root=tmp_path / "output",
        matrix_run_id="matrix_1",
        phase_c_campaign_run_id="phase_c_1",
        item=item,
        phase_d_calendar_year=2001,
        heat_representation="grouped",
        qzivr_separate=False,
        ml_policies=["monthly_distributed_holdout", "chronological_holdout", "seasonal_holdout"],
        ob_policies=["seasonal_distributed", "seasonal_block_holdout", "contiguous_identification", "custom_datetime_ranges"],
        ml_input_lag=12,
        ml_target_horizon=6,
        ml_train_fraction=0.70,
        ml_test_fraction=0.15,
        ml_validation_fraction=0.15,
        ml_sh_train_seasons="winter,spring",
        ml_sh_test_seasons="summer",
        ml_sh_validation_seasons="fall",
        sd_season_offset_days=0,
        sd_train_days=21,
        sd_test_days=7,
        sbh_train_seasons="winter,spring,fall",
        sbh_test_seasons="summer",
        ci_start_datetime="2001-04-01T00:05:00",
        ci_train_days=21,
        ci_test_days=7,
        cdr_train_ranges=["2001-01-01T00:05:00/2001-01-22T00:05:00"],
        cdr_test_ranges=["2001-01-22T00:05:00/2001-01-29T00:05:00"],
        parquet_compression="zstd",
    )
    assert cmd.count("--ml-policy") == 3
    assert cmd.count("--ob-policy") == 4
    assert "chronological_holdout" in cmd
    assert "seasonal_block_holdout" in cmd
    assert "--ci-start-datetime" in cmd
    assert "--cdr-train-range" in cmd
    assert "--cdr-test-range" in cmd



def test_d8_dataset_registry_reads_nested_temporal_manifest_fields(tmp_path: Path) -> None:
    item = MatrixAggregationRun(
        case_id="case_1", aggregation_run_id="aggr_1", aggregation_id="custom",
        aggregation_level="custom", aggregation_level_index=0,
        aggregation_family="custom", weight_mode="equal", plan_strategy="custom_groups",
        rule_set="legacy_v1", building_type="OfficeSmall", climate_zone="5A",
        weather_location="Buffalo", aggregate_zone_count=1, source_zone_count=1,
        matrix_record_order=1,
    )
    run = tmp_path / "run"
    d = run / "silos" / "ml" / "dep1" / "grp_vrin" / "l12_h6" / "ch"
    d.mkdir(parents=True)
    (d / "data.parquet").write_bytes(b"x")
    write_json(d / "manifest.json", {
        "silo": "ml_sciml", "mode": "dependent1", "independent_zone_id": None,
        "heat_representation": {"representation": "grouped_qzic_qzir", "folder_name": "grp_vrin"},
        "temporal_config": {
            "input_lag": 12, "target_horizon": 6,
            "policy_name": "chronological_holdout", "policy_token": "ch",
            "policy_realization_id": None,
            "policy_parameters": {"train_fraction": 0.7, "test_fraction": 0.15, "validation_fraction": 0.15},
        },
        "row_count": 105120, "included_row_count": 105102,
    })
    rows = collect_dataset_registry(run, item)
    assert rows[0]["policy_name"] == "chronological_holdout"
    assert rows[0]["policy_token"] == "ch"
    assert rows[0]["input_lag"] == 12
    assert rows[0]["heat_folder"] == "grp_vrin"



def test_d8_resume_completion_is_configuration_aware(tmp_path: Path) -> None:
    run = tmp_path / "run"
    config = {"ml_policies": ["monthly_distributed_holdout"], "ob_policies": ["seasonal_distributed"]}
    write_json(run / "aggregation_manifest.json", {
        "status": "completed", "final_dataset_count": 1, "runner_configuration": config
    })
    d = run / "silos" / "ml" / "dep1" / "grp_vrin" / "l12_h6" / "mdh"
    d.mkdir(parents=True)
    (d / "data.parquet").write_bytes(b"x")
    write_json(d / "manifest.json", {})
    assert is_completed_aggregation_output(run, expected_configuration=config)
    assert not is_completed_aggregation_output(
        run,
        expected_configuration={"ml_policies": ["chronological_holdout"], "ob_policies": ["seasonal_distributed"]},
    )
