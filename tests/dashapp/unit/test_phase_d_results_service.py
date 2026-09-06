from pathlib import Path
import json

import pandas as pd

from scalebridge.dashapp.services.phase_d import results_data


def _fixture(root: Path):
    campaign = root / "campaigns" / "demo"
    run = campaign / "phase_d" / "campaign_runs" / "phase_d_demo"
    run.mkdir(parents=True)
    (run / "phase_d_campaign_run_manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "demo",
                "phase_d_run_id": "phase_d_demo",
                "status": "completed",
                "dataset_count": 1,
                "ml_dataset_count": 1,
                "opt_bayes_dataset_count": 0,
                "ind_dataset_count": 1,
                "dep1_dataset_count": 0,
                "dep2_dataset_count": 0,
                "selected_aggregation_run_count": 1,
                "completed_aggregation_run_count": 1,
                "failed_aggregation_run_count": 0,
            }
        ),
        encoding="utf-8",
    )
    data_root = campaign / "phase_d" / "cases" / "case1" / "aggregation_runs" / "aggr1" / "silos" / "ml" / "ind" / "Z1" / "grp_vrin" / "l1_h1" / "mdh"
    data_root.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2001-01-01 00:05", periods=12, freq="5min"),
            "included": [True] * 9 + [False] * 3,
            "partition": ["train"] * 6 + ["validation"] * 2 + ["test"] + ["excluded"] * 3,
            "window_id": [""] * 12,
            "season": ["winter"] * 12,
            "outdoor_temperature__lag_0": range(12),
            "Z1__zone_temperature__lag_0": range(20, 32),
            "Z1__qac__lag_0": range(100, 112),
            "Z1__zone_temperature__target_1": range(21, 33),
        }
    )
    data_path = data_root / "data.parquet"
    frame.to_parquet(data_path, index=False)
    final_columns = [
        {"name": "timestamp", "base_signal": "timestamp", "temporal_role": "anchor_timestamp", "physical_role": "metadata", "offset_steps": None, "units": None, "aggregate_zone_id": None},
        {"name": "included", "base_signal": "included", "temporal_role": "selection", "physical_role": "metadata", "offset_steps": None, "units": None, "aggregate_zone_id": None},
        {"name": "partition", "base_signal": "partition", "temporal_role": "partition", "physical_role": "metadata", "offset_steps": None, "units": None, "aggregate_zone_id": None},
        {"name": "window_id", "base_signal": "window_id", "temporal_role": "selection_window", "physical_role": "metadata", "offset_steps": None, "units": None, "aggregate_zone_id": None},
        {"name": "season", "base_signal": "season", "temporal_role": "season", "physical_role": "metadata", "offset_steps": None, "units": None, "aggregate_zone_id": None},
        {"name": "outdoor_temperature__lag_0", "base_signal": "outdoor_temperature", "temporal_role": "model_input", "physical_role": "disturbance", "offset_steps": 0, "units": "degC", "aggregate_zone_id": None},
        {"name": "Z1__zone_temperature__lag_0", "base_signal": "zone_temperature", "temporal_role": "model_input", "physical_role": "state", "offset_steps": 0, "units": "degC", "aggregate_zone_id": "Z1"},
        {"name": "Z1__qac__lag_0", "base_signal": "qac", "temporal_role": "model_input", "physical_role": "control_input", "offset_steps": 0, "units": "W", "aggregate_zone_id": "Z1"},
        {"name": "Z1__zone_temperature__target_1", "base_signal": "zone_temperature", "temporal_role": "prediction_target", "physical_role": "target", "offset_steps": 1, "units": "degC", "aggregate_zone_id": "Z1"},
    ]
    manifest_path = data_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "row_count": 12,
                "included_row_count": 9,
                "excluded_row_count": 3,
                "first_timestamp": "2001-01-01T00:05:00",
                "last_timestamp": "2001-01-01T01:00:00",
                "partition_counts": {"train": 6, "validation": 2, "test": 1, "excluded": 3},
                "final_columns": final_columns,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "case_id": "case1",
                "aggregation_run_id": "aggr1",
                "aggregation_id": "identity",
                "aggregation_family": "identity",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
                "data_path": str(data_path),
                "manifest_path": str(manifest_path),
                "silo": "ml_sciml",
                "mode": "independent",
                "independent_zone_id": "Z1",
                "heat_representation": "grouped_qzic_qzir",
                "input_lag": 1,
                "target_horizon": 1,
                "policy_name": "monthly_distributed_holdout",
                "policy_token": "mdh",
                "row_count": 12,
                "included_row_count": 9,
            }
        ]
    ).to_csv(run / "dataset_registry.csv", index=False)
    (run / "phase_d_campaign_plan.json").write_text("{}", encoding="utf-8")
    return campaign, run, manifest_path


def test_results_discovery_registry_and_filtering(monkeypatch, tmp_path):
    _, _, manifest_path = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    rows = results_data.discover_phase_d_runs()
    assert len(rows) == 1
    ref = results_data.load_run_ref("demo::phase_d_demo")
    registry = results_data.dataset_registry(ref)
    assert len(registry) == 1
    assert results_data.apply_filters(registry, {"silo": "ml_sciml"}).shape[0] == 1
    assert results_data.apply_filters(registry, {"silo": "opt_bayes"}).empty
    assert results_data.dataset_options(registry)[0]["value"] == str(manifest_path)


def test_manifest_drives_signal_and_partition_choices(monkeypatch, tmp_path):
    _, _, manifest_path = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref("demo::phase_d_demo")
    manifest = results_data.load_dataset_manifest(ref, str(manifest_path))
    values = [row["value"] for row in results_data.signal_options(manifest)]
    assert "Z1__qac__lag_0" in values
    assert "included" not in values
    assert "Z1__zone_temperature__target_1" in values
    defaults = results_data.default_signal_values(manifest)
    assert "Z1__zone_temperature__lag_0" in defaults
    assert "Z1__zone_temperature__target_1" in defaults
    partitions = [row["value"] for row in results_data.partition_options(manifest)]
    assert results_data.INCLUDED in partitions
    assert "validation" in partitions


def test_plot_frame_filters_included_and_bounds_rows(monkeypatch, tmp_path):
    _, _, manifest_path = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref("demo::phase_d_demo")
    frame, meta = results_data.load_plot_frame(
        ref,
        str(manifest_path),
        signals=["Z1__qac__lag_0"],
        partition=results_data.INCLUDED,
        max_points=4,
    )
    assert meta["source_row_count"] == 9
    assert len(frame) <= 4
    assert meta["stride"] == 3


def test_selected_dataset_and_run_summary_exports(monkeypatch, tmp_path):
    from io import BytesIO
    import zipfile

    _, run, manifest_path = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref("demo::phase_d_demo")
    payload, _ = results_data.build_selected_dataset_export(ref, str(manifest_path))
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert {"data.parquet", "manifest.json", "selection_manifest.json", "README.txt"} == set(archive.namelist())
    payload, _ = results_data.build_run_summary_export(ref)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert "dataset_registry.csv" in archive.namelist()
        assert "phase_d_campaign_run_manifest.json" in archive.namelist()



def test_preview_records_reads_only_requested_rows(monkeypatch, tmp_path):
    _, _, manifest_path = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref("demo::phase_d_demo")
    records, columns = results_data.preview_records(ref, str(manifest_path), limit=5)
    assert len(records) == 5
    assert columns[0] == "timestamp"
    assert isinstance(records[0]["timestamp"], str)


def test_heat_representation_is_a_real_registry_filter(monkeypatch, tmp_path):
    _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref("demo::phase_d_demo")
    registry = results_data.dataset_registry(ref)
    assert results_data.apply_filters(
        registry,
        {"heat_representation": "grouped_qzic_qzir"},
    ).shape[0] == 1


def _cascading_registry_fixture():
    return pd.DataFrame(
        [
            {
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
                "case_id": "case_office",
                "aggregation_family": "identity",
                "aggregation_id": "identity_equal",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
                "silo": "ml_sciml",
                "mode": "independent",
                "independent_zone_id": "Z1",
                "heat_representation": "grouped_qzic_qzir",
                "policy_name": "monthly_distributed_holdout",
                "input_lag": 1,
                "target_horizon": 1,
                "manifest_path": "office_ml_l1.json",
            },
            {
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
                "case_id": "case_office",
                "aggregation_family": "identity",
                "aggregation_id": "identity_equal",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
                "silo": "ml_sciml",
                "mode": "independent",
                "independent_zone_id": "Z1",
                "heat_representation": "grouped_qzic_qzir",
                "policy_name": "monthly_distributed_holdout",
                "input_lag": 6,
                "target_horizon": 1,
                "manifest_path": "office_ml_l6.json",
            },
            {
                "building_type": "RestaurantFastFood",
                "weather_location": "Tampa",
                "case_id": "case_rff",
                "aggregation_family": "all_thermal_zones_to_one",
                "aggregation_id": "all_equal",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
                "silo": "opt_bayes",
                "mode": "dependent2",
                "independent_zone_id": None,
                "heat_representation": "grouped_qzic_qzir",
                "policy_name": "seasonal_distributed",
                "input_lag": 1,
                "target_horizon": 1,
                "manifest_path": "rff_opt_dep2.json",
            },
        ]
    )


def _option_values(options):
    return [row["value"] for row in options]


def test_cascading_filter_options_are_mutually_constrained():
    registry = _cascading_registry_fixture()
    state = results_data.cascading_filter_state(
        registry,
        {"silo": "opt_bayes"},
        preferred_column="silo",
    )

    assert len(state["matched"]) == 1
    assert _option_values(state["options"]["policy_name"]) == [
        results_data.ALL,
        "seasonal_distributed",
    ]
    assert _option_values(state["options"]["input_lag"]) == [
        results_data.ALL,
        1,
    ]
    assert _option_values(state["options"]["building_type"]) == [
        results_data.ALL,
        "RestaurantFastFood",
    ]


def test_latest_filter_selection_wins_and_clears_incompatible_older_values():
    registry = _cascading_registry_fixture()
    state = results_data.cascading_filter_state(
        registry,
        {
            "aggregation_family": "identity",
            "aggregation_id": "all_equal",
        },
        preferred_column="aggregation_id",
    )

    assert state["values"]["aggregation_id"] == "all_equal"
    assert state["values"]["aggregation_family"] == results_data.ALL
    assert len(state["matched"]) == 1
    assert _option_values(state["options"]["aggregation_family"]) == [
        results_data.ALL,
        "all_thermal_zones_to_one",
    ]


def test_clearing_filter_broadens_peer_options_again():
    registry = _cascading_registry_fixture()

    constrained = results_data.cascading_filter_state(
        registry,
        {"silo": "opt_bayes"},
        preferred_column="silo",
    )
    broadened = results_data.cascading_filter_state(
        registry,
        {"silo": results_data.ALL},
        preferred_column="silo",
    )

    assert _option_values(constrained["options"]["building_type"]) == [
        results_data.ALL,
        "RestaurantFastFood",
    ]
    assert _option_values(broadened["options"]["building_type"]) == [
        results_data.ALL,
        "OfficeSmall",
        "RestaurantFastFood",
    ]


def test_dependent_mode_removes_independent_zone_choices():
    registry = _cascading_registry_fixture()
    state = results_data.cascading_filter_state(
        registry,
        {"mode": "dependent2"},
        preferred_column="mode",
    )

    assert _option_values(state["options"]["independent_zone_id"]) == [
        results_data.ALL,
    ]
