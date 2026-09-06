from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scalebridge.dashapp.services.heat_input import results_data


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fixture(root: Path):
    campaign_id = "demo_campaign"
    phase_run_id = "phase_c_demo_20260821_120000"
    campaign_root = root / "campaigns" / campaign_id
    heat_root = campaign_root / "heat_input_regression"
    run_root = heat_root / "campaign_runs" / phase_run_id

    ids = {
        "C1": "audit_20260821_120000",
        "C2": "features_20260821_120000",
        "C3": "splits_20260821_120000",
        "C4": "datasets_20260821_120000",
        "C5": "c5_20260821_120000",
        "C6": "c6_models_20260821_120000",
        "C7": "c7_models_20260821_120000",
        "C8": "c8_models_20260821_120000",
    }
    scripts = {
        "C1": ["--audit-run-id", ids["C1"]],
        "C2": ["--feature-run-id", ids["C2"]],
        "C3": ["--split-run-id", ids["C3"]],
        "C4": ["--dataset-run-id", ids["C4"]],
        "C5": ["--output-root", str(heat_root / "model_api_validation" / ids["C5"])],
        "C6": ["--training-run-id", ids["C6"]],
        "C7": ["--evaluation-run-id", ids["C7"]],
        "C8": ["--inference-run-id", ids["C8"]],
        "C9": ["--phase-c-run-id", phase_run_id],
    }
    results = []
    for index, stage in enumerate([f"C{i}" for i in range(1, 10)], start=1):
        results.append(
            {
                "name": f"{stage} operation",
                "script": f"stage_{stage}.py",
                "command": ["python", f"stage_{stage}.py", *scripts[stage]],
                "status": "passed",
                "return_code": 0,
                "runtime_seconds": float(index),
            }
        )
        if stage in {"C3", "C4", "C6", "C7", "C8", "C9"}:
            results.append(
                {
                    "name": f"VALIDATE {stage}",
                    "script": f"validate_{stage}.py",
                    "command": ["python", f"validate_{stage}.py"],
                    "status": "passed",
                    "return_code": 0,
                    "runtime_seconds": 0.5,
                }
            )

    _write_json(
        run_root / "phase_c_campaign_run_manifest.json",
        {
            "campaign_id": campaign_id,
            "phase_c_run_id": phase_run_id,
            "matrix_run_id": "matrix_demo",
            "status": "completed",
            "created_at_utc": "2026-08-21T19:00:00+00:00",
            "runtime_seconds": 90.0,
            "command_count": len(results),
            "passed_command_count": len(results),
            "failed_command_count": 0,
            "availability_summary": {
                "candidate_model_count": 4,
                "applicable_model_count": 2,
                "structurally_inapplicable_model_count": 1,
                "invalid_model_count": 1,
                "missing_expected_data_model_count": 0,
                "trained_model_count": 2,
                "evaluated_model_count": 2,
                "inference_zone_count": 1,
                "inferred_component_count": 2,
            },
            "results": results,
        },
    )

    roots = {
        "C1": heat_root / "audit_runs" / ids["C1"],
        "C2": heat_root / "feature_runs" / ids["C2"],
        "C3": heat_root / "split_runs" / ids["C3"],
        "C4": heat_root / "dataset_runs" / ids["C4"],
        "C5": heat_root / "model_api_validation" / ids["C5"],
        "C6": heat_root / "training_runs" / ids["C6"],
        "C7": heat_root / "evaluation_runs" / ids["C7"],
        "C8": heat_root / "inference_runs" / ids["C8"],
        "C9": heat_root / "mlflow_registration_runs" / phase_run_id,
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)

    _write_csv(
        roots["C1"] / "audit_zone_results.csv",
        [
            {
                "campaign_id": campaign_id,
                "case_id": "case1",
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
                "climate_zone": "4C",
                "aggregation_id": "identity",
                "weight_mode": "equal",
                "aggregate_zone_id": "Zone1",
                "strategy": "identity",
                "rule_set": "legacy_v1",
                "candidate_model_count": 4,
                "applicable_model_count": 2,
                "structurally_inapplicable_model_count": 1,
                "invalid_model_count": 1,
                "missing_expected_data_model_count": 0,
                "status": "completed",
            }
        ],
    )
    dataset_rows = []
    for model_id in ("QAC", "PHVAC"):
        dataset_rows.append(
            {
                "campaign_id": campaign_id,
                "case_id": "case1",
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
                "climate_zone": "4C",
                "aggregation_run_id": "aggr_demo",
                "aggregation_id": "identity",
                "aggregation_family": "identity",
                "aggregation_level": "identity",
                "strategy": "identity",
                "rule_set": "legacy_v1",
                "weight_mode": "equal",
                "aggregate_zone_id": "Zone1",
                "model_id": model_id,
                "status": "completed",
                "predictor_column": "x",
                "target_column": "y",
                "source_row_count": 100,
                "valid_pair_count": 90,
                "invalid_pair_count": 10,
                "train_row_count": 60,
                "validation_row_count": 15,
                "test_row_count": 15,
                "output_root": str(
                    roots["C4"]
                    / "cases"
                    / "case1"
                    / "identity"
                    / "equal"
                    / "Zone1"
                    / "models"
                    / model_id
                ),
            }
        )
    _write_csv(roots["C4"] / "dataset_model_results.csv", dataset_rows)
    for row in dataset_rows:
        model_root = Path(row["output_root"])
        _write_csv(
            model_root / "regression_pairs_preview.csv",
            [
                {"timestamp": "2026-01-01T00:00:00", "x": 1.0, "y": 2.0},
                {"timestamp": "2026-01-01T00:05:00", "x": 2.0, "y": 4.0},
            ],
        )
        _write_json(
            model_root / "model_dataset_manifest.json",
            {
                "model_id": row["model_id"],
                "predictor_column": row["predictor_column"],
                "target_column": row["target_column"],
            },
        )

    split_zone_root = roots["C3"] / "cases" / "case1" / "Zone1"
    _write_csv(
        roots["C3"] / "split_zone_results.csv",
        [
            {
                "case_id": "case1",
                "building_type": "OfficeSmall",
                "weather_location": "Seattle",
                "climate_zone": "4C",
                "aggregation_id": "identity",
                "weight_mode": "equal",
                "aggregate_zone_id": "Zone1",
                "status": "completed",
                "assignment_row_count": 100,
                "train_row_count": 70,
                "validation_row_count": 15,
                "test_row_count": 15,
                "excluded_row_count": 0,
                "output_root": str(split_zone_root),
            }
        ],
    )
    _write_csv(
        split_zone_root / "split_summary.csv",
        [
            {
                "split": "train",
                "row_count": 70,
                "fraction_of_included": 0.70,
                "first_timestamp": "2026-01-01T00:00:00",
                "last_timestamp": "2026-12-31T23:55:00",
                "month_count": 12,
                "day_count": 365,
            },
            {
                "split": "validation",
                "row_count": 15,
                "fraction_of_included": 0.15,
                "first_timestamp": "2026-01-01T03:00:00",
                "last_timestamp": "2026-12-31T23:55:00",
                "month_count": 12,
                "day_count": 365,
            },
            {
                "split": "test",
                "row_count": 15,
                "fraction_of_included": 0.15,
                "first_timestamp": "2026-01-01T03:45:00",
                "last_timestamp": "2026-12-31T23:55:00",
                "month_count": 12,
                "day_count": 365,
            },
            {
                "split": "excluded",
                "row_count": 0,
                "fraction_of_included": 0.0,
                "first_timestamp": "",
                "last_timestamp": "",
                "month_count": 0,
                "day_count": 0,
            },
        ],
    )

    training_rows = []
    evaluation_rows = []
    for model_id in ("QAC", "PHVAC"):
        detail = (
            roots["C7"]
            / "cases"
            / "case1"
            / "identity"
            / "equal"
            / "Zone1"
            / model_id
            / "pytorch_linear_cpu"
        )
        metrics_path = detail / "split_metrics.csv"
        preview_path = detail / "test_prediction_preview.csv"
        manifest_path = detail / "evaluation_manifest.json"
        metric_rows = [
            {
                "split": "test",
                "evaluation_mode": "oracle" if model_id == "PHVAC" else "direct",
                "row_count": 3,
                "rmse": 1.0,
                "mae": 0.8,
                "r2": 0.9,
                "mean_bias_error": 0.1,
                "max_absolute_error": 1.2,
                "nrmse_by_range": 0.1,
                "nrmse_by_mean_abs_target": 0.2,
            }
        ]
        if model_id == "PHVAC":
            metric_rows.append(
                {
                    **metric_rows[0],
                    "evaluation_mode": "chained",
                    "rmse": 1.5,
                }
            )
        _write_csv(metrics_path, metric_rows)
        preview = {
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-01T00:05:00",
                "2026-01-01T00:10:00",
            ],
            "y": [10.0, 11.0, 12.0],
            "prediction": [10.1, 11.1, 12.1],
        }
        if model_id == "PHVAC":
            preview["prediction_chained"] = [9.8, 10.8, 11.8]
        _write_csv(preview_path, [dict(zip(preview, values)) for values in zip(*preview.values())])
        _write_json(
            manifest_path,
            {
                "model_id": model_id,
                "evaluation_modes": (
                    ["oracle", "chained"] if model_id == "PHVAC" else ["direct"]
                ),
                "split_metrics_path": str(metrics_path),
                "preview_paths": {"test": str(preview_path)},
                "prediction_paths": {},
            },
        )
        training_output = (
            roots["C6"]
            / "cases"
            / "case1"
            / "identity"
            / "equal"
            / "Zone1"
            / model_id
            / "pytorch_linear_cpu"
        )
        artifact_dir = training_output / "model_artifact"
        _write_json(
            artifact_dir / "model_manifest.json",
            {"model_id": model_id, "estimator_type": "pytorch_linear"},
        )
        (artifact_dir / "model_state.bin").write_bytes(b"demo-model-state")
        _write_json(
            training_output / "training_manifest.json",
            {
                "model_id": model_id,
                "estimator_type": "pytorch_linear",
                "source_dataset_manifest_payload": {"model_id": model_id},
            },
        )
        training_rows.append(
            {
                "case_id": "case1",
                "aggregation_id": "identity",
                "weight_mode": "equal",
                "aggregate_zone_id": "Zone1",
                "model_id": model_id,
                "estimator_type": "pytorch_linear",
                "requested_device": "cpu",
                "device": "cpu",
                "fit_intercept": model_id == "PHVAC",
                "coefficient": 2.0 if model_id == "PHVAC" else 1.0,
                "intercept": 0.5 if model_id == "PHVAC" else 0.0,
                "training_rmse": 0.5,
                "converged": True,
                "epochs_completed": 10,
                "reload_predictions_match": True,
                "artifact_dir": str(artifact_dir),
                "model_manifest": str(artifact_dir / "model_manifest.json"),
                "status": "completed",
            }
        )
        evaluation_rows.append(
            {
                "case_id": "case1",
                "aggregation_id": "identity",
                "weight_mode": "equal",
                "aggregate_zone_id": "Zone1",
                "model_id": model_id,
                "estimator_type": "pytorch_linear",
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "status": "completed",
                "metrics_path": str(metrics_path),
                "manifest_path": str(manifest_path),
            }
        )
    _write_csv(roots["C6"] / "training_results.csv", training_rows)
    _write_csv(roots["C7"] / "evaluation_results.csv", evaluation_rows)

    building_root = roots["C7"] / "building_phvac_reconstruction"
    _write_csv(
        building_root / "building_phvac_metrics.csv",
        [
            {
                "case_id": "case1",
                "aggregation_id": "identity",
                "weight_mode": "equal",
                "estimator_type": "pytorch_linear",
                "requested_device": "cpu",
                "split": "test",
                "evaluation_mode": "oracle",
                "aggregate_zone_count": 1,
                "rmse": 1.0,
            },
            {
                "case_id": "case1",
                "aggregation_id": "identity",
                "weight_mode": "equal",
                "estimator_type": "pytorch_linear",
                "requested_device": "cpu",
                "split": "test",
                "evaluation_mode": "chained",
                "aggregate_zone_count": 1,
                "rmse": 1.5,
            },
        ],
    )

    annual_dir = roots["C8"] / "cases" / "case1" / "identity" / "equal" / "Zone1"
    annual_manifest = annual_dir / "annual_component_predictions_manifest.json"
    annual_preview = annual_dir / "annual_component_predictions_preview.csv"
    registry = annual_dir / "component_prediction_registry.csv"
    _write_csv(
        annual_preview,
        [
            {
                "timestamp_raw": "01/01 00:00:00",
                "timestamp": "2026-01-01T00:00:00",
                "predicted_QAC": 4.0,
                "predicted_PHVAC": 3.0,
                "predicted_PHVAC_oracle": 3.2,
            },
            {
                "timestamp_raw": "01/01 00:05:00",
                "timestamp": "2026-01-01T00:05:00",
                "predicted_QAC": 4.1,
                "predicted_PHVAC": 3.1,
                "predicted_PHVAC_oracle": 3.3,
            },
        ],
    )
    _write_csv(
        registry,
        [
            {
                "model_id": "QAC",
                "output_prediction_column": "predicted_QAC",
                "prediction_units": "W",
                "estimator_type": "pytorch_linear",
                "predictor_mode": "direct",
                "oracle_output_prediction_column": "",
            },
            {
                "model_id": "PHVAC",
                "output_prediction_column": "predicted_PHVAC",
                "prediction_units": "W",
                "estimator_type": "pytorch_linear",
                "predictor_mode": "chained_from_predicted_QAC",
                "oracle_output_prediction_column": "predicted_PHVAC_oracle",
            },
        ],
    )
    _write_json(
        annual_manifest,
        {
            "stage": "C8",
            "status": "completed",
            "case_id": "case1",
            "aggregation_id": "identity",
            "weight_mode": "equal",
            "aggregate_zone_id": "Zone1",
            "component_count": 2,
            "outputs": {
                "annual_component_predictions_preview": str(annual_preview),
                "component_prediction_registry": str(registry),
            },
        },
    )
    _write_csv(
        roots["C8"] / "inference_results.csv",
        [
            {
                "case_id": "case1",
                "aggregation_id": "identity",
                "weight_mode": "equal",
                "aggregate_zone_id": "Zone1",
                "row_count": 2,
                "component_count": 2,
                "status": "completed",
                "manifest_path": str(annual_manifest),
            }
        ],
    )

    _write_csv(
        roots["C2"] / "feature_validation_results.csv",
        [{"validation_status": "passed", "check_count": 2}],
    )
    for stage, name in (
        ("C3", "split_validation_results.csv"),
        ("C4", "dataset_validation_results.csv"),
        ("C6", "training_validation_results.csv"),
        ("C7", "evaluation_validation_results.csv"),
        ("C8", "inference_validation_results.csv"),
    ):
        _write_csv(roots[stage] / name, [{"status": "passed", "check_count": 2}])
    _write_csv(
        roots["C7"] / "evaluation_validation_diagnostics.csv",
        [
            {
                "case_id": "case1",
                "aggregation_id": "identity",
                "aggregate_zone_id": "Zone1",
                "model_id": "PHVAC",
                "estimator_type": "pytorch_linear",
                "check_name": "metrics_file_exists",
                "status": "passed",
            }
        ],
    )
    _write_json(
        roots["C9"] / "phase_c_mlflow_registration_manifest.json",
        {
            "tracking_uri": "http://127.0.0.1:5000",
            "experiment_id": "7",
            "parent_run_id": "abc123",
            "stage_run_count": 8,
            "task_run_count": 4,
        },
    )
    return campaign_id, phase_run_id


def test_manifest_first_discovery_stage_roots_and_context_join(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)

    discovered = results_data.discover_phase_c_runs()
    assert len(discovered) == 1
    assert discovered[0]["phase_c_run_id"] == phase_run_id

    ref = results_data.load_run_ref(campaign_id, phase_run_id)
    roots = results_data.stage_roots(ref)
    assert roots["C6"].name == "c6_models_20260821_120000"
    assert roots["C8"].name == "c8_models_20260821_120000"

    evaluation = results_data.evaluation_catalog(ref)
    assert set(evaluation["building_type"]) == {"OfficeSmall"}
    assert set(evaluation["weather_location"]) == {"Seattle"}
    assert set(evaluation["model_id"]) == {"QAC", "PHVAC"}


def test_c7_metrics_and_phvac_modes_remain_separate(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)

    rows = results_data.load_evaluation_metrics(
        ref,
        model_ids=["PHVAC"],
        splits=["test"],
        evaluation_modes_selected=["oracle", "chained"],
    )
    assert {row["evaluation_mode"] for row in rows} == {"oracle", "chained"}
    rmse = {row["evaluation_mode"]: row["rmse"] for row in rows}
    assert rmse["oracle"] == 1.0
    assert rmse["chained"] == 1.5

    series = results_data.load_evaluation_series(
        ref,
        model_ids=["PHVAC"],
        splits=["test"],
        evaluation_modes_selected=["oracle", "chained"],
        resolution="preview",
    )
    assert len(series) == 4
    assert any("oracle" in row["name"] for row in series)
    assert any("chained" in row["name"] for row in series)


def test_c8_zone_package_is_lazy_and_oracle_chained_columns_are_distinct(
    monkeypatch, tmp_path
):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)

    options = results_data.inference_zone_options(ref)
    assert len(options) == 1
    zone = options[0]["value"]
    components = results_data.annual_component_catalog(ref, zone)
    phvac = [row for row in components if row["model_id"] == "PHVAC"]
    assert {row["evaluation_mode"] for row in phvac} == {"oracle", "chained"}

    series, summary = results_data.load_annual_series(
        ref,
        selected_zone_key=zone,
        prediction_columns=["predicted_PHVAC", "predicted_PHVAC_oracle"],
        resolution="preview",
    )
    assert len(series) == 2
    assert {row["evaluation_mode"] for row in series} == {"oracle", "chained"}
    assert len(summary) == 2


def test_validation_and_structural_absence_are_not_treated_as_failures(
    monkeypatch, tmp_path
):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)

    availability = results_data.structural_availability_rows(ref)
    assert availability[0]["structurally_inapplicable_model_count"] == 1
    assert availability[0]["invalid_model_count"] == 1

    overview = results_data.validation_overview(ref)
    assert all(row["status"] == "passed" for row in overview)
    diagnostics = results_data.validation_diagnostics(ref, "C7", model_ids=["PHVAC"])
    assert len(diagnostics) == 1
    assert diagnostics[0]["status"] == "passed"


def test_extended_inventories_are_manifest_first_and_filtered(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)

    datasets = results_data.dataset_inventory(ref, model_ids=["PHVAC"])
    assert len(datasets) == 1
    assert datasets[0]["model_id"] == "PHVAC"

    targets = results_data.target_model_inventory(ref, model_ids=["PHVAC"])
    assert len(targets) == 1
    assert targets[0]["model_id"] == "PHVAC"
    assert targets[0]["component"] == "hvac_power"
    assert targets[0]["prediction_column"] == "predicted_PHVAC"

    split_rows = results_data.split_summary_rows(ref, aggregate_zone_ids=["Zone1"])
    assert {row["split"] for row in split_rows} == {
        "train",
        "validation",
        "test",
        "excluded",
    }
    train = next(row for row in split_rows if row["split"] == "train")
    assert train["fraction_of_included"] == 0.70
    assert train["first_timestamp"] == "2026-01-01T00:00:00"
    assert train["month_count"] == 12
    assert train["detail_loaded"] is True

    artifacts = results_data.artifact_inventory(ref)
    assert {row["stage"] for row in artifacts} == {f"C{i}" for i in range(1, 10)}
    assert all("path" in row for row in artifacts)


def test_dataset_preview_requires_one_exact_dataset(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)

    rows = results_data.load_dataset_preview(ref, model_ids=["QAC"])
    assert len(rows) == 2
    assert rows[0]["x"] == 1.0

    try:
        results_data.load_dataset_preview(ref)
    except results_data.ResultSelectionTooBroad as exc:
        assert "exactly 1" in str(exc)
    else:
        raise AssertionError("Broad dataset preview should be refused")


def test_dataset_xy_trajectory_requires_one_model_and_preserves_signal_roles(
    monkeypatch, tmp_path
):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)

    series = results_data.load_dataset_series(
        ref,
        model_ids=["QAC"],
        resolution="preview",
    )
    assert len(series) == 2
    assert {row["role"] for row in series} == {"predictor", "target"}
    assert {row["signal"] for row in series} == {"x", "y"}
    assert series[0]["timestamp"][0].startswith("2026-01-01")

    with pytest.raises(results_data.ResultSelectionTooBroad, match="exactly 1"):
        results_data.load_dataset_series(ref, resolution="preview")
