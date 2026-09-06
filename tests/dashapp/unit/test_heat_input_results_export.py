from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import zipfile

from scalebridge.dashapp.services.heat_input import results_data


def _load_fixture():
    fixture_path = Path(__file__).with_name("test_heat_input_results_service.py")
    spec = spec_from_file_location("_heat_input_results_service_fixture", fixture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Results fixture module: {fixture_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fixture


_fixture = _load_fixture()


def test_evaluation_export_uses_explicit_displayed_selection(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)
    selection = {
        "filters": {"model_ids": ["PHVAC"], "estimator_types": ["pytorch_linear"]},
        "splits": ["test"],
        "evaluation_modes": ["chained"],
        "resolution": "preview",
        "range_mode": "full",
        "start": None,
        "end": None,
    }

    payload, filename = results_data.build_evaluation_export(ref, selection=selection)
    assert filename.endswith("__phase_c_evaluation_selection.zip")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert "evaluation_metrics.csv" in archive.namelist()
        assert "model_metadata.csv" in archive.namelist()
        assert "evaluation_series.csv" in archive.namelist()
        provenance = json.loads(archive.read("provenance_manifest.json"))
        assert provenance["contract"] == "selected_equals_displayed_equals_exported"
        assert provenance["selection"] == selection


def test_annual_export_preserves_phvac_column_selection(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)
    zone = results_data.inference_zone_options(ref)[0]["value"]
    selection = {
        "zone_key": zone,
        "prediction_columns": ["predicted_PHVAC", "predicted_PHVAC_oracle"],
        "resolution": "preview",
        "range_mode": "full",
        "start": None,
        "end": None,
    }

    payload, filename = results_data.build_annual_export(ref, selection=selection)
    assert filename.endswith("__phase_c_annual_selection.zip")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        provenance = json.loads(archive.read("provenance_manifest.json"))
        assert provenance["selection"] == selection
        data = archive.read("annual_inference_series.csv").decode("utf-8")
        assert "predicted_PHVAC" in data
        assert "predicted_PHVAC_oracle" in data


def test_campaign_summary_export_is_compact_and_provenance_backed(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)

    payload, filename = results_data.build_campaign_summary_export(ref)
    assert filename.endswith("__phase_c_summary.zip")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "campaign_summary.json",
            "stage_summary.csv",
            "structural_availability.csv",
            "validation_overview.csv",
            "provenance_manifest.json",
        }


def test_model_bundle_export_requires_exact_one_artifact(monkeypatch, tmp_path):
    campaign_id, phase_run_id = _fixture(tmp_path)
    monkeypatch.setattr(results_data, "resolve_generated_data_root", lambda: tmp_path)
    ref = results_data.load_run_ref(campaign_id, phase_run_id)
    selection = {
        "filters": {"model_ids": ["PHVAC"], "estimator_types": ["pytorch_linear"]}
    }

    payload, filename = results_data.build_model_bundle_export(ref, selection=selection)
    assert filename.startswith("phase_c_model_PHVAC_pytorch_linear_")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert "model_artifact/model_manifest.json" in archive.namelist()
        assert "training_manifest.json" in archive.namelist()
        assert "source_model_dataset_manifest.json" in archive.namelist()
        provenance = json.loads(archive.read("selection_manifest.json"))
        assert provenance["contract"] == "selected_equals_displayed_equals_exported"

    try:
        results_data.build_model_bundle_export(ref, selection={"filters": {}})
    except results_data.ResultSelectionTooBroad as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("Broad model export should be refused")


def test_plot_figure_export_contains_only_visible_traces_in_self_describing_zip():
    figure = {
        "data": [
            {
                "type": "scatter",
                "name": "visible trace",
                "x": ["2026-01-01", "2026-01-02"],
                "y": [1.0, 2.0],
                "visible": True,
            },
            {
                "type": "scatter",
                "name": "hidden trace",
                "x": ["2026-01-01", "2026-01-02"],
                "y": [9.0, 9.0],
                "visible": "legendonly",
            },
        ],
        "layout": {
            "xaxis": {"title": {"text": "Time"}},
            "yaxis": {"title": {"text": "Observed Y / predicted Ŷ (W)"}},
            "meta": {
                "phase_c_plot_export": {
                    "plot_key": "evaluation",
                    "run_key": "campaign::phase_c_test",
                    "selection": {
                        "filters": {"model_ids": ["QAC"]},
                        "splits": ["test"],
                        "evaluation_modes": ["direct"],
                    },
                }
            },
        },
    }
    run_ref = {
        "campaign_id": "campaign_test",
        "phase_c_run_id": "phase_c_test",
        "manifest_path": "C:/fixture/phase_c_campaign_run_manifest.json",
    }
    payload, filename = results_data.build_plot_figure_export(
        figure,
        file_format="csv",
        plot_key="evaluation",
        run_id="phase_c_test",
        run_ref=run_ref,
    )
    assert filename == "phase_c_test__evaluation__visible_plot_data_csv.zip"

    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "data/plotted_data.csv",
            "selection_manifest.json",
            "README.txt",
        }
        text = archive.read("data/plotted_data.csv").decode("utf-8")
        assert "visible trace" in text
        assert "hidden trace" not in text
        assert "trace_index,trace_name,trace_type,point_index,x,y" in text

        manifest = json.loads(archive.read("selection_manifest.json"))
        assert manifest["export_type"] == "bgirs_phase_c_visible_plot_data"
        assert manifest["campaign_id"] == "campaign_test"
        assert manifest["phase_c_run_id"] == "phase_c_test"
        assert manifest["selected_format"] == "csv"
        assert manifest["visible_trace_names"] == ["visible trace"]
        assert manifest["hidden_trace_names"] == ["hidden trace"]
        assert manifest["plot_snapshot"]["selection"]["filters"]["model_ids"] == ["QAC"]
        assert manifest["contract"] == "visible_plot_snapshot_equals_exported_data"


def test_plot_figure_export_supports_parquet_inside_zip():
    figure = {
        "data": [
            {"type": "bar", "name": "metric", "x": ["A"], "y": [3.5]}
        ],
        "layout": {
            "meta": {
                "phase_c_plot_export": {
                    "plot_key": "comparison",
                    "selection": {"comparison_kind": "estimator_metric"},
                }
            }
        },
    }
    payload, filename = results_data.build_plot_figure_export(
        figure,
        file_format="parquet",
        plot_key="comparison",
        run_id="phase_c_test",
    )
    assert filename.endswith("__comparison__visible_plot_data_parquet.zip")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert "data/plotted_data.parquet" in archive.namelist()
        assert "selection_manifest.json" in archive.namelist()
        assert "README.txt" in archive.namelist()
        manifest = json.loads(archive.read("selection_manifest.json"))
        assert manifest["selected_format"] == "parquet"
        assert manifest["plot_snapshot"]["selection"]["comparison_kind"] == "estimator_metric"
