from pathlib import Path


def test_results_callbacks_use_dash_services_not_scientific_modules():
    path = Path(
        "src/scalebridge/dashapp/pages/data_pipeline/phase_c_heat_input/results/callbacks.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "services.heat_input.results_data" in source
    assert "evaluation.heat_input_regression" not in source
    assert "inference.heat_input_regression" not in source
    assert "training.heat_input_regression" not in source
    assert "import subprocess" not in source
    assert "shell=True" not in source
    assert "components.results" in source
    assert "def _build_legend" not in source
    assert "def _legend_button_style" not in source


def test_results_service_is_read_only_and_does_not_mutate_science():
    path = Path("src/scalebridge/dashapp/services/heat_input/results_data.py")
    source = path.read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "save_definition" not in source
    assert "unlink(" not in source
    assert "rmtree(" not in source
    # Plot-data Parquet export is an in-memory BytesIO serialization only.
