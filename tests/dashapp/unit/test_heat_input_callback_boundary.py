from pathlib import Path


def test_phase_c_callbacks_do_not_import_scientific_modules_directly():
    callbacks = (
        Path(__file__).parents[3]
        / "src"
        / "scalebridge"
        / "dashapp"
        / "pages"
        / "data_pipeline"
        / "phase_c_heat_input"
        / "callbacks.py"
    ).read_text(encoding="utf-8")

    assert "scalebridge.data." not in callbacks
    assert "scalebridge.models." not in callbacks
    assert "scripts.heat_input_regression" not in callbacks
    assert "services.heat_input" in callbacks
