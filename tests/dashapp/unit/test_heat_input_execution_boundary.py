from __future__ import annotations

from pathlib import Path


def test_execution_callbacks_use_services_not_scientific_runner_directly():
    path = (
        Path(__file__).parents[3]
        / "src"
        / "scalebridge"
        / "dashapp"
        / "pages"
        / "data_pipeline"
        / "phase_c_heat_input"
        / "execution"
        / "callbacks.py"
    )
    text = path.read_text(encoding="utf-8")

    assert "scripts.heat_input_regression" not in text
    assert "import subprocess" not in text
    assert "services.heat_input" in text


def test_execution_service_delegates_to_authoritative_general_runner():
    path = (
        Path(__file__).parents[3]
        / "src"
        / "scalebridge"
        / "dashapp"
        / "services"
        / "heat_input"
        / "execution.py"
    )
    text = path.read_text(encoding="utf-8")

    assert '"run_phase_c_campaign.py"' in text
    assert '"--config"' in text
    assert "PhaseCCampaignConfig" in text
    assert "shell=True" not in text


def test_dash_execution_callbacks_force_complete_non_overwriting_phase_c():
    path = (
        Path(__file__).parents[3]
        / "src"
        / "scalebridge"
        / "dashapp"
        / "pages"
        / "data_pipeline"
        / "phase_c_heat_input"
        / "execution"
        / "callbacks.py"
    )
    text = path.read_text(encoding="utf-8")
    assert '"start_stage": "C1"' in text
    assert '"stop_stage": "C9"' in text
    assert '"overwrite_existing": False' in text
    assert 'phase-c-execution-start-stage' not in text
    assert 'phase-c-execution-stop-stage' not in text
    assert 'phase-c-execution-stdout' not in text
    assert 'phase-c-execution-stderr' not in text
