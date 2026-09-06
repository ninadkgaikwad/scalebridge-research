from __future__ import annotations

from pathlib import Path

import pytest

from scalebridge.dashapp.schemas.pipeline.phase_d import (
    PhaseDCampaignDefinition,
    PhaseDRunnerConfig,
)
from scalebridge.dashapp.services.phase_d import builder, execution


def _definition(tmp_path: Path) -> PhaseDCampaignDefinition:
    config = PhaseDRunnerConfig(
        campaign_root=str(tmp_path / "campaign"),
        output_root=str(tmp_path / "output"),
        matrix_run_id="matrix_1",
        phase_c_campaign_run_id="phase_c_1",
        ml_policies=("monthly_distributed_holdout",),
        ml_input_lags=(1, 4, 6),
        ml_target_horizons=(1,),
        ob_policies=("seasonal_distributed",),
        mlflow_enabled=True,
        mlflow_experiment_name="phase_d_test",
    )
    return PhaseDCampaignDefinition(
        phase_d_campaign_id="phase_d_exec_test",
        parent_generation_campaign_id="generation_test",
        parent_phase_c_run_key="generation_test::phase_c_1",
        machine_id="test-machine",
        runner_config=config,
    )


def test_execution_command_reuses_builder_compiler_and_adds_only_runtime_flags(
    tmp_path,
    monkeypatch,
):
    definition = _definition(tmp_path)
    fake_script = tmp_path / "run_phase_d_campaign.py"
    fake_script.write_text("# test", encoding="utf-8")

    monkeypatch.setattr(execution, "load_definition", lambda _campaign_id: definition)
    monkeypatch.setattr(builder, "runner_script", lambda: fake_script)

    cmd = execution.command_for(
        "phase_d_exec_test",
        phase_d_run_id="phase_d_20260827_103200",
        resume=True,
        continue_on_error=True,
        dry_run=True,
    )

    assert "--phase-d-run-id" in cmd
    assert cmd[cmd.index("--phase-d-run-id") + 1] == "phase_d_20260827_103200"
    assert "--resume" in cmd
    assert "--continue-on-error" in cmd
    assert "--dry-run" in cmd
    assert "--overwrite-existing" not in cmd

    # Saved scientific choices remain present and unchanged.
    assert [cmd[i + 1] for i, value in enumerate(cmd) if value == "--ml-input-lag"] == [
        "1",
        "4",
        "6",
    ]
    assert [cmd[i + 1] for i, value in enumerate(cmd) if value == "--ml-target-horizon"] == [
        "1"
    ]
    assert "--mlflow" in cmd


def test_execution_rejects_resume_plus_overwrite(tmp_path, monkeypatch):
    definition = _definition(tmp_path)
    fake_script = tmp_path / "run_phase_d_campaign.py"
    fake_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(execution, "load_definition", lambda _campaign_id: definition)
    monkeypatch.setattr(builder, "runner_script", lambda: fake_script)

    with pytest.raises(ValueError, match="mutually exclusive"):
        execution.command_for(
            "phase_d_exec_test",
            phase_d_run_id="phase_d_test_run",
            resume=True,
            overwrite_existing=True,
        )


def test_runtime_warning_detects_preexisting_selected_outputs(tmp_path, monkeypatch):
    definition = _definition(tmp_path)
    monkeypatch.setattr(execution, "load_definition", lambda _campaign_id: definition)
    monkeypatch.setattr(
        execution,
        "detect_current_machine",
        lambda: {"machine_id": "test-machine"},
    )
    monkeypatch.setattr(
        execution,
        "resolve_phase_c_context",
        lambda _run_key: {
            "aggregation_rows": [
                {
                    "case_id": "case_1",
                    "aggregation_id": "identity",
                    "weight_mode": "equal",
                    "aggregation_run_id": "aggr_1",
                }
            ]
        },
    )

    existing = (
        tmp_path
        / "output"
        / "phase_d"
        / "cases"
        / "case_1"
        / "aggregation_runs"
        / "aggr_1"
    )
    existing.mkdir(parents=True)

    warnings = execution.runtime_warnings(
        "phase_d_exec_test",
        phase_d_run_id="phase_d_new_run",
    )
    assert any(
        row["code"] == "existing_outputs_without_recovery_mode"
        and row["severity"] == "danger"
        for row in warnings
    )

    resume_warnings = execution.runtime_warnings(
        "phase_d_exec_test",
        phase_d_run_id="phase_d_new_run",
        resume=True,
    )
    assert any(row["code"] == "resume_existing_outputs" for row in resume_warnings)
