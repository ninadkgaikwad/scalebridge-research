from __future__ import annotations

from pathlib import Path

import pytest

from scalebridge.dashapp.schemas.pipeline.phase_d import PhaseDCampaignDefinition, PhaseDRunnerConfig
from scalebridge.dashapp.services.phase_d import builder
from scalebridge.dashapp.services.phase_d import definition_store


def _config(tmp_path: Path):
    return PhaseDRunnerConfig(
        campaign_root=str(tmp_path / "campaign"),
        output_root=str(tmp_path / "output"),
        matrix_run_id="matrix_1",
        phase_c_campaign_run_id="phase_c_1",
        aggregation_ids=("identity", "custom_v1"),
        weight_modes=("equal", "floor_area"),
        case_ids=("case_1",),
        max_aggregation_runs=10,
        heat_representation="components",
        qzivr_separate=True,
        ml_policies=("monthly_distributed_holdout", "chronological_holdout", "seasonal_holdout"),
        ml_input_lags=(1, 3, 12),
        ml_target_horizons=(1, 6, 12),
        ob_policies=("seasonal_distributed", "seasonal_block_holdout", "contiguous_identification", "custom_datetime_ranges"),
        ci_start_datetime="2001-04-01T00:05:00",
        cdr_train_ranges=("2001-01-01T00:05:00/2001-01-22T00:05:00",),
        cdr_test_ranges=("2001-01-22T00:05:00/2001-01-29T00:05:00",),
        mlflow_enabled=True,
        mlflow_experiment_name="PhaseD",
        mlflow_run_name="builder-test",
        mlflow_strict=True,
    )


def test_runner_config_rejects_invalid_fractions(tmp_path):
    with pytest.raises(ValueError):
        PhaseDRunnerConfig(
            campaign_root=str(tmp_path),
            matrix_run_id="m",
            phase_c_campaign_run_id="pc",
            ml_train_fraction=0.8,
            ml_test_fraction=0.15,
            ml_validation_fraction=0.15,
        )


def test_command_compiler_maps_saved_definition_to_general_runner(tmp_path, monkeypatch):
    fake_script = tmp_path / "run_phase_d_campaign.py"
    fake_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(builder, "runner_script", lambda: fake_script)
    definition = PhaseDCampaignDefinition(
        phase_d_campaign_id="phase_d_test",
        parent_generation_campaign_id="generation_test",
        parent_phase_c_run_key="generation_test::phase_c_1",
        machine_id="test-machine",
        runner_config=_config(tmp_path),
    )
    cmd = builder.command_for_definition(definition, phase_d_run_id="phase_d_20260826_160000", dry_run=True)
    assert "--campaign-root" in cmd
    assert cmd.count("--aggregation-id") == 2
    assert cmd.count("--weight-mode") == 2
    assert cmd.count("--case-id") == 1
    assert cmd.count("--ml-policy") == 3
    assert cmd.count("--ml-input-lag") == 3
    assert cmd.count("--ml-target-horizon") == 3
    assert [cmd[i + 1] for i, value in enumerate(cmd) if value == "--ml-target-horizon"] == ["1", "6", "12"]
    assert cmd.count("--ob-policy") == 4
    assert "--qzivr-separate" in cmd
    assert "--cdr-train-range" in cmd
    assert "--cdr-test-range" in cmd
    assert "--mlflow" in cmd
    assert "--mlflow-strict" in cmd
    assert "--dry-run" in cmd
    assert cmd[cmd.index("--phase-d-run-id") + 1] == "phase_d_20260826_160000"


def test_command_compiler_rejects_resume_plus_overwrite(tmp_path, monkeypatch):
    fake_script = tmp_path / "run_phase_d_campaign.py"
    fake_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(builder, "runner_script", lambda: fake_script)
    definition = PhaseDCampaignDefinition(
        phase_d_campaign_id="phase_d_test",
        parent_generation_campaign_id="generation_test",
        parent_phase_c_run_key="generation_test::phase_c_1",
        machine_id="test-machine",
        runner_config=_config(tmp_path),
    )
    with pytest.raises(ValueError):
        builder.command_for_definition(definition, resume=True, overwrite_existing=True)


def test_command_compiler_emits_only_selected_policy_parameters(tmp_path, monkeypatch):
    fake_script = tmp_path / "run_phase_d_campaign.py"
    fake_script.write_text("# test", encoding="utf-8")
    monkeypatch.setattr(builder, "runner_script", lambda: fake_script)
    config = PhaseDRunnerConfig(
        campaign_root=str(tmp_path / "campaign"),
        matrix_run_id="matrix_1",
        phase_c_campaign_run_id="phase_c_1",
        ml_policies=("seasonal_holdout",),
        ml_input_lags=(12,),
        ml_target_horizons=(6,),
        ob_policies=("contiguous_identification",),
        ci_start_datetime="2001-01-01T00:05:00",
    )
    definition = PhaseDCampaignDefinition(
        phase_d_campaign_id="phase_d_policy_test",
        parent_generation_campaign_id="generation_test",
        parent_phase_c_run_key="generation_test::phase_c_1",
        machine_id="test-machine",
        runner_config=config,
    )
    cmd = builder.command_for_definition(definition)
    assert "--ml-sh-train-seasons" in cmd
    assert "--ml-train-fraction" not in cmd
    assert "--ci-start-datetime" in cmd
    assert "--sd-train-days" not in cmd
    assert "--sbh-train-seasons" not in cmd
    assert "--cdr-train-range" not in cmd


def test_runner_config_allows_hidden_unselected_policy_values_to_be_irrelevant(tmp_path):
    config = PhaseDRunnerConfig(
        campaign_root=str(tmp_path),
        matrix_run_id="m",
        phase_c_campaign_run_id="pc",
        ml_policies=("seasonal_holdout",),
        ml_train_fraction=0.9,
        ml_test_fraction=0.9,
        ml_validation_fraction=0.9,
        ob_policies=("contiguous_identification",),
        sbh_train_seasons=("winter",),
        sbh_test_seasons=("winter",),
    )
    assert config.ml_policies == ("seasonal_holdout",)


def test_runner_config_rejects_duplicate_target_horizons(tmp_path):
    with pytest.raises(ValueError, match="ml_target_horizons cannot contain duplicates"):
        PhaseDRunnerConfig(
            campaign_root=str(tmp_path),
            matrix_run_id="m",
            phase_c_campaign_run_id="pc",
            ml_target_horizons=(1, 6, 6),
        )


def test_definition_store_migrates_legacy_scalar_target_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(definition_store, "definition_root", lambda: tmp_path)
    payload = {
        "schema_version": "phase_d_dash_campaign_definition_v1",
        "phase_d_campaign_id": "legacy_phase_d",
        "parent_generation_campaign_id": "generation_test",
        "parent_phase_c_run_key": "generation_test::phase_c_1",
        "machine_id": "test-machine",
        "runner_config": {
            "campaign_root": str(tmp_path / "campaign"),
            "matrix_run_id": "matrix_1",
            "phase_c_campaign_run_id": "phase_c_1",
            "ml_target_horizon": 6,
        },
    }
    import json
    (tmp_path / "legacy_phase_d.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = definition_store.load_definition("legacy_phase_d")
    assert loaded.runner_config.ml_target_horizons == (6,)
