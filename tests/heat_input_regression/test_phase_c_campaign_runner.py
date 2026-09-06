from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig


def _runner_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "heat_input_regression" / "run_phase_c_campaign.py"
    spec = importlib.util.spec_from_file_location("phase_c_campaign_runner_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _command(commands, prefix: str) -> list[str]:
    for name, command in commands:
        if name.startswith(prefix):
            return command
    raise AssertionError(f"Missing command with prefix {prefix!r}")


def test_config_file_plus_cli_override(tmp_path: Path) -> None:
    runner = _runner_module()
    config_path = tmp_path / "phase_c.json"
    config_path.write_text(
        json.dumps({"learning_rate": 0.01, "max_epochs": 123, "validation_profile": "some"}),
        encoding="utf-8",
    )
    args = runner.parse_args(["--config", str(config_path), "--learning-rate", "0.02", "--validation", "full"])
    config = runner.resolve_configuration(args)
    assert config.learning_rate == 0.02
    assert config.max_epochs == 123
    assert config.validation_profile == "full"


def test_runner_forwards_general_scientific_options(tmp_path: Path) -> None:
    runner = _runner_module()
    campaign_root = tmp_path / "campaign_alpha"
    campaign_root.mkdir()
    config = PhaseCCampaignConfig(
        campaign_root=str(campaign_root),
        matrix_run_id="matrix_1",
        phase_c_run_id="phase_c_general_20260820_160000",
        case_id="case_1",
        aggregation_id="identity",
        weight_mode="equal",
        aggregate_zone_id="Zone_A",
        model_ids=("QAC", "PHVAC"),
        minimum_sample_count=1200,
        internal_gain_predictor_method="contribution_sum",
        hvac_target_method="absolute_zone_sensible",
        split_strategy="chronological_fraction",
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        ridge_alpha=0.5,
        learning_rate=0.01,
        max_epochs=111,
        patience=17,
        training_seed=99,
        estimator_types=("closed_form_linear", "pytorch_linear"),
        pytorch_devices=("cpu",),
        write_full_predictions=False,
        mlflow_strict=False,
        mlflow_log_model_artifacts=True,
    )
    layout = runner.build_layout(
        campaign_root=campaign_root,
        matrix_run_id="matrix_1",
        phase_c_run_id=config.phase_c_run_id,
    )
    runner.get_help_text = lambda script: script.read_text(encoding="utf-8")
    commands = runner.build_pipeline_commands(layout, config)

    c1 = _command(commands, "C1 ")
    assert ["--case-id", "case_1"] == c1[c1.index("--case-id"):c1.index("--case-id") + 2]
    assert "--internal-gain-predictor-method" in c1
    assert "contribution_sum" in c1
    assert "--hvac-target-method" in c1
    assert "absolute_zone_sensible" in c1

    c3 = _command(commands, "C3 ")
    assert "--split-strategy" in c3 and "chronological_fraction" in c3
    assert "--train-fraction" in c3 and "0.6" in c3

    c6 = _command(commands, "C6 ")
    assert c6.count("--estimator-type") == 2
    assert c6.count("--model-id") == 2
    assert "--ridge-alpha" in c6 and "0.5" in c6
    assert "--learning-rate" in c6 and "0.01" in c6
    assert "--seed" in c6 and "99" in c6

    c7 = _command(commands, "C7 ")
    assert "--no-full-predictions" in c7
    # Do not apply a requested-device filter automatically: it would exclude
    # closed-form artifacts (which are recorded as CPU) in mixed-estimator runs.
    assert "--requested-device" not in c7

    c9 = _command(commands, "C9 ")
    assert "--non-strict" in c9
    assert "--log-model-artifacts" in c9


def test_false_fit_intercept_override_is_forwarded(tmp_path: Path) -> None:
    runner = _runner_module()
    campaign_root = tmp_path / "campaign_beta"
    campaign_root.mkdir()
    config = PhaseCCampaignConfig(
        campaign_root=str(campaign_root),
        matrix_run_id="matrix_1",
        phase_c_run_id="phase_c_general_20260820_160001",
        fit_intercept_override=False,
        validation_profile="none",
        mlflow_enabled=False,
        stop_stage="C6",
    )
    layout = runner.build_layout(campaign_root=campaign_root, matrix_run_id="matrix_1", phase_c_run_id=config.phase_c_run_id)
    runner.get_help_text = lambda script: script.read_text(encoding="utf-8")
    commands = runner.build_pipeline_commands(layout, config)
    assert "--no-fit-intercept" in _command(commands, "C6 ")


def test_max_zones_is_a_real_c1_option() -> None:
    runner = _runner_module()
    source_text = runner.require_script("audit_aggregation_for_heat_input_regression.py").read_text(encoding="utf-8")
    assert 'add_argument("--max-zones"' in source_text


def test_boolean_cli_overrides_can_disable_true_config_values(tmp_path: Path) -> None:
    runner = _runner_module()
    config_path = tmp_path / "phase_c_boolean.json"
    config_path.write_text(
        json.dumps({
            "continue_on_error": True,
            "overwrite_existing": True,
            "mlflow_enabled": True,
            "mlflow_log_model_artifacts": True,
            "run_residual_gap_audit": True,
        }),
        encoding="utf-8",
    )
    args = runner.parse_args([
        "--config", str(config_path),
        "--no-continue-on-error",
        "--no-overwrite-existing",
        "--no-mlflow-enabled",
        "--no-mlflow-log-model-artifacts",
        "--no-run-residual-gap-audit",
    ])
    config = runner.resolve_configuration(args)
    assert config.continue_on_error is False
    assert config.overwrite_existing is False
    assert config.mlflow_enabled is False
    assert config.mlflow_log_model_artifacts is False
    assert config.run_residual_gap_audit is False


def test_repeatable_downstream_zone_recovery_filter_is_forwarded(tmp_path: Path) -> None:
    runner = _runner_module()
    campaign_root = tmp_path / "campaign_gamma"
    campaign_root.mkdir()
    config = PhaseCCampaignConfig(
        campaign_root=str(campaign_root),
        matrix_run_id="matrix_1",
        phase_c_run_id="phase_c_general_20260820_160002",
        downstream_aggregate_zone_ids=("Zone_A", "Zone_B"),
        validation_profile="none",
        mlflow_enabled=False,
        start_stage="C6",
        stop_stage="C8",
    )
    layout = runner.build_layout(
        campaign_root=campaign_root,
        matrix_run_id="matrix_1",
        phase_c_run_id=config.phase_c_run_id,
    )
    runner.get_help_text = lambda script: script.read_text(encoding="utf-8")
    commands = runner.build_pipeline_commands(layout, config)
    for prefix in ("C6 ", "C7 ", "C8 "):
        cmd = _command(commands, prefix)
        assert cmd.count("--aggregate-zone-id") == 2
        assert "Zone_A" in cmd and "Zone_B" in cmd


def test_explicit_requested_device_recovery_filters_are_forwarded(tmp_path: Path) -> None:
    runner = _runner_module()
    campaign_root = tmp_path / "campaign_delta"
    campaign_root.mkdir()
    config = PhaseCCampaignConfig(
        campaign_root=str(campaign_root),
        matrix_run_id="matrix_1",
        phase_c_run_id="phase_c_general_20260820_160003",
        evaluation_requested_devices=("cpu", "auto"),
        inference_requested_devices=("cuda",),
        validation_profile="none",
        mlflow_enabled=False,
        start_stage="C7",
        stop_stage="C8",
    )
    layout = runner.build_layout(
        campaign_root=campaign_root,
        matrix_run_id="matrix_1",
        phase_c_run_id=config.phase_c_run_id,
    )
    runner.get_help_text = lambda script: script.read_text(encoding="utf-8")
    commands = runner.build_pipeline_commands(layout, config)
    c7 = _command(commands, "C7 ")
    c8 = _command(commands, "C8 ")
    assert c7.count("--requested-device") == 2
    assert "cpu" in c7 and "auto" in c7
    assert c8.count("--requested-device") == 1
    assert "cuda" in c8


def test_c1_direct_aggregation_run_root_is_forwarded(tmp_path: Path) -> None:
    runner = _runner_module()
    campaign_root = tmp_path / "campaign_epsilon"
    campaign_root.mkdir()
    direct_root = tmp_path / "aggregation_run"
    direct_root.mkdir()
    config = PhaseCCampaignConfig(
        campaign_root=str(campaign_root),
        c1_aggregation_run_root=str(direct_root),
        phase_c_run_id="phase_c_general_20260820_160004",
        validation_profile="none",
        mlflow_enabled=False,
        start_stage="C1",
        stop_stage="C1",
    )
    layout = runner.build_layout(
        campaign_root=campaign_root,
        matrix_run_id=None,
        phase_c_run_id=config.phase_c_run_id,
        direct_aggregation_run_root=config.c1_aggregation_run_root,
    )
    runner.get_help_text = lambda script: script.read_text(encoding="utf-8")
    commands = runner.build_pipeline_commands(layout, config)
    c1 = _command(commands, "C1 ")
    assert "--aggregation-run-root" in c1
    assert str(direct_root) in c1
    assert "--matrix-run-id" not in c1


def test_every_lower_level_cli_option_is_classified() -> None:
    """A new C1-C9/validator knob must be promoted or explicitly runner-owned."""
    import ast

    runner = _runner_module()
    public_options = {
        "--absolute-tolerance", "--aggregate-zone-id", "--aggregation-id",
        "--aggregation-run-root", "--campaign-id", "--campaign-root", "--case-id",
        "--coefficient-atol", "--continue-on-error", "--estimator-type",
        "--expected-cadence-seconds", "--expected-canonical-row-count", "--expected-row-count",
        "--experiment-name", "--fail-on-conflicting-source-values", "--fit-intercept",
        "--fraction-tolerance", "--generated-data-root", "--hvac-target-method",
        "--inspect-source-files", "--internal-gain-predictor-method", "--learning-rate",
        "--log-model-artifacts", "--matrix-run-id", "--max-artifact-bytes", "--max-artifacts",
        "--max-c4-models", "--max-epochs", "--max-model-datasets", "--max-zones",
        "--metric-atol", "--metric-rtol", "--minimum-sample-count", "--minimum-split-samples",
        "--model-id", "--neighbor-radius", "--no-compact-artifacts", "--no-full-predictions",
        "--non-strict", "--overwrite-existing", "--patience", "--phase-c-run-id",
        "--prediction-atol", "--prediction-preview-rows", "--prediction-rtol", "--preview-rows",
        "--pytorch-device", "--random-seed", "--relative-tolerance", "--reload-atol",
        "--reload-rtol", "--requested-device", "--ridge-alpha", "--run-name", "--seed",
        "--skip-pytorch", "--split-strategy", "--test-fraction", "--tolerance",
        "--train-fraction", "--validation-fraction", "--validation-mode", "--weight-mode",
    }
    internal_options = set(PhaseCCampaignConfig.capability_manifest()["internal_only_cli_options"])

    discovered: set[str] = set()
    for script in runner.SCRIPT_ROOT.glob("*.py"):
        if script.name == "run_phase_c_campaign.py":
            continue
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                    discovered.add(arg.value)

    assert discovered == public_options | internal_options
    assert not (public_options & internal_options)
