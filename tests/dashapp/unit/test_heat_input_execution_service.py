from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig
from scalebridge.dashapp.services.heat_input import execution


def _definition(tmp_path: Path):
    campaign_root = tmp_path / "gen_demo"
    campaign_root.mkdir(parents=True, exist_ok=True)
    config = PhaseCCampaignConfig(
        campaign_root=str(campaign_root),
        campaign_id="gen_demo",
        matrix_run_id="aggregation_matrix_demo",
        mlflow_enabled=False,
        mlflow_validation_mode="none",
    )
    return SimpleNamespace(
        phase_c_campaign_id="phase_c_demo",
        parent_aggregation_campaign_id="agg_demo",
        parent_generation_campaign_id="gen_demo",
        machine_id="laptop",
        display_name="Demo",
        runner_config=config,
    )


def test_suggested_run_id_has_runner_timestamp_suffix():
    value = execution.suggested_run_id(datetime(2026, 8, 21, 11, 12, 13))
    assert value == "phase_c_20260821_111213"


def test_effective_config_only_applies_execution_time_overrides(monkeypatch, tmp_path):
    fake = _definition(tmp_path)
    monkeypatch.setattr(execution, "load_definition", lambda _: fake)

    resolved = execution.effective_config(
        "phase_c_demo",
        phase_c_run_id="phase_c_20260821_111213",
        start_stage="C6",
        stop_stage="C8",
        overwrite_existing=True,
    )

    assert resolved.phase_c_run_id == "phase_c_20260821_111213"
    assert resolved.start_stage == "C6"
    assert resolved.stop_stage == "C8"
    assert resolved.overwrite_existing is True
    assert resolved.matrix_run_id == "aggregation_matrix_demo"
    assert resolved.campaign_id == "gen_demo"


def test_effective_config_rejects_invalid_stage_order(monkeypatch, tmp_path):
    fake = _definition(tmp_path)
    monkeypatch.setattr(execution, "load_definition", lambda _: fake)

    with pytest.raises(ValueError, match="start_stage"):
        execution.effective_config(
            "phase_c_demo",
            start_stage="C8",
            stop_stage="C6",
        )


def test_command_uses_current_python_general_runner_and_config(monkeypatch, tmp_path):
    fake = _definition(tmp_path)
    fake_script = tmp_path / "run_phase_c_campaign.py"
    fake_script.write_text("raise SystemExit(0)", encoding="utf-8")
    fake_config = tmp_path / "phase_c_effective_config.json"

    monkeypatch.setattr(execution, "load_definition", lambda _: fake)
    monkeypatch.setattr(execution, "runner_script", lambda: fake_script)
    monkeypatch.setattr(execution, "execution_config_path", lambda *_: fake_config)

    command = execution.command_for(
        "phase_c_demo",
        phase_c_run_id="phase_c_20260821_111213",
        dry_run=True,
    )

    assert command == [
        sys.executable,
        "-u",
        str(fake_script),
        "--config",
        str(fake_config),
        "--dry-run",
    ]


def test_materialize_effective_config_writes_raw_runner_contract(monkeypatch, tmp_path):
    fake = _definition(tmp_path)
    target = tmp_path / "request" / "phase_c_effective_config.json"

    monkeypatch.setattr(execution, "load_definition", lambda _: fake)
    monkeypatch.setattr(execution, "execution_config_path", lambda *_: target)

    config, path = execution.materialize_effective_config(
        "phase_c_demo",
        phase_c_run_id="phase_c_20260821_111213",
        start_stage="C5",
        stop_stage="C7",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == target
    assert payload["schema_version"] == "0.2.0"
    assert payload["phase_c_run_id"] == "phase_c_20260821_111213"
    assert payload["start_stage"] == "C5"
    assert payload["stop_stage"] == "C7"
    assert "phase_c_campaign_id" not in payload
    assert config.model_dump(mode="json") == payload


def test_confirmation_reasons_cover_overwrite_recovery_and_machine(monkeypatch, tmp_path):
    fake = _definition(tmp_path)
    fake.machine_id = "lab-pc"
    monkeypatch.setattr(execution, "load_definition", lambda _: fake)
    monkeypatch.setattr(
        execution,
        "detect_current_machine",
        lambda: {"machine_id": "laptop"},
    )

    reasons = execution.confirmation_reasons(
        "phase_c_demo",
        start_stage="C6",
        stop_stage="C8",
        overwrite_existing=True,
    )
    codes = {row["code"] for row in reasons}

    assert {"machine_mismatch", "recovery_range", "overwrite"}.issubset(codes)

def test_effective_config_rejects_run_id_without_timestamp_suffix(monkeypatch, tmp_path):
    fake = _definition(tmp_path)
    monkeypatch.setattr(execution, "load_definition", lambda _: fake)

    with pytest.raises(ValueError, match="YYYYMMDD_HHMMSS"):
        execution.effective_config(
            "phase_c_demo",
            phase_c_run_id="phase_c_bad_id",
        )


def test_existing_run_directory_is_a_confirmation_warning(monkeypatch, tmp_path):
    fake = _definition(tmp_path)
    run_id = "phase_c_20260821_111213"
    run_root = (
        Path(fake.runner_config.campaign_root)
        / "heat_input_regression"
        / "campaign_runs"
        / run_id
    )
    run_root.mkdir(parents=True)
    monkeypatch.setattr(execution, "load_definition", lambda _: fake)
    monkeypatch.setattr(
        execution,
        "detect_current_machine",
        lambda: {"machine_id": "laptop"},
    )

    reasons = execution.confirmation_reasons(
        "phase_c_demo",
        phase_c_run_id=run_id,
    )
    assert "run_id_exists" in {row["code"] for row in reasons}

