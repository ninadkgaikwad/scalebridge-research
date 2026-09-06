from __future__ import annotations

from pathlib import Path
import sys

import pytest

from scalebridge.dashapp.services.aggregation.execution import (
    AggregationProcessManager,
    command_for,
    command_text,
    runner_script,
)


def test_initial_snapshot_has_terminal_or_not_started_status():
    manager = AggregationProcessManager()
    snapshot = manager.snapshot()
    assert snapshot["status"] == "not_started"
    assert snapshot["pid"] is None
    assert snapshot["console"] == ""


def test_runner_script_is_authoritative_b2_cli():
    path = runner_script()
    assert path.name == "run_aggregation_campaign.py"
    assert path.parent.name == "aggregation"
    assert path.is_file()


def test_command_uses_current_python_and_saved_definition(monkeypatch, tmp_path):
    import scalebridge.dashapp.services.aggregation.execution as execution

    fake_script = tmp_path / "scripts" / "aggregation" / "run_aggregation_campaign.py"
    fake_script.parent.mkdir(parents=True)
    fake_script.write_text("raise SystemExit(0)", encoding="utf-8")
    fake_definition = tmp_path / "campaign.json"
    fake_definition.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(execution, "runner_script", lambda: fake_script)
    monkeypatch.setattr(execution, "definition_path", lambda _: fake_definition)

    command = command_for("demo")
    assert command == [
        sys.executable,
        "-u",
        str(fake_script),
        "--campaign-definition",
        str(fake_definition),
    ]
    assert "--campaign-definition" in command_text("demo")


def test_manager_rejects_duplicate_active_launch(monkeypatch):
    import scalebridge.dashapp.services.aggregation.execution as execution

    class FakeProcess:
        pid = 123
        stdout = iter(())
        def poll(self):
            return None

    manager = AggregationProcessManager()
    manager._s.process = FakeProcess()

    with pytest.raises(RuntimeError, match="already running"):
        manager.start("demo")
