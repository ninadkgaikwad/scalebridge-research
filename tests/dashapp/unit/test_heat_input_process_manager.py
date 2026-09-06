from __future__ import annotations

from types import SimpleNamespace

import pytest

from scalebridge.dashapp.services.heat_input.execution import (
    PhaseCProcessManager,
)


def test_initial_snapshot_is_not_started():
    manager = PhaseCProcessManager()
    snapshot = manager.snapshot()

    assert snapshot["status"] == "not_started"
    assert snapshot["pid"] is None
    assert snapshot["stdout"] == ""
    assert snapshot["stderr"] == ""
    assert snapshot["stage_statuses"]["C1"] == "pending"


def test_process_manager_rejects_duplicate_active_launch(monkeypatch):
    class FakeProcess:
        def poll(self):
            return None

    manager = PhaseCProcessManager()
    manager._s.process = FakeProcess()

    with pytest.raises(RuntimeError, match="already running"):
        manager.start("phase_c_demo")


def test_progress_parser_tracks_runner_command_markers():
    manager = PhaseCProcessManager()
    manager._update_progress("[1/19] C1 audit")
    snapshot = manager.snapshot()

    assert snapshot["current_stage"] == "C1"
    assert snapshot["current_command"] == "C1 audit"
    assert snapshot["command_sequence"] == 1
    assert snapshot["command_total"] == 19
    assert snapshot["stage_statuses"]["C1"] == "running"

    manager._update_progress("[4/19] C2 feature construction")
    snapshot = manager.snapshot()

    assert snapshot["stage_statuses"]["C1"] == "completed"
    assert snapshot["stage_statuses"]["C2"] == "running"


def test_progress_parser_ignores_non_runner_lines():
    manager = PhaseCProcessManager()
    manager._update_progress("ordinary scientific log output")
    snapshot = manager.snapshot()

    assert snapshot["current_stage"] is None
    assert all(value == "pending" for value in snapshot["stage_statuses"].values())
