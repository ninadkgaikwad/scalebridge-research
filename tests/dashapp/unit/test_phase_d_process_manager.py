from __future__ import annotations

import pytest

from scalebridge.dashapp.services.phase_d.execution import PhaseDProcessManager


def test_phase_d_process_manager_initial_snapshot_is_lightweight():
    manager = PhaseDProcessManager()
    snapshot = manager.snapshot()

    assert snapshot["status"] == "not_started"
    assert snapshot["pid"] is None
    assert snapshot["console"] == ""
    assert snapshot["phase_d_run_id"] is None


def test_phase_d_process_manager_rejects_duplicate_active_launch():
    class FakeProcess:
        def poll(self):
            return None

    manager = PhaseDProcessManager()
    manager._s.process = FakeProcess()

    with pytest.raises(RuntimeError, match="already running"):
        manager.start("phase_d_demo")
