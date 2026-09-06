from __future__ import annotations

from scalebridge.dashapp.help.registry.heat_input_help import HEAT_INPUT_HELP_ENTRIES


def test_execution_help_matches_the_simplified_complete_phase_c_surface():
    required = {
        "heat_input.page.execution",
        "heat_input.execution.saved_definition",
        "heat_input.execution.phase_c_run_id",
        "heat_input.execution.dry_run",
        "heat_input.execution.command",
        "heat_input.execution.progress",
        "heat_input.execution.console",
    }
    assert required.issubset(HEAT_INPUT_HELP_ENTRIES)
    assert "heat_input.execution.start_stage" not in HEAT_INPUT_HELP_ENTRIES
    assert "heat_input.execution.stop_stage" not in HEAT_INPUT_HELP_ENTRIES
    assert "heat_input.execution.overwrite" not in HEAT_INPUT_HELP_ENTRIES
