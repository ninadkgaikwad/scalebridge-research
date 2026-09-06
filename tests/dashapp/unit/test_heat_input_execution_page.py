from __future__ import annotations

from scalebridge.dashapp.pages.data_pipeline.phase_c_heat_input.execution.page import (
    build_layout,
)


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _ids(layout):
    return {
        component.id: component
        for component in _walk(layout)
        if isinstance(getattr(component, "id", None), str)
    }


def test_execution_page_exposes_simple_complete_phase_c_contract():
    ids = _ids(build_layout())
    required = {
        "phase-c-execution-campaign",
        "phase-c-execution-refresh",
        "phase-c-execution-definition-summary",
        "phase-c-execution-run-id",
        "phase-c-execution-dry-run",
        "phase-c-execution-runtime-warnings",
        "phase-c-execution-command",
        "phase-c-execution-effective-config",
        "phase-c-execution-start",
        "phase-c-execution-stop",
        "phase-c-execution-status",
        "phase-c-execution-stage-progress",
        "phase-c-execution-console",
        "phase-c-execution-confirm-modal",
        "phase-c-execution-pending-action",
        "phase-c-execution-poll",
    }
    assert required.issubset(ids)


def test_execution_page_hides_recovery_and_separate_stream_controls():
    ids = _ids(build_layout())
    for removed in (
        "phase-c-execution-start-stage",
        "phase-c-execution-stop-stage",
        "phase-c-execution-overwrite",
        "phase-c-execution-stdout",
        "phase-c-execution-stderr",
    ):
        assert removed not in ids
    assert ids["phase-c-execution-dry-run"].__class__.__name__ == "Checkbox"


def test_execution_page_has_one_console_and_safe_initial_stop_state():
    ids = _ids(build_layout())
    assert ids["phase-c-execution-stop"].disabled is True
    assert "phase-c-execution-console" in ids
