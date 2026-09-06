from __future__ import annotations

from scalebridge.dashapp.pages.data_pipeline.phase_d_thermal_model_data.execution import page


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


def test_phase_d_execution_exposes_only_runtime_controls(monkeypatch):
    monkeypatch.setattr(page, "list_execution_definitions", lambda: [])
    layout = page.build_layout()
    ids = _ids(layout)

    required = {
        "phase-d-execution-campaign",
        "phase-d-execution-refresh",
        "phase-d-execution-run-id",
        "phase-d-execution-dry-run",
        "phase-d-execution-resume",
        "phase-d-execution-overwrite",
        "phase-d-execution-continue-on-error",
        "phase-d-execution-command",
        "phase-d-execution-start",
        "phase-d-execution-stop",
        "phase-d-execution-status",
        "phase-d-execution-progress",
        "phase-d-execution-console",
        "phase-d-execution-poll",
        "phase-d-execution-confirm-modal",
    }
    assert required.issubset(ids)

    for forbidden in (
        "phase-d-execution-ml-policy",
        "phase-d-execution-ml-input-lag",
        "phase-d-execution-ml-target-horizon",
        "phase-d-execution-ob-policy",
        "phase-d-execution-heat-representation",
        "phase-d-execution-mlflow",
    ):
        assert forbidden not in ids


def test_phase_d_execution_uses_checkboxes_not_switches(monkeypatch):
    monkeypatch.setattr(page, "list_execution_definitions", lambda: [])
    layout = page.build_layout()
    ids = _ids(layout)
    for component_id in (
        "phase-d-execution-dry-run",
        "phase-d-execution-resume",
        "phase-d-execution-overwrite",
        "phase-d-execution-continue-on-error",
    ):
        assert ids[component_id].__class__.__name__ == "Checkbox"
    assert not any(component.__class__.__name__ == "Switch" for component in _walk(layout))
