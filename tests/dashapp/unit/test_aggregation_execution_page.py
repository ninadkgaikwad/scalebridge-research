from __future__ import annotations

from scalebridge.dashapp.pages.data_pipeline.phase_b_aggregation.execution.page import (
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


def test_execution_page_exposes_managed_process_controls():
    ids = _ids(build_layout())
    required = {
        "aggregation-execution-campaign",
        "aggregation-execution-refresh",
        "aggregation-execution-definition-summary",
        "aggregation-execution-command",
        "aggregation-execution-start",
        "aggregation-execution-stop",
        "aggregation-execution-status",
        "aggregation-execution-console",
        "aggregation-execution-poll",
    }
    assert required.issubset(ids)


def test_stop_is_disabled_initially():
    ids = _ids(build_layout())
    assert ids["aggregation-execution-stop"].disabled is True


def test_execution_alerts_use_wrapping_contract():
    from pathlib import Path

    callbacks = (
        Path(__file__).parents[3]
        / "src"
        / "scalebridge"
        / "dashapp"
        / "pages"
        / "data_pipeline"
        / "phase_b_aggregation"
        / "callbacks.py"
    ).read_text(encoding="utf-8")

    assert callbacks.count('className="aggregation-wrap-alert"') >= 3
