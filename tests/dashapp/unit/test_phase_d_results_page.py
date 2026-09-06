from dash import dcc

from scalebridge.dashapp.components.results.scroll_legend import build_scroll_legend
from scalebridge.dashapp.pages.data_pipeline.phase_d_thermal_model_data.results.page import (
    _graph_with_legend,
    build_layout,
)


def _walk(component):
    # Shared result helpers may return a list/tuple of Dash children rather than
    # one wrapper component. Traverse that root sequence explicitly.
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)
        return

    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if isinstance(child, (list, tuple)) or hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


def _collect_ids(component):
    return [
        getattr(item, "id")
        for item in _walk(component)
        if getattr(item, "id", None) is not None
    ]


def test_phase_d_results_page_has_registry_dataset_plot_and_download_contract():
    ids = _collect_ids(build_layout())

    # BGIRS intentionally uses dict pattern IDs for contextual help, so only
    # string IDs belong in this direct component-ID contract.
    string_ids = {cid for cid in ids if isinstance(cid, str)}
    pattern_ids = [cid for cid in ids if isinstance(cid, dict)]

    required = {
        "phase-d-results-run",
        "phase-d-results-dataset",
        "phase-d-results-signals",
        "phase-d-results-partition",
        "phase-d-results-graph",
        "phase-d-results-legend",
        "phase-d-results-preview",
        "phase-d-results-download-plot",
        "phase-d-results-download-dataset",
        "phase-d-results-download-summary",
    }
    assert required.issubset(string_ids)
    assert any(row.get("type") == "context-help-button" for row in pattern_ids)


def test_phase_d_results_main_graph_uses_locked_75_25_external_legend_layout():
    row = _graph_with_legend()
    assert len(row.children) == 2
    assert row.children[0].width == 9
    assert row.children[1].width == 3

    left_ids = {
        cid for cid in _collect_ids(row.children[0]) if isinstance(cid, str)
    }
    right_ids = {
        cid for cid in _collect_ids(row.children[1]) if isinstance(cid, str)
    }
    assert "phase-d-results-graph" in left_ids
    assert "phase-d-results-legend" in right_ids


def test_phase_d_clickable_legend_uses_valid_dash_pattern_ids():
    legend = build_scroll_legend(
        [
            {
                "index": 0,
                "visible": True,
                "name": "Zone temperature",
                "primary_label": "zone_temperature",
                "secondary_label": "Z1 | model_input | degC",
                "color": "#123456",
            }
        ],
        toggle_type="phase-d-results-trace",
    )
    assert isinstance(legend, list)
    ids = _collect_ids(legend)
    assert {"type": "phase-d-results-trace", "index": 0} in ids
