from __future__ import annotations

from scalebridge.dashapp.pages.data_pipeline.phase_b_aggregation.results.page import build_layout


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


def test_final_results_exposes_all_locked_filters_plot_and_download_controls():
    ids = _ids(build_layout())
    required = {
        "aggregation-results-campaign",
        "aggregation-results-building",
        "aggregation-results-weather",
        "aggregation-results-climate",
        "aggregation-results-strategy",
        "aggregation-results-weight",
        "aggregation-results-ruleset",
        "aggregation-results-zone",
        "aggregation-results-variable",
        "aggregation-results-variable-column",
        "aggregation-results-run",
        "aggregation-results-range-mode",
        "aggregation-results-start",
        "aggregation-results-end",
        "aggregation-results-plot-button",
        "aggregation-results-graph",
        "aggregation-results-custom-legend",
        "aggregation-results-export-format",
        "aggregation-results-download-button",
        "aggregation-results-download",
    }
    assert required.issubset(ids)


def test_all_locked_dropdown_filters_are_multiselect_and_always_present():
    ids = _ids(build_layout())
    for component_id in (
        "aggregation-results-campaign",
        "aggregation-results-building",
        "aggregation-results-weather",
        "aggregation-results-climate",
        "aggregation-results-strategy",
        "aggregation-results-weight",
        "aggregation-results-ruleset",
        "aggregation-results-zone",
        "aggregation-results-variable",
        "aggregation-results-variable-column",
        "aggregation-results-run",
    ):
        assert component_id in ids
        assert ids[component_id].multi is True


def test_plot_uses_locked_75_25_graph_legend_columns():
    layout = build_layout()
    columns = [
        component
        for component in _walk(layout)
        if component.__class__.__name__ == "Col"
    ]
    graph_col = next(
        col for col in columns
        if any(
            getattr(child, "id", None) == "aggregation-results-graph"
            for child in _walk(col)
        )
    )
    legend_col = next(
        col for col in columns
        if any(
            getattr(child, "id", None) == "aggregation-results-custom-legend"
            for child in _walk(col)
        )
    )
    assert graph_col.width == 9
    assert legend_col.width == 3
