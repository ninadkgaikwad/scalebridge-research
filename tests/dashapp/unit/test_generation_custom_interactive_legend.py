from scalebridge.dashapp.pages.data_pipeline.phase_a_generation.results.page import build_layout
from scalebridge.dashapp.pages.data_pipeline.phase_a_generation.callbacks import (
    _build_custom_legend,
    _apply_custom_results_plot_layout,
)
import plotly.graph_objects as go


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            if hasattr(child, "__class__"):
                yield from _walk(child)
    elif hasattr(children, "__class__"):
        yield from _walk(children)


def test_results_layout_has_graph_75_legend_25_panels():
    layout = build_layout()
    components = list(_walk(layout))
    graph = next(c for c in components if getattr(c, "id", None) == "generation-results-graph")
    legend = next(c for c in components if getattr(c, "id", None) == "generation-results-custom-legend")
    graph_col = next(c for c in components if getattr(c, "children", None) is graph)
    assert getattr(graph_col, "width", None) == 9
    assert legend.style["overflowY"] == "auto"


def test_custom_plot_hides_native_plotly_legend():
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[2, 3], name="trace")
    _apply_custom_results_plot_layout(fig)
    assert fig.layout.showlegend is False
    assert fig.layout.hovermode == "x unified"


def test_custom_legend_items_are_clickable_and_dim_hidden_traces():
    items = [
        {
            "index": 0,
            "visible": True,
            "color": "#123456",
            "full_name": "OfficeSmall | Weather | case | run | Variable",
            "primary_label": "OfficeSmall | Tampa, FL",
            "variable_name": "Variable",
        },
        {
            "index": 1,
            "visible": False,
            "color": "#654321",
            "full_name": "RestaurantFastFood | Weather | case | run | Variable",
            "primary_label": "RestaurantFastFood | Tampa, FL",
            "variable_name": "Variable",
        },
    ]
    children = _build_custom_legend(items)
    buttons = [c for c in children if c.__class__.__name__ == "Button"]
    assert len(buttons) == 2
    assert buttons[0].id == {"type": "generation-results-legend-toggle", "index": 0}
    assert buttons[0].style["opacity"] == 1.0
    assert buttons[1].style["opacity"] < 1.0
