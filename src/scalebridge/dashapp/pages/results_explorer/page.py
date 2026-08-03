"""Major-page composition for Results Explorer."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .single_experiment.page import build_layout as build_single_experiment_layout
from .cross_experiment.page import build_layout as build_cross_experiment_layout
from .time_series.page import build_layout as build_time_series_layout
from .metrics.page import build_layout as build_metrics_layout
from .parameters.page import build_layout as build_parameters_layout
from .diagnostics.page import build_layout as build_diagnostics_layout
from .simulator_results.page import build_layout as build_simulator_results_layout

_SUBPAGE_BUILDERS = {
    "single_experiment": build_single_experiment_layout,
    "cross_experiment": build_cross_experiment_layout,
    "time_series": build_time_series_layout,
    "metrics": build_metrics_layout,
    "parameters": build_parameters_layout,
    "diagnostics": build_diagnostics_layout,
    "simulator_results": build_simulator_results_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Results Explorer page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Results Explorer", className="page-title"),
                            help_button("page.results_explorer", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("results_explorer", selected),
            html.Div(builder(), id="results_explorer-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
