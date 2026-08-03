"""Major-page composition for Thermal Modeling."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .datasets.page import build_layout as build_datasets_layout
from .ann_sequence_models.page import build_layout as build_ann_sequence_models_layout
from .scientific_ml.page import build_layout as build_scientific_ml_layout
from .optimization.page import build_layout as build_optimization_layout
from .bayesian_inference.page import build_layout as build_bayesian_inference_layout
from .run_monitor.page import build_layout as build_run_monitor_layout

_SUBPAGE_BUILDERS = {
    "datasets": build_datasets_layout,
    "ann_sequence_models": build_ann_sequence_models_layout,
    "scientific_ml": build_scientific_ml_layout,
    "optimization": build_optimization_layout,
    "bayesian_inference": build_bayesian_inference_layout,
    "run_monitor": build_run_monitor_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Thermal Modeling page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Thermal Modeling", className="page-title"),
                            help_button("page.thermal_modeling", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("thermal_modeling", selected),
            html.Div(builder(), id="thermal_modeling-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
