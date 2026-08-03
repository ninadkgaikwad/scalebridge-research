"""Major-page composition for Data Pipeline."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .phase_a_generation.page import build_layout as build_phase_a_generation_layout
from .phase_b_aggregation.page import build_layout as build_phase_b_aggregation_layout
from .phase_c_heat_input.page import build_layout as build_phase_c_heat_input_layout
from .phase_d_thermal_model_data.page import build_layout as build_phase_d_thermal_model_data_layout

_SUBPAGE_BUILDERS = {
    "phase_a_generation": build_phase_a_generation_layout,
    "phase_b_aggregation": build_phase_b_aggregation_layout,
    "phase_c_heat_input": build_phase_c_heat_input_layout,
    "phase_d_thermal_model_data": build_phase_d_thermal_model_data_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Data Pipeline page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Data Pipeline", className="page-title"),
                            help_button("page.data_pipeline", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("data_pipeline", selected),
            html.Div(builder(), id="data_pipeline-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
