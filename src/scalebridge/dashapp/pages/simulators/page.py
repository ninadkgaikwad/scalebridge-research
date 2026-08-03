"""Major-page composition for Simulators."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .smartcommunitysim.page import build_layout as build_smartcommunitysim_layout
from .smartbuildingssim.page import build_layout as build_smartbuildingssim_layout
from .opendsssim.page import build_layout as build_opendsssim_layout
from .co_simulationsim.page import build_layout as build_co_simulationsim_layout

_SUBPAGE_BUILDERS = {
    "smartcommunitysim": build_smartcommunitysim_layout,
    "smartbuildingssim": build_smartbuildingssim_layout,
    "opendsssim": build_opendsssim_layout,
    "co_simulationsim": build_co_simulationsim_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Simulators page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Simulators", className="page-title"),
                            help_button("page.simulators", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("simulators", selected),
            html.Div(builder(), id="simulators-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
