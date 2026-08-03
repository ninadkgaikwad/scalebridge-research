"""Major-page composition for Model Catalog."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .browse.page import build_layout as build_browse_layout
from .model_details.page import build_layout as build_model_details_layout
from .portability.page import build_layout as build_portability_layout
from .validation.page import build_layout as build_validation_layout
from .comparison_tray.page import build_layout as build_comparison_tray_layout

_SUBPAGE_BUILDERS = {
    "browse": build_browse_layout,
    "model_details": build_model_details_layout,
    "portability": build_portability_layout,
    "validation": build_validation_layout,
    "comparison_tray": build_comparison_tray_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Model Catalog page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Model Catalog", className="page-title"),
                            help_button("page.model_catalog", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("model_catalog", selected),
            html.Div(builder(), id="model_catalog-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
