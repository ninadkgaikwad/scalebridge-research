"""Major-page composition for Campaigns."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .browser.page import build_layout as build_browser_layout
from .builder.page import build_layout as build_builder_layout
from .matrix_preview.page import build_layout as build_matrix_preview_layout
from .monitor.page import build_layout as build_monitor_layout
from .templates.page import build_layout as build_templates_layout

_SUBPAGE_BUILDERS = {
    "browser": build_browser_layout,
    "builder": build_builder_layout,
    "matrix_preview": build_matrix_preview_layout,
    "monitor": build_monitor_layout,
    "templates": build_templates_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Campaigns page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Campaigns", className="page-title"),
                            help_button("page.campaigns", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("campaigns", selected),
            html.Div(builder(), id="campaigns-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
