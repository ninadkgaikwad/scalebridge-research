"""Major-page composition for Publication Studio."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .figure_builder.page import build_layout as build_figure_builder_layout
from .table_builder.page import build_layout as build_table_builder_layout
from .saved_specifications.page import build_layout as build_saved_specifications_layout
from .export_history.page import build_layout as build_export_history_layout

_SUBPAGE_BUILDERS = {
    "figure_builder": build_figure_builder_layout,
    "table_builder": build_table_builder_layout,
    "saved_specifications": build_saved_specifications_layout,
    "export_history": build_export_history_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Publication Studio page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Publication Studio", className="page-title"),
                            help_button("page.publication_studio", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("publication_studio", selected),
            html.Div(builder(), id="publication_studio-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
