"""Horizontal subpage tabs."""

from dash import dcc, html

from ...components.help import help_button
from ...pages.registry import SUBPAGES


def build_horizontal_tabs(page_id: str, active_tab: str | None = None):
    """Build horizontal tabs for one major page."""
    items = SUBPAGES[page_id]
    selected = active_tab or items[0]["id"]

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Subpages", className="horizontal-tabs-label"),
                    help_button("navigation.tabs"),
                ],
                className="horizontal-tabs-heading",
            ),
            dcc.Tabs(
                id={"type": "major-page-tabs", "page_id": page_id},
                value=selected,
                persistence=True,
                persistence_type="session",
                className="horizontal-tabs",
                parent_className="horizontal-tabs-parent",
                children=[
                    dcc.Tab(
                        label=item["label"],
                        value=item["id"],
                        className="horizontal-tab",
                        selected_className="horizontal-tab-selected",
                    )
                    for item in items
                ],
            ),
        ],
        className="horizontal-tabs-container",
    )
