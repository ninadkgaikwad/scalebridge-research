"""Major-page composition for Home."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .overview.page import build_layout as build_overview_layout
from .recent_activity.page import build_layout as build_recent_activity_layout
from .system_status.page import build_layout as build_system_status_layout
from .quick_actions.page import build_layout as build_quick_actions_layout

_SUBPAGE_BUILDERS = {
    "overview": build_overview_layout,
    "recent_activity": build_recent_activity_layout,
    "system_status": build_system_status_layout,
    "quick_actions": build_quick_actions_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Home page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Home", className="page-title"),
                            help_button("page.home", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("home", selected),
            html.Div(builder(), id="home-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
