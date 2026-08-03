"""Major-page composition for Settings."""

from dash import html

from ...components.help import help_button
from ...layout.navigation import build_horizontal_tabs
from .paths.page import build_layout as build_paths_layout
from .machines.page import build_layout as build_machines_layout
from .environments.page import build_layout as build_environments_layout
from .mlflow.page import build_layout as build_mlflow_layout
from .visualization.page import build_layout as build_visualization_layout
from .help.page import build_layout as build_help_layout

_SUBPAGE_BUILDERS = {
    "paths": build_paths_layout,
    "machines": build_machines_layout,
    "environments": build_environments_layout,
    "mlflow": build_mlflow_layout,
    "visualization": build_visualization_layout,
    "help": build_help_layout,
}


def build_page(active_tab: str | None = None):
    """Build the Settings page with horizontal tabs."""
    selected = active_tab or next(iter(_SUBPAGE_BUILDERS))
    builder = _SUBPAGE_BUILDERS.get(selected, next(iter(_SUBPAGE_BUILDERS.values())))
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("Settings", className="page-title"),
                            help_button("page.settings", compact=False),
                        ],
                        className="title-with-help",
                    ),
                ],
                className="page-heading",
            ),
            build_horizontal_tabs("settings", selected),
            html.Div(builder(), id="settings-subpage-content"),
        ],
        className="major-page",
    )


def get_subpage_builder(tab_id: str):
    """Return the builder for a selected horizontal tab."""
    return _SUBPAGE_BUILDERS.get(tab_id, next(iter(_SUBPAGE_BUILDERS.values())))
