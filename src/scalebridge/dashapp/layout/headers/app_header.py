"""Top application header."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from ...constants import APP_NAME, APP_SUBTITLE, THEME_OPTIONS
from ...components.help import help_button


def build_app_header():
    """Build the responsive application header."""
    return html.Header(
        [
            html.Div(
                [
                    dbc.Button(
                        html.I(
                            className="bi bi-list",
                            **{"aria-hidden": "true"},
                        ),
                        id="sidebar-toggle",
                        color="link",
                        className="header-icon-button",
                        title="Toggle Sidebar",
                    ),
                    html.Div(
                        [
                            html.H1(APP_NAME, className="app-title"),
                            html.P(APP_SUBTITLE, className="app-subtitle"),
                        ],
                        className="app-title-group",
                    ),
                ],
                className="app-header-left",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.I(
                                className="bi bi-circle-half theme-control-icon",
                                **{"aria-hidden": "true"},
                            ),
                            dcc.Dropdown(
                                id="theme-selector",
                                options=list(THEME_OPTIONS),
                                value="system",
                                clearable=False,
                                searchable=False,
                                persistence=True,
                                persistence_type="local",
                                className="theme-selector",
                            ),
                            help_button("theme.selector"),
                        ],
                        className="theme-control",
                    ),
                ],
                className="app-header-actions",
            ),
        ],
        className="app-header",
    )
