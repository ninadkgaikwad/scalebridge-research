"""Application shell layout and shell callbacks."""

from dash import Input, Output, State, callback, clientside_callback, dcc, html
import dash_bootstrap_components as dbc

from ..headers import build_app_header
from ..navigation import build_sidebar
from ...components.help import build_help_modal
from ...constants import (
    APP_NAME,
    APP_DESCRIPTION,
    APP_SUBTITLE,
    AUTHOR_NAME,
    AUTHOR_ROLE,
    AUTHOR_UNIT,
    AUTHOR_LAB,
    AUTHOR_INSTITUTION,
    EXTERNAL_LINKS,
)


def build_about_modal():
    """Build the application About dialog."""
    link_rows = [
        ("Ninad Kiran Gaikwad", EXTERNAL_LINKS["author"], "bi-person-badge"),
        ("SCALE Laboratory", EXTERNAL_LINKS["scale_lab"], "bi-lightbulb"),
        ("WSU School of Electrical Engineering and Computer Science", EXTERNAL_LINKS["eecs"], "bi-mortarboard"),
        ("Washington State University", EXTERNAL_LINKS["wsu"], "bi-building"),
    ]
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(APP_NAME)),
            dbc.ModalBody(
                [
                    html.P(APP_SUBTITLE, className="about-subtitle"),
                    html.P(APP_DESCRIPTION),
                    html.Hr(),
                    html.H5("Research and Software Development"),
                    html.P(
                        [
                            html.Strong(AUTHOR_NAME),
                            html.Br(),
                            AUTHOR_ROLE,
                            html.Br(),
                            AUTHOR_UNIT,
                            html.Br(),
                            AUTHOR_LAB,
                            html.Br(),
                            AUTHOR_INSTITUTION,
                        ]
                    ),
                    html.Div(
                        [
                            html.A(
                                [
                                    html.I(className=f"bi {icon} me-2"),
                                    label,
                                    html.I(className="bi bi-box-arrow-up-right ms-2"),
                                ],
                                href=url,
                                target="_blank",
                                rel="noopener noreferrer",
                                className="about-link",
                            )
                            for label, url, icon in link_rows
                        ],
                        className="about-links",
                    ),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-about-modal", color="secondary")
            ),
        ],
        id="about-modal",
        is_open=False,
        centered=True,
        scrollable=True,
        size="lg",
    )


def build_app_shell():
    """Build the full application shell."""
    return html.Div(
        [
            dcc.Location(id="app-location", refresh=False),
            dcc.Store(id="sidebar-state", storage_type="local", data={"collapsed": False}),
            dcc.Store(id="resolved-theme", storage_type="memory"),
            build_sidebar(),
            html.Div(
                [
                    build_app_header(),
                    html.Main(id="page-content", className="app-main-content"),
                ],
                id="app-workspace",
                className="app-workspace",
            ),
            build_help_modal(),
            build_about_modal(),
        ],
        id="app-root",
        className="app-root",
    )


def register_shell_callbacks() -> None:
    """Register callbacks controlling shell interaction."""

    @callback(
        Output("sidebar-state", "data"),
        Input("sidebar-toggle", "n_clicks"),
        State("sidebar-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(_clicks, state):
        collapsed = bool((state or {}).get("collapsed", False))
        return {"collapsed": not collapsed}

    @callback(
        Output("app-root", "className"),
        Input("sidebar-state", "data"),
    )
    def apply_sidebar_state(state):
        collapsed = bool((state or {}).get("collapsed", False))
        return "app-root sidebar-collapsed" if collapsed else "app-root"

    @callback(
        Output("about-modal", "is_open"),
        Input("open-about-modal", "n_clicks"),
        Input("close-about-modal", "n_clicks"),
        State("about-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_about(open_clicks, close_clicks, is_open):
        return not is_open

    clientside_callback(
        """
        function(themeValue) {
            const requested = themeValue || "system";
            const prefersDark = window.matchMedia &&
                window.matchMedia("(prefers-color-scheme: dark)").matches;
            const resolved = requested === "system"
                ? (prefersDark ? "dark" : "light")
                : requested;

            document.documentElement.setAttribute("data-theme", resolved);
            document.documentElement.setAttribute("data-theme-preference", requested);
            localStorage.setItem("bgirs-theme", requested);
            return resolved;
        }
        """,
        Output("resolved-theme", "data"),
        Input("theme-selector", "value"),
    )
