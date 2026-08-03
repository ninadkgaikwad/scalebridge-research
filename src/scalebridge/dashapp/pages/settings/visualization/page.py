"""User-controlled visualization preferences."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from ..components import settings_heading


def _control(label, component, description):
    return html.Div(
        [
            html.Label(label, className="settings-control-label"),
            component,
            html.Small(description, className="settings-control-help"),
        ],
        className="settings-control-group",
    )


def build_layout():
    """Build browser-persisted visualization controls."""
    return html.Div(
        [
            settings_heading(
                "Visualization",
                (
                    "Browser-persisted display and publication-export "
                    "preferences. These controls are independent of machine "
                    "identity."
                ),
                "bi-palette",
                "subpage.settings.visualization",
            ),
            dbc.Alert(
                (
                    "Visualization preferences are stored in this browser. "
                    "They do not modify machine paths, environments, or MLflow."
                ),
                color="info",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Display Preferences"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    _control(
                                        "Application Theme",
                                        dcc.Dropdown(
                                            id="settings-visual-theme",
                                            options=[
                                                {
                                                    "label": "System Preference",
                                                    "value": "system",
                                                },
                                                {
                                                    "label": "Light Mode",
                                                    "value": "light",
                                                },
                                                {
                                                    "label": "Dark Mode",
                                                    "value": "dark",
                                                },
                                            ],
                                            value="system",
                                            clearable=False,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Controls the preferred application color mode.",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    _control(
                                        "Display Unit System",
                                        dcc.Dropdown(
                                            id="settings-visual-display-units",
                                            options=[
                                                {
                                                    "label": "User Selectable",
                                                    "value": "selectable",
                                                },
                                                {
                                                    "label": "SI",
                                                    "value": "si",
                                                },
                                                {
                                                    "label": "IP / US Customary",
                                                    "value": "ip",
                                                },
                                            ],
                                            value="selectable",
                                            clearable=False,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Default units used in interactive displays.",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    _control(
                                        "Publication Unit System",
                                        dcc.Dropdown(
                                            id="settings-visual-publication-units",
                                            options=[
                                                {"label": "SI", "value": "si"},
                                                {
                                                    "label": "IP / US Customary",
                                                    "value": "ip",
                                                },
                                            ],
                                            value="si",
                                            clearable=False,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Default unit system for paper-ready exports.",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    _control(
                                        "Interactive Figure Width",
                                        dcc.Dropdown(
                                            id="settings-visual-figure-width",
                                            options=[
                                                {
                                                    "label": "Responsive",
                                                    "value": "responsive",
                                                },
                                                {
                                                    "label": "Single Column",
                                                    "value": "single-column",
                                                },
                                                {
                                                    "label": "Double Column",
                                                    "value": "double-column",
                                                },
                                                {
                                                    "label": "Full Page",
                                                    "value": "full-page",
                                                },
                                            ],
                                            value="responsive",
                                            clearable=False,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Default sizing mode for interactive figures.",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        )
                    ),
                ],
                className="studio-card",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Publication and Table Exports"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    _control(
                                        "PNG Resolution",
                                        dcc.Dropdown(
                                            id="settings-visual-png-dpi",
                                            options=[
                                                {"label": "150 DPI", "value": 150},
                                                {"label": "300 DPI", "value": 300},
                                                {"label": "600 DPI", "value": 600},
                                            ],
                                            value=300,
                                            clearable=False,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Raster resolution for publication exports.",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    _control(
                                        "Vector Export Formats",
                                        dcc.Checklist(
                                            id="settings-visual-vector-formats",
                                            options=[
                                                {"label": " SVG", "value": "svg"},
                                                {"label": " PDF", "value": "pdf"},
                                            ],
                                            value=["svg", "pdf"],
                                            inline=True,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Preferred vector formats for publication figures.",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    _control(
                                        "Table Decimal Places",
                                        dcc.Dropdown(
                                            id="settings-visual-decimals",
                                            options=[
                                                {"label": str(value), "value": value}
                                                for value in range(0, 7)
                                            ],
                                            value=3,
                                            clearable=False,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Default numerical precision in result tables.",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    _control(
                                        "Rows per Table Page",
                                        dcc.Dropdown(
                                            id="settings-visual-page-size",
                                            options=[
                                                {"label": str(value), "value": value}
                                                for value in (10, 25, 50, 100)
                                            ],
                                            value=25,
                                            clearable=False,
                                            persistence=True,
                                            persistence_type="local",
                                        ),
                                        "Default number of rows shown per table page.",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        )
                    ),
                ],
                className="studio-card",
            ),
        ],
        className="page-content-container settings-page",
    )
