"""Final Phase B Aggregation Results: filtering, plotting, and selected-data export."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.aggregation import result_campaign_options


def _selector(label, component_id, *, multi=True, md=4, help_key=None):
    title = [html.Label(label)]
    if help_key:
        title.append(help_button(help_key, compact=True))
    return dbc.Col(
        [
            html.Div(title, className="aggregation-label-with-help"),
            dcc.Dropdown(
                id=component_id,
                multi=multi,
                placeholder=f"Select {label.lower()}",
            ),
        ],
        md=md,
    )


def build_layout():
    return html.Div(
        [
            html.Div(
                [html.H3("Results"), help_button("aggregation.page.results")],
                className="title-with-help",
            ),
            html.P(
                "Select one or more Aggregation campaigns, intersect all multi-select "
                "filters, plot the selected stored signals, and export exactly "
                "the same selected data.",
                className="page-description",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label("Aggregation Campaign"),
                                    help_button(
                                        "aggregation.results.campaign", compact=True
                                    ),
                                ],
                                className="aggregation-label-with-help",
                            ),
                            dcc.Dropdown(
                                id="aggregation-results-campaign",
                                options=result_campaign_options(),
                                multi=True,
                                placeholder="Select Aggregation campaign(s)",
                            ),
                        ],
                        md=10,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Refresh",
                            id="aggregation-results-refresh",
                            color="secondary",
                            outline=True,
                            className="mt-4",
                        ),
                        md=2,
                    ),
                ],
                className="g-3",
            ),
            html.Div(id="aggregation-results-metadata", className="mt-3"),
            dbc.Row(
                [
                    _selector(
                        "Building Type",
                        "aggregation-results-building",
                        md=6,
                        help_key="aggregation.results.building",
                    ),
                    _selector(
                        "Weather Location",
                        "aggregation-results-weather",
                        md=6,
                        help_key="aggregation.results.weather",
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    _selector(
                        "Strategy",
                        "aggregation-results-strategy",
                        help_key="aggregation.results.strategy",
                    ),
                    _selector(
                        "Weight Mode",
                        "aggregation-results-weight",
                        help_key="aggregation.results.weight",
                    ),
                    _selector(
                        "Rule Set",
                        "aggregation-results-ruleset",
                        help_key="aggregation.results.ruleset",
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    _selector(
                        "Aggregation Zone",
                        "aggregation-results-zone",
                        help_key="aggregation.results.zone",
                    ),
                    _selector(
                        "Variable",
                        "aggregation-results-variable",
                        help_key="aggregation.results.variable",
                    ),
                    _selector(
                        "Variable Column",
                        "aggregation-results-variable-column",
                        help_key="aggregation.results.variable_column",
                    ),
                ],
                className="g-3 mt-1",
            ),
            dbc.Row(
                [
                    _selector(
                        "Run",
                        "aggregation-results-run",
                        help_key="aggregation.results.run",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Label("Time Range"),
                                    help_button(
                                        "aggregation.results.time_range", compact=True
                                    ),
                                ],
                                className="aggregation-label-with-help",
                            ),
                            dcc.RadioItems(
                                id="aggregation-results-range-mode",
                                options=[
                                    {"label": "Full range", "value": "full"},
                                    {"label": "Custom range", "value": "custom"},
                                ],
                                value="full",
                                inline=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3 mt-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Start datetime"),
                            dbc.Input(
                                id="aggregation-results-start",
                                type="datetime-local",
                                disabled=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("End datetime"),
                            dbc.Input(
                                id="aggregation-results-end",
                                type="datetime-local",
                                disabled=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3 mt-1",
            ),
            dbc.Button(
                "Load and Plot Selected Signals",
                id="aggregation-results-plot-button",
                color="primary",
                className="mt-3",
            ),
            dbc.Button(
                "Clear Signal Selection",
                id="aggregation-results-clear",
                outline=True,
                className="mt-3 ms-2",
            ),
            html.Div(id="aggregation-results-message", className="mt-2"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id="aggregation-results-graph",
                            style={"height": "620px"},
                            config={"displaylogo": False, "responsive": True},
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H5("Plotted signals", className="mb-1"),
                                html.P(
                                    "Click a signal to show or hide its trace.",
                                    className="text-muted small mb-2",
                                ),
                                html.Div(
                                    id="aggregation-results-custom-legend",
                                    children=html.Div(
                                        "Plot signals to populate the legend.",
                                        className="text-muted small",
                                    ),
                                    style={
                                        "height": "555px",
                                        "overflowY": "auto",
                                        "overflowX": "hidden",
                                        "paddingRight": "0.35rem",
                                    },
                                ),
                            ],
                            style={
                                "height": "620px",
                                "borderLeft": "1px solid rgba(120,120,120,0.25)",
                                "paddingLeft": "1rem",
                                "paddingTop": "0.75rem",
                            },
                        ),
                        width=3,
                    ),
                ],
                className="g-0 mt-2 align-items-stretch",
            ),
            dcc.Store(id="aggregation-results-index", data=[]),
            dcc.Store(id="aggregation-results-variable-catalog", data=[]),
            dcc.Store(id="aggregation-results-legend-state", data=[]),
            html.Hr(),
            html.H5("Download Selected Data"),
            html.P(
                "Export the same Aggregation runs, zones, variables, variable columns, "
                "and datetime range used for visualization. The ZIP includes a lineage "
                "manifest."
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Export format"),
                            dcc.Dropdown(
                                id="aggregation-results-export-format",
                                options=[
                                    {"label": "CSV ZIP", "value": "csv"},
                                    {"label": "Parquet ZIP", "value": "parquet"},
                                ],
                                value="csv",
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Download Selected Data",
                            id="aggregation-results-download-button",
                            color="secondary",
                            className="mt-4",
                        ),
                        md=4,
                    ),
                ],
                className="g-3",
            ),
            html.Div(id="aggregation-results-download-message", className="mt-2"),
            dcc.Download(id="aggregation-results-download"),
        ],
        className="page-content-container aggregation-page",
    )
