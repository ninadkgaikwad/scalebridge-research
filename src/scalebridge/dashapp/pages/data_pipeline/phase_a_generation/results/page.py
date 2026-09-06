from dash import dcc, html
import dash_bootstrap_components as dbc
from scalebridge.dashapp.services.generation import campaign_options


def dd(label, id, multi=True):
    return dbc.Col([html.Label(label), dcc.Dropdown(id=id, multi=multi)], md=4)


def build_layout():
    return html.Div(
        [
            html.H3("Results"),
            html.P(
                "Select one generated campaign. Metadata and parquet signals are "
                "loaded only for that campaign."
            ),
            dcc.Dropdown(
                id="generation-results-campaign",
                options=campaign_options(),
                placeholder="Select a campaign",
            ),
            html.Div(id="generation-results-metadata", className="mt-3"),
            dbc.Row(
                [
                    dd("Building type", "generation-results-building"),
                    dd("Weather location", "generation-results-weather"),
                    dd("Case ID", "generation-results-case"),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dd("Run ID", "generation-results-run"),
                    dd("Variable name", "generation-results-variable"),
                    dd("Variable column / key", "generation-results-key"),
                ],
                className="g-3 mt-1",
            ),
            dcc.RadioItems(
                id="generation-results-range-mode",
                options=[
                    {"label": "Full range", "value": "full"},
                    {"label": "Custom range", "value": "custom"},
                ],
                value="full",
                inline=True,
                className="mt-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Start datetime"),
                            dbc.Input(
                                id="generation-results-start",
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
                                id="generation-results-end",
                                type="datetime-local",
                                disabled=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                id="generation-results-range-row",
                className="g-3 mt-1",
            ),
            dbc.Button(
                "Load and Plot Selected Signals",
                id="generation-results-plot-button",
                color="primary",
                className="mt-3",
            ),
            dbc.Button(
                "Clear Selection",
                id="generation-results-clear",
                outline=True,
                className="mt-3 ms-2",
            ),
            html.Div(id="generation-results-message", className="mt-2"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id="generation-results-graph",
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
                                    id="generation-results-custom-legend",
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
            dcc.Store(id="generation-results-legend-state", data=[]),
            html.Hr(),
            html.H5("Download Selected Data"),
            html.P(
                "Export the same filtered signals and datetime range used for "
                "visualization. The ZIP preserves Generation nomenclature and "
                "includes a provenance manifest for external publication plotting."
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Export format"),
                            dcc.Dropdown(
                                id="generation-results-export-format",
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
                            id="generation-results-download-button",
                            color="secondary",
                            className="mt-4",
                        ),
                        md=4,
                    ),
                ],
                className="g-3",
            ),
            html.Div(id="generation-results-download-message", className="mt-2"),
            dcc.Download(id="generation-results-download"),
            dcc.Store(id="generation-results-index"),
        ],
        className="page-content-container generation-page",
    )
