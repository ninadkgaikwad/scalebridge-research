"""Phase D Results browser for runner-created thermal-model-ready datasets."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.phase_d.results_data import ALL, run_options


_FILTERS = [
    ("Building", "building_type"),
    ("Weather", "weather_location"),
    ("Case", "case_id"),
    ("Aggregation Family", "aggregation_family"),
    ("Aggregation ID", "aggregation_id"),
    ("Weight Mode", "weight_mode"),
    ("Rule Set", "rule_set"),
    ("Silo", "silo"),
    ("Relationship", "mode"),
    ("Independent Zone", "independent_zone_id"),
    ("Heat Representation", "heat_representation"),
    ("Policy", "policy_name"),
    ("Input Lag", "input_lag"),
    ("Target Horizon", "target_horizon"),
]


def _label(text: str, help_key: str):
    return html.Div(
        [html.Strong(text), help_button(help_key, compact=True)],
        className="phase-d-label-with-help",
    )


def _filter_control(label: str, column: str):
    return dbc.Col(
        [
            html.Label(label),
            dcc.Dropdown(
                id=f"phase-d-results-filter-{column.replace('_', '-')}",
                options=[{"label": "All", "value": ALL}],
                value=ALL,
                clearable=False,
            ),
        ],
        lg=3,
        md=4,
        sm=6,
    )


def _graph_with_legend():
    return dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    id="phase-d-results-graph",
                    style={"height": "600px"},
                    config={"displaylogo": False, "responsive": True},
                ),
                width=9,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H5("Plotted traces", className="mb-1"),
                        html.P("Click a trace to show or hide it.", className="text-muted small mb-2"),
                        html.Div(
                            id="phase-d-results-legend",
                            children=html.Div("Select and load a dataset.", className="text-muted small"),
                            className="phase-d-results-scroll-legend",
                        ),
                    ],
                    className="phase-d-results-legend-panel",
                ),
                width=3,
            ),
        ],
        className="g-0 mt-2 align-items-stretch",
    )


def build_layout():
    options = run_options()
    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.H3("Results"), help_button("phase_d.page.results")], className="title-with-help"),
                    html.P(
                        "Filter the compact Phase D dataset registry, select one final realization, and inspect exactly the time-series columns recorded by its manifest.",
                        className="text-muted mb-0",
                    ),
                ],
                className="phase-d-subpage-heading",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.Strong("Phase D Run"), help_button("phase_d.results.run", compact=True)], className="title-with-help"),
                        html.Div(
                            [
                                dcc.Dropdown(
                                    id="phase-d-results-run",
                                    options=options,
                                    placeholder="Select completed Phase D run",
                                    className="flex-grow-1",
                                ),
                                dbc.Button("Refresh", id="phase-d-results-refresh", color="secondary", outline=True),
                            ],
                            className="d-flex gap-2 align-items-end",
                        ),
                        html.Div(id="phase-d-results-run-summary", className="mt-3"),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.Strong("Dataset Filters"), help_button("phase_d.results.filters", compact=True)], className="title-with-help"),
                        html.P(
                            "Filters are mutually constrained by dataset_registry.csv: each selection immediately removes incompatible choices from the other filters. Valid selections are preserved, incompatible older selections are cleared, and All leaves that dimension unrestricted.",
                            className="small text-muted",
                        ),
                        dbc.Row([_filter_control(label, column) for label, column in _FILTERS], className="g-3"),
                        html.Div(id="phase-d-results-match-count", className="small text-muted mt-3"),
                        html.Hr(),
                        _label("Selected Final Dataset", "phase_d.results.dataset"),
                        dcc.Dropdown(
                            id="phase-d-results-dataset",
                            placeholder="Choose one matching final Phase D dataset",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.Strong("Dataset Summary"), help_button("phase_d.results.dataset_summary", compact=True)], className="title-with-help"),
                        html.Div(id="phase-d-results-dataset-summary"),
                        html.Div(id="phase-d-results-partition-summary", className="mt-3"),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.Strong("Thermal-Model Time Series"), help_button("phase_d.results.timeseries", compact=True)], className="title-with-help"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label("Signals", "phase_d.results.signals"),
                                        dcc.Dropdown(id="phase-d-results-signals", multi=True),
                                    ],
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        _label("Partition", "phase_d.results.partition"),
                                        dcc.Dropdown(id="phase-d-results-partition", clearable=False),
                                    ],
                                    lg=3,
                                ),
                                dbc.Col(
                                    [
                                        _label("Maximum plotted rows", "phase_d.results.max_points"),
                                        dcc.Dropdown(
                                            id="phase-d-results-max-points",
                                            options=[
                                                {"label": "5,000", "value": 5000},
                                                {"label": "20,000", "value": 20000},
                                                {"label": "50,000", "value": 50000},
                                                {"label": "All selected rows", "value": 0},
                                            ],
                                            value=20000,
                                            clearable=False,
                                        ),
                                    ],
                                    lg=3,
                                ),
                            ],
                            className="g-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Time Range"),
                                        dcc.RadioItems(
                                            id="phase-d-results-range-mode",
                                            options=[
                                                {"label": "Full", "value": "full"},
                                                {"label": "Custom", "value": "custom"},
                                            ],
                                            value="full",
                                            inline=True,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Input(id="phase-d-results-start", type="datetime-local", disabled=True),
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Input(id="phase-d-results-end", type="datetime-local", disabled=True),
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Button("Load Plot", id="phase-d-results-load-plot", color="primary", className="w-100"),
                                    md=3,
                                ),
                            ],
                            className="g-3 mt-1 align-items-end",
                        ),
                        html.Div(id="phase-d-results-plot-message", className="mt-2"),
                        _graph_with_legend(),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Dropdown(
                                            id="phase-d-results-plot-download-format",
                                            options=[{"label": "CSV", "value": "csv"}, {"label": "Parquet", "value": "parquet"}],
                                            value="csv",
                                            clearable=False,
                                            style={"minWidth": "10rem"},
                                        ),
                                        dbc.Button(
                                            "Download visible plotted data",
                                            id="phase-d-results-download-plot",
                                            color="secondary",
                                            outline=True,
                                        ),
                                    ],
                                    className="d-flex gap-2 align-items-center flex-wrap",
                                ),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Download selected dataset",
                                            id="phase-d-results-download-dataset",
                                            color="secondary",
                                            outline=True,
                                        ),
                                        dbc.Button(
                                            "Download Phase D run summary",
                                            id="phase-d-results-download-summary",
                                            color="secondary",
                                            outline=True,
                                        ),
                                    ],
                                    className="d-flex gap-2 flex-wrap",
                                ),
                            ],
                            className="d-flex justify-content-between gap-3 flex-wrap mt-3",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.Strong("Dataset Preview"), help_button("phase_d.results.preview", compact=True)], className="title-with-help"),
                        html.P("First 200 rows of the exact selected final data.parquet.", className="small text-muted"),
                        html.Div(id="phase-d-results-preview"),
                    ]
                ),
                className="phase-d-card",
            ),
            dcc.Store(id="phase-d-results-legend-state", data=[]),
            dcc.Download(id="phase-d-results-download"),
        ],
        className="phase-d-page",
    )
