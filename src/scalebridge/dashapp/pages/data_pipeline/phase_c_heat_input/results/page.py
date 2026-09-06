"""Phase C Tab 3: model-oriented, lazy, read-only Results workspace."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.heat_input.results_data import run_options


def _label(text: str, help_key: str):
    return html.Div(
        [html.Label(text), help_button(help_key, compact=True)],
        className="heat-input-label-with-help",
    )


def _multi_filter(label: str, component_id: str, help_key: str, md: int = 4):
    return dbc.Col(
        [
            _label(label, help_key),
            dcc.Dropdown(id=component_id, multi=True, placeholder=f"All {label.lower()}"),
        ],
        md=md,
    )


def _graph_with_legend(graph_id: str, legend_id: str, *, height: int = 580):
    """Match the Phase A/B 75/25 graph + external scrollable legend pattern."""
    return dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    id=graph_id,
                    style={"height": f"{height}px"},
                    config={"displaylogo": False, "responsive": True},
                ),
                width=9,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H5("Plotted traces", className="mb-1"),
                        html.P(
                            "Click a trace to show or hide it.",
                            className="text-muted small mb-2",
                        ),
                        html.Div(
                            id=legend_id,
                            children=html.Div(
                                "Load a selected result to populate the legend.",
                                className="text-muted small",
                            ),
                            className="heat-input-scroll-legend",
                        ),
                    ],
                    className="heat-input-legend-panel",
                ),
                width=3,
            ),
        ],
        className="g-0 mt-2 align-items-stretch",
    )


def _plot_download_controls(prefix: str):
    """Controls that download exactly the currently visible plot traces."""
    help_key = "heat_input.results.plot_download"
    return html.Div(
        [
            html.Div(
                [
                    html.Span("Plot data", className="heat-input-download-label"),
                    help_button(help_key, compact=True),
                ],
                className="heat-input-download-heading",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id=f"phase-c-results-{prefix}-download-format",
                            options=[
                                {"label": "CSV", "value": "csv"},
                                {"label": "Parquet", "value": "parquet"},
                            ],
                            value="csv",
                            clearable=False,
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Download plotted data",
                            id=f"phase-c-results-{prefix}-download-plot-data",
                            color="secondary",
                            outline=True,
                            className="w-100",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            id=f"phase-c-results-{prefix}-plot-export-message",
                            className="small",
                        ),
                        md=5,
                    ),
                ],
                className="g-2 align-items-center",
            ),
            dcc.Download(id=f"phase-c-results-{prefix}-plot-download"),
        ],
        className="heat-input-plot-download-bar",
    )


def build_layout():
    """Build the simplified Phase C Results tab."""
    return html.Div(
        [
            html.Div(
                [html.H3("Results"), help_button("heat_input.page.results")],
                className="title-with-help",
            ),
            html.P(
                (
                    "Select a Phase C run, narrow to a model/context, and inspect the "
                    "dataset, fitted prediction, and full-year prediction trajectories. "
                    "Large artifacts are opened only after an explicit load action."
                ),
                className="page-description",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Phase C Run"),
                                help_button("heat_input.results.run", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="phase-c-results-run",
                                        options=run_options(),
                                        placeholder="Select a Phase C run",
                                    ),
                                    md=10,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Refresh",
                                        id="phase-c-results-refresh",
                                        color="secondary",
                                        outline=True,
                                    ),
                                    md=2,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(id="phase-c-results-run-summary", className="mt-3"),
                        html.Div(id="phase-c-results-mlflow", className="mt-2"),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Phase C Status"),
                                    html.Div(id="phase-c-results-stage-summary"),
                                ]
                            ),
                            className="heat-input-card heat-input-results-section h-100",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Model Availability"),
                                    html.P(
                                        (
                                            "Structural absence is a valid scientific state "
                                            "and is kept separate from invalid/missing data."
                                        ),
                                        className="small text-muted",
                                    ),
                                    html.Div(id="phase-c-results-availability-summary"),
                                ]
                            ),
                            className="heat-input-card heat-input-results-section h-100",
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Model / Context Filters"),
                                help_button("heat_input.results.filters", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "Weather is the location/climate selector; a separate "
                                "climate filter is unnecessary."
                            ),
                            className="small text-muted",
                        ),
                        dbc.Row(
                            [
                                _multi_filter(
                                    "Building Type",
                                    "phase-c-results-building",
                                    "heat_input.results.filters",
                                ),
                                _multi_filter(
                                    "Weather",
                                    "phase-c-results-weather",
                                    "heat_input.results.filters",
                                ),
                                _multi_filter(
                                    "Case ID",
                                    "phase-c-results-case",
                                    "heat_input.results.filters",
                                ),
                            ],
                            className="g-3",
                        ),
                        dbc.Row(
                            [
                                _multi_filter(
                                    "Aggregation",
                                    "phase-c-results-aggregation",
                                    "heat_input.results.filters",
                                ),
                                _multi_filter(
                                    "Weight Mode",
                                    "phase-c-results-weight",
                                    "heat_input.results.filters",
                                ),
                                _multi_filter(
                                    "Aggregate Zone",
                                    "phase-c-results-zone",
                                    "heat_input.results.filters",
                                ),
                            ],
                            className="g-3 mt-1",
                        ),
                        dbc.Row(
                            [
                                _multi_filter(
                                    "Model",
                                    "phase-c-results-model",
                                    "heat_input.results.filters",
                                    md=6,
                                ),
                                _multi_filter(
                                    "Estimator",
                                    "phase-c-results-estimator",
                                    "heat_input.results.filters",
                                    md=6,
                                ),
                            ],
                            className="g-3 mt-1",
                        ),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
            html.Div(
                [
                    html.H4("Model Trajectories", className="mt-4 mb-1"),
                    html.P(
                        (
                            "For one selected model/context: inspect the C4 regression "
                            "dataset X/Y, the C7 observed Y versus predicted Ŷ, and the C8 "
                            "full-year predicted Ŷ trajectory."
                        ),
                        className="text-muted",
                    ),
                ]
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Dataset X / Y Trajectory"),
                                help_button(
                                    "heat_input.results.dataset_trajectory", compact=True
                                ),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "C4 regression-pair data for exactly one selected model. "
                                "X is the model predictor and Y is the fitted target."
                            ),
                            className="small text-muted",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Dataset Data"),
                                        dcc.Dropdown(
                                            id="phase-c-results-dataset-resolution",
                                            options=[
                                                {"label": "Preview rows", "value": "preview"},
                                                {"label": "Full dataset", "value": "full"},
                                            ],
                                            value="preview",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Time Range"),
                                        dcc.RadioItems(
                                            id="phase-c-results-dataset-range-mode",
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
                                    dbc.Input(
                                        id="phase-c-results-dataset-start",
                                        type="datetime-local",
                                        disabled=True,
                                    ),
                                    md=2,
                                ),
                                dbc.Col(
                                    dbc.Input(
                                        id="phase-c-results-dataset-end",
                                        type="datetime-local",
                                        disabled=True,
                                    ),
                                    md=2,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Load X / Y",
                                        id="phase-c-results-load-dataset",
                                        color="primary",
                                    ),
                                    md=2,
                                ),
                            ],
                            className="g-3 align-items-end",
                        ),
                        html.Div(id="phase-c-results-dataset-message", className="mt-2"),
                        _graph_with_legend(
                            "phase-c-results-dataset-graph",
                            "phase-c-results-dataset-legend",
                        ),
                        _plot_download_controls("dataset"),
                        dcc.Store(id="phase-c-results-dataset-legend-state", data=[]),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Observed Y vs Predicted Ŷ"),
                                help_button("heat_input.results.evaluation", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "C7 fitted-model evaluation. The time-series view is the "
                                "default; scatter and residual diagnostics remain available."
                            ),
                            className="small text-muted",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Split"),
                                        dcc.Dropdown(
                                            id="phase-c-results-split",
                                            options=[
                                                {"label": "Train", "value": "train"},
                                                {"label": "Validation", "value": "validation"},
                                                {"label": "Test", "value": "test"},
                                            ],
                                            value=["test"],
                                            multi=True,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Evaluation Mode",
                                            "heat_input.results.phvac_modes",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-results-evaluation-mode",
                                            options=[
                                                {"label": "Direct", "value": "direct"},
                                                {"label": "Oracle", "value": "oracle"},
                                                {"label": "Chained", "value": "chained"},
                                            ],
                                            multi=True,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Prediction Data"),
                                        dcc.Dropdown(
                                            id="phase-c-results-evaluation-resolution",
                                            options=[
                                                {"label": "Preview rows", "value": "preview"},
                                                {"label": "Full selected split", "value": "full"},
                                            ],
                                            value="preview",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Plot"),
                                        dcc.Dropdown(
                                            id="phase-c-results-evaluation-plot-kind",
                                            options=[
                                                {
                                                    "label": "Y / Ŷ Time Trajectory",
                                                    "value": "time_series",
                                                },
                                                {
                                                    "label": "Y vs Ŷ Scatter",
                                                    "value": "scatter",
                                                },
                                                {
                                                    "label": "Residual Time Trajectory",
                                                    "value": "residual_time_series",
                                                },
                                                {
                                                    "label": "Residual Distribution",
                                                    "value": "residual_distribution",
                                                },
                                            ],
                                            value="time_series",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
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
                                            id="phase-c-results-evaluation-range-mode",
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
                                    dbc.Input(
                                        id="phase-c-results-evaluation-start",
                                        type="datetime-local",
                                        disabled=True,
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Input(
                                        id="phase-c-results-evaluation-end",
                                        type="datetime-local",
                                        disabled=True,
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Load Y / Ŷ",
                                        id="phase-c-results-load-evaluation",
                                        color="primary",
                                    ),
                                    md=3,
                                ),
                            ],
                            className="g-3 mt-1 align-items-end",
                        ),
                        html.Div(id="phase-c-results-evaluation-message", className="mt-2"),
                        _graph_with_legend(
                            "phase-c-results-evaluation-graph",
                            "phase-c-results-evaluation-legend",
                        ),
                        _plot_download_controls("evaluation"),
                        dcc.Store(id="phase-c-results-evaluation-legend-state", data=[]),
                        dcc.Store(id="phase-c-results-evaluation-selection", data={}),
                        html.H5("Evaluation Metrics", className="mt-3"),
                        html.Div(id="phase-c-results-metrics-table"),
                        html.H5("Model Coefficients / Metadata", className="mt-3"),
                        html.Div(id="phase-c-results-model-table"),
                        html.H5("Building PHVAC Reconstruction", className="mt-3"),
                        html.Div(id="phase-c-results-building-phvac-table"),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Full-Year Predicted Ŷ"),
                                help_button("heat_input.results.annual", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "C8 full-year inference is selected by zone package and "
                                "component. The selected model filter is also applied to "
                                "the available component list."
                            ),
                            className="small text-muted",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Annual Zone Package"),
                                        dcc.Dropdown(
                                            id="phase-c-results-annual-zone",
                                            placeholder="Select one C8 zone package",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Predicted Y Component(s)"),
                                        dcc.Dropdown(
                                            id="phase-c-results-annual-components",
                                            multi=True,
                                            placeholder="Select up to 8 components",
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Prediction Data"),
                                        dcc.Dropdown(
                                            id="phase-c-results-annual-resolution",
                                            options=[
                                                {"label": "Preview rows", "value": "preview"},
                                                {"label": "Full annual table", "value": "full"},
                                            ],
                                            value="preview",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Time Range"),
                                        dcc.RadioItems(
                                            id="phase-c-results-annual-range-mode",
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
                                    dbc.Input(
                                        id="phase-c-results-annual-start",
                                        type="datetime-local",
                                        disabled=True,
                                    ),
                                    md=2,
                                ),
                                dbc.Col(
                                    dbc.Input(
                                        id="phase-c-results-annual-end",
                                        type="datetime-local",
                                        disabled=True,
                                    ),
                                    md=2,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Load Full-Year Ŷ",
                                        id="phase-c-results-load-annual",
                                        color="primary",
                                    ),
                                    md=2,
                                ),
                            ],
                            className="g-3 mt-1 align-items-end",
                        ),
                        html.Div(id="phase-c-results-annual-message", className="mt-2"),
                        _graph_with_legend(
                            "phase-c-results-annual-graph",
                            "phase-c-results-annual-legend",
                        ),
                        _plot_download_controls("annual"),
                        dcc.Store(id="phase-c-results-annual-legend-state", data=[]),
                        dcc.Store(id="phase-c-results-annual-selection", data={}),
                        html.H5("Annual Component Summary", className="mt-3"),
                        html.Div(id="phase-c-results-annual-summary"),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Metrics & Comparative Diagnostics"),
                                help_button("heat_input.results.inventory", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="phase-c-results-comparison-kind",
                                        options=[
                                            {
                                                "label": "Estimator Metric Comparison",
                                                "value": "estimator_metric",
                                            },
                                            {
                                                "label": "Coefficient / Intercept",
                                                "value": "coefficient",
                                            },
                                            {
                                                "label": "Error by Building / Weather / Zone",
                                                "value": "error_context",
                                            },
                                            {"label": "Split Coverage", "value": "split_coverage"},
                                            {
                                                "label": "Component Availability",
                                                "value": "availability",
                                            },
                                            {
                                                "label": "PHVAC Building Reconstruction",
                                                "value": "building_phvac",
                                            },
                                        ],
                                        value="estimator_metric",
                                        clearable=False,
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Plot Metric / Diagnostic",
                                        id="phase-c-results-plot-comparison",
                                        color="primary",
                                    ),
                                    md=3,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Load Inventories",
                                        id="phase-c-results-load-inventories",
                                        color="secondary",
                                        outline=True,
                                    ),
                                    md=3,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(id="phase-c-results-comparison-message", className="mt-2"),
                        _graph_with_legend(
                            "phase-c-results-comparison-graph",
                            "phase-c-results-comparison-legend",
                        ),
                        _plot_download_controls("comparison"),
                        dcc.Store(id="phase-c-results-comparison-legend-state", data=[]),
                        html.Div(id="phase-c-results-inventory-message", className="mt-2"),
                        html.Details(
                            [
                                html.Summary("Lineage and artifact inventories"),
                                html.H5("Lineage", className="mt-3"),
                                html.Div(id="phase-c-results-lineage-table"),
                                html.H5("Dataset Inventory", className="mt-3"),
                                html.Div(id="phase-c-results-dataset-inventory"),
                                html.H5("Target / Model Inventory", className="mt-3"),
                                html.Div(id="phase-c-results-target-model-inventory"),
                                html.H5("Split Summary", className="mt-3"),
                                html.Div(id="phase-c-results-split-summary"),
                                html.H5("Model Inventory", className="mt-3"),
                                html.Div(id="phase-c-results-model-inventory"),
                                html.H5("Generalization", className="mt-3"),
                                html.Div(id="phase-c-results-generalization-table"),
                                html.H5("Artifact Inventory", className="mt-3"),
                                html.Div(id="phase-c-results-artifact-inventory"),
                            ],
                            className="mt-3",
                        ),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Artifacts & Run Information"),
                                help_button(
                                    "heat_input.results.artifact_downloads",
                                    compact=True,
                                ),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "These downloads are run/model artifacts rather than data "
                                "from a particular plot. Plot-specific data downloads live "
                                "directly beneath each graph."
                            ),
                            className="small text-muted",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Strong("Run summary"),
                                                help_button(
                                                    "heat_input.results.summary_download",
                                                    compact=True,
                                                ),
                                            ],
                                            className="heat-input-download-heading",
                                        ),
                                        dbc.Button(
                                            "Download run summary",
                                            id="phase-c-results-download-summary",
                                            outline=True,
                                            color="secondary",
                                            className="w-100",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Strong("Selected model"),
                                                help_button(
                                                    "heat_input.results.model_download",
                                                    compact=True,
                                                ),
                                            ],
                                            className="heat-input-download-heading",
                                        ),
                                        dbc.Button(
                                            "Download selected model bundle",
                                            id="phase-c-results-download-model-bundle",
                                            outline=True,
                                            color="secondary",
                                            className="w-100",
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            id="phase-c-results-artifact-export-message",
                            className="mt-2",
                        ),
                        dcc.Download(id="phase-c-results-summary-download"),
                        dcc.Download(id="phase-c-results-model-bundle-download"),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Validation"),
                                help_button("heat_input.results.validation", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.Div(id="phase-c-results-validation-overview"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="phase-c-results-validation-stage",
                                        options=[
                                            {"label": stage, "value": stage}
                                            for stage in ("C3", "C4", "C6", "C7", "C8")
                                        ],
                                        placeholder="Choose validator stage",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Load Diagnostics",
                                        id="phase-c-results-load-validation",
                                        outline=True,
                                        color="secondary",
                                    ),
                                    md=4,
                                ),
                            ],
                            className="g-3 mt-2",
                        ),
                        html.Div(id="phase-c-results-validation-table", className="mt-3"),
                    ]
                ),
                className="heat-input-card heat-input-results-section",
            ),
        ],
        className="page-content-container heat-input-page",
    )
