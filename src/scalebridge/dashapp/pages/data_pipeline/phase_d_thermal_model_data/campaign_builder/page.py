"""Campaign Builder for Phase D Thermal-Model Data."""
from __future__ import annotations

import os

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.phase_d import phase_c_run_options


ML_POLICY_OPTIONS = [
    {"label": "Monthly Distributed Holdout", "value": "monthly_distributed_holdout"},
    {"label": "Chronological Holdout", "value": "chronological_holdout"},
    {"label": "Seasonal Holdout", "value": "seasonal_holdout"},
]
OB_POLICY_OPTIONS = [
    {"label": "Seasonal Distributed", "value": "seasonal_distributed"},
    {"label": "Seasonal Block Holdout", "value": "seasonal_block_holdout"},
    {"label": "Contiguous Identification", "value": "contiguous_identification"},
    {"label": "Custom Datetime Ranges", "value": "custom_datetime_ranges"},
]
SEASON_OPTIONS = [
    {"label": value.title(), "value": value}
    for value in ("winter", "spring", "summer", "fall")
]


def _label(title: str, help_key: str):
    return html.Div(
        [html.Strong(title), help_button(help_key, compact=True)],
        className="phase-d-label-with-help",
    )


def _number(component_id: str, value, *, min_value=None, step=None):
    return dbc.Input(
        id=component_id,
        type="number",
        value=value,
        min=min_value,
        step=step,
    )


def build_layout():
    machine_id = os.getenv("SCALEBRIDGE_MACHINE_ID") or "unidentified-machine"
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H3("Campaign Builder"), help_button("phase_d.page.campaign_builder")],
                        className="title-with-help",
                    ),
                    html.P(
                        "Select a completed Phase C run and configure the options accepted by the general Phase D campaign runner. Upstream Generation/Aggregation lineage and required artifacts are resolved automatically.",
                        className="text-muted mb-0",
                    ),
                ],
                className="phase-d-subpage-heading",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("Campaign"), help_button("phase_d.section.identity", compact=True)], className="title-with-help"),
                        dbc.Row(
                            [
                                dbc.Col([_label("Phase D Campaign ID", "phase_d.input.campaign_id"), dbc.Input(id="phase-d-builder-campaign-id", placeholder="phase_d_thermal_data_v1")], md=4),
                                dbc.Col([_label("Display Name", "phase_d.input.display_name"), dbc.Input(id="phase-d-builder-display-name", placeholder="Optional")], md=4),
                                dbc.Col([_label("Machine ID", "phase_d.input.machine_id"), dbc.Input(id="phase-d-builder-machine-id", value=machine_id, readonly=True)], md=4),
                            ],
                            className="g-3",
                        ),
                        html.Div([_label("Notes", "phase_d.input.notes"), dbc.Textarea(id="phase-d-builder-notes", rows=2, placeholder="Optional campaign notes")], className="mt-3"),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("Upstream Phase C Run"), help_button("phase_d.section.upstream", compact=True)], className="title-with-help"),
                        html.P("Choose one completed Phase C run. Phase D uses its recorded campaign root and Aggregation matrix rather than asking you to choose A, B, and C independently.", className="small text-muted"),
                        dbc.Row(
                            [
                                dbc.Col([_label("Completed Phase C Campaign / Run", "phase_d.input.phase_c_run"), dcc.Dropdown(id="phase-d-builder-phase-c-run", options=phase_c_run_options(), placeholder="Select completed Phase C run")], md=9),
                                dbc.Col(dbc.Button("Refresh", id="phase-d-builder-refresh-phase-c", color="secondary", outline=True, className="w-100"), md=3, className="d-flex align-items-end"),
                            ],
                            className="g-3",
                        ),
                        html.Div(id="phase-d-builder-upstream-lineage", className="mt-3"),
                        dcc.Store(id="phase-d-builder-upstream-cache", data={}),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("Campaign Scope"), help_button("phase_d.section.scope", compact=True)], className="title-with-help"),
                        html.P("These are the exact repeatable scope filters supported by the Phase D runner. Leave a selector empty to include all matching successful Aggregation rows.", className="small text-muted"),
                        dbc.Row(
                            [
                                dbc.Col([_label("Generation Case(s)", "phase_d.input.case_ids"), dcc.Dropdown(id="phase-d-builder-case-ids", multi=True, placeholder="All cases")], md=4),
                                dbc.Col([_label("Aggregation ID(s)", "phase_d.input.aggregation_ids"), dcc.Dropdown(id="phase-d-builder-aggregation-ids", multi=True, placeholder="All aggregation IDs")], md=4),
                                dbc.Col([_label("Weight Mode(s)", "phase_d.input.weight_modes"), dcc.Dropdown(id="phase-d-builder-weight-modes", multi=True, placeholder="All weight modes")], md=4),
                            ],
                            className="g-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col([_label("Maximum Aggregation Runs", "phase_d.input.max_aggregation_runs"), _number("phase-d-builder-max-runs", None, min_value=1, step=1)], md=4),
                                dbc.Col(html.Div(id="phase-d-builder-match-count", className="phase-d-readonly-box mt-4"), md=8),
                            ],
                            className="g-3 mt-1",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("Thermal-Model Data Representation"), help_button("phase_d.section.representation", compact=True)], className="title-with-help"),
                        dbc.Row(
                            [
                                dbc.Col([_label("Heat Representation", "phase_d.input.heat_representation"), dcc.Dropdown(id="phase-d-builder-heat-representation", options=[{"label": "Grouped ZIC / ZIR", "value": "grouped"}, {"label": "Individual Heat Components", "value": "components"}], value="grouped", clearable=False)], md=6),
                                dbc.Col([_label("Visible-Lighting Radiant Gain", "phase_d.input.qzivr_separate"), dbc.Checkbox(id="phase-d-builder-qzivr-separate", label="Keep QZivr_L separate from grouped ZIR", value=False)], md=6),
                            ],
                            className="g-3",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("ML / SciML Dataset Policies"), help_button("phase_d.section.ml", compact=True)], className="title-with-help"),
                        dbc.Row(
                            [
                                dbc.Col([_label("Policies", "phase_d.input.ml_policies"), dcc.Dropdown(id="phase-d-builder-ml-policies", options=ML_POLICY_OPTIONS, value=["monthly_distributed_holdout"], multi=True, clearable=False)], md=6),
                                dbc.Col([_label("Input Lag(s)", "phase_d.input.ml_input_lags"), dbc.Input(id="phase-d-builder-ml-input-lags", value="12", placeholder="Comma-separated, e.g. 1,3,6,12")], md=3),
                                dbc.Col([_label("Target Horizon(s)", "phase_d.input.ml_target_horizons"), dbc.Input(id="phase-d-builder-ml-target-horizons", value="6", placeholder="Comma-separated, e.g. 1,6,12")], md=3),
                            ], className="g-3"),
                        html.Div(
                            [
                                html.H6("Fraction-based policy options", className="mt-3"),
                                html.Small("Used by Monthly Distributed Holdout and/or Chronological Holdout when selected.", className="text-muted"),
                                dbc.Row(
                                    [
                                        dbc.Col([_label("Train Fraction", "phase_d.input.ml_fractions"), _number("phase-d-builder-ml-train-fraction", 0.70, min_value=0.000001, step=0.01)], md=4),
                                        dbc.Col([_label("Test Fraction", "phase_d.input.ml_fractions"), _number("phase-d-builder-ml-test-fraction", 0.15, min_value=0.000001, step=0.01)], md=4),
                                        dbc.Col([_label("Validation Fraction", "phase_d.input.ml_fractions"), _number("phase-d-builder-ml-validation-fraction", 0.15, min_value=0.000001, step=0.01)], md=4),
                                    ], className="g-3 mt-1"),
                            ],
                            id="phase-d-builder-ml-fraction-options",
                            hidden=False,
                        ),
                        html.Div(
                            [
                                html.H6("Seasonal Holdout options", className="mt-3"),
                                html.Small("Assign complete meteorological seasons to train, test, and validation.", className="text-muted"),
                                dbc.Row(
                                    [
                                        dbc.Col([_label("Train Seasons", "phase_d.input.ml_sh_seasons"), dcc.Dropdown(id="phase-d-builder-ml-sh-train", options=SEASON_OPTIONS, value=["winter", "spring"], multi=True)], md=4),
                                        dbc.Col([_label("Test Seasons", "phase_d.input.ml_sh_seasons"), dcc.Dropdown(id="phase-d-builder-ml-sh-test", options=SEASON_OPTIONS, value=["summer"], multi=True)], md=4),
                                        dbc.Col([_label("Validation Seasons", "phase_d.input.ml_sh_seasons"), dcc.Dropdown(id="phase-d-builder-ml-sh-validation", options=SEASON_OPTIONS, value=["fall"], multi=True)], md=4),
                                    ], className="g-3"),
                            ],
                            id="phase-d-builder-ml-seasonal-holdout-options",
                            hidden=True,
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("Optimization / Bayesian Dataset Policies"), help_button("phase_d.section.ob", compact=True)], className="title-with-help"),
                        dbc.Row([dbc.Col([_label("Policies", "phase_d.input.ob_policies"), dcc.Dropdown(id="phase-d-builder-ob-policies", options=OB_POLICY_OPTIONS, value=["seasonal_distributed"], multi=True, clearable=False)], md=12)], className="g-3"),
                        html.Div(
                            [
                                html.H6("Seasonal Distributed options", className="mt-3"),
                                html.Small("Within each meteorological season: skip the offset, then select Train Days followed immediately by Test Days.", className="text-muted"),
                                dbc.Row(
                                    [
                                        dbc.Col([_label("Season Offset Days", "phase_d.input.sd"), _number("phase-d-builder-sd-offset", 0, min_value=0, step=1)], md=4),
                                        dbc.Col([_label("Train Days", "phase_d.input.sd"), _number("phase-d-builder-sd-train-days", 21, min_value=1, step=1)], md=4),
                                        dbc.Col([_label("Test Days", "phase_d.input.sd"), _number("phase-d-builder-sd-test-days", 7, min_value=1, step=1)], md=4),
                                    ], className="g-3"),
                            ],
                            id="phase-d-builder-ob-sd-options",
                            hidden=False,
                        ),
                        html.Div(
                            [
                                html.H6("Seasonal Block Holdout options", className="mt-3"),
                                html.Small("Uses whole meteorological seasons; this policy does not take arbitrary start/end dates.", className="text-muted"),
                                dbc.Row(
                                    [
                                        dbc.Col([_label("Train Seasons", "phase_d.input.sbh"), dcc.Dropdown(id="phase-d-builder-sbh-train", options=SEASON_OPTIONS, value=["winter", "spring", "fall"], multi=True)], md=6),
                                        dbc.Col([_label("Test Seasons", "phase_d.input.sbh"), dcc.Dropdown(id="phase-d-builder-sbh-test", options=SEASON_OPTIONS, value=["summer"], multi=True)], md=6),
                                    ], className="g-3"),
                            ],
                            id="phase-d-builder-ob-sbh-options",
                            hidden=True,
                        ),
                        html.Div(
                            [
                                html.H6("Contiguous Identification options", className="mt-3"),
                                html.Small("One train block starts at Start Datetime and is followed immediately by one test block. Format: YYYY-MM-DDTHH:MM:SS. Blank uses the first canonical timestamp.", className="text-muted"),
                                dbc.Row(
                                    [
                                        dbc.Col([_label("Start Datetime", "phase_d.input.ci"), dbc.Input(id="phase-d-builder-ci-start", placeholder="2001-01-01T00:05:00")], md=6),
                                        dbc.Col([_label("Train Days", "phase_d.input.ci"), _number("phase-d-builder-ci-train-days", 21, min_value=1, step=1)], md=3),
                                        dbc.Col([_label("Test Days", "phase_d.input.ci"), _number("phase-d-builder-ci-test-days", 7, min_value=1, step=1)], md=3),
                                    ], className="g-3"),
                            ],
                            id="phase-d-builder-ob-ci-options",
                            hidden=True,
                        ),
                        html.Div(
                            [
                                html.H6("Custom Datetime Ranges options", className="mt-3"),
                                html.Small("Separate policy for arbitrary explicit windows. Enter one half-open [START, END) range per line as YYYY-MM-DDTHH:MM:SS/YYYY-MM-DDTHH:MM:SS. Ranges may repeat but cannot overlap.", className="text-muted"),
                                dbc.Row(
                                    [
                                        dbc.Col([_label("Train Ranges", "phase_d.input.cdr"), dbc.Textarea(id="phase-d-builder-cdr-train", rows=4, placeholder="2001-01-01T00:05:00/2001-01-08T00:05:00\n2001-02-01T00:05:00/2001-02-08T00:05:00")], md=6),
                                        dbc.Col([_label("Test Ranges", "phase_d.input.cdr"), dbc.Textarea(id="phase-d-builder-cdr-test", rows=4, placeholder="2001-01-08T00:05:00/2001-01-10T00:05:00\n2001-02-08T00:05:00/2001-02-10T00:05:00")], md=6),
                                    ], className="g-3"),
                            ],
                            id="phase-d-builder-ob-cdr-options",
                            hidden=True,
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("Advanced Runner Options"), help_button("phase_d.section.advanced", compact=True)], className="title-with-help"),
                        dbc.Row(
                            [
                                dbc.Col([_label("Phase D Calendar Year", "phase_d.input.calendar_year"), _number("phase-d-builder-calendar-year", 2001, min_value=1, step=1)], md=3),
                                dbc.Col([_label("Parquet Compression", "phase_d.input.parquet_compression"), dbc.Input(id="phase-d-builder-parquet-compression", value="zstd")], md=3),
                                dbc.Col([_label("Output Root", "phase_d.input.output_root"), dbc.Input(id="phase-d-builder-output-root", placeholder="Blank = selected campaign root")], md=6),
                            ], className="g-3"),
                        html.H6("MLflow", className="mt-3"),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Checkbox(id="phase-d-builder-mlflow-enabled", label="Enable campaign-level Phase D MLflow tracking", value=False), md=4),
                                dbc.Col([_label("Experiment Name", "phase_d.input.mlflow"), dbc.Input(id="phase-d-builder-mlflow-experiment", placeholder="Optional")], md=4),
                                dbc.Col([_label("Run Name", "phase_d.input.mlflow"), dbc.Input(id="phase-d-builder-mlflow-run-name", placeholder="Optional")], md=4),
                            ], className="g-3"),
                        html.Div(dbc.Checkbox(id="phase-d-builder-mlflow-strict", label="Treat MLflow setup/logging failure as a Phase D failure", value=False), className="mt-2"),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div([html.H5("Definition Preview & Save"), help_button("phase_d.section.save", compact=True)], className="title-with-help"),
                        dbc.Row(
                            [
                                dbc.Col(dbc.Button("Preview Campaign", id="phase-d-builder-preview", color="secondary", outline=True, className="w-100"), md=4),
                                dbc.Col(dbc.Button("Save Campaign", id="phase-d-builder-save", color="primary", className="w-100"), md=4),
                                dbc.Col(dbc.Checkbox(id="phase-d-builder-replace", label="Replace existing definition", value=False), md=4, className="d-flex align-items-center"),
                            ], className="g-3"),
                        html.Div(id="phase-d-builder-definition-preview", className="mt-3"),
                        html.Div(id="phase-d-builder-save-status", className="mt-2"),
                    ]
                ),
                className="phase-d-card",
            ),
        ],
        className="phase-d-page",
    )
