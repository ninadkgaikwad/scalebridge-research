"""Simplified Campaign Builder for Phase C Heat-Input Regression."""
from __future__ import annotations

import os

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.heat_input import model_id_options, parent_aggregation_options


def _label(title: str, help_key: str):
    return html.Div(
        [html.Strong(title), help_button(help_key, compact=True)],
        className="heat-input-label-with-help",
    )


def build_layout():
    """Build the curated Phase C campaign-level configuration surface."""
    machine_id = os.getenv("SCALEBRIDGE_MACHINE_ID") or "unidentified-machine"
    parent_options = parent_aggregation_options()

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Campaign Builder"),
                            help_button("heat_input.page.campaign_builder"),
                        ],
                        className="title-with-help",
                    ),
                    html.P(
                        (
                            "Choose Aggregation outputs and the scientific choices that "
                            "define a Phase C campaign. Stage-level C1-C9 implementation "
                            "details remain automatic."
                        ),
                        className="text-muted mb-0",
                    ),
                ],
                className="heat-input-subpage-heading",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Campaign"),
                                help_button("heat_input.section.identity", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Phase C Campaign ID",
                                            "heat_input.input.phase_c_campaign_id",
                                        ),
                                        dbc.Input(
                                            id="phase-c-builder-campaign-id",
                                            placeholder="phase_c_heat_input_v1",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Display Name",
                                            "heat_input.input.display_name",
                                        ),
                                        dbc.Input(
                                            id="phase-c-builder-display-name",
                                            placeholder="Optional",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        _label("Machine ID", "heat_input.input.machine_id"),
                                        dbc.Input(
                                            id="phase-c-builder-machine-id",
                                            value=machine_id,
                                            readonly=True,
                                        ),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            [
                                _label("Notes", "heat_input.input.notes"),
                                dbc.Textarea(
                                    id="phase-c-builder-notes",
                                    rows=2,
                                    placeholder="Optional campaign notes",
                                ),
                            ],
                            className="mt-3",
                        ),
                    ]
                ),
                className="heat-input-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Aggregation Source"),
                                help_button("heat_input.section.upstream", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "All artifact-discoverable Aggregation campaigns are shown, "
                                "including legacy matrices that predate the Phase B UI "
                                "definition store."
                            ),
                            className="small text-muted",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Aggregation Campaign",
                                            "heat_input.input.parent_aggregation_campaign",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-builder-parent-aggregation",
                                            options=parent_options,
                                            placeholder="Select Aggregation campaign",
                                        ),
                                    ],
                                    md=9,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Refresh",
                                        id="phase-c-builder-refresh-parents",
                                        color="secondary",
                                        outline=True,
                                        className="w-100",
                                    ),
                                    md=3,
                                    className="d-flex align-items-end",
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            id="phase-c-builder-parent-lineage",
                            className="mt-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Aggregation Matrix Run",
                                            "heat_input.input.matrix_run",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-builder-matrix-run",
                                            options=[],
                                            placeholder="Select matrix run",
                                        ),
                                    ],
                                    md=12,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(id="phase-c-builder-matrix-summary", className="mt-3"),
                    ]
                ),
                className="heat-input-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Campaign Scope"),
                                help_button("heat_input.section.scope", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "Generation Case and Weight Mode may be left empty to include "
                                "all applicable rows. Aggregation Strategy, Custom Grouping ID, "
                                "and Rule Set are shown separately so Phase B lineage is explicit."
                            ),
                            className="small text-muted",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Generation Case",
                                            "heat_input.input.case",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-builder-case",
                                            placeholder="All Generation cases",
                                            clearable=True,
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Aggregation Strategy",
                                            "heat_input.input.aggregation_strategy",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-builder-strategy",
                                            placeholder="All aggregation strategies",
                                            clearable=True,
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
                                        _label(
                                            "Custom Grouping ID",
                                            "heat_input.input.custom_grouping_id",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-builder-custom-grouping",
                                            placeholder="Not applicable unless custom_groups",
                                            clearable=True,
                                            disabled=True,
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Weight Mode",
                                            "heat_input.input.weight_mode",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-builder-weight",
                                            placeholder="All weight modes",
                                            clearable=True,
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Rule Set",
                                            "heat_input.input.rule_set",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-builder-rule-set",
                                            placeholder="All rule sets",
                                            clearable=True,
                                        ),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="g-3 mt-1",
                        ),
                        html.Div(
                            [
                                html.Label("Model Relationships"),
                                dcc.Dropdown(
                                    id="phase-c-builder-model-ids",
                                    options=model_id_options(),
                                    multi=True,
                                    placeholder="All applicable Phase C models",
                                ),
                                html.Div(
                                    (
                                        "Empty means all scientifically applicable models; "
                                        "structural absence remains an explicit state."
                                    ),
                                    className="form-text",
                                ),
                            ],
                            className="mt-3",
                        ),
                    ]
                ),
                className="heat-input-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Scientific Setup"),
                                help_button("heat_input.section.scientific", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Internal-Gain Predictor"),
                                        dcc.Dropdown(
                                            id="phase-c-builder-internal-gain-method",
                                            options=[
                                                {
                                                    "label": "Aggregate average",
                                                    "value": "aggregate_average",
                                                },
                                                {
                                                    "label": "Contribution sum",
                                                    "value": "contribution_sum",
                                                },
                                            ],
                                            value="aggregate_average",
                                            clearable=False,
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("HVAC Target"),
                                        dcc.Dropdown(
                                            id="phase-c-builder-hvac-target-method",
                                            options=[
                                                {
                                                    "label": "Signed zone sensible",
                                                    "value": "signed_zone_sensible",
                                                },
                                                {
                                                    "label": "Absolute zone sensible",
                                                    "value": "absolute_zone_sensible",
                                                },
                                            ],
                                            value="signed_zone_sensible",
                                            clearable=False,
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
                                        html.Label("Split Strategy"),
                                        dcc.Dropdown(
                                            id="phase-c-builder-split-strategy",
                                            options=[
                                                {
                                                    "label": "Monthly distributed holdout",
                                                    "value": "monthly_distributed_holdout",
                                                },
                                                {
                                                    "label": "Chronological fraction",
                                                    "value": "chronological_fraction",
                                                },
                                            ],
                                            value="monthly_distributed_holdout",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Train"),
                                        dbc.Input(
                                            id="phase-c-builder-train-fraction",
                                            type="number",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.70,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Validation"),
                                        dbc.Input(
                                            id="phase-c-builder-validation-fraction",
                                            type="number",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.15,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Test"),
                                        dbc.Input(
                                            id="phase-c-builder-test-fraction",
                                            type="number",
                                            min=0,
                                            max=1,
                                            step=0.01,
                                            value=0.15,
                                        ),
                                    ],
                                    md=3,
                                ),
                            ],
                            className="g-3 mt-1",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Estimator(s)"),
                                        dcc.Dropdown(
                                            id="phase-c-builder-estimators",
                                            options=[
                                                {
                                                    "label": "PyTorch linear",
                                                    "value": "pytorch_linear",
                                                },
                                                {
                                                    "label": "Closed-form linear",
                                                    "value": "closed_form_linear",
                                                },
                                            ],
                                            value=["pytorch_linear"],
                                            multi=True,
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("PyTorch Device(s)"),
                                        dcc.Dropdown(
                                            id="phase-c-builder-devices",
                                            options=[
                                                {"label": "Auto", "value": "auto"},
                                                {"label": "CPU", "value": "cpu"},
                                                {"label": "CUDA", "value": "cuda"},
                                            ],
                                            value=["auto"],
                                            multi=True,
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3 mt-1",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Checkbox(
                                        id="phase-c-builder-validation-enabled",
                                        value=True,
                                        label="Run Phase C validation",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Checkbox(
                                        id="phase-c-builder-mlflow-enabled",
                                        value=True,
                                        label="Register completed Phase C run with MLflow",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3 mt-3",
                        ),
                    ]
                ),
                className="heat-input-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H5("Preview and Save"),
                                help_button("heat_input.section.preview_save", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Preview Campaign",
                                        id="phase-c-builder-preview",
                                        color="secondary",
                                        outline=True,
                                        className="w-100",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Save Campaign",
                                        id="phase-c-builder-save",
                                        color="primary",
                                        className="w-100",
                                    ),
                                    md=4,
                                ),
                                dbc.Col(
                                    dbc.Checkbox(
                                        id="phase-c-builder-replace",
                                        value=False,
                                        label="Replace existing definition",
                                    ),
                                    md=4,
                                    className="d-flex align-items-center",
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(id="phase-c-builder-save-status", className="mt-3"),
                        html.Div(
                            id="phase-c-builder-definition-preview",
                            className="mt-3",
                        ),
                    ]
                ),
                className="heat-input-card",
            ),
            dcc.Store(id="phase-c-builder-parent-cache"),
            dcc.Store(id="phase-c-builder-matrix-cache"),
        ],
        className="page-content-container heat-input-page",
    )
