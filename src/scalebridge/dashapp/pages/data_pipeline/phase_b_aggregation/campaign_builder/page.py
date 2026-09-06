"""Campaign Builder tab for Phase B Aggregation."""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.aggregation import parent_campaign_options


def _label(text: str, help_id: str):
    return html.Div(
        [
            html.Label(text, className="form-label mb-0"),
            help_button(help_id),
        ],
        className="aggregation-label-with-help",
    )


def build_layout():
    """Build the Aggregation Campaign Builder UI."""
    parent_options = parent_campaign_options()

    return html.Div(
        [
            html.Div(
                [
                    html.H3("Campaign Builder"),
                    html.P(
                        "Select a successful Generation campaign, choose source cases, "
                        "configure Aggregation plans, and save a reusable Phase B definition.",
                        className="text-muted mb-0",
                    ),
                ],
                className="aggregation-subpage-heading",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("1. Upstream Generation Campaign"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Parent Generation Campaign",
                                            "aggregation.input.parent_generation_campaign",
                                        ),
                                        dcc.Dropdown(
                                            id="aggregation-builder-parent-campaign",
                                            options=parent_options,
                                            placeholder="Select a Generation campaign",
                                            clearable=True,
                                        ),
                                    ],
                                    md=8,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Refresh",
                                            "aggregation.action.refresh_generation_campaigns",
                                        ),
                                        dbc.Button(
                                            "Refresh Generation Campaigns",
                                            id="aggregation-builder-refresh-campaigns",
                                            color="secondary",
                                            outline=True,
                                            className="w-100",
                                        ),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            id="aggregation-builder-parent-status",
                            className="mt-3",
                        ),
                    ]
                ),
                className="aggregation-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("2. Generation Case Selection"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Building Type",
                                            "aggregation.input.building_type_filter",
                                        ),
                                        dcc.Dropdown(
                                            id="aggregation-builder-building-filter",
                                            multi=True,
                                            placeholder="All building types",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Weather Location",
                                            "aggregation.input.weather_filter",
                                        ),
                                        dcc.Dropdown(
                                            id="aggregation-builder-weather-filter",
                                            multi=True,
                                            placeholder="All weather locations",
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            [
                                _label(
                                    "Generation Cases",
                                    "aggregation.input.case_selection",
                                ),
                                dcc.Dropdown(
                                    id="aggregation-builder-cases",
                                    multi=True,
                                    placeholder="Select one or more eligible Generation cases",
                                ),
                            ],
                            className="mt-3",
                        ),
                        html.Div(
                            id="aggregation-builder-case-summary",
                            className="mt-3",
                        ),
                    ]
                ),
                className="aggregation-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("3. Aggregation Plan Requests"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Strategies",
                                            "aggregation.input.strategies",
                                        ),
                                        dcc.Dropdown(
                                            id="aggregation-builder-strategies",
                                            options=[
                                                {
                                                    "label": "All thermal zones → one zone",
                                                    "value": "all_thermal_zones_to_one",
                                                },
                                                {
                                                    "label": "Custom groups",
                                                    "value": "custom_groups",
                                                },
                                                {
                                                    "label": "Identity",
                                                    "value": "identity",
                                                },
                                            ],
                                            value=["all_thermal_zones_to_one"],
                                            multi=True,
                                            clearable=False,
                                        ),
                                    ],
                                    md=5,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Weight Modes",
                                            "aggregation.input.weight_modes",
                                        ),
                                        dcc.Dropdown(
                                            id="aggregation-builder-weight-modes",
                                            options=[
                                                {"label": "Equal", "value": "equal"},
                                                {
                                                    "label": "Floor area",
                                                    "value": "floor_area",
                                                },
                                                {"label": "Volume", "value": "volume"},
                                            ],
                                            value=["equal"],
                                            multi=True,
                                            clearable=False,
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Rule Set",
                                            "aggregation.input.rule_set",
                                        ),
                                        dcc.Dropdown(
                                            id="aggregation-builder-rule-set",
                                            options=[
                                                {
                                                    "label": "legacy_v1",
                                                    "value": "legacy_v1",
                                                }
                                            ],
                                            value="legacy_v1",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            [
                                html.H6("Custom Zone Groups"),
                                html.P(
                                    "Shown only when custom_groups is selected. "
                                    "Every thermal zone in every selected case must be "
                                    "assigned to exactly one aggregate-zone name.",
                                    className="text-muted small",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                _label(
                                                    "Custom Aggregation ID",
                                                    "aggregation.input.custom_aggregation_id",
                                                ),
                                                dbc.Input(
                                                    id="aggregation-builder-custom-id",
                                                    value="custom_v1",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        dbc.Col(
                                            [
                                                _label(
                                                    "Grouping Table",
                                                    "aggregation.input.custom_grouping",
                                                ),
                                                html.Div(
                                                    "Edit only the Aggregate Zone column.",
                                                    className="small text-muted pt-2",
                                                ),
                                            ],
                                            md=8,
                                        ),
                                    ],
                                    className="g-3 mb-2",
                                ),
                                dash_table.DataTable(
                                    id="aggregation-builder-custom-table",
                                    columns=[
                                        {
                                            "name": "Case ID",
                                            "id": "case_id",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Source Zone",
                                            "id": "source_zone_name",
                                            "editable": False,
                                        },
                                        {
                                            "name": "Aggregate Zone",
                                            "id": "aggregate_zone_name",
                                            "editable": True,
                                        },
                                    ],
                                    data=[],
                                    editable=True,
                                    page_size=15,
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "textAlign": "left",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                ),
                                html.Div(
                                    id="aggregation-builder-custom-status",
                                    className="mt-2",
                                ),
                            ],
                            id="aggregation-builder-custom-panel",
                            className="aggregation-custom-panel mt-3",
                            style={"display": "none"},
                        ),
                        html.Div(
                            id="aggregation-builder-plan-summary",
                            className="mt-3",
                        ),
                    ]
                ),
                className="aggregation-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("4. Campaign and Execution Definition"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Aggregation Campaign ID",
                                            "aggregation.input.campaign_id",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-campaign-id",
                                            placeholder="aggregation_campaign_v1",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Machine ID",
                                            "aggregation.input.machine_id",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-machine-id",
                                            value="laptop",
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Case Limit",
                                            "aggregation.input.case_limit",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-case-limit",
                                            type="number",
                                            min=1,
                                            placeholder="All selected",
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
                                        _label(
                                            "Variable Limit",
                                            "aggregation.input.variable_limit",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-variable-limit",
                                            type="number",
                                            min=1,
                                            placeholder="All variables",
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Preview Rows",
                                            "aggregation.input.preview_rows",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-preview-rows",
                                            type="number",
                                            min=0,
                                            value=100,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Checkbox(
                                            id="aggregation-builder-pickles",
                                            value=False,
                                            label="Write legacy pickle",
                                        ),
                                        dbc.Checkbox(
                                            id="aggregation-builder-continue",
                                            value=True,
                                            label="Continue on error",
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Checkbox(
                                            id="aggregation-builder-mlflow",
                                            value=True,
                                            label="Enable MLflow",
                                        ),
                                        dbc.Checkbox(
                                            id="aggregation-builder-mlflow-strict",
                                            value=False,
                                            label="Strict MLflow",
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "Aggregate Zone Stem",
                                            "aggregation.input.aggregate_zone_stem",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-zone-stem",
                                            value="Aggregated_Zone",
                                        ),
                                    ],
                                    md=3,
                                ),
                            ],
                            className="g-3 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "System Node Pattern",
                                            "aggregation.input.system_node_pattern",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-system-node-pattern",
                                            value="DIRECT AIR INLET NODE",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "MLflow Tracking URI",
                                            "aggregation.input.mlflow_uri",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-mlflow-uri",
                                            value="http://127.0.0.1:5000",
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3 mt-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "MLflow Experiment",
                                            "aggregation.input.mlflow_experiment",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-mlflow-experiment",
                                            placeholder="<campaign-id>_aggregation",
                                        ),
                                    ],
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        _label(
                                            "MLflow Run Name",
                                            "aggregation.input.mlflow_run_name",
                                        ),
                                        dbc.Input(
                                            id="aggregation-builder-mlflow-run-name",
                                            placeholder="Optional",
                                        ),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-3 mt-2",
                        ),
                    ]
                ),
                className="aggregation-card",
            ),
            html.Div(id="aggregation-builder-definition-summary"),
            dbc.Button(
                "Save Campaign Definition",
                id="aggregation-builder-save",
                color="primary",
                className="mt-2",
            ),
            html.Div(
                id="aggregation-builder-save-status",
                className="mt-2",
            ),
            dcc.Store(id="aggregation-builder-generation-cache"),
            dcc.Store(id="aggregation-builder-generation-issues"),
        ],
        className="page-content-container aggregation-page",
    )
