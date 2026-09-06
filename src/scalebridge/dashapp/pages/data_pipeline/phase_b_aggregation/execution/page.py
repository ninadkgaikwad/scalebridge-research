"""B7 managed execution UI for saved Phase B Aggregation definitions."""

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.aggregation import list_definitions


def _definition_options():
    rows = list_definitions()
    return [
        {
            "label": (
                f"{row['campaign_id']} | parent={row['parent_generation_campaign_id']} | "
                f"{row['plan_request_count']} plan request(s)"
            ),
            "value": row["campaign_id"],
        }
        for row in rows
    ]


def build_layout():
    options = _definition_options()
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Execution"),
                    help_button("aggregation.page.execution"),
                ],
                className="title-with-help",
            ),
            html.P(
                "Select a saved Aggregation campaign definition, review its Phase A lineage "
                "and exact B2 runner command, then launch or stop the managed subprocess.",
                className="page-description",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Strong("Saved Aggregation Campaign"),
                                                help_button(
                                                    "aggregation.execution.saved_definition",
                                                    compact=True,
                                                ),
                                            ],
                                            className="aggregation-label-with-help",
                                        ),
                                        dcc.Dropdown(
                                            id="aggregation-execution-campaign",
                                            options=options,
                                            placeholder="Select a saved Aggregation campaign",
                                        ),
                                    ],
                                    className="flex-grow-1",
                                ),
                                dbc.Button(
                                    "Refresh Definitions",
                                    id="aggregation-execution-refresh",
                                    color="secondary",
                                    outline=True,
                                ),
                            ],
                            className="d-flex gap-2 align-items-end flex-wrap",
                        ),
                        html.Div(
                            id="aggregation-execution-definition-summary",
                            className="mt-3",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Strong("Resolved B2 Runner Command"),
                                        help_button(
                                            "aggregation.execution.command",
                                            compact=True,
                                        ),
                                    ],
                                    className="aggregation-label-with-help",
                                ),
                                html.Pre(
                                    id="aggregation-execution-command",
                                    className="aggregation-command-preview",
                                ),
                            ],
                            className="mt-3",
                        ),
                    ]
                ),
                className="aggregation-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                dbc.Button(
                                    "Start Execution",
                                    id="aggregation-execution-start",
                                    color="success",
                                    disabled=not bool(options),
                                ),
                                dbc.Button(
                                    "Stop Execution",
                                    id="aggregation-execution-stop",
                                    color="danger",
                                    outline=True,
                                    disabled=True,
                                ),
                            ],
                            className="d-flex gap-2 flex-wrap",
                        ),
                        html.Div(
                            id="aggregation-execution-status",
                            className="mt-3",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Strong("Live Console"),
                                        help_button(
                                            "aggregation.execution.console",
                                            compact=True,
                                        ),
                                    ],
                                    className="aggregation-label-with-help",
                                ),
                                html.Pre(
                                    id="aggregation-execution-console",
                                    className="aggregation-live-console",
                                ),
                            ],
                            className="mt-3",
                        ),
                    ]
                ),
                className="aggregation-card",
            ),
            dcc.Interval(
                id="aggregation-execution-poll",
                interval=1500,
                n_intervals=0,
            ),
        ],
        className="page-content-container aggregation-page",
    )
