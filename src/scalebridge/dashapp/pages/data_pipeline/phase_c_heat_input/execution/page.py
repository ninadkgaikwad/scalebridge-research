"""Simplified managed Execution UI for saved Phase C definitions."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.heat_input import list_execution_definitions, suggested_run_id


def _definition_options():
    rows = list_execution_definitions()
    return [
        {
            "label": (
                f"{row['phase_c_campaign_id']} | "
                f"parent={row['parent_generation_campaign_id']} | "
                f"matrix={row.get('matrix_run_id') or '—'}"
            ),
            "value": row["phase_c_campaign_id"],
        }
        for row in rows
    ]


def _label(title: str, help_key: str):
    return html.Div(
        [html.Strong(title), help_button(help_key, compact=True)],
        className="heat-input-label-with-help",
    )


def build_layout():
    """Build Tab 2 as one complete-Phase-C execution surface."""
    options = _definition_options()
    return html.Div(
        [
            html.Div(
                [html.H3("Execution"), help_button("heat_input.page.execution")],
                className="title-with-help",
            ),
            html.P(
                (
                    "Select a saved campaign and run the complete Phase C workflow. "
                    "Stage selection and recovery controls stay in the CLI rather than the "
                    "normal Dash workflow."
                ),
                className="page-description",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        _label(
                                            "Saved Phase C Campaign",
                                            "heat_input.execution.saved_definition",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-c-execution-campaign",
                                            options=options,
                                            placeholder="Select saved Phase C campaign",
                                        ),
                                    ],
                                    className="flex-grow-1",
                                ),
                                dbc.Button(
                                    "Refresh",
                                    id="phase-c-execution-refresh",
                                    color="secondary",
                                    outline=True,
                                ),
                            ],
                            className="d-flex gap-2 align-items-end flex-wrap",
                        ),
                        html.Div(
                            id="phase-c-execution-definition-summary",
                            className="mt-3",
                        ),
                    ]
                ),
                className="heat-input-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Phase C Run ID",
                                            "heat_input.execution.phase_c_run_id",
                                        ),
                                        dbc.Input(
                                            id="phase-c-execution-run-id",
                                            value=suggested_run_id(),
                                            type="text",
                                        ),
                                    ],
                                    md=7,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            dbc.Checkbox(
                                                id="phase-c-execution-dry-run",
                                                value=False,
                                                label=(
                                                    "Dry run (plan only; do not execute Phase C)"
                                                ),
                                            ),
                                            help_button(
                                                "heat_input.execution.dry_run",
                                                compact=True,
                                            ),
                                        ],
                                        className="d-flex gap-2 align-items-center",
                                    ),
                                    md=5,
                                    className="d-flex align-items-end pb-2",
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            id="phase-c-execution-runtime-warnings",
                            className="mt-3",
                        ),
                        html.Details(
                            [
                                html.Summary("Technical details"),
                                html.Div(
                                    [
                                        html.Strong("Resolved runner command"),
                                        html.Pre(
                                            id="phase-c-execution-command",
                                            className="heat-input-command-preview",
                                        ),
                                        html.Strong("Effective Phase C configuration"),
                                        html.Pre(
                                            id="phase-c-execution-effective-config",
                                            className="heat-input-definition-preview",
                                        ),
                                    ],
                                    className="mt-2",
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
                                dbc.Button(
                                    "Start Phase C",
                                    id="phase-c-execution-start",
                                    color="success",
                                    disabled=not bool(options),
                                ),
                                dbc.Button(
                                    "Stop",
                                    id="phase-c-execution-stop",
                                    color="danger",
                                    outline=True,
                                    disabled=True,
                                ),
                            ],
                            className="d-flex gap-2 flex-wrap",
                        ),
                        html.Div(
                            id="phase-c-execution-action-message",
                            className="mt-3",
                        ),
                        html.Div(id="phase-c-execution-status", className="mt-3"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Strong("Phase C Progress"),
                                        help_button(
                                            "heat_input.execution.progress",
                                            compact=True,
                                        ),
                                    ],
                                    className="title-with-help",
                                ),
                                html.Div(
                                    id="phase-c-execution-stage-progress",
                                    className="heat-input-stage-progress",
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
                                html.Strong("Live Console"),
                                help_button("heat_input.execution.console", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.Pre(
                            id="phase-c-execution-console",
                            className="heat-input-live-console",
                        ),
                    ]
                ),
                className="heat-input-card",
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Confirm Phase C Action")),
                    dbc.ModalBody(id="phase-c-execution-confirm-body"),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="phase-c-execution-confirm-cancel",
                                color="secondary",
                                outline=True,
                            ),
                            dbc.Button(
                                "Confirm",
                                id="phase-c-execution-confirm-accept",
                                color="danger",
                            ),
                        ]
                    ),
                ],
                id="phase-c-execution-confirm-modal",
                is_open=False,
                centered=True,
            ),
            dcc.Store(id="phase-c-execution-pending-action"),
            dcc.Interval(id="phase-c-execution-poll", interval=1500, n_intervals=0),
        ],
        className="page-content-container heat-input-page",
    )
