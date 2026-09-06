"""Managed Execution UI for saved Phase D Campaign Builder definitions."""
from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from .....components.help import help_button
from .....services.phase_d import list_execution_definitions, suggested_run_id


def _definition_options():
    rows = list_execution_definitions()
    return [
        {
            "label": (
                f"{row['phase_d_campaign_id']} | "
                f"Phase C={row['phase_c_campaign_run_id']} | "
                f"matrix={row['matrix_run_id']}"
            ),
            "value": row["phase_d_campaign_id"],
        }
        for row in rows
    ]


def _label(title: str, help_key: str):
    return html.Div(
        [html.Strong(title), help_button(help_key, compact=True)],
        className="phase-d-label-with-help",
    )


def _runtime_checkbox(component_id: str, label: str, help_key: str):
    return html.Div(
        [
            dbc.Checkbox(id=component_id, value=False, label=label),
            help_button(help_key, compact=True),
        ],
        className="d-flex gap-2 align-items-center",
    )


def build_layout():
    """Build Tab 2 as a thin managed interface over the saved runner definition."""
    options = _definition_options()
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H3("Execution"), help_button("phase_d.page.execution")],
                        className="title-with-help",
                    ),
                    html.P(
                        (
                            "Load a saved Phase D campaign definition and execute that exact "
                            "general-runner configuration. Scientific settings remain owned "
                            "by Campaign Builder; this tab exposes run-time controls only."
                        ),
                        className="text-muted mb-0",
                    ),
                ],
                className="phase-d-subpage-heading",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        _label(
                                            "Saved Phase D Campaign",
                                            "phase_d.execution.saved_definition",
                                        ),
                                        dcc.Dropdown(
                                            id="phase-d-execution-campaign",
                                            options=options,
                                            placeholder="Select saved Phase D campaign",
                                        ),
                                    ],
                                    className="flex-grow-1",
                                ),
                                dbc.Button(
                                    "Refresh",
                                    id="phase-d-execution-refresh",
                                    color="secondary",
                                    outline=True,
                                ),
                            ],
                            className="d-flex gap-2 align-items-end flex-wrap",
                        ),
                        html.Div(
                            id="phase-d-execution-definition-summary",
                            className="mt-3",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Strong("Execution Request"),
                                help_button("phase_d.execution.request", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        _label(
                                            "Phase D Run ID",
                                            "phase_d.execution.run_id",
                                        ),
                                        dbc.Input(
                                            id="phase-d-execution-run-id",
                                            value=suggested_run_id(),
                                            type="text",
                                        ),
                                    ],
                                    lg=6,
                                ),
                                dbc.Col(
                                    [
                                        _runtime_checkbox(
                                            "phase-d-execution-dry-run",
                                            "Dry Run",
                                            "phase_d.execution.dry_run",
                                        ),
                                        _runtime_checkbox(
                                            "phase-d-execution-continue-on-error",
                                            "Continue on Error",
                                            "phase_d.execution.continue_on_error",
                                        ),
                                    ],
                                    lg=3,
                                    className="d-flex flex-column justify-content-end gap-2 pb-1",
                                ),
                                dbc.Col(
                                    [
                                        _runtime_checkbox(
                                            "phase-d-execution-resume",
                                            "Resume",
                                            "phase_d.execution.resume",
                                        ),
                                        _runtime_checkbox(
                                            "phase-d-execution-overwrite",
                                            "Overwrite Existing",
                                            "phase_d.execution.overwrite",
                                        ),
                                    ],
                                    lg=3,
                                    className="d-flex flex-column justify-content-end gap-2 pb-1",
                                ),
                            ],
                            className="g-3",
                        ),
                        html.Div(
                            id="phase-d-execution-runtime-warnings",
                            className="mt-3",
                        ),
                        html.Details(
                            [
                                html.Summary("Technical runner command"),
                                html.Pre(
                                    id="phase-d-execution-command",
                                    className="phase-d-command-preview mt-2",
                                ),
                            ],
                            className="mt-3",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                dbc.Button(
                                    "Start Phase D",
                                    id="phase-d-execution-start",
                                    color="success",
                                    disabled=not bool(options),
                                ),
                                dbc.Button(
                                    "Stop",
                                    id="phase-d-execution-stop",
                                    color="danger",
                                    outline=True,
                                    disabled=True,
                                ),
                            ],
                            className="d-flex gap-2 flex-wrap",
                        ),
                        html.Div(
                            id="phase-d-execution-action-message",
                            className="mt-3",
                        ),
                        html.Div(id="phase-d-execution-status", className="mt-3"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Strong("Aggregation Progress"),
                                        help_button(
                                            "phase_d.execution.progress",
                                            compact=True,
                                        ),
                                    ],
                                    className="title-with-help",
                                ),
                                dbc.Progress(
                                    "0 / 0",
                                    id="phase-d-execution-progress",
                                    value=0,
                                    striped=True,
                                    animated=False,
                                    className="phase-d-execution-progress",
                                ),
                            ],
                            className="mt-3",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Strong("Live Console"),
                                help_button("phase_d.execution.console", compact=True),
                            ],
                            className="title-with-help",
                        ),
                        html.P(
                            (
                                "The console combines the top-level Phase D runner output "
                                "with a tail of the currently active per-aggregation log."
                            ),
                            className="text-muted small",
                        ),
                        html.Pre(
                            id="phase-d-execution-console",
                            className="phase-d-live-console",
                        ),
                    ]
                ),
                className="phase-d-card",
            ),
            dcc.Interval(
                id="phase-d-execution-poll",
                interval=2000,
                n_intervals=0,
            ),
            dcc.Store(id="phase-d-execution-pending-action"),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Confirm Phase D Execution")),
                    dbc.ModalBody(id="phase-d-execution-confirm-body"),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id="phase-d-execution-confirm-cancel",
                                color="secondary",
                                outline=True,
                            ),
                            dbc.Button(
                                "Confirm",
                                id="phase-d-execution-confirm-accept",
                                color="danger",
                            ),
                        ]
                    ),
                ],
                id="phase-d-execution-confirm-modal",
                is_open=False,
                centered=True,
            ),
        ],
        className="phase-d-page",
    )
