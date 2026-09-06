"""Callbacks dedicated to the simplified Phase C Tab 2 execution surface."""
from __future__ import annotations

from dash import Input, Output, State, callback, ctx, html
import dash_bootstrap_components as dbc

from .....services.heat_input import (
    ACTIVE_STATUSES,
    MANAGER,
    command_text,
    confirmation_reasons,
    effective_config_text,
    execution_definition_summary,
    list_execution_definitions,
    runtime_warnings,
    suggested_run_id,
)


_REGISTERED = False


def _definition_options():
    return [
        {
            "label": (
                f"{row['phase_c_campaign_id']} | "
                f"parent={row['parent_generation_campaign_id']} | "
                f"matrix={row.get('matrix_run_id') or '—'}"
            ),
            "value": row["phase_c_campaign_id"],
        }
        for row in list_execution_definitions()
    ]


def _runtime_kwargs(run_id):
    """Dash always runs the complete non-overwriting Phase C workflow."""
    return {
        "phase_c_run_id": str(run_id or "").strip() or None,
        "start_stage": "C1",
        "stop_stage": "C9",
        "overwrite_existing": False,
    }


def _warning_components(rows):
    return [
        dbc.Alert(
            row["message"],
            color={"danger": "danger", "warning": "warning"}.get(
                row.get("severity"),
                "secondary",
            ),
            className="heat-input-wrap-alert mb-2",
        )
        for row in rows
    ]


def _status_card(snapshot):
    status = str(snapshot["status"])
    color = {
        "running": "primary",
        "dry_running": "info",
        "completed": "success",
        "dry_run_completed": "success",
        "failed": "danger",
        "stop_requested": "warning",
        "stopped": "secondary",
    }.get(status, "secondary")
    runtime = snapshot.get("runtime_seconds")
    runtime_text = "—" if runtime is None else f"{runtime:.1f} s"
    mode = snapshot.get("mode") or "—"
    if mode == "dry_run":
        mode = "Dry run"
    elif mode == "execute":
        mode = "Complete Phase C"
    return dbc.Alert(
        [
            html.Div([html.Strong("Status: "), status]),
            html.Div([html.Strong("Mode: "), mode]),
            html.Div([html.Strong("Run ID: "), snapshot.get("phase_c_run_id") or "—"]),
            html.Div([html.Strong("PID: "), str(snapshot.get("pid") or "—")]),
            html.Div([html.Strong("Runtime: "), runtime_text]),
            html.Div(
                [
                    html.Strong("Current step: "),
                    snapshot.get("current_command") or "—",
                ]
            ),
            html.Div(
                [
                    html.Strong("Return code: "),
                    (
                        "—"
                        if snapshot.get("return_code") is None
                        else str(snapshot["return_code"])
                    ),
                ]
            ),
        ],
        color=color,
        className="heat-input-wrap-alert",
    )


def _stage_components(snapshot):
    statuses = snapshot.get("stage_statuses") or {}
    colors = {
        "pending": "secondary",
        "running": "primary",
        "completed": "success",
        "failed": "danger",
        "stopped": "warning",
        "skipped": "light",
    }
    return [
        dbc.Badge(
            f"{stage}: {statuses.get(stage, 'pending')}",
            color=colors.get(statuses.get(stage), "secondary"),
            text_color="dark" if statuses.get(stage) == "skipped" else None,
            className="heat-input-progress-badge",
        )
        for stage in [f"C{i}" for i in range(1, 10)]
    ]


def register_execution_callbacks():
    """Register Tab-2 callbacks exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    @callback(
        Output("phase-c-execution-campaign", "options"),
        Input("phase-c-execution-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_definitions(_):
        return _definition_options()

    @callback(
        Output("phase-c-execution-definition-summary", "children"),
        Output("phase-c-execution-run-id", "value"),
        Input("phase-c-execution-campaign", "value"),
    )
    def load_selected_definition(campaign_id):
        if not campaign_id:
            return "", suggested_run_id()
        try:
            summary = execution_definition_summary(campaign_id)
            run_id = summary["saved_phase_c_run_id"] or suggested_run_id()
            machine_color = "success" if summary["machine_match"] else "warning"
            validation_enabled = summary.get("validation_profile") != "none"
            body = dbc.Alert(
                [
                    html.Div(
                        [
                            html.Strong("Campaign: "),
                            summary["phase_c_campaign_id"],
                            (
                                f" — {summary['display_name']}"
                                if summary.get("display_name")
                                else ""
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Aggregation → Generation: "),
                            f"{summary['parent_aggregation_campaign_id']} → "
                            f"{summary['parent_generation_campaign_id']}",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Matrix: "),
                            summary.get("matrix_run_id") or "—",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Estimators: "),
                            ", ".join(summary["estimators"]),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Validation / MLflow: "),
                            f"{'enabled' if validation_enabled else 'disabled'} / "
                            f"{'enabled' if summary['mlflow_enabled'] else 'disabled'}",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Saved / current machine: "),
                            f"{summary['saved_machine_id']} / "
                            f"{summary['current_machine_id']}",
                        ]
                    ),
                ],
                color=machine_color,
                className="heat-input-wrap-alert",
            )
            return body, run_id
        except Exception as exc:
            return (
                dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
                suggested_run_id(),
            )

    @callback(
        Output("phase-c-execution-command", "children"),
        Output("phase-c-execution-effective-config", "children"),
        Output("phase-c-execution-runtime-warnings", "children"),
        Input("phase-c-execution-campaign", "value"),
        Input("phase-c-execution-run-id", "value"),
    )
    def preview_execution(campaign_id, run_id):
        if not campaign_id:
            return "", "", ""
        kwargs = _runtime_kwargs(run_id)
        try:
            command = command_text(campaign_id, **kwargs)
            config_text = effective_config_text(campaign_id, **kwargs)
            warnings = _warning_components(runtime_warnings(campaign_id, **kwargs))
            return command, config_text, warnings
        except Exception as exc:
            message = dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
                className="heat-input-wrap-alert",
            )
            return "", "", message

    @callback(
        Output("phase-c-execution-confirm-modal", "is_open"),
        Output("phase-c-execution-confirm-body", "children"),
        Output("phase-c-execution-pending-action", "data"),
        Output("phase-c-execution-action-message", "children"),
        Input("phase-c-execution-start", "n_clicks"),
        Input("phase-c-execution-stop", "n_clicks"),
        Input("phase-c-execution-confirm-accept", "n_clicks"),
        Input("phase-c-execution-confirm-cancel", "n_clicks"),
        State("phase-c-execution-campaign", "value"),
        State("phase-c-execution-run-id", "value"),
        State("phase-c-execution-dry-run", "value"),
        State("phase-c-execution-pending-action", "data"),
        prevent_initial_call=True,
    )
    def execution_action(
        _start_clicks,
        _stop_clicks,
        _confirm_clicks,
        _cancel_clicks,
        campaign_id,
        run_id,
        dry_run,
        pending,
    ):
        trigger = ctx.triggered_id
        if trigger == "phase-c-execution-confirm-cancel":
            return False, "", None, dbc.Alert("Action cancelled.", color="secondary")

        if trigger == "phase-c-execution-stop":
            snapshot = MANAGER.snapshot()
            if snapshot["status"] not in ACTIVE_STATUSES:
                return False, "", None, dbc.Alert(
                    "No active Phase C process to stop.",
                    color="secondary",
                )
            return (
                True,
                (
                    "Stop the active Phase C subprocess tree? Existing artifacts are "
                    "left in place; this action does not delete outputs."
                ),
                {"action": "stop"},
                "",
            )

        if trigger == "phase-c-execution-start":
            if not campaign_id:
                return False, "", None, dbc.Alert(
                    "Select a saved Phase C campaign.",
                    color="warning",
                )
            kwargs = _runtime_kwargs(run_id)
            dry_run = bool(dry_run)
            try:
                reasons = confirmation_reasons(campaign_id, **kwargs)
                if reasons and not dry_run:
                    body = html.Div(
                        [
                            html.P("Confirm execution with the following warning(s):"),
                            *_warning_components(reasons),
                        ]
                    )
                    return (
                        True,
                        body,
                        {
                            "action": "start",
                            "campaign_id": campaign_id,
                            "kwargs": kwargs,
                            "dry_run": dry_run,
                        },
                        "",
                    )
                MANAGER.start(campaign_id, dry_run=dry_run, **kwargs)
                text = (
                    "Dry-run Phase C plan started."
                    if dry_run
                    else "Complete Phase C execution started."
                )
                return False, "", None, dbc.Alert(
                    text,
                    color="info" if dry_run else "success",
                )
            except Exception as exc:
                return False, "", None, dbc.Alert(
                    f"{type(exc).__name__}: {exc}",
                    color="danger",
                )

        if trigger == "phase-c-execution-confirm-accept":
            pending = pending or {}
            action = pending.get("action")
            try:
                if action == "stop":
                    MANAGER.stop()
                    return False, "", None, dbc.Alert(
                        "Stop requested. The process tree is being terminated.",
                        color="warning",
                    )
                if action == "start":
                    dry_run = bool(pending.get("dry_run"))
                    MANAGER.start(
                        pending["campaign_id"],
                        dry_run=dry_run,
                        **dict(pending.get("kwargs") or {}),
                    )
                    return False, "", None, dbc.Alert(
                        (
                            "Dry-run Phase C plan started after confirmation."
                            if dry_run
                            else "Complete Phase C execution started after confirmation."
                        ),
                        color="info" if dry_run else "success",
                    )
            except Exception as exc:
                return False, "", None, dbc.Alert(
                    f"{type(exc).__name__}: {exc}",
                    color="danger",
                )

        return False, "", None, ""

    @callback(
        Output("phase-c-execution-status", "children"),
        Output("phase-c-execution-stage-progress", "children"),
        Output("phase-c-execution-console", "children"),
        Output("phase-c-execution-dry-run", "disabled"),
        Output("phase-c-execution-start", "disabled"),
        Output("phase-c-execution-stop", "disabled"),
        Input("phase-c-execution-poll", "n_intervals"),
        State("phase-c-execution-campaign", "value"),
    )
    def poll_execution(_, campaign_id):
        snapshot = MANAGER.snapshot()
        active = snapshot["status"] in ACTIVE_STATUSES
        selectable = bool(campaign_id)
        return (
            _status_card(snapshot),
            _stage_components(snapshot),
            snapshot["console"],
            active,
            active or not selectable,
            not active,
        )
