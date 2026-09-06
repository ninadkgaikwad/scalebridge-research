"""Callbacks dedicated to Phase D Tab 2 managed execution."""
from __future__ import annotations

from dash import Input, Output, State, callback, ctx, html
import dash_bootstrap_components as dbc

from .....services.phase_d import (
    ACTIVE_STATUSES,
    MANAGER,
    command_text,
    confirmation_reasons,
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
                f"{row['phase_d_campaign_id']} | "
                f"Phase C={row['phase_c_campaign_run_id']} | "
                f"matrix={row['matrix_run_id']}"
            ),
            "value": row["phase_d_campaign_id"],
        }
        for row in list_execution_definitions()
    ]


def _runtime_kwargs(run_id, resume, overwrite, continue_on_error, dry_run):
    return {
        "phase_d_run_id": str(run_id or "").strip() or None,
        "resume": bool(resume),
        "overwrite_existing": bool(overwrite),
        "continue_on_error": bool(continue_on_error),
        "dry_run": bool(dry_run),
    }


def _warning_components(rows):
    colors = {
        "danger": "danger",
        "warning": "warning",
        "info": "info",
    }
    return [
        dbc.Alert(
            row["message"],
            color=colors.get(row.get("severity"), "secondary"),
            className="phase-d-wrap-alert mb-2",
        )
        for row in rows
    ]


def _definition_card(summary):
    color = "success" if summary["machine_match"] else "warning"
    return dbc.Alert(
        [
            html.Div(
                [
                    html.Strong("Campaign: "),
                    summary["phase_d_campaign_id"],
                    (
                        f" — {summary['display_name']}"
                        if summary.get("display_name")
                        else ""
                    ),
                ]
            ),
            html.Div(
                [
                    html.Strong("Parent Phase C run: "),
                    summary["parent_phase_c_run_key"],
                ]
            ),
            html.Div(
                [
                    html.Strong("Generation / matrix: "),
                    f"{summary['parent_generation_campaign_id']} / "
                    f"{summary['matrix_run_id']}",
                ]
            ),
            html.Div(
                [
                    html.Strong("Matched aggregation runs: "),
                    str(summary["matched_aggregation_runs"]),
                ]
            ),
            html.Div(
                [
                    html.Strong("Heat representation: "),
                    summary["heat_representation"],
                ]
            ),
            html.Div(
                [
                    html.Strong("ML/SciML: "),
                    f"policies={summary['ml_policies']} | "
                    f"lags={summary['ml_input_lags']} | "
                    f"horizons={summary['ml_target_horizons']}",
                ]
            ),
            html.Div(
                [
                    html.Strong("Optimization/Bayesian: "),
                    f"policies={summary['ob_policies']}",
                ]
            ),
            html.Div(
                [
                    html.Strong("MLflow: "),
                    "enabled" if summary["mlflow_enabled"] else "disabled",
                ]
            ),
            html.Div(
                [
                    html.Strong("Saved / current machine: "),
                    f"{summary['saved_machine_id']} / "
                    f"{summary['current_machine_id']}",
                ]
            ),
            html.Div(
                [
                    html.Strong("Output root: "),
                    summary["output_root"],
                ]
            ),
        ],
        color=color,
        className="phase-d-wrap-alert",
    )


def _status_card(snapshot):
    status = str(snapshot.get("status") or "not_started")
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
        mode = "Dry Run"
    elif mode == "execute":
        mode = "Phase D execution"

    total = int(snapshot.get("selected_aggregation_count") or 0)
    done = int(snapshot.get("finished_aggregation_count") or 0)
    datasets = int(snapshot.get("dataset_count") or 0)
    failed = int(snapshot.get("failed_aggregation_count") or 0)

    current = snapshot.get("current_aggregation_run_id") or "—"
    sequence = snapshot.get("current_sequence")
    if sequence and total:
        current = f"{sequence}/{total} — {current}"

    return dbc.Alert(
        [
            html.Div([html.Strong("Status: "), status]),
            html.Div([html.Strong("Mode: "), mode]),
            html.Div(
                [
                    html.Strong("Run ID: "),
                    snapshot.get("phase_d_run_id") or "—",
                ]
            ),
            html.Div([html.Strong("PID: "), str(snapshot.get("pid") or "—")]),
            html.Div([html.Strong("Runtime: "), runtime_text]),
            html.Div(
                [
                    html.Strong("Aggregation progress: "),
                    f"{done}/{total}" if total else "—",
                ]
            ),
            html.Div([html.Strong("Current aggregation: "), current]),
            html.Div([html.Strong("Datasets registered: "), str(datasets)]),
            html.Div([html.Strong("Failed aggregations: "), str(failed)]),
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
            html.Div(
                [
                    html.Strong("Campaign run root: "),
                    snapshot.get("campaign_run_root") or "—",
                ]
            ),
        ],
        color=color,
        className="phase-d-wrap-alert",
    )


def register_execution_callbacks():
    """Register Phase D Tab-2 callbacks exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    @callback(
        Output("phase-d-execution-campaign", "options"),
        Input("phase-d-execution-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_definitions(_):
        return _definition_options()

    @callback(
        Output("phase-d-execution-definition-summary", "children"),
        Output("phase-d-execution-run-id", "value"),
        Input("phase-d-execution-campaign", "value"),
    )
    def load_selected_definition(campaign_id):
        if not campaign_id:
            return "", suggested_run_id()
        try:
            summary = execution_definition_summary(campaign_id)
            return _definition_card(summary), suggested_run_id()
        except Exception as exc:
            return (
                dbc.Alert(
                    f"{type(exc).__name__}: {exc}",
                    color="danger",
                    className="phase-d-wrap-alert",
                ),
                suggested_run_id(),
            )

    @callback(
        Output("phase-d-execution-command", "children"),
        Output("phase-d-execution-runtime-warnings", "children"),
        Input("phase-d-execution-campaign", "value"),
        Input("phase-d-execution-run-id", "value"),
        Input("phase-d-execution-resume", "value"),
        Input("phase-d-execution-overwrite", "value"),
        Input("phase-d-execution-continue-on-error", "value"),
        Input("phase-d-execution-dry-run", "value"),
    )
    def preview_execution(
        campaign_id,
        run_id,
        resume,
        overwrite,
        continue_on_error,
        dry_run,
    ):
        if not campaign_id:
            return "", ""
        kwargs = _runtime_kwargs(
            run_id,
            resume,
            overwrite,
            continue_on_error,
            dry_run,
        )
        try:
            command = command_text(campaign_id, **kwargs)
            warnings = _warning_components(runtime_warnings(campaign_id, **kwargs))
            return command, warnings
        except Exception as exc:
            return "", dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
                className="phase-d-wrap-alert",
            )

    @callback(
        Output("phase-d-execution-confirm-modal", "is_open"),
        Output("phase-d-execution-confirm-body", "children"),
        Output("phase-d-execution-pending-action", "data"),
        Output("phase-d-execution-action-message", "children"),
        Input("phase-d-execution-start", "n_clicks"),
        Input("phase-d-execution-stop", "n_clicks"),
        Input("phase-d-execution-confirm-accept", "n_clicks"),
        Input("phase-d-execution-confirm-cancel", "n_clicks"),
        State("phase-d-execution-campaign", "value"),
        State("phase-d-execution-run-id", "value"),
        State("phase-d-execution-resume", "value"),
        State("phase-d-execution-overwrite", "value"),
        State("phase-d-execution-continue-on-error", "value"),
        State("phase-d-execution-dry-run", "value"),
        State("phase-d-execution-pending-action", "data"),
        prevent_initial_call=True,
    )
    def execution_action(
        _start_clicks,
        _stop_clicks,
        _confirm_clicks,
        _cancel_clicks,
        campaign_id,
        run_id,
        resume,
        overwrite,
        continue_on_error,
        dry_run,
        pending,
    ):
        trigger = ctx.triggered_id

        if trigger == "phase-d-execution-confirm-cancel":
            return False, "", None, dbc.Alert(
                "Action cancelled.",
                color="secondary",
                className="phase-d-wrap-alert",
            )

        if trigger == "phase-d-execution-stop":
            snapshot = MANAGER.snapshot()
            if snapshot["status"] not in ACTIVE_STATUSES:
                return False, "", None, dbc.Alert(
                    "No active Phase D process to stop.",
                    color="secondary",
                    className="phase-d-wrap-alert",
                )
            return (
                True,
                (
                    "Stop the active Phase D subprocess tree? Existing Phase D "
                    "artifacts already written by the scientific runner are left in place."
                ),
                {"action": "stop"},
                "",
            )

        if trigger == "phase-d-execution-start":
            if not campaign_id:
                return False, "", None, dbc.Alert(
                    "Select a saved Phase D campaign.",
                    color="warning",
                    className="phase-d-wrap-alert",
                )
            kwargs = _runtime_kwargs(
                run_id,
                resume,
                overwrite,
                continue_on_error,
                dry_run,
            )
            try:
                reasons = confirmation_reasons(campaign_id, **kwargs)
                if reasons:
                    body = html.Div(
                        [
                            html.P(
                                "Confirm execution with the following warning(s):"
                            ),
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
                        },
                        "",
                    )
                MANAGER.start(campaign_id, **kwargs)
                text = (
                    "Phase D Dry Run started."
                    if kwargs["dry_run"]
                    else "Phase D execution started."
                )
                return False, "", None, dbc.Alert(
                    text,
                    color="info" if kwargs["dry_run"] else "success",
                    className="phase-d-wrap-alert",
                )
            except Exception as exc:
                return False, "", None, dbc.Alert(
                    f"{type(exc).__name__}: {exc}",
                    color="danger",
                    className="phase-d-wrap-alert",
                )

        if trigger == "phase-d-execution-confirm-accept":
            pending = pending or {}
            action = pending.get("action")
            try:
                if action == "stop":
                    MANAGER.stop()
                    return False, "", None, dbc.Alert(
                        "Stop requested. The Phase D process tree is being terminated.",
                        color="warning",
                        className="phase-d-wrap-alert",
                    )
                if action == "start":
                    campaign = pending.get("campaign_id")
                    kwargs = dict(pending.get("kwargs") or {})
                    MANAGER.start(campaign, **kwargs)
                    text = (
                        "Phase D Dry Run started."
                        if kwargs.get("dry_run")
                        else "Phase D execution started."
                    )
                    return False, "", None, dbc.Alert(
                        text,
                        color="info" if kwargs.get("dry_run") else "success",
                        className="phase-d-wrap-alert",
                    )
            except Exception as exc:
                return False, "", None, dbc.Alert(
                    f"{type(exc).__name__}: {exc}",
                    color="danger",
                    className="phase-d-wrap-alert",
                )

        return False, "", None, ""

    @callback(
        Output("phase-d-execution-status", "children"),
        Output("phase-d-execution-progress", "value"),
        Output("phase-d-execution-progress", "children"),
        Output("phase-d-execution-progress", "animated"),
        Output("phase-d-execution-console", "children"),
        Output("phase-d-execution-start", "disabled"),
        Output("phase-d-execution-stop", "disabled"),
        Input("phase-d-execution-poll", "n_intervals"),
        State("phase-d-execution-campaign", "value"),
    )
    def poll_execution(_tick, selected_campaign):
        snapshot = MANAGER.snapshot()
        active = snapshot["status"] in ACTIVE_STATUSES

        total = int(snapshot.get("selected_aggregation_count") or 0)
        done = int(snapshot.get("finished_aggregation_count") or 0)
        percent = float(snapshot.get("progress_percent") or 0.0)
        progress_text = f"{done} / {total}" if total else "0 / 0"

        console = str(snapshot.get("console") or "").rstrip()
        latest_log = str(snapshot.get("latest_log_tail") or "").rstrip()
        latest_path = snapshot.get("latest_log_path")
        if latest_log:
            section = (
                f"--- Current aggregation log: {latest_path} ---\n"
                f"{latest_log}"
            )
            console = f"{console}\n\n{section}".strip() if console else section

        return (
            _status_card(snapshot),
            percent,
            progress_text,
            active,
            console,
            active or not bool(selected_campaign),
            not active,
        )
