"""Callbacks for the Phase D Thermal-Model Data workspace."""
from __future__ import annotations

from dash import Input, Output, State, callback, ctx, html
import dash_bootstrap_components as dbc

from ....services.phase_d import (
    aggregation_options,
    build_definition,
    case_options,
    definition_summary,
    phase_c_run_options,
    resolve_phase_c_context,
    save_definition,
    selected_aggregation_count,
)
from .execution.callbacks import register_execution_callbacks
from .results.callbacks import register_results_callbacks
from .page import get_tab_builder

_REGISTERED = False


def _split_csv(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _split_lines(value):
    return [item.strip() for item in str(value or "").splitlines() if item.strip()]


def _preview(summary):
    return dbc.Alert(
        [
            html.Div([html.Strong("Phase D campaign: "), summary["phase_d_campaign_id"]]),
            html.Div([html.Strong("Parent Phase C run: "), summary["parent_phase_c_run_key"]]),
            html.Div([html.Strong("Generation campaign: "), summary["parent_generation_campaign_id"]]),
            html.Div([html.Strong("Aggregation matrix: "), summary["matrix_run_id"]]),
            html.Div([html.Strong("Matched aggregation runs: "), str(summary["matched_aggregation_runs"])]),
            html.Div([html.Strong("Scope: "), f"cases={summary['case_ids'] or 'all'} | aggregation_ids={summary['aggregation_ids'] or 'all'} | weights={summary['weight_modes'] or 'all'}"]),
            html.Div([html.Strong("Heat representation: "), f"{summary['heat_representation']} | QZivr separate={summary['qzivr_separate']}"]),
            html.Div([html.Strong("ML/SciML: "), f"policies={summary['ml_policies']} | lags={summary['ml_input_lags']} | horizon={summary['ml_target_horizons']}"]),
            html.Div([html.Strong("Optimization/Bayesian: "), f"policies={summary['ob_policies']}"]),
            html.Div([html.Strong("MLflow: "), str(summary["mlflow_enabled"])]),
            html.Details([html.Summary("Technical runner command"), html.Pre(summary["command"], className="phase-d-command-preview mt-2")], className="mt-2"),
        ],
        color="info",
        className="phase-d-wrap-alert",
    )


def register_phase_d_callbacks():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    register_execution_callbacks()
    register_results_callbacks()

    @callback(
        Output("phase-d-workspace-content", "children"),
        Input("phase-d-workspace-tabs", "value"),
        prevent_initial_call=True,
    )
    def tab(value):
        return get_tab_builder(value)()

    @callback(
        Output("phase-d-builder-phase-c-run", "options"),
        Input("phase-d-builder-refresh-phase-c", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_phase_c(_):
        return phase_c_run_options()

    @callback(
        Output("phase-d-builder-upstream-lineage", "children"),
        Output("phase-d-builder-upstream-cache", "data"),
        Output("phase-d-builder-case-ids", "options"),
        Output("phase-d-builder-case-ids", "value"),
        Output("phase-d-builder-aggregation-ids", "options"),
        Output("phase-d-builder-aggregation-ids", "value"),
        Output("phase-d-builder-weight-modes", "options"),
        Output("phase-d-builder-weight-modes", "value"),
        Input("phase-d-builder-phase-c-run", "value"),
    )
    def load_upstream(run_key):
        if not run_key:
            return "", {}, [], [], [], [], [], []
        try:
            context = resolve_phase_c_context(run_key)
            lineage = dbc.Alert(
                [
                    html.Div([html.Strong("Generation campaign: "), context["parent_generation_campaign_id"]]),
                    html.Div([html.Strong("Phase C run: "), context["phase_c_campaign_run_id"]]),
                    html.Div([html.Strong("Aggregation matrix: "), context["matrix_run_id"]]),
                    html.Div([html.Strong("Successful aggregation runs available: "), str(context["aggregation_run_count"])]),
                    html.Div([html.Strong("Buildings / weather: "), f"{', '.join(context['buildings']) or '—'} / {', '.join(context['weather_locations']) or '—'}"]),
                    html.Div([html.Strong("Recorded strategies / rule sets: "), f"{', '.join(context['strategies']) or '—'} / {', '.join(context['rule_sets']) or '—'}"]),
                ],
                color="light",
                className="phase-d-wrap-alert",
            )
            weights = [{"label": value, "value": value} for value in context["weight_modes"]]
            # Dash Store must receive JSON-safe primitives only.
            cache = {key: value for key, value in context.items() if key != "aggregation_rows"}
            cache["aggregation_rows"] = context["aggregation_rows"]
            return lineage, cache, case_options(context), [], aggregation_options(context), [], weights, []
        except Exception as exc:
            return dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"), {}, [], [], [], [], [], []

    @callback(
        Output("phase-d-builder-match-count", "children"),
        Input("phase-d-builder-upstream-cache", "data"),
        Input("phase-d-builder-case-ids", "value"),
        Input("phase-d-builder-aggregation-ids", "value"),
        Input("phase-d-builder-weight-modes", "value"),
        Input("phase-d-builder-max-runs", "value"),
    )
    def update_match_count(context, cases, aggregations, weights, maximum):
        if not context:
            return "Select a completed Phase C run to discover available Phase D scope."
        count = selected_aggregation_count(
            context,
            case_ids=cases,
            aggregation_ids=aggregations,
            weight_modes=weights,
            max_aggregation_runs=maximum,
        )
        total = int(context.get("aggregation_run_count") or 0)
        return f"Selected runner scope: {count} of {total} successful aggregation runs"

    @callback(
        Output("phase-d-builder-ml-fraction-options", "hidden"),
        Output("phase-d-builder-ml-seasonal-holdout-options", "hidden"),
        Output("phase-d-builder-ob-sd-options", "hidden"),
        Output("phase-d-builder-ob-sbh-options", "hidden"),
        Output("phase-d-builder-ob-ci-options", "hidden"),
        Output("phase-d-builder-ob-cdr-options", "hidden"),
        Input("phase-d-builder-ml-policies", "value"),
        Input("phase-d-builder-ob-policies", "value"),
    )
    def policy_specific_visibility(ml_policies, ob_policies):
        ml = set(ml_policies or [])
        ob = set(ob_policies or [])
        return (
            not bool({"monthly_distributed_holdout", "chronological_holdout"} & ml),
            "seasonal_holdout" not in ml,
            "seasonal_distributed" not in ob,
            "seasonal_block_holdout" not in ob,
            "contiguous_identification" not in ob,
            "custom_datetime_ranges" not in ob,
        )

    @callback(
        Output("phase-d-builder-definition-preview", "children"),
        Output("phase-d-builder-save-status", "children"),
        Input("phase-d-builder-preview", "n_clicks"),
        Input("phase-d-builder-save", "n_clicks"),
        State("phase-d-builder-campaign-id", "value"),
        State("phase-d-builder-display-name", "value"),
        State("phase-d-builder-notes", "value"),
        State("phase-d-builder-machine-id", "value"),
        State("phase-d-builder-phase-c-run", "value"),
        State("phase-d-builder-case-ids", "value"),
        State("phase-d-builder-aggregation-ids", "value"),
        State("phase-d-builder-weight-modes", "value"),
        State("phase-d-builder-max-runs", "value"),
        State("phase-d-builder-heat-representation", "value"),
        State("phase-d-builder-qzivr-separate", "value"),
        State("phase-d-builder-ml-policies", "value"),
        State("phase-d-builder-ml-input-lags", "value"),
        State("phase-d-builder-ml-target-horizons", "value"),
        State("phase-d-builder-ml-train-fraction", "value"),
        State("phase-d-builder-ml-test-fraction", "value"),
        State("phase-d-builder-ml-validation-fraction", "value"),
        State("phase-d-builder-ml-sh-train", "value"),
        State("phase-d-builder-ml-sh-test", "value"),
        State("phase-d-builder-ml-sh-validation", "value"),
        State("phase-d-builder-ob-policies", "value"),
        State("phase-d-builder-sd-offset", "value"),
        State("phase-d-builder-sd-train-days", "value"),
        State("phase-d-builder-sd-test-days", "value"),
        State("phase-d-builder-sbh-train", "value"),
        State("phase-d-builder-sbh-test", "value"),
        State("phase-d-builder-ci-start", "value"),
        State("phase-d-builder-ci-train-days", "value"),
        State("phase-d-builder-ci-test-days", "value"),
        State("phase-d-builder-cdr-train", "value"),
        State("phase-d-builder-cdr-test", "value"),
        State("phase-d-builder-calendar-year", "value"),
        State("phase-d-builder-parquet-compression", "value"),
        State("phase-d-builder-output-root", "value"),
        State("phase-d-builder-mlflow-enabled", "value"),
        State("phase-d-builder-mlflow-experiment", "value"),
        State("phase-d-builder-mlflow-run-name", "value"),
        State("phase-d-builder-mlflow-strict", "value"),
        State("phase-d-builder-replace", "value"),
        prevent_initial_call=True,
    )
    def preview_or_save(
        _preview_clicks, _save_clicks, campaign_id, display_name, notes, machine_id, phase_c_run,
        case_ids, aggregation_ids, weight_modes, max_runs, heat_representation,
        qzivr_separate, ml_policies, ml_input_lags, ml_target_horizons,
        ml_train_fraction, ml_test_fraction, ml_validation_fraction,
        ml_sh_train, ml_sh_test, ml_sh_validation, ob_policies, sd_offset,
        sd_train_days, sd_test_days, sbh_train, sbh_test, ci_start,
        ci_train_days, ci_test_days, cdr_train, cdr_test, calendar_year,
        parquet_compression, output_root, mlflow_enabled, mlflow_experiment,
        mlflow_run_name, mlflow_strict, replace,
    ):
        try:
            values = {
                "case_ids": list(case_ids or []),
                "aggregation_ids": list(aggregation_ids or []),
                "weight_modes": list(weight_modes or []),
                "max_aggregation_runs": max_runs,
                "heat_representation": heat_representation,
                "qzivr_separate": bool(qzivr_separate),
                "ml_policies": list(ml_policies or []),
                "ml_input_lags": _split_csv(ml_input_lags),
                "ml_target_horizons": _split_csv(ml_target_horizons),
                "ml_train_fraction": ml_train_fraction,
                "ml_test_fraction": ml_test_fraction,
                "ml_validation_fraction": ml_validation_fraction,
                "ml_sh_train_seasons": list(ml_sh_train or []),
                "ml_sh_test_seasons": list(ml_sh_test or []),
                "ml_sh_validation_seasons": list(ml_sh_validation or []),
                "ob_policies": list(ob_policies or []),
                "sd_season_offset_days": sd_offset,
                "sd_train_days": sd_train_days,
                "sd_test_days": sd_test_days,
                "sbh_train_seasons": list(sbh_train or []),
                "sbh_test_seasons": list(sbh_test or []),
                "ci_start_datetime": ci_start,
                "ci_train_days": ci_train_days,
                "ci_test_days": ci_test_days,
                "cdr_train_ranges": _split_lines(cdr_train),
                "cdr_test_ranges": _split_lines(cdr_test),
                "phase_d_calendar_year": calendar_year,
                "parquet_compression": parquet_compression,
                "output_root": output_root,
                "mlflow_enabled": bool(mlflow_enabled),
                "mlflow_experiment_name": mlflow_experiment,
                "mlflow_run_name": mlflow_run_name,
                "mlflow_strict": bool(mlflow_strict),
            }
            definition = build_definition(
                phase_d_campaign_id=campaign_id,
                parent_phase_c_run_key=phase_c_run,
                machine_id=machine_id,
                display_name=display_name,
                notes=notes,
                values=values,
            )
            preview = _preview(definition_summary(definition))
            if ctx.triggered_id == "phase-d-builder-save":
                path = save_definition(definition, replace=bool(replace))
                return preview, dbc.Alert(f"Saved Phase D campaign definition: {path}", color="success", className="phase-d-wrap-alert")
            return preview, ""
        except Exception as exc:
            return "", dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger", className="phase-d-wrap-alert")
