"""Callbacks for the Phase C Heat-Input Regression workspace."""
from __future__ import annotations

from dash import Input, Output, State, callback, ctx, html, no_update
import dash_bootstrap_components as dbc

from ....services.heat_input import (
    build_definition,
    definition_summary,
    matrix_run_options,
    matrix_summary,
    parent_aggregation_options,
    resolve_parent_context,
    resolve_scope_selection,
    save_definition,
    scope_options,
)
from .execution import register_execution_callbacks
from .page import get_tab_builder
from .results import register_results_callbacks


_REGISTERED = False


def _options(values):
    return [{"label": str(value), "value": str(value)} for value in values or []]


def _definition_preview(summary):
    models = summary.get("model_ids") or []
    scope = [
        summary.get("case_id") or "all cases",
        summary.get("aggregation_strategy") or "all strategies",
        summary.get("custom_grouping_id") or "no custom grouping",
        summary.get("weight_mode") or "all weights",
        summary.get("rule_set") or "all rule sets",
    ]
    return dbc.Alert(
        [
            html.Div(
                [
                    html.Strong("Phase C campaign: "),
                    summary["phase_c_campaign_id"],
                ]
            ),
            html.Div(
                [
                    html.Strong("Aggregation → Generation: "),
                    f"{summary['parent_aggregation_campaign_id']} → "
                    f"{summary['parent_generation_campaign_id']}",
                ]
            ),
            html.Div([html.Strong("Matrix: "), summary.get("matrix_run_id") or "—"]),
            html.Div([html.Strong("Scope: "), " | ".join(scope)]),
            html.Div(
                [
                    html.Strong("Models: "),
                    ", ".join(models) if models else "all applicable models",
                ]
            ),
            html.Div(
                [
                    html.Strong("Features / target: "),
                    f"{summary['internal_gain_predictor_method']} / "
                    f"{summary['hvac_target_method']}",
                ]
            ),
            html.Div([html.Strong("Split: "), summary["split"]]),
            html.Div(
                [
                    html.Strong("Estimators / devices: "),
                    f"{', '.join(summary['estimators'])} / "
                    f"{', '.join(summary['devices']) or 'not applicable'}",
                ]
            ),
            html.Div(
                [
                    html.Strong("Validation / MLflow: "),
                    f"{summary['validation_enabled']} / {summary['mlflow_enabled']}",
                ]
            ),
        ],
        color="info",
        className="heat-input-wrap-alert",
    )


def register_heat_input_callbacks():
    """Register Phase C callbacks exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    register_execution_callbacks()
    register_results_callbacks()

    @callback(
        Output("phase-c-workspace-content", "children"),
        Input("phase-c-workspace-tabs", "value"),
        prevent_initial_call=True,
    )
    def tab(value):
        return get_tab_builder(value)()

    @callback(
        Output("phase-c-builder-parent-aggregation", "options"),
        Input("phase-c-builder-refresh-parents", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_parent_aggregation(_):
        return parent_aggregation_options()

    @callback(
        Output("phase-c-builder-matrix-run", "options"),
        Output("phase-c-builder-matrix-run", "value"),
        Output("phase-c-builder-parent-lineage", "children"),
        Output("phase-c-builder-parent-cache", "data"),
        Input("phase-c-builder-parent-aggregation", "value"),
    )
    def load_parent(aggregation_campaign_id):
        if not aggregation_campaign_id:
            return [], None, "", {}
        try:
            context = resolve_parent_context(aggregation_campaign_id)
            options = matrix_run_options(aggregation_campaign_id)
            selected = options[0]["value"] if options else None
            source = (
                "legacy artifact discovery"
                if context.get("legacy_artifact_only")
                else (
                    "Phase B definition + artifacts"
                    if context.get("definition_available")
                    else "Aggregation artifacts"
                )
            )
            color = "success" if context["campaign_root_exists"] else "warning"
            lineage = dbc.Alert(
                [
                    html.Div(
                        [
                            html.Strong("Generation parent: "),
                            context["parent_generation_campaign_id"],
                        ]
                    ),
                    html.Div([html.Strong("Discovery: "), source]),
                    html.Div(
                        f"{len(options)} matrix run(s) available for Phase C."
                    ),
                ],
                color=color,
                className="heat-input-wrap-alert",
            )
            return options, selected, lineage, context
        except Exception as exc:
            return (
                [],
                None,
                dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
                {},
            )

    @callback(
        Output("phase-c-builder-matrix-summary", "children"),
        Output("phase-c-builder-matrix-cache", "data"),
        Output("phase-c-builder-case", "options"),
        Output("phase-c-builder-case", "value"),
        Output("phase-c-builder-strategy", "options"),
        Output("phase-c-builder-strategy", "value"),
        Output("phase-c-builder-custom-grouping", "options"),
        Output("phase-c-builder-custom-grouping", "value"),
        Output("phase-c-builder-weight", "options"),
        Output("phase-c-builder-weight", "value"),
        Output("phase-c-builder-rule-set", "options"),
        Output("phase-c-builder-rule-set", "value"),
        Input("phase-c-builder-parent-aggregation", "value"),
        Input("phase-c-builder-matrix-run", "value"),
    )
    def load_matrix(aggregation_campaign_id, matrix_run_id):
        empty = ("", {}, [], None, [], None, [], None, [], None, [], None)
        if not aggregation_campaign_id or not matrix_run_id:
            return empty
        try:
            summary = matrix_summary(aggregation_campaign_id, matrix_run_id)
            readiness = str(summary["readiness"])
            color = {
                "ready": "success",
                "partial": "warning",
                "unusable": "danger",
            }.get(readiness, "secondary")
            body = dbc.Alert(
                [
                    html.Div(
                        [
                            html.Strong("Readiness: "),
                            readiness,
                            " | ",
                            html.Strong("Plans: "),
                            (
                                f"{summary['successful_plan_count']} successful / "
                                f"{summary['selected_plan_count']} selected / "
                                f"{summary['failed_plan_count']} failed"
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Buildings: "),
                            ", ".join(summary.get("building_types") or []) or "—",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Weather: "),
                            ", ".join(summary.get("weather_locations") or []) or "—",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Aggregation strategies: "),
                            ", ".join(summary.get("strategies") or []) or "—",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Custom grouping IDs: "),
                            ", ".join(summary.get("custom_grouping_ids") or []) or "none",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Weight modes / rule sets: "),
                            f"{', '.join(summary.get('weight_modes') or []) or '—'} / "
                            f"{', '.join(summary.get('rule_sets') or []) or '—'}",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Successful case-plan rows: "),
                            str(summary.get("successful_case_plan_rows") or 0),
                        ]
                    ),
                ],
                color=color,
                className="heat-input-wrap-alert",
            )
            return (
                body,
                summary,
                scope_options(summary, "case_id"),
                None,
                scope_options(summary, "strategy"),
                None,
                scope_options(summary, "custom_grouping_id"),
                None,
                scope_options(summary, "weight_mode"),
                None,
                scope_options(summary, "rule_set"),
                None,
            )
        except Exception as exc:
            return (
                dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
                {},
                [],
                None,
                [],
                None,
                [],
                None,
                [],
                None,
                [],
                None,
            )

    @callback(
        Output("phase-c-builder-custom-grouping", "disabled"),
        Output("phase-c-builder-custom-grouping", "value", allow_duplicate=True),
        Input("phase-c-builder-strategy", "value"),
        prevent_initial_call=True,
    )
    def custom_grouping_applicability(strategy):
        applicable = strategy == "custom_groups"
        return (not applicable), (None if not applicable else no_update)

    @callback(
        Output("phase-c-builder-definition-preview", "children"),
        Output("phase-c-builder-save-status", "children"),
        Input("phase-c-builder-preview", "n_clicks"),
        Input("phase-c-builder-save", "n_clicks"),
        State("phase-c-builder-campaign-id", "value"),
        State("phase-c-builder-display-name", "value"),
        State("phase-c-builder-notes", "value"),
        State("phase-c-builder-machine-id", "value"),
        State("phase-c-builder-parent-aggregation", "value"),
        State("phase-c-builder-matrix-run", "value"),
        State("phase-c-builder-case", "value"),
        State("phase-c-builder-strategy", "value"),
        State("phase-c-builder-custom-grouping", "value"),
        State("phase-c-builder-weight", "value"),
        State("phase-c-builder-rule-set", "value"),
        State("phase-c-builder-matrix-cache", "data"),
        State("phase-c-builder-model-ids", "value"),
        State("phase-c-builder-internal-gain-method", "value"),
        State("phase-c-builder-hvac-target-method", "value"),
        State("phase-c-builder-split-strategy", "value"),
        State("phase-c-builder-train-fraction", "value"),
        State("phase-c-builder-validation-fraction", "value"),
        State("phase-c-builder-test-fraction", "value"),
        State("phase-c-builder-estimators", "value"),
        State("phase-c-builder-devices", "value"),
        State("phase-c-builder-validation-enabled", "value"),
        State("phase-c-builder-mlflow-enabled", "value"),
        State("phase-c-builder-replace", "value"),
        prevent_initial_call=True,
    )
    def preview_or_save(
        _preview_clicks,
        _save_clicks,
        phase_c_campaign_id,
        display_name,
        notes,
        machine_id,
        parent_aggregation_campaign_id,
        matrix_run_id,
        case_id,
        aggregation_strategy,
        custom_grouping_id,
        weight_mode,
        rule_set,
        matrix_cache,
        model_ids,
        internal_gain_method,
        hvac_target_method,
        split_strategy,
        train_fraction,
        validation_fraction,
        test_fraction,
        estimator_types,
        pytorch_devices,
        validation_enabled,
        mlflow_enabled,
        replace,
    ):
        try:
            estimators = list(estimator_types or [])
            if not estimators:
                raise ValueError("Select at least one estimator")
            devices = list(pytorch_devices or [])
            if "pytorch_linear" in estimators and not devices:
                raise ValueError("Select at least one PyTorch device")
            if "pytorch_linear" not in estimators:
                devices = []

            resolved_scope = resolve_scope_selection(
                matrix_cache or {},
                case_id=case_id,
                strategy=aggregation_strategy,
                custom_grouping_id=custom_grouping_id,
                weight_mode=weight_mode,
                rule_set=rule_set,
            )
            config_values = {
                **resolved_scope,
                "model_ids": list(model_ids or []),
                "internal_gain_predictor_method": internal_gain_method,
                "hvac_target_method": hvac_target_method,
                "split_strategy": split_strategy,
                "train_fraction": train_fraction,
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "estimator_types": estimators,
                "pytorch_devices": devices,
                "validation_profile": "full" if validation_enabled else "none",
                "mlflow_enabled": bool(mlflow_enabled),
            }
            definition = build_definition(
                phase_c_campaign_id=phase_c_campaign_id,
                parent_aggregation_campaign_id=parent_aggregation_campaign_id,
                matrix_run_id=matrix_run_id,
                machine_id=machine_id,
                config_values=config_values,
                display_name=display_name,
                notes=notes,
            )
            summary = definition_summary(definition)
            summary.update(
                {
                    "aggregation_strategy": aggregation_strategy or None,
                    "custom_grouping_id": custom_grouping_id or None,
                    "rule_set": rule_set or None,
                }
            )
            preview = _definition_preview(summary)
            if ctx.triggered_id == "phase-c-builder-save":
                path = save_definition(definition, replace=bool(replace))
                return preview, dbc.Alert(
                    f"Saved Phase C campaign definition: {path}",
                    color="success",
                    className="heat-input-wrap-alert",
                )
            return preview, ""
        except Exception as exc:
            return "", dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
                className="heat-input-wrap-alert",
            )
