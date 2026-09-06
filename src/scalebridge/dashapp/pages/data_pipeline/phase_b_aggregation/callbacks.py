"""Callbacks for the Phase B Aggregation workspace."""

from __future__ import annotations

import json

from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

from ....services.aggregation import (
    MANAGER,
    build_definition,
    build_plan_requests,
    build_selected_aggregation_data_export,
    command_text,
    definition_summary,
    discover_generation_cases,
    discover_result_zones,
    filter_generation_cases,
    filter_result_index,
    list_definitions,
    load_selected_signals,
    parent_campaign_options,
    relative_custom_grouping_path,
    result_campaign_options,
    result_index,
    result_options,
    result_run_options,
    result_variable_catalog,
    result_variable_column_options,
    result_variable_options,
    save_builder_definition,
    selection_facets,
    validate_custom_group_rows,
)
from .page import get_tab_builder


_REGISTERED = False


def _options(values):
    return [{"label": value, "value": value} for value in values]


def _selected_rows(rows, case_ids):
    wanted = set(case_ids or [])
    return [row for row in (rows or []) if row.get("case_id") in wanted]


def _custom_table_rows(rows, case_ids):
    output = []
    for row in _selected_rows(rows, case_ids):
        for zone_name in row.get("thermal_zone_names", []) or []:
            output.append(
                {
                    "case_id": row.get("case_id", ""),
                    "source_zone_name": zone_name,
                    "aggregate_zone_name": "",
                }
            )
    return output


def _merge_custom_table_rows(rows, case_ids, current_rows):
    """Rebuild expected rows while preserving edits for still-selected cases."""
    existing = {
        (
            str(row.get("case_id") or ""),
            str(row.get("source_zone_name") or ""),
        ): str(row.get("aggregate_zone_name") or "")
        for row in (current_rows or [])
    }
    merged = _custom_table_rows(rows, case_ids)
    for row in merged:
        key = (str(row["case_id"]), str(row["source_zone_name"]))
        row["aggregate_zone_name"] = existing.get(key, "")
    return merged


def _custom_grouping_metrics(rows, case_ids, table_rows):
    """Return authoritative custom-table completeness for the selected cases."""
    expected = _custom_table_rows(rows, case_ids)
    expected_keys = {
        (str(row["case_id"]), str(row["source_zone_name"])) for row in expected
    }
    table_by_key = {
        (
            str(row.get("case_id") or ""),
            str(row.get("source_zone_name") or ""),
        ): str(row.get("aggregate_zone_name") or "").strip()
        for row in (table_rows or [])
    }
    assigned = sum(1 for key in expected_keys if table_by_key.get(key, ""))
    distinct_groups = len(
        {
            (case_id, table_by_key[key])
            for case_id, source_zone in expected_keys
            for key in [(case_id, source_zone)]
            if table_by_key.get(key, "")
        }
    )
    total = len(expected_keys)
    return {
        "selected_cases": len(set(case_ids or [])),
        "source_zones": total,
        "assigned": assigned,
        "unassigned": total - assigned,
        "distinct_case_local_groups": distinct_groups,
        "valid": bool(case_ids) and total > 0 and assigned == total,
    }



def _aggregation_legend_style(color: str, visible: bool):
    return {
        "width": "100%",
        "display": "flex",
        "alignItems": "flex-start",
        "gap": "0.55rem",
        "padding": "0.55rem 0.6rem",
        "marginBottom": "0.4rem",
        "border": "1px solid rgba(120,120,120,0.25)",
        "borderRadius": "0.4rem",
        "background": "rgba(255,255,255,0.92)" if visible else "rgba(230,230,230,0.55)",
        "opacity": 1.0 if visible else 0.48,
        "cursor": "pointer",
        "textAlign": "left",
        "whiteSpace": "normal",
    }


def _aggregation_legend(items):
    if not items:
        return html.Div("Plot signals to populate the legend.", className="text-muted small")
    visible_count = sum(1 for item in items if item.get("visible", True))
    children = [
        html.Div(
            f"{visible_count} of {len(items)} traces visible",
            className="small text-muted mb-2",
        )
    ]
    for item in items:
        visible = bool(item.get("visible", True))
        color = str(item.get("color") or "#666")
        children.append(
            html.Button(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "1rem",
                            "minWidth": "1rem",
                            "height": "0.28rem",
                            "marginTop": "0.45rem",
                            "borderRadius": "0.2rem",
                            "backgroundColor": color,
                        }
                    ),
                    html.Span(
                        [
                            html.Div(item.get("primary_label", ""), style={"fontWeight": 600}),
                            html.Div(item.get("variable_name", ""), className="small"),
                            html.Div(
                                f"Column: {item.get('variable_column','')}",
                                className="small text-muted",
                            ),
                        ],
                        style={"minWidth": 0, "overflowWrap": "anywhere"},
                    ),
                ],
                id={"type": "aggregation-results-legend-toggle", "index": int(item["index"])},
                n_clicks=0,
                title=item.get("full_name", ""),
                style=_aggregation_legend_style(color, visible),
            )
        )
    return children

def register_aggregation_callbacks():
    """Register B6 callbacks exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    @callback(
        Output("aggregation-workspace-content", "children"),
        Input("aggregation-workspace-tabs", "value"),
        prevent_initial_call=True,
    )
    def tab(value):
        return get_tab_builder(value)()

    @callback(
        Output("aggregation-builder-parent-campaign", "options"),
        Input("aggregation-builder-refresh-campaigns", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_parent_campaigns(_):
        return parent_campaign_options()

    @callback(
        Output("aggregation-builder-generation-cache", "data"),
        Output("aggregation-builder-generation-issues", "data"),
        Output("aggregation-builder-parent-status", "children"),
        Input("aggregation-builder-parent-campaign", "value"),
    )
    def load_parent_campaign(campaign_id):
        if not campaign_id:
            return [], [], ""

        result = discover_generation_cases(campaign_id, include_zone_inventory=True)
        rows = result["cases"]
        issues = result["issues"]
        status = dbc.Alert(
            f"Discovered {len(rows)} eligible latest Generation case(s); "
            f"{len(issues)} discovery issue(s). "
            "completed and completed_with_warnings are eligible.",
            color="info" if rows else "warning",
        )
        return rows, issues, status

    @callback(
        Output("aggregation-builder-building-filter", "options"),
        Output("aggregation-builder-building-filter", "value"),
        Input("aggregation-builder-generation-cache", "data"),
    )
    def load_building_filter(rows):
        facets = selection_facets(rows or [])
        values = facets["building_types"]
        return _options(values), values

    @callback(
        Output("aggregation-builder-weather-filter", "options"),
        Output("aggregation-builder-weather-filter", "value"),
        Input("aggregation-builder-generation-cache", "data"),
        Input("aggregation-builder-building-filter", "value"),
    )
    def load_weather_filter(rows, building_types):
        if not rows or not building_types:
            return [], []
        filtered = filter_generation_cases(
            rows,
            building_types=building_types,
        )
        values = selection_facets(filtered)["weather_locations"]
        return _options(values), values

    @callback(
        Output("aggregation-builder-cases", "options"),
        Output("aggregation-builder-cases", "value"),
        Input("aggregation-builder-generation-cache", "data"),
        Input("aggregation-builder-building-filter", "value"),
        Input("aggregation-builder-weather-filter", "value"),
    )
    def load_case_selection(rows, building_types, weather_locations):
        if not rows or not building_types or not weather_locations:
            return [], []
        filtered = filter_generation_cases(
            rows,
            building_types=building_types,
            weather_locations=weather_locations,
        )
        options = [
            {
                "label": (
                    f"{row.get('building_type') or 'Unknown building'} | "
                    f"{row.get('weather_location') or 'Unknown weather'} | "
                    f"{row['case_id']} | {row['status']}"
                ),
                "value": row["case_id"],
            }
            for row in filtered
        ]
        # Filter changes deliberately select every matching case. Manual case
        # de-selection remains possible afterward and is authoritative.
        return options, [row["case_id"] for row in filtered]

    @callback(
        Output("aggregation-builder-case-summary", "children"),
        Input("aggregation-builder-cases", "options"),
        Input("aggregation-builder-cases", "value"),
    )
    def case_summary(case_options, selected_case_ids):
        allowed = {
            str(option.get("value"))
            for option in (case_options or [])
            if isinstance(option, dict) and option.get("value") is not None
        }
        selected = {
            str(case_id)
            for case_id in (selected_case_ids or [])
            if str(case_id) in allowed
        }
        return dbc.Alert(
            f"{len(allowed)} case(s) match the current Building/Weather filters; "
            f"{len(selected)} selected.",
            color="secondary",
        )

    @callback(
        Output("aggregation-builder-custom-panel", "style"),
        Input("aggregation-builder-strategies", "value"),
    )
    def custom_panel(strategies):
        return (
            {"display": "block"}
            if "custom_groups" in (strategies or [])
            else {"display": "none"}
        )

    @callback(
        Output("aggregation-builder-custom-table", "data"),
        Input("aggregation-builder-generation-cache", "data"),
        Input("aggregation-builder-cases", "value"),
        State("aggregation-builder-custom-table", "data"),
    )
    def custom_table(rows, case_ids, current_rows):
        return _merge_custom_table_rows(
            rows or [],
            case_ids or [],
            current_rows or [],
        )

    @callback(
        Output("aggregation-builder-plan-summary", "children"),
        Output("aggregation-builder-custom-status", "children"),
        Output("aggregation-builder-save", "disabled"),
        Input("aggregation-builder-strategies", "value"),
        Input("aggregation-builder-weight-modes", "value"),
        Input("aggregation-builder-generation-cache", "data"),
        Input("aggregation-builder-cases", "value"),
        Input("aggregation-builder-custom-table", "data"),
    )
    def plan_and_custom_summary(strategies, weights, rows, cases, table_rows):
        n_requests = len(strategies or []) * len(weights or [])
        n_cases = len(cases or [])
        n_executions = n_requests * n_cases
        plan = dbc.Alert(
            f"Definition requests {n_requests} plan configuration(s) across "
            f"{n_cases} selected Generation case(s); "
            f"{n_executions} total case-plan execution(s).",
            color="secondary",
        )

        uses_custom = "custom_groups" in (strategies or [])
        if not uses_custom:
            return plan, "", False

        metrics = _custom_grouping_metrics(
            rows or [],
            cases or [],
            table_rows or [],
        )
        status = dbc.Alert(
            [
                html.Div(
                    f"{metrics['selected_cases']} selected Generation case(s) | "
                    f"{metrics['source_zones']} source zone(s) | "
                    f"{metrics['assigned']} assigned | "
                    f"{metrics['unassigned']} unassigned"
                ),
                html.Div(
                    f"{metrics['distinct_case_local_groups']} distinct case-local "
                    f"aggregate zone(s) | "
                    f"Custom grouping {'VALID' if metrics['valid'] else 'INCOMPLETE'}"
                ),
            ],
            color="success" if metrics["valid"] else "warning",
        )
        return plan, status, not metrics["valid"]

    @callback(
        Output("aggregation-builder-definition-summary", "children"),
        Output("aggregation-builder-save-status", "children"),
        Input("aggregation-builder-save", "n_clicks"),
        State("aggregation-builder-parent-campaign", "value"),
        State("aggregation-builder-generation-cache", "data"),
        State("aggregation-builder-cases", "value"),
        State("aggregation-builder-strategies", "value"),
        State("aggregation-builder-weight-modes", "value"),
        State("aggregation-builder-rule-set", "value"),
        State("aggregation-builder-custom-id", "value"),
        State("aggregation-builder-custom-table", "data"),
        State("aggregation-builder-campaign-id", "value"),
        State("aggregation-builder-machine-id", "value"),
        State("aggregation-builder-case-limit", "value"),
        State("aggregation-builder-variable-limit", "value"),
        State("aggregation-builder-preview-rows", "value"),
        State("aggregation-builder-pickles", "value"),
        State("aggregation-builder-continue", "value"),
        State("aggregation-builder-zone-stem", "value"),
        State("aggregation-builder-system-node-pattern", "value"),
        State("aggregation-builder-mlflow", "value"),
        State("aggregation-builder-mlflow-uri", "value"),
        State("aggregation-builder-mlflow-experiment", "value"),
        State("aggregation-builder-mlflow-run-name", "value"),
        State("aggregation-builder-mlflow-strict", "value"),
        prevent_initial_call=True,
    )
    def save(
        _,
        parent_campaign_id,
        generation_rows,
        case_ids,
        strategies,
        weight_modes,
        rule_set,
        custom_id,
        custom_table_rows,
        campaign_id,
        machine_id,
        case_limit,
        variable_limit,
        preview_rows,
        pickles,
        continue_on_error,
        zone_stem,
        system_node_pattern,
        mlflow_enabled,
        mlflow_uri,
        mlflow_experiment,
        mlflow_run_name,
        mlflow_strict,
    ):
        try:
            if not parent_campaign_id:
                raise ValueError("Select a Parent Generation Campaign")
            if not case_ids:
                raise ValueError("Select at least one Generation case")

            requests = build_plan_requests(
                strategies=strategies or [],
                weight_modes=weight_modes or [],
                rule_set=rule_set,
                custom_aggregation_id=custom_id,
            )

            uses_custom = "custom_groups" in (strategies or [])
            validated_custom_rows = None
            custom_relative_path = None
            if uses_custom:
                validated_custom_rows = validate_custom_group_rows(
                    rows=custom_table_rows or [],
                    selected_case_rows=_selected_rows(generation_rows or [], case_ids),
                    aggregation_id=custom_id,
                )
                custom_relative_path = relative_custom_grouping_path(campaign_id)

            definition = build_definition(
                aggregation_campaign_id=campaign_id,
                parent_generation_campaign_id=parent_campaign_id,
                machine_id=machine_id,
                case_ids=case_ids,
                plan_requests=requests,
                custom_zone_groups_path=custom_relative_path,
                case_limit=case_limit,
                max_variables=variable_limit,
                preview_rows=preview_rows,
                write_legacy_pickle=pickles,
                continue_on_error=continue_on_error,
                aggregate_zone_name_stem=zone_stem,
                system_node_name_pattern=system_node_pattern,
                mlflow_enabled=mlflow_enabled,
                mlflow_tracking_uri=mlflow_uri,
                mlflow_experiment_name=mlflow_experiment,
                mlflow_run_name=mlflow_run_name,
                mlflow_strict=mlflow_strict,
            )
            definition_path, custom_path = save_builder_definition(
                definition=definition,
                custom_rows=validated_custom_rows,
            )
            payload = json.dumps(definition.model_dump(mode="json"), indent=2)
            status_children = [
                html.Div(
                    [
                        html.Strong("Saved Aggregation definition:"),
                        html.Div(
                            str(definition_path),
                            className="aggregation-path-text",
                        ),
                    ]
                )
            ]
            if custom_path is not None:
                status_children.append(
                    html.Div(
                        [
                            html.Strong("Custom grouping:"),
                            html.Div(
                                str(custom_path),
                                className="aggregation-path-text",
                            ),
                        ],
                        className="mt-2",
                    )
                )
            return (
                html.Pre(payload, className="aggregation-definition-preview"),
                dbc.Alert(
                    status_children,
                    color="success",
                    className="aggregation-wrap-alert",
                ),
            )
        except Exception as exc:
            return (
                "",
                dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
            )

    @callback(
        Output("aggregation-execution-campaign", "options"),
        Input("aggregation-execution-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_execution_definitions(_):
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

    @callback(
        Output("aggregation-execution-definition-summary", "children"),
        Output("aggregation-execution-command", "children"),
        Output("aggregation-execution-start", "disabled", allow_duplicate=True),
        Input("aggregation-execution-campaign", "value"),
        prevent_initial_call=True,
    )
    def show_execution_definition(campaign_id):
        if not campaign_id:
            return "", "", True
        try:
            summary = definition_summary(campaign_id)
            lineage = dbc.Alert(
                [
                    html.Div(
                        [
                            html.Strong("Aggregation Campaign: "),
                            str(summary["aggregation_campaign_id"]),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Parent Generation Campaign: "),
                            str(summary["parent_generation_campaign_id"]),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Cases / Plan Requests: "),
                            f"{summary['selected_case_count']} / {summary['plan_request_count']}",
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Strategies: "),
                            ", ".join(summary["strategies"]),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Weight Modes: "),
                            ", ".join(summary["weight_modes"]),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("Machine ID: "),
                            str(summary["machine_id"]),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong("MLflow: "),
                            "enabled" if summary["mlflow_enabled"] else "disabled",
                        ]
                    ),
                ],
                color="info",
                className="aggregation-wrap-alert",
            )
            active = MANAGER.snapshot()["status"] in MANAGER.ACTIVE_STATUSES
            return lineage, str(summary["command"]), active
        except Exception as exc:
            return (
                dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
                "",
                True,
            )

    @callback(
        Output("aggregation-execution-status", "children", allow_duplicate=True),
        Input("aggregation-execution-start", "n_clicks"),
        State("aggregation-execution-campaign", "value"),
        prevent_initial_call=True,
    )
    def start_aggregation_execution(_, campaign_id):
        if not campaign_id:
            return dbc.Alert("Select a saved Aggregation campaign.", color="warning")
        try:
            MANAGER.start(campaign_id)
            return dbc.Alert(
                f"Started Aggregation campaign {campaign_id}.",
                color="success",
                className="aggregation-wrap-alert",
            )
        except Exception as exc:
            return dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")

    @callback(
        Output("aggregation-execution-status", "children", allow_duplicate=True),
        Input("aggregation-execution-stop", "n_clicks"),
        prevent_initial_call=True,
    )
    def stop_aggregation_execution(_):
        MANAGER.stop()
        return dbc.Alert(
            "Stop requested for the complete Aggregation process tree.",
            color="warning",
        )

    @callback(
        Output("aggregation-execution-console", "children"),
        Output("aggregation-execution-status", "children"),
        Output("aggregation-execution-start", "disabled"),
        Output("aggregation-execution-stop", "disabled"),
        Input("aggregation-execution-poll", "n_intervals"),
    )
    def poll_aggregation_execution(_):
        snapshot = MANAGER.snapshot()
        active = snapshot["status"] in MANAGER.ACTIVE_STATUSES
        details = (
            f"Status: {snapshot['status']} | "
            f"Campaign: {snapshot['campaign_id']} | "
            f"PID: {snapshot['pid']} | "
            f"Started: {snapshot['started_at']} | "
            f"Return code: {snapshot['return_code']}"
        )
        return (
            snapshot["console"],
            dbc.Alert(
                details,
                color="info" if active else "secondary",
                className="aggregation-wrap-alert",
            ),
            active,
            not active,
        )

    @callback(
        Output("aggregation-results-campaign", "options"),
        Input("aggregation-results-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_results_campaigns(_):
        return result_campaign_options()

    @callback(
        Output("aggregation-results-index", "data"),
        Output("aggregation-results-metadata", "children"),
        Input("aggregation-results-campaign", "value"),
    )
    def load_results_index(campaign_ids):
        if not campaign_ids:
            return [], ""
        rows = result_index(campaign_ids)
        return rows, dbc.Alert(
            f"Discovered {len(rows)} completed Aggregation case/plan run(s) "
            f"across {len(campaign_ids)} selected campaign(s).",
            color="info" if rows else "warning",
        )

    @callback(
        Output("aggregation-results-building", "options"),
        Output("aggregation-results-building", "value"),
        Input("aggregation-results-index", "data"),
    )
    def result_building_options(rows):
        opts = result_options(rows or [], "building_type")
        return opts, [item["value"] for item in opts]

    @callback(
        Output("aggregation-results-weather", "options"),
        Output("aggregation-results-weather", "value"),
        Input("aggregation-results-index", "data"),
        Input("aggregation-results-building", "value"),
    )
    def result_weather_options(rows, buildings):
        selected = filter_result_index(rows or [], building_types=buildings)
        opts = result_options(selected, "weather_location")
        return opts, [item["value"] for item in opts]

    @callback(
        Output("aggregation-results-strategy", "options"),
        Output("aggregation-results-strategy", "value"),
        Input("aggregation-results-index", "data"),
        Input("aggregation-results-building", "value"),
        Input("aggregation-results-weather", "value"),
    )
    def result_strategy_options(rows, buildings, weather):
        selected = filter_result_index(
            rows or [],
            building_types=buildings,
            weather_locations=weather,
        )
        opts = result_options(selected, "strategy")
        return opts, [item["value"] for item in opts]

    @callback(
        Output("aggregation-results-weight", "options"),
        Output("aggregation-results-weight", "value"),
        Input("aggregation-results-index", "data"),
        Input("aggregation-results-building", "value"),
        Input("aggregation-results-weather", "value"),
        Input("aggregation-results-strategy", "value"),
    )
    def result_weight_options(rows, buildings, weather, strategies):
        selected = filter_result_index(
            rows or [],
            building_types=buildings,
            weather_locations=weather,
            strategies=strategies,
        )
        opts = result_options(selected, "weight_mode")
        return opts, [item["value"] for item in opts]

    @callback(
        Output("aggregation-results-ruleset", "options"),
        Output("aggregation-results-ruleset", "value"),
        Input("aggregation-results-index", "data"),
        Input("aggregation-results-building", "value"),
        Input("aggregation-results-weather", "value"),
        Input("aggregation-results-strategy", "value"),
        Input("aggregation-results-weight", "value"),
    )
    def result_ruleset_options(
        rows, buildings, weather, strategies, weights
    ):
        selected = filter_result_index(
            rows or [],
            building_types=buildings,
            weather_locations=weather,
            strategies=strategies,
            weight_modes=weights,
        )
        opts = result_options(selected, "rule_set")
        return opts, [item["value"] for item in opts]

    @callback(
        Output("aggregation-results-run", "options"),
        Output("aggregation-results-run", "value"),
        Input("aggregation-results-index", "data"),
        Input("aggregation-results-building", "value"),
        Input("aggregation-results-weather", "value"),
        Input("aggregation-results-strategy", "value"),
        Input("aggregation-results-weight", "value"),
        Input("aggregation-results-ruleset", "value"),
    )
    def result_run_options_callback(
        rows, buildings, weather, strategies, weights, rule_sets
    ):
        selected = filter_result_index(
            rows or [],
            building_types=buildings,
            weather_locations=weather,
            strategies=strategies,
            weight_modes=weights,
            rule_sets=rule_sets,
        )
        opts = result_run_options(selected)
        return opts, [item["value"] for item in opts]

    @callback(
        Output("aggregation-results-zone", "options"),
        Output("aggregation-results-zone", "value"),
        Input("aggregation-results-index", "data"),
        Input("aggregation-results-building", "value"),
        Input("aggregation-results-weather", "value"),
        Input("aggregation-results-strategy", "value"),
        Input("aggregation-results-weight", "value"),
        Input("aggregation-results-ruleset", "value"),
        Input("aggregation-results-run", "value"),
    )
    def result_zone_options(
        rows, buildings, weather, strategies, weights, rule_sets, runs
    ):
        selected = filter_result_index(
            rows or [],
            building_types=buildings,
            weather_locations=weather,
            strategies=strategies,
            weight_modes=weights,
            rule_sets=rule_sets,
            run_tokens=runs,
        )
        zones = discover_result_zones(selected)
        opts = [{"label": zone, "value": zone} for zone in zones]
        return opts, zones

    @callback(
        Output("aggregation-results-variable-catalog", "data"),
        Output("aggregation-results-variable", "options"),
        Output("aggregation-results-variable", "value"),
        Input("aggregation-results-index", "data"),
        Input("aggregation-results-building", "value"),
        Input("aggregation-results-weather", "value"),
        Input("aggregation-results-strategy", "value"),
        Input("aggregation-results-weight", "value"),
        Input("aggregation-results-ruleset", "value"),
        Input("aggregation-results-run", "value"),
        Input("aggregation-results-zone", "value"),
    )
    def result_variables(
        rows,
        buildings,
        weather,
        strategies,
        weights,
        rule_sets,
        runs,
        zones,
    ):
        selected = filter_result_index(
            rows or [],
            building_types=buildings,
            weather_locations=weather,
            strategies=strategies,
            weight_modes=weights,
            rule_sets=rule_sets,
            run_tokens=runs,
        )
        catalog = result_variable_catalog(selected, zones=zones)
        opts = result_variable_options(catalog)
        # Variables remain explicit selections to avoid immediately drawing every
        # stored signal in large campaigns.
        return catalog, opts, []

    @callback(
        Output("aggregation-results-variable-column", "options"),
        Output("aggregation-results-variable-column", "value"),
        Input("aggregation-results-variable-catalog", "data"),
        Input("aggregation-results-variable", "value"),
    )
    def result_variable_columns(catalog, variables):
        if not variables:
            return [], []
        opts = result_variable_column_options(catalog or [], variables=variables)
        # Every output column belonging to the explicitly selected variables is
        # selected by default, including multi-column Schedule Value outputs.
        return opts, [item["value"] for item in opts]

    @callback(
        Output("aggregation-results-start", "disabled"),
        Output("aggregation-results-end", "disabled"),
        Input("aggregation-results-range-mode", "value"),
    )
    def result_range_mode(value):
        disabled = value != "custom"
        return disabled, disabled

    @callback(
        Output("aggregation-results-variable", "value", allow_duplicate=True),
        Output("aggregation-results-variable-column", "value", allow_duplicate=True),
        Input("aggregation-results-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_result_signals(_):
        return [], []

    def _selected_result_rows(
        rows,
        campaign_ids,
        buildings,
        weather,
        strategies,
        weights,
        rule_sets,
        runs,
    ):
        return filter_result_index(
            rows or [],
            aggregation_campaign_ids=campaign_ids,
            building_types=buildings,
            weather_locations=weather,
            strategies=strategies,
            weight_modes=weights,
            rule_sets=rule_sets,
            run_tokens=runs,
        )

    @callback(
        Output("aggregation-results-graph", "figure"),
        Output("aggregation-results-custom-legend", "children"),
        Output("aggregation-results-legend-state", "data"),
        Output("aggregation-results-message", "children"),
        Input("aggregation-results-plot-button", "n_clicks"),
        State("aggregation-results-campaign", "value"),
        State("aggregation-results-index", "data"),
        State("aggregation-results-building", "value"),
        State("aggregation-results-weather", "value"),
        State("aggregation-results-strategy", "value"),
        State("aggregation-results-weight", "value"),
        State("aggregation-results-ruleset", "value"),
        State("aggregation-results-run", "value"),
        State("aggregation-results-zone", "value"),
        State("aggregation-results-variable", "value"),
        State("aggregation-results-variable-column", "value"),
        State("aggregation-results-range-mode", "value"),
        State("aggregation-results-start", "value"),
        State("aggregation-results-end", "value"),
        prevent_initial_call=True,
    )
    def plot_aggregation_results(
        _,
        campaign_ids,
        rows,
        buildings,
        weather,
        strategies,
        weights,
        rule_sets,
        runs,
        zones,
        variables,
        variable_columns,
        range_mode,
        start,
        end,
    ):
        selected = _selected_result_rows(
            rows,
            campaign_ids,
            buildings,
            weather,
            strategies,
            weights,
            rule_sets,
            runs,
        )
        if not selected:
            return no_update, no_update, no_update, dbc.Alert(
                "The current multi-filter intersection contains no completed "
                "Aggregation runs.",
                color="warning",
            )
        try:
            frame = load_selected_signals(
                selected,
                zones=zones or [],
                variables=variables or [],
                variable_columns=variable_columns or [],
                start=start if range_mode == "custom" else None,
                end=end if range_mode == "custom" else None,
            )
            if frame.empty:
                return no_update, no_update, no_update, dbc.Alert(
                    "No rows matched the selected zones, variables, variable "
                    "columns, and datetime range.",
                    color="warning",
                )
            fig = px.line(frame, x="timestamp", y="value", color="series")
            fig.update_layout(
                showlegend=False,
                margin=dict(l=70, r=25, t=45, b=65),
                hovermode="x unified",
                autosize=True,
            )
            legend_items = []
            first_by_series = (
                frame.sort_values("timestamp")
                .groupby("series", sort=False)
                .first()
                .to_dict("index")
            )
            for index, trace in enumerate(fig.data):
                full_name = str(trace.name)
                meta = first_by_series.get(full_name, {})
                units = str(meta.get("units") or "")
                trace.hovertemplate = (
                    f"{full_name}<br>Datetime=%{{x}}<br>"
                    f"Value=%{{y}} {units}<extra></extra>"
                )
                color = getattr(getattr(trace, "line", None), "color", None) or "#666"
                legend_items.append(
                    {
                        "index": index,
                        "visible": True,
                        "color": color,
                        "full_name": full_name,
                        "primary_label": (
                            f"{meta.get('aggregate_zone_id','')} | "
                            f"{units or 'unitless'}"
                        ),
                        "variable_name": meta.get("source_variable_name", ""),
                        "variable_column": meta.get("output_variable_name", ""),
                    }
                )
            return (
                fig,
                _aggregation_legend(legend_items),
                legend_items,
                dbc.Alert(
                    f"Displayed {len(fig.data)} trace(s), {len(frame)} rows from "
                    "the exact current filter intersection.",
                    color="success",
                ),
            )
        except Exception as exc:
            return no_update, no_update, no_update, dbc.Alert(
                f"{type(exc).__name__}: {exc}", color="danger"
            )

    @callback(
        Output("aggregation-results-graph", "figure", allow_duplicate=True),
        Output("aggregation-results-custom-legend", "children", allow_duplicate=True),
        Output("aggregation-results-legend-state", "data", allow_duplicate=True),
        Input({"type": "aggregation-results-legend-toggle", "index": ALL}, "n_clicks"),
        State("aggregation-results-graph", "figure"),
        State("aggregation-results-legend-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_aggregation_trace(_clicks, figure, items):
        triggered = ctx.triggered_id
        if (
            not isinstance(triggered, dict)
            or triggered.get("type") != "aggregation-results-legend-toggle"
        ):
            return no_update, no_update, no_update
        if not figure or not items:
            return no_update, no_update, no_update
        index = int(triggered["index"])
        if index < 0 or index >= len(items) or index >= len(figure.get("data", [])):
            return no_update, no_update, no_update
        items = [dict(item) for item in items]
        items[index]["visible"] = not bool(items[index].get("visible", True))
        figure = dict(figure)
        figure["data"] = [dict(trace) for trace in figure.get("data", [])]
        figure["data"][index]["visible"] = True if items[index]["visible"] else False
        return figure, _aggregation_legend(items), items

    @callback(
        Output("aggregation-results-download", "data"),
        Output("aggregation-results-download-message", "children"),
        Input("aggregation-results-download-button", "n_clicks"),
        State("aggregation-results-campaign", "value"),
        State("aggregation-results-index", "data"),
        State("aggregation-results-building", "value"),
        State("aggregation-results-weather", "value"),
        State("aggregation-results-strategy", "value"),
        State("aggregation-results-weight", "value"),
        State("aggregation-results-ruleset", "value"),
        State("aggregation-results-run", "value"),
        State("aggregation-results-zone", "value"),
        State("aggregation-results-variable", "value"),
        State("aggregation-results-variable-column", "value"),
        State("aggregation-results-range-mode", "value"),
        State("aggregation-results-start", "value"),
        State("aggregation-results-end", "value"),
        State("aggregation-results-export-format", "value"),
        prevent_initial_call=True,
    )
    def download_aggregation_results(
        _,
        campaign_ids,
        rows,
        buildings,
        weather,
        strategies,
        weights,
        rule_sets,
        runs,
        zones,
        variables,
        variable_columns,
        range_mode,
        start,
        end,
        export_format,
    ):
        if not campaign_ids:
            return no_update, dbc.Alert(
                "Select at least one Aggregation campaign before downloading data.",
                color="warning",
            )
        selected = _selected_result_rows(
            rows,
            campaign_ids,
            buildings,
            weather,
            strategies,
            weights,
            rule_sets,
            runs,
        )
        try:
            payload, filename = build_selected_aggregation_data_export(
                aggregation_campaign_ids=campaign_ids,
                rows=selected,
                zones=zones or [],
                variables=variables or [],
                variable_columns=variable_columns or [],
                export_format=export_format or "csv",
                range_mode=range_mode or "full",
                start=start,
                end=end,
            )
            return dcc.send_bytes(payload, filename), dbc.Alert(
                f"Prepared exactly the selected/displayed filter intersection as "
                f"{filename}.",
                color="success",
            )
        except Exception as exc:
            return no_update, dbc.Alert(
                f"{type(exc).__name__}: {exc}", color="danger"
            )

