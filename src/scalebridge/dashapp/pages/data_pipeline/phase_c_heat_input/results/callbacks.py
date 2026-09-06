"""Callbacks dedicated to the simplified Phase C Tab 3 read-only Results."""
from __future__ import annotations

from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from .....components.results import toggle_trace_visibility
from .....services.heat_input.results_data import (
    ResultSelectionTooBroad,
    annual_component_catalog,
    artifact_inventory,
    building_phvac_metrics,
    build_annual_export,
    build_campaign_summary_export,
    build_evaluation_export,
    build_model_bundle_export,
    build_plot_figure_export,
    dataset_inventory,
    faceted_filter_options,
    generalization_metrics,
    inference_zone_options,
    lineage_summary,
    load_annual_series,
    load_building_phvac_series,
    load_dataset_series,
    load_evaluation_metrics,
    load_evaluation_series,
    load_model_metadata,
    load_run_ref_from_key,
    run_options,
    run_summary,
    split_summary_rows,
    stage_summary,
    structural_availability_rows,
    target_model_inventory,
    validation_diagnostics,
    validation_overview,
)


_REGISTERED = False


def _options(values):
    return [{"label": str(value), "value": str(value)} for value in values or []]


def _filters(building, weather, case, aggregation, weight, zone, model, estimator):
    return {
        "building_types": building,
        "weather_locations": weather,
        "case_ids": case,
        "aggregation_ids": aggregation,
        "weight_modes": weight,
        "aggregate_zone_ids": zone,
        "model_ids": model,
        "estimator_types": estimator,
    }


def _display_value(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _attach_plot_export_snapshot(
    figure,
    *,
    plot_key: str,
    run_key: str | None,
    selection: dict | None,
):
    """Persist the exact selector/filter snapshot that produced a Plotly figure."""
    snapshot = {
        "plot_key": str(plot_key),
        "run_key": str(run_key or ""),
        "selection": dict(selection or {}),
    }
    figure.update_layout(meta={"phase_c_plot_export": snapshot})
    return figure


_TRACE_COLORS = tuple(qualitative.Plotly)


def _trace_color(trace, index: int) -> str:
    for holder_name in ("line", "marker"):
        holder = getattr(trace, holder_name, None)
        color = getattr(holder, "color", None) if holder is not None else None
        if isinstance(color, str) and color.strip():
            return color
    return _TRACE_COLORS[index % len(_TRACE_COLORS)]


def _apply_trace_colors(figure):
    for index, trace in enumerate(figure.data):
        color = _trace_color(trace, index)
        trace_type = str(getattr(trace, "type", "") or "")
        mode = str(getattr(trace, "mode", "") or "")
        if trace_type in {"bar", "histogram"}:
            trace.marker.color = color
        else:
            if "lines" in mode or trace_type in {"scatter", "scattergl"}:
                trace.line.color = color
            if "markers" in mode:
                trace.marker.color = color
    return figure


def _build_colored_scroll_legend(items, *, toggle_type: str):
    children = []
    visible_count = sum(bool(item.get("visible", True)) for item in items or [])
    children.append(
        html.Div(
            f"{visible_count} of {len(items or [])} trace(s) visible",
            className="heat-input-legend-count",
        )
    )
    for item in items or []:
        index = int(item.get("index", 0))
        visible = bool(item.get("visible", True))
        color = str(item.get("color") or _TRACE_COLORS[index % len(_TRACE_COLORS)])
        children.append(
            html.Button(
                [
                    html.Span(
                        className="heat-input-legend-swatch",
                        style={"backgroundColor": color, "borderColor": color},
                    ),
                    html.Span(
                        [
                            html.Span(
                                str(item.get("primary_label") or item.get("name") or "trace"),
                                className="heat-input-legend-primary",
                            ),
                            html.Span(
                                str(item.get("secondary_label") or ""),
                                className="heat-input-legend-secondary",
                            ),
                        ],
                        className="heat-input-legend-copy",
                    ),
                ],
                id={"type": toggle_type, "index": index},
                n_clicks=0,
                className=(
                    "heat-input-legend-item"
                    + ("" if visible else " heat-input-legend-item-hidden")
                ),
                title="Click to show or hide this trace.",
                type="button",
            )
        )
    return children


def _pretty_column(column: str) -> str:
    aliases = {
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "case_id": "Case ID",
        "model_id": "Model ID",
        "aggregation_id": "Aggregation",
        "aggregate_zone_id": "Aggregate Zone",
        "weather_location": "Weather",
        "weight_mode": "Weight Mode",
        "estimator_type": "Estimator",
        "evaluation_mode": "Evaluation Mode",
        "row_count": "Rows",
    }
    return aliases.get(str(column), str(column).replace("_", " ").title())


def _table(rows, *, columns=None, limit=100, empty="No rows available."):
    rows = list(rows or [])
    if not rows:
        return html.Div(empty, className="text-muted small heat-input-results-empty")
    if columns is None:
        columns = list(rows[0].keys())
    columns = [column for column in columns if any(column in row for row in rows)]
    body = rows[:limit]
    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(
                            _pretty_column(column),
                            title=str(column),
                            className="heat-input-results-header",
                        )
                        for column in columns
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(
                                _display_value(row.get(column)),
                                className=(
                                    "heat-input-results-cell "
                                    + (
                                        "heat-input-results-cell-number"
                                        if isinstance(row.get(column), (int, float))
                                        and not isinstance(row.get(column), bool)
                                        else ""
                                    )
                                ),
                                title=str(row.get(column) if row.get(column) is not None else ""),
                            )
                            for column in columns
                        ]
                    )
                    for row in body
                ]
            ),
        ],
        striped=True,
        hover=True,
        responsive=False,
        size="sm",
        className="heat-input-results-table",
    )
    footer = None
    if len(rows) > limit:
        footer = html.Div(
            f"Showing first {limit} of {len(rows)} rows.",
            className="small text-muted heat-input-table-note",
        )
    return html.Div(
        [table, footer] if footer is not None else [table],
        className="heat-input-results-table-wrap",
    )


def _figure_from_series(series, *, y_title):
    figure = go.Figure()
    items = []
    for index, item in enumerate(series or []):
        role = str(item.get("role") or item.get("evaluation_mode") or "prediction")
        dash = "dot" if role == "target" else "solid"
        figure.add_trace(
            go.Scattergl(
                x=item.get("timestamp") or [],
                y=item.get("value") or [],
                mode="lines",
                name=str(item.get("name") or f"trace-{index}"),
                line={"dash": dash},
                hovertemplate="%{x}<br>%{y}<extra></extra>",
            )
        )
        items.append(
            {
                "index": index,
                "visible": True,
                "name": str(item.get("name") or f"trace-{index}"),
                "primary_label": str(
                    item.get("identity")
                    or item.get("model_id")
                    or item.get("prediction_column")
                    or "trace"
                ),
                "secondary_label": str(
                    item.get("signal")
                    or item.get("role")
                    or item.get("evaluation_mode")
                    or item.get("prediction_column")
                    or ""
                ),
            }
        )
    _apply_trace_colors(figure)
    for item in items:
        index = int(item["index"])
        item["color"] = _trace_color(figure.data[index], index)
    figure.update_layout(
        showlegend=False,
        hovermode="x unified",
        margin={"l": 70, "r": 25, "t": 45, "b": 65},
        xaxis_title="Time",
        yaxis_title=y_title,
        autosize=True,
    )
    return figure, items


def _legend_items_from_figure(figure, *, secondary_label: str):
    _apply_trace_colors(figure)
    items = []
    for index, trace in enumerate(figure.data):
        name = str(trace.name or f"trace-{index}")
        items.append(
            {
                "index": index,
                "visible": True,
                "name": name,
                "primary_label": name,
                "secondary_label": secondary_label,
                "color": _trace_color(trace, index),
            }
        )
    figure.update_layout(showlegend=False)
    return items


def _paired_evaluation_series(series):
    groups = {}
    for item in series or []:
        groups.setdefault(str(item.get("identity") or ""), {})[
            str(item.get("role"))
        ] = item
    return [
        (identity, group)
        for identity, group in groups.items()
        if "target" in group and "prediction" in group
    ]


def _evaluation_figure(series, *, kind: str):
    if kind == "time_series":
        return _figure_from_series(series, y_title="Observed Y / predicted Ŷ (W)")
    figure = go.Figure()
    legend_items = []
    for index, (identity, group) in enumerate(_paired_evaluation_series(series)):
        y = np.asarray(group["target"].get("value") or [], dtype=float)
        prediction = np.asarray(group["prediction"].get("value") or [], dtype=float)
        n = min(len(y), len(prediction))
        y, prediction = y[:n], prediction[:n]
        finite = np.isfinite(y) & np.isfinite(prediction)
        y, prediction = y[finite], prediction[finite]
        if kind == "scatter":
            figure.add_trace(
                go.Scattergl(x=y, y=prediction, mode="markers", name=identity)
            )
        elif kind == "residual_time_series":
            timestamp = (group["target"].get("timestamp") or [])[:n]
            timestamp = [value for value, keep in zip(timestamp, finite) if keep]
            figure.add_trace(
                go.Scattergl(
                    x=timestamp,
                    y=prediction - y,
                    mode="lines",
                    name=identity,
                )
            )
        elif kind == "residual_distribution":
            figure.add_trace(
                go.Histogram(x=prediction - y, name=identity, opacity=0.65)
            )
        legend_items.append(
            {
                "index": index,
                "visible": True,
                "name": identity,
                "primary_label": identity,
                "secondary_label": kind.replace("_", " "),
            }
        )
    if kind == "scatter" and figure.data:
        values = []
        for trace in figure.data:
            values.extend(list(trace.x or []))
            values.extend(list(trace.y or []))
        finite_values = [float(value) for value in values if np.isfinite(value)]
        if finite_values:
            lo, hi = min(finite_values), max(finite_values)
            figure.add_shape(
                type="line",
                x0=lo,
                y0=lo,
                x1=hi,
                y1=hi,
                line={"dash": "dash"},
            )
        figure.update_layout(xaxis_title="Observed Y (W)", yaxis_title="Predicted Ŷ (W)")
    elif kind == "residual_time_series":
        figure.update_layout(
            xaxis_title="Time",
            yaxis_title="Residual = Ŷ - Y (W)",
        )
    elif kind == "residual_distribution":
        figure.update_layout(
            xaxis_title="Residual = Ŷ - Y (W)",
            yaxis_title="Count",
            barmode="overlay",
        )
    _apply_trace_colors(figure)
    for item in legend_items:
        index = int(item["index"])
        if index < len(figure.data):
            item["color"] = _trace_color(figure.data[index], index)
    figure.update_layout(
        showlegend=False,
        margin={"l": 70, "r": 25, "t": 45, "b": 65},
    )
    return figure, legend_items


def _comparison_figure(kind, *, metrics, models, splits, availability, phvac_series):
    if kind == "building_phvac":
        return _figure_from_series(
            phvac_series,
            y_title="Building HVAC electric power (W)",
        )[0]

    figure = go.Figure()
    if kind == "estimator_metric":
        for row in metrics:
            figure.add_trace(
                go.Bar(
                    x=[str(row.get("estimator_type"))],
                    y=[row.get("rmse")],
                    name=(
                        f"{row.get('model_id')} | {row.get('split')} | "
                        f"{row.get('evaluation_mode')}"
                    ),
                )
            )
        figure.update_layout(xaxis_title="Estimator", yaxis_title="RMSE (W)")
    elif kind == "coefficient":
        labels = [
            (
                f"{row.get('model_id')} | {row.get('estimator_type')} | "
                f"{row.get('aggregate_zone_id')}"
            )
            for row in models
        ]
        figure.add_trace(
            go.Bar(
                x=labels,
                y=[row.get("coefficient") for row in models],
                name="coefficient",
            )
        )
        figure.add_trace(
            go.Bar(
                x=labels,
                y=[row.get("intercept") for row in models],
                name="intercept",
            )
        )
        figure.update_layout(xaxis_title="Model identity", yaxis_title="Fitted value")
    elif kind == "error_context":
        labels = [
            (
                f"{row.get('building_type')} | {row.get('weather_location')} | "
                f"{row.get('aggregate_zone_id')}"
            )
            for row in metrics
        ]
        figure.add_trace(
            go.Bar(
                x=labels,
                y=[row.get("rmse") for row in metrics],
                name="RMSE",
            )
        )
        figure.update_layout(
            xaxis_title="Building | weather | zone",
            yaxis_title="RMSE (W)",
        )
    elif kind == "split_coverage":
        for split in ("train", "validation", "test", "excluded"):
            rows = [row for row in splits if row.get("split") == split]
            figure.add_trace(
                go.Bar(
                    x=[str(row.get("aggregate_zone_id")) for row in rows],
                    y=[row.get("row_count") for row in rows],
                    name=split,
                )
            )
        figure.update_layout(
            barmode="stack",
            xaxis_title="Aggregate zone",
            yaxis_title="Rows",
        )
    elif kind == "availability":
        labels = [str(row.get("aggregate_zone_id")) for row in availability]
        for field, label in (
            ("applicable_model_count", "applicable"),
            ("structurally_inapplicable_model_count", "structural"),
            ("invalid_model_count", "invalid"),
            ("missing_expected_data_model_count", "missing"),
        ):
            figure.add_trace(
                go.Bar(x=labels, y=[row.get(field) for row in availability], name=label)
            )
        figure.update_layout(
            barmode="stack",
            xaxis_title="Aggregate zone",
            yaxis_title="Component count",
        )
    _apply_trace_colors(figure)
    figure.update_layout(
        showlegend=False,
        margin={"l": 70, "r": 25, "t": 45, "b": 95},
    )
    return figure


def _summary_component(summary):
    availability = summary.get("availability_summary") or {}
    return dbc.Alert(
        [
            html.Div([html.Strong("Campaign: "), summary["campaign_id"]]),
            html.Div([html.Strong("Phase C run: "), summary["phase_c_run_id"]]),
            html.Div([html.Strong("Status: "), summary.get("status") or "—"]),
            html.Div([html.Strong("Matrix: "), summary.get("matrix_run_id") or "—"]),
            html.Div(
                [
                    html.Strong("Commands: "),
                    (
                        f"{summary['passed_command_count']} passed / "
                        f"{summary['failed_command_count']} failed / "
                        f"{summary['command_count']} total"
                    ),
                ]
            ),
            html.Div(
                [
                    html.Strong("Models: "),
                    (
                        f"{availability.get('applicable_model_count', 0)} applicable, "
                        f"{availability.get('structurally_inapplicable_model_count', 0)} "
                        "structurally absent, "
                        f"{availability.get('invalid_model_count', 0)} invalid, "
                        f"{availability.get('missing_expected_data_model_count', 0)} missing"
                    ),
                ]
            ),
            html.Div(
                [
                    html.Strong("Full-year inference: "),
                    (
                        f"{availability.get('inference_zone_count', 0)} zone packages / "
                        f"{availability.get('inferred_component_count', 0)} components"
                    ),
                ]
            ),
        ],
        color="success" if summary.get("status") == "completed" else "warning",
        className="heat-input-wrap-alert",
    )


def _mlflow_component(summary):
    if not summary.get("mlflow_tracking_uri"):
        return dbc.Alert(
            "No C9 MLflow registration metadata is available.",
            color="secondary",
        )
    children = [
        html.Div(
            [
                html.Strong("Experiment / parent run: "),
                f"{summary.get('mlflow_experiment_id')} / "
                f"{summary.get('mlflow_parent_run_id')}",
            ]
        ),
        html.Div(
            [
                html.Strong("Registered stage / task runs: "),
                (
                    f"{summary.get('mlflow_stage_run_count') or 0} / "
                    f"{summary.get('mlflow_task_run_count') or 0}"
                ),
            ]
        ),
    ]
    if summary.get("mlflow_url"):
        children.append(
            html.A(
                "Open MLflow parent run",
                href=summary["mlflow_url"],
                target="_blank",
                rel="noopener noreferrer",
            )
        )
    return dbc.Alert(children, color="info")


def _selection_from_states(
    *,
    filters,
    splits,
    modes,
    resolution,
    range_mode,
    start,
    end,
):
    return {
        "filters": filters,
        "splits": splits or [],
        "evaluation_modes": modes or [],
        "resolution": resolution or "preview",
        "range_mode": range_mode or "full",
        "start": start if range_mode == "custom" else None,
        "end": end if range_mode == "custom" else None,
    }


def register_results_callbacks():
    """Register Tab-3 callbacks exactly once."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    @callback(
        Output("phase-c-results-run", "options"),
        Input("phase-c-results-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_runs(_):
        return run_options()

    @callback(
        Output("phase-c-results-run-summary", "children"),
        Output("phase-c-results-stage-summary", "children"),
        Output("phase-c-results-availability-summary", "children"),
        Output("phase-c-results-validation-overview", "children"),
        Output("phase-c-results-mlflow", "children"),
        Input("phase-c-results-run", "value"),
    )
    def load_run(value):
        if not value:
            return "", "", "", "", ""
        try:
            ref = load_run_ref_from_key(value)
            summary = run_summary(ref)
            stages = stage_summary(ref)
            availability = structural_availability_rows(ref)
            validation = validation_overview(ref)
            return (
                _summary_component(summary),
                _table(
                    stages,
                    columns=[
                        "stage",
                        "status",
                        "runtime_seconds",
                        "validation_status",
                        "validation_count",
                        "validation_failure_count",
                    ],
                    limit=20,
                ),
                _table(
                    availability,
                    columns=[
                        "building_type",
                        "weather_location",
                        "aggregation_id",
                        "aggregate_zone_id",
                        "candidate_model_count",
                        "applicable_model_count",
                        "structurally_inapplicable_model_count",
                        "invalid_model_count",
                        "missing_expected_data_model_count",
                    ],
                    limit=100,
                ),
                _table(validation, limit=20),
                _mlflow_component(summary),
            )
        except Exception as exc:
            error = dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")
            return error, "", "", "", ""

    @callback(
        Output("phase-c-results-building", "options"),
        Output("phase-c-results-weather", "options"),
        Output("phase-c-results-case", "options"),
        Output("phase-c-results-aggregation", "options"),
        Output("phase-c-results-weight", "options"),
        Output("phase-c-results-zone", "options"),
        Output("phase-c-results-model", "options"),
        Output("phase-c-results-estimator", "options"),
        Output("phase-c-results-annual-zone", "options"),
        Input("phase-c-results-run", "value"),
        Input("phase-c-results-building", "value"),
        Input("phase-c-results-weather", "value"),
        Input("phase-c-results-case", "value"),
        Input("phase-c-results-aggregation", "value"),
        Input("phase-c-results-weight", "value"),
        Input("phase-c-results-zone", "value"),
        Input("phase-c-results-model", "value"),
        Input("phase-c-results-estimator", "value"),
    )
    def cascade_filters(
        value,
        building,
        weather,
        case,
        aggregation,
        weight,
        zone,
        model,
        estimator,
    ):
        if not value:
            return [], [], [], [], [], [], [], [], []
        try:
            ref = load_run_ref_from_key(value)
            filters = _filters(
                building,
                weather,
                case,
                aggregation,
                weight,
                zone,
                model,
                estimator,
            )
            values = faceted_filter_options(ref, **filters)
            annual = inference_zone_options(ref, **filters)
            return (
                _options(values.get("building_type")),
                _options(values.get("weather_location")),
                _options(values.get("case_id")),
                _options(values.get("aggregation_id")),
                _options(values.get("weight_mode")),
                _options(values.get("aggregate_zone_id")),
                _options(values.get("model_id")),
                _options(values.get("estimator_type")),
                annual,
            )
        except Exception:
            return [], [], [], [], [], [], [], [], []

    @callback(
        Output("phase-c-results-dataset-start", "disabled"),
        Output("phase-c-results-dataset-end", "disabled"),
        Input("phase-c-results-dataset-range-mode", "value"),
    )
    def dataset_range_mode(value):
        disabled = value != "custom"
        return disabled, disabled

    @callback(
        Output("phase-c-results-dataset-graph", "figure"),
        Output("phase-c-results-dataset-legend", "children"),
        Output("phase-c-results-dataset-legend-state", "data"),
        Output("phase-c-results-dataset-message", "children"),
        Input("phase-c-results-load-dataset", "n_clicks"),
        State("phase-c-results-run", "value"),
        State("phase-c-results-building", "value"),
        State("phase-c-results-weather", "value"),
        State("phase-c-results-case", "value"),
        State("phase-c-results-aggregation", "value"),
        State("phase-c-results-weight", "value"),
        State("phase-c-results-zone", "value"),
        State("phase-c-results-model", "value"),
        State("phase-c-results-estimator", "value"),
        State("phase-c-results-dataset-resolution", "value"),
        State("phase-c-results-dataset-range-mode", "value"),
        State("phase-c-results-dataset-start", "value"),
        State("phase-c-results-dataset-end", "value"),
        prevent_initial_call=True,
    )
    def load_dataset(
        _,
        value,
        building,
        weather,
        case,
        aggregation,
        weight,
        zone,
        model,
        estimator,
        resolution,
        range_mode,
        start,
        end,
    ):
        if not value:
            return no_update, no_update, no_update, dbc.Alert(
                "Select a Phase C run.",
                color="warning",
            )
        if not model:
            return no_update, no_update, no_update, dbc.Alert(
                "Select exactly one model/context before loading dataset X/Y.",
                color="warning",
            )
        filters = _filters(
            building,
            weather,
            case,
            aggregation,
            weight,
            zone,
            model,
            estimator,
        )
        try:
            ref = load_run_ref_from_key(value)
            series = load_dataset_series(
                ref,
                resolution=resolution or "preview",
                start=start if range_mode == "custom" else None,
                end=end if range_mode == "custom" else None,
                **filters,
            )
            if not series:
                raise ValueError("No C4 X/Y trajectory rows are available for this selection.")
            figure, items = _figure_from_series(
                series,
                y_title="Dataset X / Y (native units)",
            )
            figure = _attach_plot_export_snapshot(
                figure,
                plot_key="dataset",
                run_key=value,
                selection={
                    "filters": filters,
                    "resolution": resolution or "preview",
                    "range_mode": range_mode or "full",
                    "start": start if range_mode == "custom" else None,
                    "end": end if range_mode == "custom" else None,
                },
            )
            return (
                figure,
                _build_colored_scroll_legend(
                    items,
                    toggle_type="phase-c-results-dataset-legend-toggle",
                ),
                items,
                dbc.Alert(
                    f"Loaded {len(series)} C4 X/Y trajectory trace(s).",
                    color="success",
                ),
            )
        except (ResultSelectionTooBroad, ValueError) as exc:
            return no_update, no_update, no_update, dbc.Alert(str(exc), color="warning")
        except Exception as exc:
            return no_update, no_update, no_update, dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
            )

    @callback(
        Output("phase-c-results-dataset-graph", "figure", allow_duplicate=True),
        Output("phase-c-results-dataset-legend", "children", allow_duplicate=True),
        Output("phase-c-results-dataset-legend-state", "data", allow_duplicate=True),
        Input({"type": "phase-c-results-dataset-legend-toggle", "index": ALL}, "n_clicks"),
        State("phase-c-results-dataset-graph", "figure"),
        State("phase-c-results-dataset-legend-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_dataset_trace(_clicks, figure, items):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not figure or not items:
            return no_update, no_update, no_update
        index = int(triggered.get("index", -1))
        updated_figure, updated_items = toggle_trace_visibility(figure, items, index)
        if updated_figure is figure and updated_items is items:
            return no_update, no_update, no_update
        return (
            updated_figure,
            _build_colored_scroll_legend(
                updated_items,
                toggle_type="phase-c-results-dataset-legend-toggle",
            ),
            updated_items,
        )

    @callback(
        Output("phase-c-results-lineage-table", "children"),
        Output("phase-c-results-dataset-inventory", "children"),
        Output("phase-c-results-target-model-inventory", "children"),
        Output("phase-c-results-split-summary", "children"),
        Output("phase-c-results-model-inventory", "children"),
        Output("phase-c-results-generalization-table", "children"),
        Output("phase-c-results-artifact-inventory", "children"),
        Output("phase-c-results-inventory-message", "children"),
        Input("phase-c-results-load-inventories", "n_clicks"),
        State("phase-c-results-run", "value"),
        State("phase-c-results-building", "value"),
        State("phase-c-results-weather", "value"),
        State("phase-c-results-case", "value"),
        State("phase-c-results-aggregation", "value"),
        State("phase-c-results-weight", "value"),
        State("phase-c-results-zone", "value"),
        State("phase-c-results-model", "value"),
        State("phase-c-results-estimator", "value"),
        State("phase-c-results-evaluation-mode", "value"),
        prevent_initial_call=True,
    )
    def load_inventories(
        _,
        value,
        building,
        weather,
        case,
        aggregation,
        weight,
        zone,
        model,
        estimator,
        modes,
    ):
        if not value:
            return *([""] * 7), dbc.Alert(
                "Select a Phase C run first.",
                color="warning",
            )
        filters = _filters(
            building,
            weather,
            case,
            aggregation,
            weight,
            zone,
            model,
            estimator,
        )
        try:
            ref = load_run_ref_from_key(value)
            lineage = lineage_summary(ref)
            datasets = dataset_inventory(ref, **filters)
            targets = target_model_inventory(ref, **filters)
            split_rows = split_summary_rows(ref, **filters)
            artifacts = artifact_inventory(ref)
            try:
                models = load_model_metadata(ref, max_rows=500, **filters)
            except ResultSelectionTooBroad as exc:
                models = [{"status": "selection-too-broad", "detail": str(exc)}]
            try:
                generalization = generalization_metrics(
                    ref,
                    evaluation_modes_selected=modes,
                    **filters,
                )
            except ResultSelectionTooBroad as exc:
                generalization = [
                    {"status": "selection-too-broad", "detail": str(exc)}
                ]
            message = dbc.Alert(
                (
                    f"Loaded compact inventories for {len(datasets)} C4 dataset row(s). "
                    "Detailed scientific files remain lazy."
                ),
                color="success",
            )
            return (
                _table(lineage, limit=20),
                _table(datasets, limit=250),
                _table(targets, limit=100),
                _table(split_rows, limit=250),
                _table(models, limit=500),
                _table(generalization, limit=500),
                _table(artifacts, limit=100),
                message,
            )
        except Exception as exc:
            return *([""] * 7), dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
            )

    @callback(
        Output("phase-c-results-comparison-graph", "figure"),
        Output("phase-c-results-comparison-legend", "children"),
        Output("phase-c-results-comparison-legend-state", "data"),
        Output("phase-c-results-comparison-message", "children"),
        Input("phase-c-results-plot-comparison", "n_clicks"),
        State("phase-c-results-comparison-kind", "value"),
        State("phase-c-results-run", "value"),
        State("phase-c-results-building", "value"),
        State("phase-c-results-weather", "value"),
        State("phase-c-results-case", "value"),
        State("phase-c-results-aggregation", "value"),
        State("phase-c-results-weight", "value"),
        State("phase-c-results-zone", "value"),
        State("phase-c-results-model", "value"),
        State("phase-c-results-estimator", "value"),
        State("phase-c-results-split", "value"),
        State("phase-c-results-evaluation-mode", "value"),
        prevent_initial_call=True,
    )
    def plot_comparison(
        _,
        kind,
        value,
        building,
        weather,
        case,
        aggregation,
        weight,
        zone,
        model,
        estimator,
        splits,
        modes,
    ):
        if not value:
            return no_update, no_update, no_update, dbc.Alert(
                "Select a Phase C run.",
                color="warning",
            )
        filters = _filters(
            building,
            weather,
            case,
            aggregation,
            weight,
            zone,
            model,
            estimator,
        )
        try:
            ref = load_run_ref_from_key(value)
            metrics = []
            models = []
            split_rows = []
            availability = []
            phvac_series = []
            if kind in {"estimator_metric", "error_context"}:
                if not model:
                    raise ValueError(
                        "Select at least one model before loading detailed metrics."
                    )
                metrics = load_evaluation_metrics(
                    ref,
                    splits=splits,
                    evaluation_modes_selected=modes,
                    **filters,
                )
            elif kind == "coefficient":
                if not model:
                    raise ValueError(
                        "Select at least one model before plotting coefficients."
                    )
                models = load_model_metadata(ref, max_rows=100, **filters)
            elif kind == "split_coverage":
                split_rows = split_summary_rows(ref, **filters)
            elif kind == "availability":
                availability = structural_availability_rows(ref)
            elif kind == "building_phvac":
                phvac_series = load_building_phvac_series(
                    ref,
                    split=(splits or ["test"])[0],
                    evaluation_modes_selected=modes,
                    **filters,
                )
            figure = _comparison_figure(
                kind,
                metrics=metrics,
                models=models,
                splits=split_rows,
                availability=availability,
                phvac_series=phvac_series,
            )
            figure = _attach_plot_export_snapshot(
                figure,
                plot_key="comparison",
                run_key=value,
                selection={
                    "filters": filters,
                    "comparison_kind": kind,
                    "splits": splits or [],
                    "evaluation_modes": modes or [],
                },
            )
            items = _legend_items_from_figure(
                figure,
                secondary_label=str(kind).replace("_", " "),
            )
            return (
                figure,
                _build_colored_scroll_legend(
                    items,
                    toggle_type="phase-c-results-comparison-legend-toggle",
                ),
                items,
                dbc.Alert(
                    (
                        f"Loaded {str(kind).replace('_', ' ')} diagnostic from "
                        "the current filters."
                    ),
                    color="success",
                ),
            )
        except (ResultSelectionTooBroad, ValueError) as exc:
            return no_update, no_update, no_update, dbc.Alert(str(exc), color="warning")
        except Exception as exc:
            return no_update, no_update, no_update, dbc.Alert(
                f"{type(exc).__name__}: {exc}",
                color="danger",
            )

    @callback(
        Output("phase-c-results-comparison-graph", "figure", allow_duplicate=True),
        Output("phase-c-results-comparison-legend", "children", allow_duplicate=True),
        Output("phase-c-results-comparison-legend-state", "data", allow_duplicate=True),
        Input(
            {"type": "phase-c-results-comparison-legend-toggle", "index": ALL},
            "n_clicks",
        ),
        State("phase-c-results-comparison-graph", "figure"),
        State("phase-c-results-comparison-legend-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_comparison_trace(_clicks, figure, items):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not figure or not items:
            return no_update, no_update, no_update
        index = int(triggered.get("index", -1))
        updated_figure, updated_items = toggle_trace_visibility(figure, items, index)
        if updated_figure is figure and updated_items is items:
            return no_update, no_update, no_update
        return (
            updated_figure,
            _build_colored_scroll_legend(
                updated_items,
                toggle_type="phase-c-results-comparison-legend-toggle",
            ),
            updated_items,
        )

    @callback(
        Output("phase-c-results-evaluation-start", "disabled"),
        Output("phase-c-results-evaluation-end", "disabled"),
        Input("phase-c-results-evaluation-range-mode", "value"),
    )
    def evaluation_range_mode(value):
        disabled = value != "custom"
        return disabled, disabled

    @callback(
        Output("phase-c-results-annual-start", "disabled"),
        Output("phase-c-results-annual-end", "disabled"),
        Input("phase-c-results-annual-range-mode", "value"),
    )
    def annual_range_mode(value):
        disabled = value != "custom"
        return disabled, disabled

    @callback(
        Output("phase-c-results-annual-components", "options"),
        Output("phase-c-results-annual-components", "value"),
        Input("phase-c-results-run", "value"),
        Input("phase-c-results-annual-zone", "value"),
        Input("phase-c-results-model", "value"),
    )
    def annual_components(value, selected_zone, selected_models):
        if not value or not selected_zone:
            return [], []
        try:
            ref = load_run_ref_from_key(value)
            rows = annual_component_catalog(ref, selected_zone)
            model_values = {str(item) for item in (selected_models or [])}
            if model_values:
                rows = [
                    row for row in rows if str(row.get("model_id")) in model_values
                ]
            options = [
                {"label": row["label"], "value": row["prediction_column"]}
                for row in rows
            ]
            default = [row["prediction_column"] for row in rows[:2]]
            return options, default
        except Exception:
            return [], []

    @callback(
        Output("phase-c-results-evaluation-graph", "figure"),
        Output("phase-c-results-evaluation-legend", "children"),
        Output("phase-c-results-evaluation-legend-state", "data"),
        Output("phase-c-results-metrics-table", "children"),
        Output("phase-c-results-model-table", "children"),
        Output("phase-c-results-building-phvac-table", "children"),
        Output("phase-c-results-evaluation-message", "children"),
        Output("phase-c-results-evaluation-selection", "data"),
        Input("phase-c-results-load-evaluation", "n_clicks"),
        State("phase-c-results-run", "value"),
        State("phase-c-results-building", "value"),
        State("phase-c-results-weather", "value"),
        State("phase-c-results-case", "value"),
        State("phase-c-results-aggregation", "value"),
        State("phase-c-results-weight", "value"),
        State("phase-c-results-zone", "value"),
        State("phase-c-results-model", "value"),
        State("phase-c-results-estimator", "value"),
        State("phase-c-results-split", "value"),
        State("phase-c-results-evaluation-mode", "value"),
        State("phase-c-results-evaluation-resolution", "value"),
        State("phase-c-results-evaluation-plot-kind", "value"),
        State("phase-c-results-evaluation-range-mode", "value"),
        State("phase-c-results-evaluation-start", "value"),
        State("phase-c-results-evaluation-end", "value"),
        prevent_initial_call=True,
    )
    def load_evaluation(
        _,
        value,
        building,
        weather,
        case,
        aggregation,
        weight,
        zone,
        model,
        estimator,
        splits,
        modes,
        resolution,
        plot_kind,
        range_mode,
        start,
        end,
    ):
        empty_result = (no_update, no_update, no_update, "", "", "")
        if not value:
            return (
                *empty_result,
                dbc.Alert("Select a Phase C run.", color="warning"),
                {},
            )
        if not model:
            return (
                *empty_result,
                dbc.Alert(
                    "Select at least one model before loading observed/predicted Y.",
                    color="warning",
                ),
                {},
            )
        filters = _filters(
            building,
            weather,
            case,
            aggregation,
            weight,
            zone,
            model,
            estimator,
        )
        selection = _selection_from_states(
            filters=filters,
            splits=splits,
            modes=modes,
            resolution=resolution,
            range_mode=range_mode,
            start=start,
            end=end,
        )
        try:
            ref = load_run_ref_from_key(value)
            metrics = load_evaluation_metrics(
                ref,
                splits=selection["splits"],
                evaluation_modes_selected=selection["evaluation_modes"],
                **filters,
            )
            models = load_model_metadata(ref, **filters)
            series = load_evaluation_series(
                ref,
                splits=selection["splits"],
                evaluation_modes_selected=selection["evaluation_modes"],
                resolution=selection["resolution"],
                start=selection["start"],
                end=selection["end"],
                **filters,
            )
            building_rows = building_phvac_metrics(
                ref,
                splits=selection["splits"],
                evaluation_modes_selected=selection["evaluation_modes"],
                **filters,
            )
            figure, legend_items = _evaluation_figure(
                series,
                kind=plot_kind or "time_series",
            )
            selection["run_key"] = value
            selection["plot_kind"] = plot_kind or "time_series"
            figure = _attach_plot_export_snapshot(
                figure,
                plot_key="evaluation",
                run_key=value,
                selection=selection,
            )
            message = dbc.Alert(
                (
                    f"Loaded {len(metrics)} metric row(s), {len(models)} model row(s), "
                    f"and {len(series)} trajectory/diagnostic trace(s). The displayed "
                    "plot-data download below mirrors the visible traces in this graph."
                ),
                color="success",
            )
            return (
                figure,
                _build_colored_scroll_legend(
                    legend_items,
                    toggle_type="phase-c-results-eval-legend-toggle",
                ),
                legend_items,
                _table(metrics, limit=250),
                _table(models, limit=250),
                _table(building_rows, limit=100),
                message,
                selection,
            )
        except (ResultSelectionTooBroad, ValueError) as exc:
            return (
                *empty_result,
                dbc.Alert(str(exc), color="warning"),
                {},
            )
        except Exception as exc:
            return (
                *empty_result,
                dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
                {},
            )

    @callback(
        Output("phase-c-results-evaluation-graph", "figure", allow_duplicate=True),
        Output("phase-c-results-evaluation-legend", "children", allow_duplicate=True),
        Output("phase-c-results-evaluation-legend-state", "data", allow_duplicate=True),
        Input({"type": "phase-c-results-eval-legend-toggle", "index": ALL}, "n_clicks"),
        State("phase-c-results-evaluation-graph", "figure"),
        State("phase-c-results-evaluation-legend-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_evaluation_trace(_clicks, figure, items):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not figure or not items:
            return no_update, no_update, no_update
        index = int(triggered.get("index", -1))
        updated_figure, updated_items = toggle_trace_visibility(figure, items, index)
        if updated_figure is figure and updated_items is items:
            return no_update, no_update, no_update
        return (
            updated_figure,
            _build_colored_scroll_legend(
                updated_items,
                toggle_type="phase-c-results-eval-legend-toggle",
            ),
            updated_items,
        )

    @callback(
        Output("phase-c-results-validation-table", "children"),
        Input("phase-c-results-load-validation", "n_clicks"),
        State("phase-c-results-run", "value"),
        State("phase-c-results-validation-stage", "value"),
        State("phase-c-results-building", "value"),
        State("phase-c-results-weather", "value"),
        State("phase-c-results-case", "value"),
        State("phase-c-results-aggregation", "value"),
        State("phase-c-results-weight", "value"),
        State("phase-c-results-zone", "value"),
        State("phase-c-results-model", "value"),
        State("phase-c-results-estimator", "value"),
        prevent_initial_call=True,
    )
    def load_validation(
        _,
        value,
        stage,
        building,
        weather,
        case,
        aggregation,
        weight,
        zone,
        model,
        estimator,
    ):
        if not value or not stage:
            return dbc.Alert(
                "Select a Phase C run and validator stage.",
                color="warning",
            )
        try:
            ref = load_run_ref_from_key(value)
            filters = _filters(
                building,
                weather,
                case,
                aggregation,
                weight,
                zone,
                model,
                estimator,
            )
            rows = validation_diagnostics(ref, stage, **filters)
            return _table(rows, limit=500, empty=f"No {stage} diagnostics recorded.")
        except Exception as exc:
            return dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")

    @callback(
        Output("phase-c-results-annual-graph", "figure"),
        Output("phase-c-results-annual-legend", "children"),
        Output("phase-c-results-annual-legend-state", "data"),
        Output("phase-c-results-annual-summary", "children"),
        Output("phase-c-results-annual-message", "children"),
        Output("phase-c-results-annual-selection", "data"),
        Input("phase-c-results-load-annual", "n_clicks"),
        State("phase-c-results-run", "value"),
        State("phase-c-results-annual-zone", "value"),
        State("phase-c-results-annual-components", "value"),
        State("phase-c-results-annual-resolution", "value"),
        State("phase-c-results-annual-range-mode", "value"),
        State("phase-c-results-annual-start", "value"),
        State("phase-c-results-annual-end", "value"),
        prevent_initial_call=True,
    )
    def load_annual(
        _,
        value,
        selected_zone,
        components,
        resolution,
        range_mode,
        start,
        end,
    ):
        if not value or not selected_zone:
            return (
                no_update,
                no_update,
                no_update,
                "",
                dbc.Alert(
                    "Select a Phase C run and C8 zone package.",
                    color="warning",
                ),
                {},
            )
        selection = {
            "run_key": value,
            "zone_key": selected_zone,
            "prediction_columns": components or [],
            "resolution": resolution or "preview",
            "range_mode": range_mode or "full",
            "start": start if range_mode == "custom" else None,
            "end": end if range_mode == "custom" else None,
        }
        try:
            ref = load_run_ref_from_key(value)
            series, summary = load_annual_series(
                ref,
                selected_zone_key=selected_zone,
                prediction_columns=selection["prediction_columns"],
                resolution=selection["resolution"],
                start=selection["start"],
                end=selection["end"],
            )
            figure, legend_items = _figure_from_series(
                series,
                y_title="Full-year predicted Ŷ",
            )
            figure = _attach_plot_export_snapshot(
                figure,
                plot_key="annual",
                run_key=value,
                selection=selection,
            )
            message = dbc.Alert(
                (
                    f"Loaded {len(series)} full-year predicted-Y trace(s). PHVAC "
                    "oracle/chained traces remain separately labeled. The displayed "
                    "plot-data download below mirrors the visible traces in this graph."
                ),
                color="success",
            )
            return (
                figure,
                _build_colored_scroll_legend(
                    legend_items,
                    toggle_type="phase-c-results-annual-legend-toggle",
                ),
                legend_items,
                _table(summary, limit=50),
                message,
                selection,
            )
        except (ResultSelectionTooBroad, ValueError) as exc:
            return (
                no_update,
                no_update,
                no_update,
                "",
                dbc.Alert(str(exc), color="warning"),
                {},
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                "",
                dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger"),
                {},
            )

    @callback(
        Output("phase-c-results-annual-graph", "figure", allow_duplicate=True),
        Output("phase-c-results-annual-legend", "children", allow_duplicate=True),
        Output("phase-c-results-annual-legend-state", "data", allow_duplicate=True),
        Input({"type": "phase-c-results-annual-legend-toggle", "index": ALL}, "n_clicks"),
        State("phase-c-results-annual-graph", "figure"),
        State("phase-c-results-annual-legend-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_annual_trace(_clicks, figure, items):
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict) or not figure or not items:
            return no_update, no_update, no_update
        index = int(triggered.get("index", -1))
        updated_figure, updated_items = toggle_trace_visibility(figure, items, index)
        if updated_figure is figure and updated_items is items:
            return no_update, no_update, no_update
        return (
            updated_figure,
            _build_colored_scroll_legend(
                updated_items,
                toggle_type="phase-c-results-annual-legend-toggle",
            ),
            updated_items,
        )

    def _register_plot_download(prefix: str, plot_label: str):
        @callback(
            Output(f"phase-c-results-{prefix}-plot-download", "data"),
            Output(f"phase-c-results-{prefix}-plot-export-message", "children"),
            Input(f"phase-c-results-{prefix}-download-plot-data", "n_clicks"),
            State(f"phase-c-results-{prefix}-graph", "figure"),
            State(f"phase-c-results-{prefix}-download-format", "value"),
            State("phase-c-results-run", "value"),
            prevent_initial_call=True,
        )
        def _download_plot(_, figure, file_format, run_key):
            if not figure or not (figure.get("data") or []):
                return no_update, dbc.Alert(
                    "Load this plot first; the download mirrors the visible graph.",
                    color="warning",
                    className="py-1 mb-0",
                )
            try:
                run_id = "phase_c"
                run_ref = None
                if run_key:
                    run_ref = load_run_ref_from_key(run_key)
                    run_id = str(run_ref.get("phase_c_run_id") or run_id)
                payload, filename = build_plot_figure_export(
                    figure,
                    file_format=file_format or "csv",
                    plot_key=prefix,
                    run_id=run_id,
                    run_ref=run_ref,
                )
                return dcc.send_bytes(payload, filename), dbc.Alert(
                    (
                        f"Prepared {plot_label} visible data ZIP with "
                        f"{str(file_format or 'csv').upper()} data and selection metadata."
                    ),
                    color="success",
                    className="py-1 mb-0",
                )
            except Exception as exc:
                return no_update, dbc.Alert(
                    f"{type(exc).__name__}: {exc}",
                    color="danger",
                    className="py-1 mb-0",
                )

    for _prefix, _label in (
        ("dataset", "Dataset X/Y"),
        ("evaluation", "Evaluation"),
        ("annual", "Annual inference"),
        ("comparison", "Metric/diagnostic"),
    ):
        _register_plot_download(_prefix, _label)

    @callback(
        Output("phase-c-results-summary-download", "data"),
        Input("phase-c-results-download-summary", "n_clicks"),
        State("phase-c-results-run", "value"),
        prevent_initial_call=True,
    )
    def download_summary(_, value):
        if not value:
            return no_update
        try:
            payload, filename = build_campaign_summary_export(
                load_run_ref_from_key(value)
            )
            return dcc.send_bytes(payload, filename)
        except Exception:
            return no_update

    @callback(
        Output("phase-c-results-model-bundle-download", "data"),
        Output(
            "phase-c-results-artifact-export-message",
            "children",
            allow_duplicate=True,
        ),
        Input("phase-c-results-download-model-bundle", "n_clicks"),
        State("phase-c-results-evaluation-selection", "data"),
        prevent_initial_call=True,
    )
    def download_model_bundle(_, selection):
        if not selection or not selection.get("run_key"):
            return no_update, dbc.Alert(
                "Load an exact evaluation selection first. Model bundles require one model.",
                color="warning",
            )
        try:
            ref = load_run_ref_from_key(selection["run_key"])
            payload, filename = build_model_bundle_export(ref, selection=selection)
            return dcc.send_bytes(payload, filename), dbc.Alert(
                f"Prepared {filename} for the exact selected model artifact.",
                color="success",
            )
        except (ResultSelectionTooBroad, ValueError) as exc:
            return no_update, dbc.Alert(str(exc), color="warning")
        except Exception as exc:
            return no_update, dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")
