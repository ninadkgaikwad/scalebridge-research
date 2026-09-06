"""Callbacks for Phase D Tab 3 Results."""
from __future__ import annotations

from dash import ALL as DASH_ALL, Input, Output, State, callback, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from .....components.results.scroll_legend import build_scroll_legend, toggle_trace_visibility
from .....services.phase_d import results_data


_REGISTERED = False
_FILTER_COLUMNS = list(results_data.FILTER_COLUMNS)


def _filter_id(column: str) -> str:
    return f"phase-d-results-filter-{column.replace('_', '-')}"


def _run_summary_component(summary):
    runtime = summary.get("runtime_seconds")
    runtime_text = "—" if runtime is None else f"{float(runtime) / 3600:.2f} h"
    return dbc.Alert(
        [
            html.Div([html.Strong("Status: "), summary["status"]]),
            html.Div([html.Strong("Campaign / Run: "), f"{summary['campaign_id']} / {summary['phase_d_run_id']}"]),
            html.Div([html.Strong("Phase C / matrix: "), f"{summary['phase_c_campaign_run_id']} / {summary['matrix_run_id']}"]),
            html.Div([html.Strong("Aggregations: "), f"{summary['completed_aggregation_run_count']}/{summary['selected_aggregation_run_count']} completed | {summary['failed_aggregation_run_count']} failed"]),
            html.Div([html.Strong("Datasets: "), f"{summary['dataset_count']} total | {summary['ml_dataset_count']} ML/SciML | {summary['opt_bayes_dataset_count']} Opt/Bayes"]),
            html.Div([html.Strong("Relationships: "), f"{summary['ind_dataset_count']} IND | {summary['dep1_dataset_count']} DEP1 | {summary['dep2_dataset_count']} DEP2"]),
            html.Div([html.Strong("Runtime: "), runtime_text]),
            html.Div([html.Strong("MLflow: "), f"{'enabled' if summary['mlflow_enabled'] else 'disabled'} | run={summary['mlflow_run_id'] or '—'}"]),
        ],
        color="success" if summary["status"] == "completed" else "warning",
        className="phase-d-wrap-alert",
    )


def _partition_table(counts):
    if not counts:
        return html.Div("No partition counts recorded.", className="text-muted small")
    total = sum(int(v or 0) for v in counts.values())
    header = html.Thead(html.Tr([html.Th("Partition"), html.Th("Rows"), html.Th("% of annual rows")]))
    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(str(name)),
                    html.Td(f"{int(count):,}"),
                    html.Td(f"{100.0 * int(count) / total:.2f}%" if total else "—"),
                ]
            )
            for name, count in counts.items()
        ]
    )
    return dbc.Table([header, body], bordered=True, hover=True, responsive=True, size="sm")


def _dataset_summary(details):
    row, manifest = details["registry"], details["manifest"]
    return dbc.Alert(
        [
            html.Div([html.Strong("Building / weather: "), f"{row.get('building_type')} / {row.get('weather_location')}"]),
            html.Div([html.Strong("Aggregation: "), f"{row.get('aggregation_family')} | {row.get('aggregation_id')} | {row.get('weight_mode')} | {row.get('rule_set')}"]),
            html.Div([html.Strong("Silo / relationship: "), f"{row.get('silo')} / {row.get('mode')}"]),
            html.Div([html.Strong("Independent zone: "), str(row.get("independent_zone_id") or "—")]),
            html.Div([html.Strong("Policy / temporal grid: "), f"{row.get('policy_name')} | lag={row.get('input_lag')} | horizon={row.get('target_horizon')}"]),
            html.Div([html.Strong("Rows: "), f"{details['row_count']:,} annual | {details['included_row_count']:,} included | {details['excluded_row_count']:,} excluded"]),
            html.Div([html.Strong("Time coverage: "), f"{details['first_timestamp']} → {details['last_timestamp']}"]),
            html.Div([html.Strong("Final columns: "), str(details["final_column_count"])]),
            html.Div([html.Strong("Data path: "), str(row.get("data_path") or "")]),
        ],
        color="light",
        className="phase-d-wrap-alert",
    )


def _preview_table(records, columns):
    if not records:
        return html.Div("No preview rows.", className="text-muted small")
    header = html.Thead(html.Tr([html.Th(column) for column in columns]))
    body = html.Tbody(
        [
            html.Tr([html.Td(record.get(column)) for column in columns])
            for record in records
        ]
    )
    return html.Div(
        dbc.Table([header, body], bordered=True, hover=True, striped=True, size="sm"),
        className="phase-d-results-table-wrap",
    )


def _trace_label(column_meta):
    zone = str(column_meta.get("aggregate_zone_id") or "Common")
    base = str(column_meta.get("base_signal") or column_meta.get("name") or "signal")
    role = str(column_meta.get("temporal_role") or "")
    offset = column_meta.get("offset_steps")
    if role == "prediction_target":
        temporal = f"target +{offset}"
    elif offset in (None, 0):
        temporal = "lag 0"
    else:
        temporal = f"lag {abs(int(offset))}"
    units = str(column_meta.get("units") or "")
    return f"{zone} | {base} | {temporal}" + (f" [{units}]" if units else "")


def _build_figure(frame, metadata, *, selection):
    figure = go.Figure()
    items = []
    colmeta = metadata["column_metadata"]
    units_seen = []
    for signal in metadata["signals"]:
        meta = colmeta.get(signal) or {}
        units = str(meta.get("units") or "")
        if units and units not in units_seen:
            units_seen.append(units)

    axis_for_units = {}
    if units_seen:
        axis_for_units[units_seen[0]] = "y"
    if len(units_seen) >= 2:
        axis_for_units[units_seen[1]] = "y2"
    for units in units_seen[2:]:
        axis_for_units[units] = "y"

    for index, signal in enumerate(metadata["signals"]):
        meta = colmeta.get(signal) or {}
        units = str(meta.get("units") or "")
        label = _trace_label(meta)
        x_values = [
            value.isoformat() if isinstance(value, pd.Timestamp) else str(value)
            for value in frame["timestamp"].tolist()
        ]
        y_values = [
            None if pd.isna(value) else value.item() if hasattr(value, "item") else value
            for value in frame[signal].tolist()
        ]
        trace = go.Scattergl(
            x=x_values,
            y=y_values,
            mode="lines",
            name=label,
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        )
        if axis_for_units.get(units) == "y2":
            trace.update(yaxis="y2")
        figure.add_trace(trace)
        items.append(
            {
                "index": index,
                "visible": True,
                "name": label,
                "primary_label": str(meta.get("base_signal") or signal),
                "secondary_label": f"{meta.get('aggregate_zone_id') or 'Common'} | {meta.get('temporal_role')} | {units or 'unitless'}",
            }
        )

    # Lock final trace colors before rendering the external legend.
    palette = tuple(qualitative.Plotly)
    for index, trace in enumerate(figure.data):
        color = palette[index % len(palette)]
        trace.line.color = color
        items[index]["color"] = color

    layout = {
        "showlegend": False,
        "hovermode": "x unified",
        "margin": {"l": 70, "r": 70, "t": 45, "b": 65},
        "xaxis_title": "Timestamp",
        "autosize": True,
        "meta": {
            "phase_d_plot_export": {
                "selection": selection,
                "source_row_count": metadata["source_row_count"],
                "plotted_row_count": metadata["plotted_row_count"],
                "stride": metadata["stride"],
            }
        },
    }
    if units_seen:
        layout["yaxis"] = {"title": units_seen[0]}
    if len(units_seen) >= 2:
        layout["yaxis2"] = {
            "title": units_seen[1],
            "overlaying": "y",
            "side": "right",
        }
    figure.update_layout(**layout)
    return figure, items


def register_results_callbacks():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    @callback(
        Output("phase-d-results-run", "options"),
        Input("phase-d-results-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_runs(_):
        return results_data.run_options()

    @callback(
        Output("phase-d-results-run-summary", "children"),
        Input("phase-d-results-run", "value"),
    )
    def load_run_summary(run_key):
        if not run_key:
            return ""
        try:
            ref = results_data.load_run_ref(run_key)
            return _run_summary_component(results_data.run_summary(ref))
        except Exception as exc:
            return dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")

    cascade_outputs = []
    for column in _FILTER_COLUMNS:
        cascade_outputs.append(Output(_filter_id(column), "options"))
        cascade_outputs.append(Output(_filter_id(column), "value"))
    cascade_outputs += [
        Output("phase-d-results-dataset", "options"),
        Output("phase-d-results-dataset", "value"),
        Output("phase-d-results-match-count", "children"),
    ]
    filter_inputs = [Input(_filter_id(column), "value") for column in _FILTER_COLUMNS]

    @callback(
        *cascade_outputs,
        Input("phase-d-results-run", "value"),
        *filter_inputs,
    )
    def cascade_filters(run_key, *values):
        values = list(values[: len(_FILTER_COLUMNS)])

        if not run_key:
            result = []
            for _ in _FILTER_COLUMNS:
                result += [
                    [{"label": "All", "value": results_data.ALL}],
                    results_data.ALL,
                ]
            return tuple(result + [[], None, "Select a Phase D run."])

        try:
            ref = results_data.load_run_ref(run_key)
            frame = results_data.dataset_registry(ref)

            trigger = ctx.triggered_id
            if trigger == "phase-d-results-run":
                raw_filters = {
                    column: results_data.ALL
                    for column in _FILTER_COLUMNS
                }
                preferred_column = None
            else:
                raw_filters = dict(zip(_FILTER_COLUMNS, values))
                preferred_column = None
                if isinstance(trigger, str):
                    for column in _FILTER_COLUMNS:
                        if trigger == _filter_id(column):
                            preferred_column = column
                            break

            state = results_data.cascading_filter_state(
                frame,
                raw_filters,
                preferred_column=preferred_column,
            )
            matched = state["matched"]

            dataset_rows = results_data.dataset_options(matched)

            result = []
            for column in _FILTER_COLUMNS:
                resolved_value = state["values"][column]
                current_value = raw_filters.get(column, results_data.ALL)
                if trigger == "phase-d-results-run":
                    value_output = resolved_value
                elif current_value == resolved_value:
                    value_output = no_update
                else:
                    value_output = resolved_value
                result += [
                    state["options"][column],
                    value_output,
                ]
            result += [
                dataset_rows,
                None,
                f"{len(matched)} of {len(frame)} final datasets match the current filters.",
            ]
            return tuple(result)
        except Exception as exc:
            result = []
            for _ in _FILTER_COLUMNS:
                result += [
                    [{"label": "All", "value": results_data.ALL}],
                    results_data.ALL,
                ]
            return tuple(
                result
                + [
                    [],
                    None,
                    f"{type(exc).__name__}: {exc}",
                ]
            )

    @callback(
        Output("phase-d-results-dataset-summary", "children"),
        Output("phase-d-results-partition-summary", "children"),
        Output("phase-d-results-signals", "options"),
        Output("phase-d-results-signals", "value"),
        Output("phase-d-results-partition", "options"),
        Output("phase-d-results-partition", "value"),
        Output("phase-d-results-start", "value"),
        Output("phase-d-results-end", "value"),
        Output("phase-d-results-preview", "children"),
        Input("phase-d-results-dataset", "value"),
        State("phase-d-results-run", "value"),
    )
    def load_dataset(manifest_key, run_key):
        if not manifest_key or not run_key:
            return "", "", [], [], [], None, None, None, ""
        try:
            ref = results_data.load_run_ref(run_key)
            details = results_data.dataset_details(ref, manifest_key)
            manifest = details["manifest"]
            records, columns = results_data.preview_records(ref, manifest_key, limit=200)
            return (
                _dataset_summary(details),
                _partition_table(details["partition_counts"]),
                results_data.signal_options(manifest),
                results_data.default_signal_values(manifest),
                results_data.partition_options(manifest),
                results_data.INCLUDED,
                details["first_timestamp"],
                details["last_timestamp"],
                _preview_table(records, columns),
            )
        except Exception as exc:
            alert = dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")
            return alert, "", [], [], [], None, None, None, ""

    @callback(
        Output("phase-d-results-start", "disabled"),
        Output("phase-d-results-end", "disabled"),
        Input("phase-d-results-range-mode", "value"),
    )
    def toggle_range(mode):
        disabled = mode != "custom"
        return disabled, disabled

    @callback(
        Output("phase-d-results-graph", "figure"),
        Output("phase-d-results-legend-state", "data"),
        Output("phase-d-results-legend", "children"),
        Output("phase-d-results-plot-message", "children"),
        Input("phase-d-results-load-plot", "n_clicks"),
        State("phase-d-results-run", "value"),
        State("phase-d-results-dataset", "value"),
        State("phase-d-results-signals", "value"),
        State("phase-d-results-partition", "value"),
        State("phase-d-results-range-mode", "value"),
        State("phase-d-results-start", "value"),
        State("phase-d-results-end", "value"),
        State("phase-d-results-max-points", "value"),
        prevent_initial_call=True,
    )
    def load_plot(_, run_key, manifest_key, signals, partition, range_mode, start, end, max_points):
        if not run_key or not manifest_key:
            return go.Figure(), [], build_scroll_legend([], toggle_type="phase-d-results-trace"), dbc.Alert("Select a Phase D run and final dataset.", color="warning")
        try:
            ref = results_data.load_run_ref(run_key)
            start_value = start if range_mode == "custom" else None
            end_value = end if range_mode == "custom" else None
            limit = None if not max_points else int(max_points)
            frame, metadata = results_data.load_plot_frame(
                ref,
                manifest_key,
                signals=list(signals or []),
                partition=partition or results_data.INCLUDED,
                start=start_value,
                end=end_value,
                max_points=limit,
            )
            selection = {
                "dataset_manifest": manifest_key,
                "signals": list(signals or []),
                "partition": partition,
                "range_mode": range_mode,
                "start": start_value,
                "end": end_value,
                "max_points": limit,
            }
            figure, items = _build_figure(frame, metadata, selection=selection)
            if metadata["stride"] > 1:
                message = dbc.Alert(
                    f"Displayed {metadata['plotted_row_count']:,} of {metadata['source_row_count']:,} selected rows using deterministic stride={metadata['stride']}. Plot-data download uses this exact displayed sampling; selected-dataset download remains full resolution.",
                    color="info",
                )
            else:
                message = dbc.Alert(f"Displayed all {metadata['plotted_row_count']:,} selected rows.", color="success")
            return figure, items, build_scroll_legend(items, toggle_type="phase-d-results-trace"), message
        except Exception as exc:
            return go.Figure(), [], build_scroll_legend([], toggle_type="phase-d-results-trace"), dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")

    @callback(
        Output("phase-d-results-graph", "figure", allow_duplicate=True),
        Output("phase-d-results-legend-state", "data", allow_duplicate=True),
        Output("phase-d-results-legend", "children", allow_duplicate=True),
        Input({"type": "phase-d-results-trace", "index": DASH_ALL}, "n_clicks"),
        State("phase-d-results-graph", "figure"),
        State("phase-d-results-legend-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_trace(_clicks, figure, items):
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict):
            return figure, items, build_scroll_legend(items, toggle_type="phase-d-results-trace")
        updated_figure, updated_items = toggle_trace_visibility(figure, items, int(trigger["index"]))
        return updated_figure, updated_items, build_scroll_legend(updated_items, toggle_type="phase-d-results-trace")

    @callback(
        Output("phase-d-results-download", "data"),
        Input("phase-d-results-download-plot", "n_clicks"),
        Input("phase-d-results-download-dataset", "n_clicks"),
        Input("phase-d-results-download-summary", "n_clicks"),
        State("phase-d-results-run", "value"),
        State("phase-d-results-dataset", "value"),
        State("phase-d-results-graph", "figure"),
        State("phase-d-results-plot-download-format", "value"),
        prevent_initial_call=True,
    )
    def download(_plot, _dataset, _summary, run_key, manifest_key, figure, fmt):
        if not run_key:
            return None
        ref = results_data.load_run_ref(run_key)
        trigger = ctx.triggered_id
        if trigger == "phase-d-results-download-summary":
            payload, filename = results_data.build_run_summary_export(ref)
        elif trigger == "phase-d-results-download-dataset":
            if not manifest_key:
                return None
            payload, filename = results_data.build_selected_dataset_export(ref, manifest_key)
        else:
            if not manifest_key or not figure:
                return None
            payload, filename = results_data.build_visible_plot_export(
                figure,
                file_format=fmt,
                run_ref=ref,
                manifest_key=manifest_key,
            )
        return dcc.send_bytes(payload, filename)
