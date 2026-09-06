"""Read-only Results access for Phase D thermal-model-ready time-series datasets."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path, PureWindowsPath
from typing import Any
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


RUN_MANIFEST = "phase_d_campaign_run_manifest.json"
RUN_PLAN = "phase_d_campaign_plan.json"
DATASET_REGISTRY = "dataset_registry.csv"
AGGREGATION_REGISTRY = "aggregation_run_registry.csv"
FAILURES = "failures.csv"

ALL = "__ALL__"
INCLUDED = "__INCLUDED__"

FILTER_COLUMNS = (
    "building_type",
    "weather_location",
    "case_id",
    "aggregation_family",
    "aggregation_id",
    "weight_mode",
    "rule_set",
    "silo",
    "mode",
    "independent_zone_id",
    "heat_representation",
    "policy_name",
    "input_lag",
    "target_horizon",
)


def campaigns_root() -> Path:
    return resolve_generated_data_root() / "campaigns"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def discover_phase_d_runs() -> list[dict[str, Any]]:
    """Discover compact Phase D campaign runs without opening dataset Parquets."""
    rows: list[dict[str, Any]] = []
    root = campaigns_root()
    if not root.is_dir():
        return rows
    for campaign_root in sorted(p for p in root.iterdir() if p.is_dir()):
        runs_root = campaign_root / "phase_d" / "campaign_runs"
        if not runs_root.is_dir():
            continue
        for run_root in sorted(
            (p for p in runs_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        ):
            manifest_path = run_root / RUN_MANIFEST
            manifest = _read_json(manifest_path)
            if not manifest:
                continue
            rows.append(
                {
                    "campaign_id": str(manifest.get("campaign_id") or campaign_root.name),
                    "phase_d_run_id": str(manifest.get("phase_d_run_id") or run_root.name),
                    "phase_c_campaign_run_id": str(manifest.get("phase_c_campaign_run_id") or ""),
                    "matrix_run_id": str(manifest.get("matrix_run_id") or ""),
                    "status": str(manifest.get("status") or ""),
                    "created_at_utc": str(manifest.get("created_at_utc") or ""),
                    "dataset_count": int(manifest.get("dataset_count") or 0),
                    "completed_aggregation_run_count": int(manifest.get("completed_aggregation_run_count") or 0),
                    "failed_aggregation_run_count": int(manifest.get("failed_aggregation_run_count") or 0),
                    "run_root": str(run_root),
                    "manifest_path": str(manifest_path),
                }
            )
    rows.sort(key=lambda r: (r["created_at_utc"], r["phase_d_run_id"]), reverse=True)
    return rows


def encode_run_key(campaign_id: str, phase_d_run_id: str) -> str:
    return f"{campaign_id}::{phase_d_run_id}"


def decode_run_key(value: str) -> tuple[str, str]:
    text = str(value or "")
    if "::" not in text:
        raise ValueError("Expected Phase D run key '<campaign_id>::<phase_d_run_id>'")
    campaign_id, run_id = text.split("::", 1)
    if not campaign_id or not run_id:
        raise ValueError(f"Invalid Phase D run key: {value!r}")
    return campaign_id, run_id


def run_options() -> list[dict[str, str]]:
    options = []
    for row in discover_phase_d_runs():
        label = (
            f"{row['phase_d_run_id']} | {row['campaign_id']} | "
            f"{row['status'] or 'unknown'} | datasets={row['dataset_count']}"
        )
        options.append({"label": label, "value": encode_run_key(row["campaign_id"], row["phase_d_run_id"])})
    return options


def load_run_ref(run_key: str) -> dict[str, Any]:
    campaign_id, run_id = decode_run_key(run_key)
    campaign_root = campaigns_root() / campaign_id
    run_root = campaign_root / "phase_d" / "campaign_runs" / run_id
    manifest_path = run_root / RUN_MANIFEST
    manifest = _read_json(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Phase D run manifest not found or unreadable: {manifest_path}")
    return {
        "campaign_id": campaign_id,
        "phase_d_run_id": run_id,
        "campaign_root": campaign_root,
        "run_root": run_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def run_summary(run_ref: dict[str, Any]) -> dict[str, Any]:
    manifest = run_ref["manifest"]
    return {
        "campaign_id": run_ref["campaign_id"],
        "phase_d_run_id": run_ref["phase_d_run_id"],
        "status": str(manifest.get("status") or ""),
        "created_at_utc": str(manifest.get("created_at_utc") or ""),
        "matrix_run_id": str(manifest.get("matrix_run_id") or ""),
        "phase_c_campaign_run_id": str(manifest.get("phase_c_campaign_run_id") or ""),
        "selected_aggregation_run_count": int(manifest.get("selected_aggregation_run_count") or 0),
        "completed_aggregation_run_count": int(manifest.get("completed_aggregation_run_count") or 0),
        "failed_aggregation_run_count": int(manifest.get("failed_aggregation_run_count") or 0),
        "dataset_count": int(manifest.get("dataset_count") or 0),
        "ml_dataset_count": int(manifest.get("ml_dataset_count") or 0),
        "opt_bayes_dataset_count": int(manifest.get("opt_bayes_dataset_count") or 0),
        "ind_dataset_count": int(manifest.get("ind_dataset_count") or 0),
        "dep1_dataset_count": int(manifest.get("dep1_dataset_count") or 0),
        "dep2_dataset_count": int(manifest.get("dep2_dataset_count") or 0),
        "runtime_seconds": manifest.get("runtime_seconds"),
        "mlflow_enabled": bool(manifest.get("mlflow_enabled")),
        "mlflow_experiment_name": str(manifest.get("mlflow_experiment_name") or ""),
        "mlflow_run_id": str(manifest.get("mlflow_run_id") or ""),
        "run_root": str(run_ref["run_root"]),
    }


def dataset_registry(run_ref: dict[str, Any]) -> pd.DataFrame:
    frame = _read_csv(Path(run_ref["run_root"]) / DATASET_REGISTRY)
    if frame.empty:
        return frame
    return frame.reset_index(drop=True)


def _option_label(column: str, value: Any) -> str:
    text = str(value)
    if column == "silo":
        return {"ml_sciml": "ML / SciML", "opt_bayes": "Optimization / Bayesian"}.get(text, text)
    if column == "mode":
        return {"independent": "IND — Independent", "dependent1": "DEP1 — Dependent 1", "dependent2": "DEP2 — Dependent 2"}.get(text, text)
    if column == "policy_name":
        return text.replace("_", " ").title()
    if column == "heat_representation":
        return {
            "grouped_qzic_qzir": "Grouped QZIC + QZIR",
            "component_heat_inputs": "Component heat inputs",
        }.get(text, text.replace("_", " ").title())
    if column == "aggregation_family":
        return text.replace("_", " ").title()
    return text


def filter_options(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return [{"label": "All", "value": ALL}]
    values = frame[column].dropna().drop_duplicates().tolist()
    try:
        values = sorted(values)
    except Exception:
        pass
    return [{"label": "All", "value": ALL}] + [
        {"label": _option_label(column, value), "value": value}
        for value in values
    ]


def all_filter_options(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    return {column: filter_options(frame, column) for column in FILTER_COLUMNS}


def _is_unrestricted(value: Any) -> bool:
    return value is None or value == "" or value == ALL


def apply_filters(frame: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    work = frame.copy()
    for column, value in (filters or {}).items():
        if column not in work.columns or _is_unrestricted(value):
            continue
        if pd.isna(value):
            work = work[work[column].isna()]
        else:
            work = work[work[column].astype(str) == str(value)]
    return work.reset_index(drop=True)


def reconcile_filter_values(
    frame: pd.DataFrame,
    filters: dict[str, Any] | None,
    *,
    preferred_column: str | None = None,
) -> dict[str, Any]:
    """Return a feasible single-value filter state.

    The most recently changed filter may be supplied as ``preferred_column``.
    Its selection is evaluated first, so a valid new user choice wins over
    older incompatible selections. Older selections are preserved only while
    they keep at least one registry row feasible; otherwise they are reset to
    ``ALL``. Clearing a filter therefore broadens the feasible registry again.
    """
    selected = {
        column: (filters or {}).get(column, ALL)
        for column in FILTER_COLUMNS
    }
    for column, value in list(selected.items()):
        if _is_unrestricted(value) or column not in frame.columns:
            selected[column] = ALL

    order = list(FILTER_COLUMNS)
    if (
        preferred_column in FILTER_COLUMNS
        and not _is_unrestricted(selected.get(preferred_column))
    ):
        order.remove(preferred_column)
        order.insert(0, preferred_column)

    feasible = frame.copy()
    for column in order:
        value = selected[column]
        if _is_unrestricted(value):
            continue
        candidate = apply_filters(feasible, {column: value})
        if candidate.empty:
            selected[column] = ALL
        else:
            feasible = candidate

    return selected


def faceted_filter_options(
    frame: pd.DataFrame,
    filters: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Build mutually constrained option lists from the feasible registry.

    Each filter's own active value is excluded while its options are computed.
    This lets a user change that dimension without first clearing it, while
    every *other* active selection still constrains the option list.
    """
    selected = {
        column: (filters or {}).get(column, ALL)
        for column in FILTER_COLUMNS
    }
    options: dict[str, list[dict[str, Any]]] = {}
    for column in FILTER_COLUMNS:
        other_filters = {
            other: value
            for other, value in selected.items()
            if other != column
        }
        feasible = apply_filters(frame, other_filters)
        options[column] = filter_options(feasible, column)
    return options


def cascading_filter_state(
    frame: pd.DataFrame,
    filters: dict[str, Any] | None,
    *,
    preferred_column: str | None = None,
) -> dict[str, Any]:
    """Resolve a complete mutually constrained Results-filter state."""
    values = reconcile_filter_values(
        frame,
        filters,
        preferred_column=preferred_column,
    )
    matched = apply_filters(frame, values)
    return {
        "values": values,
        "options": faceted_filter_options(frame, values),
        "matched": matched,
    }


def _dataset_label(row: pd.Series) -> str:
    parts = [
        str(row.get("building_type") or ""),
        str(row.get("weather_location") or ""),
        f"{row.get('aggregation_family')}/{row.get('weight_mode')}",
        _option_label("silo", row.get("silo")),
        _option_label("mode", row.get("mode")),
        _option_label("heat_representation", row.get("heat_representation")),
    ]
    zone = row.get("independent_zone_id")
    if pd.notna(zone) and str(zone).strip():
        parts.append(str(zone))
    parts += [
        f"lag={row.get('input_lag')}",
        f"h={row.get('target_horizon')}",
        str(row.get("policy_token") or row.get("policy_name") or ""),
    ]
    return " | ".join(part for part in parts if part and part != "nan")


def dataset_options(frame: pd.DataFrame) -> list[dict[str, str]]:
    if frame.empty or "manifest_path" not in frame.columns:
        return []
    options = []
    for _, row in frame.iterrows():
        value = str(row["manifest_path"])
        options.append({"label": _dataset_label(row), "value": value})
    return options


def selected_registry_row(run_ref: dict[str, Any], manifest_key: str) -> dict[str, Any]:
    frame = dataset_registry(run_ref)
    if frame.empty:
        raise ValueError("Phase D dataset registry is empty")
    matches = frame[frame["manifest_path"].astype(str) == str(manifest_key)]
    if matches.empty:
        raise KeyError("Selected Phase D dataset is not present in this run registry")
    row = matches.iloc[0]
    return {
        key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in row.to_dict().items()
    }


def _localize_artifact_path(stored_path: str | Path, run_ref: dict[str, Any]) -> Path:
    raw = str(stored_path or "").strip()
    if not raw:
        raise ValueError("Empty Phase D artifact path")
    direct = Path(raw)
    if direct.is_file():
        return direct

    parts = list(PureWindowsPath(raw.replace("/", "\\")).parts)
    lower = [part.casefold() for part in parts]
    try:
        phase_d_index = lower.index("phase_d")
    except ValueError:
        raise FileNotFoundError(f"Could not localize Phase D artifact: {raw}") from None

    localized = Path(run_ref["campaign_root"]) / Path(*parts[phase_d_index:])
    if not localized.is_file():
        raise FileNotFoundError(f"Phase D artifact not found: {localized}")
    return localized


def load_dataset_manifest(run_ref: dict[str, Any], manifest_key: str) -> dict[str, Any]:
    row = selected_registry_row(run_ref, manifest_key)
    path = _localize_artifact_path(row["manifest_path"], run_ref)
    payload = _read_json(path)
    if not payload:
        raise ValueError(f"Unreadable Phase D dataset manifest: {path}")
    return payload


def dataset_details(run_ref: dict[str, Any], manifest_key: str) -> dict[str, Any]:
    row = selected_registry_row(run_ref, manifest_key)
    manifest = load_dataset_manifest(run_ref, manifest_key)
    partition_counts = manifest.get("partition_counts") or {}
    return {
        "registry": row,
        "manifest": manifest,
        "partition_counts": partition_counts,
        "first_timestamp": manifest.get("first_timestamp"),
        "last_timestamp": manifest.get("last_timestamp"),
        "included_row_count": int(manifest.get("included_row_count") or row.get("included_row_count") or 0),
        "excluded_row_count": int(manifest.get("excluded_row_count") or 0),
        "row_count": int(manifest.get("row_count") or row.get("row_count") or 0),
        "final_column_count": len(manifest.get("final_columns") or []),
    }


def signal_options(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    options = []
    for column in manifest.get("final_columns") or []:
        if not isinstance(column, dict):
            continue
        role = str(column.get("temporal_role") or "")
        if role not in {"model_input", "prediction_target"}:
            continue
        name = str(column.get("name") or "")
        if not name:
            continue
        zone = str(column.get("aggregate_zone_id") or "Common")
        signal = str(column.get("base_signal") or name)
        units = str(column.get("units") or "")
        offset = column.get("offset_steps")
        if role == "prediction_target":
            temporal = f"target +{offset} step" if offset is not None else "target"
        elif offset in (None, 0):
            temporal = "lag 0"
        else:
            temporal = f"lag {abs(int(offset))}"
        label = f"{zone} | {signal} | {temporal}"
        if units:
            label += f" | {units}"
        options.append({"label": label, "value": name})
    return options


def default_signal_values(manifest: dict[str, Any]) -> list[str]:
    """Choose a compact, scientifically useful default trace set.

    Prioritize current zone temperatures and one-step targets, followed by
    anchor-time control/disturbance signals. Historical lag columns remain
    available in the selector but are not all plotted by default.
    """
    columns = [
        row for row in (manifest.get("final_columns") or [])
        if isinstance(row, dict) and row.get("name")
    ]

    def select(*, base_signal=None, role=None, offset=None):
        values = []
        for row in columns:
            if base_signal is not None and row.get("base_signal") != base_signal:
                continue
            if role is not None and row.get("temporal_role") != role:
                continue
            if offset is not None and row.get("offset_steps") != offset:
                continue
            values.append(str(row["name"]))
        return values

    preferred: list[str] = []
    preferred += select(base_signal="zone_temperature", role="model_input", offset=0)
    preferred += select(base_signal="zone_temperature", role="prediction_target", offset=1)
    preferred += select(base_signal="qac", role="model_input", offset=0)
    preferred += select(base_signal="outdoor_temperature", role="model_input", offset=0)
    preferred += select(base_signal="zic", role="model_input", offset=0)
    preferred += select(base_signal="zir", role="model_input", offset=0)
    preferred += [
        str(row["name"])
        for row in columns
        if row.get("temporal_role") == "model_input"
        and row.get("offset_steps") == 0
    ]
    preferred += [
        str(row["name"])
        for row in columns
        if row.get("temporal_role") == "prediction_target"
    ]

    return list(dict.fromkeys(value for value in preferred if value))[:12]


def partition_options(manifest: dict[str, Any]) -> list[dict[str, str]]:
    counts = manifest.get("partition_counts") or {}
    options = [
        {"label": "Included rows", "value": INCLUDED},
        {"label": "All annual rows", "value": ALL},
    ]
    preferred_order = ["train", "validation", "test", "excluded"]
    ordered = [name for name in preferred_order if name in counts]
    ordered += [str(name) for name in counts if str(name) not in ordered]
    for partition in ordered:
        options.append(
            {
                "label": f"{str(partition).title()} ({int(counts[partition]):,})",
                "value": str(partition),
            }
        )
    return options


def column_metadata(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("name")): dict(row)
        for row in (manifest.get("final_columns") or [])
        if isinstance(row, dict) and row.get("name")
    }


def load_plot_frame(
    run_ref: dict[str, Any],
    manifest_key: str,
    *,
    signals: list[str],
    partition: str = INCLUDED,
    start: str | None = None,
    end: str | None = None,
    max_points: int | None = 20_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not signals:
        raise ValueError("Select at least one Phase D signal to plot")
    row = selected_registry_row(run_ref, manifest_key)
    manifest = load_dataset_manifest(run_ref, manifest_key)
    data_path = _localize_artifact_path(row["data_path"], run_ref)
    available = {str(item.get("name")) for item in manifest.get("final_columns") or [] if isinstance(item, dict)}
    missing = [signal for signal in signals if signal not in available]
    if missing:
        raise ValueError(f"Selected signal(s) are not present in this dataset: {missing}")

    required = ["timestamp", "included", "partition"] + list(dict.fromkeys(signals))
    frame = pd.read_parquet(data_path, columns=required)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")

    if partition == INCLUDED:
        frame = frame[frame["included"].astype(bool)]
    elif partition not in (None, "", ALL):
        frame = frame[frame["partition"].astype(str) == str(partition)]

    start_ts = pd.Timestamp(start) if start else None
    end_ts = pd.Timestamp(end) if end else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("Custom plot start datetime must be <= end datetime")
    if start_ts is not None:
        frame = frame[frame["timestamp"] >= start_ts]
    if end_ts is not None:
        frame = frame[frame["timestamp"] <= end_ts]

    source_rows = len(frame)
    stride = 1
    if max_points and source_rows > int(max_points):
        stride = max(1, (source_rows + int(max_points) - 1) // int(max_points))
        frame = frame.iloc[::stride].copy()

    metadata = {
        "source_row_count": int(source_rows),
        "plotted_row_count": int(len(frame)),
        "stride": int(stride),
        "partition": partition,
        "start": start,
        "end": end,
        "max_points": max_points,
        "signals": list(signals),
        "column_metadata": column_metadata(manifest),
    }
    return frame.reset_index(drop=True), metadata


def preview_records(
    run_ref: dict[str, Any],
    manifest_key: str,
    *,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read only a bounded first batch from the selected final Parquet."""
    requested = max(1, int(limit))
    row = selected_registry_row(run_ref, manifest_key)
    manifest = load_dataset_manifest(run_ref, manifest_key)
    data_path = _localize_artifact_path(row["data_path"], run_ref)
    columns = [
        str(item.get("name"))
        for item in (manifest.get("final_columns") or [])
        if isinstance(item, dict) and item.get("name")
    ]

    parquet = pq.ParquetFile(data_path)
    batch = next(
        parquet.iter_batches(batch_size=requested, columns=columns),
        None,
    )
    if batch is None:
        return [], columns

    frame = batch.to_pandas().head(requested).copy()
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned[key] = None
            elif isinstance(value, pd.Timestamp):
                cleaned[key] = value.isoformat()
            elif hasattr(value, "item"):
                cleaned[key] = value.item()
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records, list(frame.columns)


def build_visible_plot_export(
    figure: dict[str, Any],
    *,
    file_format: str,
    run_ref: dict[str, Any],
    manifest_key: str,
) -> tuple[bytes, str]:
    fmt = str(file_format or "csv").lower()
    if fmt not in {"csv", "parquet"}:
        raise ValueError("Plot export format must be csv or parquet")
    traces = list((figure or {}).get("data") or [])
    visible_rows: list[dict[str, Any]] = []
    visible_names: list[str] = []
    hidden_names: list[str] = []
    for index, trace in enumerate(traces):
        name = str(trace.get("name") or f"trace-{index}")
        visible = trace.get("visible", True) not in {False, "legendonly"}
        if not visible:
            hidden_names.append(name)
            continue
        visible_names.append(name)
        xs, ys = list(trace.get("x") or []), list(trace.get("y") or [])
        for point_index, (x, y) in enumerate(zip(xs, ys)):
            visible_rows.append(
                {
                    "trace_index": index,
                    "trace_name": name,
                    "point_index": point_index,
                    "timestamp": x,
                    "value": y,
                }
            )
    frame = pd.DataFrame(visible_rows)
    details = dataset_details(run_ref, manifest_key)
    meta = (((figure or {}).get("layout") or {}).get("meta") or {}).get("phase_d_plot_export") or {}
    manifest = {
        "export_type": "bgirs_phase_d_visible_plot_data",
        "contract": "visible_plot_snapshot_equals_exported_data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": run_ref["campaign_id"],
        "phase_d_run_id": run_ref["phase_d_run_id"],
        "selected_dataset_manifest": manifest_key,
        "selected_format": fmt,
        "visible_trace_names": visible_names,
        "hidden_trace_names": hidden_names,
        "plot_snapshot": meta,
        "dataset_registry_row": details["registry"],
    }
    readme = (
        "BGIRS Phase D visible plot data export\n\n"
        "This ZIP contains exactly the traces visible in the displayed Phase D plot. "
        "Hidden traces are recorded in selection_manifest.json but are not exported as plotted data. "
        "Use the separate selected-dataset download for the complete scientific Parquet.\n"
    )
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if fmt == "csv":
            archive.writestr("data/plotted_data.csv", frame.to_csv(index=False))
        else:
            buffer = BytesIO()
            frame.to_parquet(buffer, index=False)
            archive.writestr("data/plotted_data.parquet", buffer.getvalue())
        archive.writestr("selection_manifest.json", json.dumps(manifest, indent=2, default=str))
        archive.writestr("README.txt", readme)
    filename = f"{run_ref['phase_d_run_id']}__thermal_model_data__visible_plot_data_{fmt}.zip"
    return payload.getvalue(), filename


def build_selected_dataset_export(
    run_ref: dict[str, Any],
    manifest_key: str,
) -> tuple[bytes, str]:
    row = selected_registry_row(run_ref, manifest_key)
    data_path = _localize_artifact_path(row["data_path"], run_ref)
    manifest_path = _localize_artifact_path(row["manifest_path"], run_ref)
    payload = BytesIO()
    selection = {
        "contract": "selected_dataset_equals_exported_dataset",
        "campaign_id": run_ref["campaign_id"],
        "phase_d_run_id": run_ref["phase_d_run_id"],
        "dataset_registry_row": row,
    }
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(data_path, "data.parquet")
        archive.write(manifest_path, "manifest.json")
        archive.writestr("selection_manifest.json", json.dumps(selection, indent=2, default=str))
        archive.writestr(
            "README.txt",
            "Exact selected Phase D thermal-model dataset. data.parquet is the authoritative final realization and manifest.json describes its scientific contract.\n",
        )
    safe_mode = str(row.get("mode") or "dataset").replace("dependent", "dep")
    filename = (
        f"{run_ref['phase_d_run_id']}__{row.get('silo')}__{safe_mode}"
        f"__l{row.get('input_lag')}_h{row.get('target_horizon')}__selected_dataset.zip"
    )
    return payload.getvalue(), filename


def build_run_summary_export(run_ref: dict[str, Any]) -> tuple[bytes, str]:
    payload = BytesIO()
    root = Path(run_ref["run_root"])
    names = [RUN_MANIFEST, RUN_PLAN, AGGREGATION_REGISTRY, DATASET_REGISTRY, FAILURES]
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = root / name
            if path.is_file():
                archive.write(path, name)
        archive.writestr(
            "README.txt",
            "Compact Phase D campaign-run summary. This download intentionally excludes the individual large final dataset Parquets.\n",
        )
    return payload.getvalue(), f"{run_ref['phase_d_run_id']}__phase_d_summary.zip"
