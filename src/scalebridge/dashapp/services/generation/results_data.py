from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

import pandas as pd

from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


def campaign_options():
    root = resolve_generated_data_root() / "campaigns"
    if not root.is_dir():
        return []
    return [
        {"label": p.name, "value": p.name}
        for p in sorted(root.iterdir())
        if (p / "generation" / "cases").is_dir()
    ]


def campaign_index(campaign_id):
    root = resolve_generated_data_root() / "campaigns" / campaign_id / "generation" / "cases"
    rows = []
    if not root.is_dir():
        return rows

    for case_root in sorted(p for p in root.iterdir() if p.is_dir()):
        latest = case_root / "latest_run.json"
        if not latest.is_file():
            continue
        try:
            lp = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_id = str(lp.get("run_id", ""))
        run_root = case_root / "runs" / run_id
        manifest = run_root / "run_manifest.json"
        payload = {}
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                payload = {}

        spec = payload.get("case_spec", {})
        tags = spec.get("tags", {})
        run_period = spec.get("run_period", {})
        year = run_period.get("calendar_year") or 2013
        variable_manifest = run_root / "canonical" / "variable_manifest.json"
        artifacts = []
        if variable_manifest.is_file():
            try:
                artifacts = json.loads(variable_manifest.read_text(encoding="utf-8")).get(
                    "artifacts", []
                )
            except Exception:
                artifacts = []

        if artifacts:
            candidates = [
                (
                    a.get("variable_name") or a.get("variable_id"),
                    Path(a.get("canonical_parquet_path", "")),
                    a.get("units") or a.get("unit") or "",
                    a.get("variable_id") or "",
                )
                for a in artifacts
            ]
        else:
            var_root = run_root / "canonical" / "variables"
            candidates = (
                [(pq.stem, pq, "", pq.stem) for pq in sorted(var_root.glob("*.parquet"))]
                if var_root.is_dir()
                else []
            )

        for variable_name, pq, units, variable_id in candidates:
            if not pq.is_absolute():
                pq = (run_root / pq).resolve()
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "building_type": spec.get("building_type")
                    or tags.get("source_idf_name")
                    or case_root.name,
                    "weather_location": spec.get("weather_location")
                    or tags.get("source_weather_name")
                    or "",
                    "case_id": case_root.name,
                    "run_id": run_id,
                    "variable_name": str(variable_name),
                    "variable_id": str(variable_id),
                    "units": str(units),
                    "parquet_path": str(pq),
                    "status": lp.get("status", ""),
                    "calendar_year": int(year),
                }
            )
    return rows


def filter_index_rows(
    rows,
    *,
    building_types=None,
    weather_locations=None,
    case_ids=None,
    run_ids=None,
    variable_names=None,
):
    selected = list(rows or [])
    for key, values in (
        ("building_type", building_types),
        ("weather_location", weather_locations),
        ("case_id", case_ids),
        ("run_id", run_ids),
        ("variable_name", variable_names),
    ):
        if values:
            selected = [row for row in selected if row.get(key) in values]
    return selected


@lru_cache(maxsize=2048)
def _cached_key_values(path_text: str, mtime_ns: int) -> tuple[str, ...]:
    path = Path(path_text)
    try:
        frame = pd.read_parquet(path, columns=["key_value"])
    except Exception:
        frame = pd.read_parquet(path)
        if "key_value" not in frame.columns:
            return ("*",)
    values = sorted(
        {
            str(value)
            for value in frame["key_value"].dropna().astype(str)
            if str(value).strip()
        },
        key=str.casefold,
    )
    return tuple(values or ["*"])


def discover_key_values(rows) -> list[str]:
    """Return the union of EnergyPlus key_value columns in selected variable artifacts."""
    values: set[str] = set()
    for row in rows or []:
        path = Path(row["parquet_path"])
        if not path.is_file():
            continue
        values.update(_cached_key_values(str(path.resolve()), path.stat().st_mtime_ns))
    return sorted(values, key=str.casefold)


def _parse_timestamp_raw(series, year):
    text = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    is_24 = text.str.contains(r"\s24:", regex=True)
    normalized = text.str.replace(r"\s24:", " 00:", regex=True)
    parsed = pd.to_datetime(
        str(year) + "/" + normalized,
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
    )
    parsed.loc[is_24 & parsed.notna()] = (
        parsed.loc[is_24 & parsed.notna()] + pd.Timedelta(days=1)
    )
    return parsed


def _time_value_columns(df):
    if "timestamp_raw" in df.columns:
        return "timestamp_raw", "value" if "value" in df.columns else None
    time_candidates = ["timestamp", "datetime", "date_time", "Date/Time", "time"]
    t = next((c for c in time_candidates if c in df.columns), None)
    if t is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                t = c
                break
    numeric = [
        c for c in df.columns if c != t and pd.api.types.is_numeric_dtype(df[c])
    ]
    v = next(
        (c for c in ("value", "Value", "variable_value") if c in numeric),
        numeric[0] if numeric else None,
    )
    if t is None or v is None:
        raise ValueError(f"Could not infer timestamp/value columns from {list(df.columns)}")
    return t, v


def _load_one_series(row, start=None, end=None, key_values=None):
    df = pd.read_parquet(row["parquet_path"])
    if "key_value" not in df.columns:
        df = df.copy()
        df["key_value"] = "*"
    df["key_value"] = df["key_value"].fillna("*").astype(str)
    if key_values:
        selected_keys = {str(value) for value in key_values}
        df = df[df["key_value"].isin(selected_keys)]

    t, v = _time_value_columns(df)
    timestamp = (
        _parse_timestamp_raw(df[t], row.get("calendar_year", 2013))
        if t == "timestamp_raw"
        else pd.to_datetime(df[t], errors="coerce")
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "key_value": df["key_value"].astype(str),
            "value": pd.to_numeric(df[v], errors="coerce"),
        }
    ).dropna(subset=["timestamp"])
    detected_units = ""
    if "units" in df.columns:
        nonempty_units = df["units"].dropna().astype(str)
        if not nonempty_units.empty:
            detected_units = nonempty_units.iloc[0]
    frame.attrs["units"] = detected_units
    if start:
        frame = frame[frame.timestamp >= pd.Timestamp(start)]
    if end:
        frame = frame[frame.timestamp <= pd.Timestamp(end)]
    return frame.reset_index(drop=True)


def load_series(rows, start=None, end=None, key_values=None):
    """Load only selected EnergyPlus key-value columns and keep each as its own trace."""
    frames = []
    for row in rows:
        frame = _load_one_series(row, start=start, end=end, key_values=key_values)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["series"] = frame["key_value"].map(
            lambda key: (
                f"{row['building_type']} | {row['weather_location']} | "
                f"{row['case_id']} | {row['run_id']} | {row['variable_name']} | {key}"
            )
        )
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["timestamp", "key_value", "value", "series"])
    )


def _safe_component(value: object, *, max_length: int = 100) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._") or "unknown"
    return text[:max_length]


def _signal_export_name(row, key_value: str, suffix: str) -> str:
    parts = [
        row.get("building_type"),
        row.get("weather_location"),
        row.get("case_id"),
        row.get("run_id"),
        row.get("variable_name"),
        key_value,
    ]
    return "__".join(_safe_component(part) for part in parts) + suffix


def _export_frame(frame: pd.DataFrame, row: dict, key_value: str) -> pd.DataFrame:
    detected_units = frame.attrs.get("units", "")
    return pd.DataFrame(
        {
            "campaign_id": row.get("campaign_id", ""),
            "building_type": row.get("building_type", ""),
            "weather_location": row.get("weather_location", ""),
            "case_id": row.get("case_id", ""),
            "run_id": row.get("run_id", ""),
            "variable_name": row.get("variable_name", ""),
            "variable_id": row.get("variable_id", ""),
            "key_value": key_value,
            "units": row.get("units", "") or detected_units,
            "timestamp": frame["timestamp"].to_numpy(),
            "value": frame["value"].to_numpy(),
        }
    )


def build_selected_data_export(
    *,
    campaign_id: str,
    rows,
    export_format: str,
    range_mode: str,
    start=None,
    end=None,
    key_values=None,
):
    """Build a ZIP containing only the currently selected variable/key series."""
    if not rows:
        raise ValueError("No generated variables are selected for export.")
    if key_values is None:
        key_values = discover_key_values(rows)
    elif not key_values:
        raise ValueError("Select at least one variable column / key for export.")
    if export_format not in {"csv", "parquet"}:
        raise ValueError("export_format must be 'csv' or 'parquet'")
    if range_mode not in {"full", "custom"}:
        raise ValueError("range_mode must be 'full' or 'custom'")

    selected_keys = [str(value) for value in key_values]
    effective_start = start if range_mode == "custom" else None
    effective_end = end if range_mode == "custom" else None
    if effective_start and effective_end and pd.Timestamp(effective_end) < pd.Timestamp(effective_start):
        raise ValueError("Custom end datetime must not precede start datetime.")

    manifest_signals = []
    combined_frames = []
    archive_buffer = BytesIO()
    suffix = ".csv" if export_format == "csv" else ".parquet"

    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names: set[str] = set()
        signal_index = 0
        for row in rows:
            frame = _load_one_series(
                row,
                start=effective_start,
                end=effective_end,
                key_values=selected_keys,
            )
            if frame.empty:
                continue
            effective_units = row.get("units", "") or frame.attrs.get("units", "")
            for key_value, key_frame in frame.groupby("key_value", sort=True):
                key_value = str(key_value)
                if key_value not in selected_keys:
                    continue
                signal_index += 1
                key_frame = key_frame.reset_index(drop=True)
                key_frame.attrs["units"] = effective_units
                exported = _export_frame(key_frame, row, key_value)
                combined_frames.append(exported)

                base_name = _signal_export_name(row, key_value, suffix)
                archive_name = f"signals/{base_name}"
                if archive_name.casefold() in used_names:
                    stem = base_name[: -len(suffix)]
                    archive_name = f"signals/{stem}__{signal_index:03d}{suffix}"
                used_names.add(archive_name.casefold())

                if export_format == "csv":
                    archive.writestr(archive_name, exported.to_csv(index=False))
                else:
                    parquet_buffer = BytesIO()
                    exported.to_parquet(parquet_buffer, index=False)
                    archive.writestr(archive_name, parquet_buffer.getvalue())

                manifest_signals.append(
                    {
                        "campaign_id": campaign_id,
                        "building_type": row.get("building_type", ""),
                        "weather_location": row.get("weather_location", ""),
                        "case_id": row.get("case_id", ""),
                        "run_id": row.get("run_id", ""),
                        "variable_name": row.get("variable_name", ""),
                        "variable_id": row.get("variable_id", ""),
                        "key_value": key_value,
                        "units": effective_units,
                        "source_parquet_path": row.get("parquet_path", ""),
                        "calendar_year": row.get("calendar_year"),
                        "export_file": archive_name,
                        "row_count": int(len(exported)),
                        "exported_timestamp_min": (
                            exported["timestamp"].min().isoformat()
                            if not exported.empty
                            else None
                        ),
                        "exported_timestamp_max": (
                            exported["timestamp"].max().isoformat()
                            if not exported.empty
                            else None
                        ),
                    }
                )

        if not manifest_signals:
            raise ValueError("The selected variable columns / keys contain no data.")

        combined = pd.concat(combined_frames, ignore_index=True)
        combined_name = f"selected_signals_combined{suffix}"
        if export_format == "csv":
            archive.writestr(combined_name, combined.to_csv(index=False))
        else:
            parquet_buffer = BytesIO()
            combined.to_parquet(parquet_buffer, index=False)
            archive.writestr(combined_name, parquet_buffer.getvalue())

        manifest = {
            "schema_version": "0.2.0",
            "export_type": "bgirs_generation_selected_data",
            "campaign_id": campaign_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "export_format": export_format,
            "range_mode": range_mode,
            "requested_start_datetime": effective_start,
            "requested_end_datetime": effective_end,
            "selected_key_values": selected_keys,
            "signal_series_count": len(manifest_signals),
            "combined_row_count": int(len(combined)),
            "combined_file": combined_name,
            "nomenclature": {
                "series_label": "Building | Weather | Case | Run | Variable | Key",
                "signal_filename": (
                    "<building>__<weather>__<case_id>__<run_id>__<variable_name>__<key_value>"
                    f"{suffix}"
                ),
            },
            "signals": manifest_signals,
        }
        archive.writestr(
            "selection_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    filename = f"{_safe_component(campaign_id)}__generation_selected_data__{export_format}.zip"
    return archive_buffer.getvalue(), filename
