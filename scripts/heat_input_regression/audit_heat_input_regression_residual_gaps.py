#!/usr/bin/env python
"""Deep residual-gap audit for ScaleBridge heat-input regression C8 outputs.

This script is intentionally adaptive because C8 artifact layouts may evolve.
It discovers inference artifacts, existing missing-value audit files, C2 feature
parquets, and manifests. It then produces a consolidated evidence package for
every residual affected timestamp.

Outputs
-------
residual_gap_events.csv
residual_gap_component_summary.csv
residual_gap_timestamp_summary.csv
residual_gap_source_family_summary.csv
residual_gap_neighbor_context.csv
residual_gap_recommendations.csv
residual_gap_audit_manifest.json
residual_gap_audit_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TIME_NAMES = (
    "timestamp",
    "datetime",
    "date_time",
    "time",
    "timestamp_raw",
)

PREDICTION_TOKENS = (
    "prediction",
    "predicted",
    "y_pred",
    "estimate",
)

TARGET_TOKENS = (
    "target",
    "actual",
    "observed",
    "truth",
    "y_true",
)

PREDICTOR_TOKENS = (
    "predictor",
    "feature",
    "input",
)

META_COLUMNS = {
    "case_id",
    "campaign_id",
    "aggregation_id",
    "aggregation_run_id",
    "aggregate_zone_id",
    "zone_id",
    "component_id",
    "model_id",
    "estimator",
    "device",
    "split",
    "row_index",
}


def _safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not math.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
    raise TypeError(type(value).__name__)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table: {path}")


def _find_time_column(frame: pd.DataFrame) -> str | None:
    lowered = {str(c).lower(): str(c) for c in frame.columns}
    for candidate in TIME_NAMES:
        if candidate in lowered:
            return lowered[candidate]
    for col in frame.columns:
        name = str(col).lower()
        if "timestamp" in name or "datetime" in name:
            return str(col)
    return None


def _parse_time(series: pd.Series) -> pd.Series:
    """Parse canonical ISO timestamps and EnergyPlus month/day timestamps."""

    cleaned = series.astype(str).str.strip()

    parsed = pd.to_datetime(
        cleaned,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    unresolved = parsed.isna()
    if unresolved.any():
        iso_alt = pd.to_datetime(
            cleaned[unresolved],
            format="%Y-%m-%dT%H:%M:%S",
            errors="coerce",
        )
        parsed.loc[unresolved] = iso_alt

    unresolved = parsed.isna()
    if unresolved.any():
        energyplus = cleaned[unresolved].str.replace(r"\s+", " ", regex=True)
        md = pd.to_datetime(
            "2001/" + energyplus,
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce",
        )
        parsed.loc[unresolved] = md

    unresolved = parsed.isna()
    if unresolved.any():
        parsed.loc[unresolved] = pd.to_datetime(
            cleaned[unresolved],
            errors="coerce",
            format="mixed",
        )

    return parsed


def _classify_source_family(name: str) -> str:
    n = name.lower()
    if any(t in n for t in ("ghi", "dni", "dhi", "solar", "irradiance", "sun")):
        return "weather_solar"
    if "schedule" in n or any(
        t in n for t in ("people", "lights", "equipment", "occup")
    ):
        return "schedule_internal_gain"
    if any(
        t in n
        for t in (
            "hvac",
            "mass_flow",
            "node_temperature",
            "sensible",
            "cooling",
            "heating",
            "supply",
            "air_system",
        )
    ):
        return "hvac"
    if any(t in n for t in ("temperature", "humidity", "pressure", "wind")):
        return "weather_environment"
    if any(t in n for t in PREDICTION_TOKENS):
        return "model_prediction"
    return "other"


def _recommend(family: str, missing_run_length: int, neighbors_available: bool) -> tuple[str, str]:
    if family in {"weather_solar", "weather_environment"}:
        if missing_run_length <= 2 and neighbors_available:
            return (
                "interpolate_time",
                "Short isolated environmental gap with valid bracketing neighbors.",
            )
        return (
            "inspect_and_interpolate",
            "Environmental gap requires continuity and boundary checks before filling.",
        )
    if family == "schedule_internal_gain":
        if missing_run_length <= 2 and neighbors_available:
            return (
                "calendar_neighbor_or_linear_fill",
                "Schedule-derived value is isolated; preserve schedule pattern and level.",
            )
        return (
            "reconstruct_from_schedule_source",
            "Prefer rebuilding from schedule fraction and nominal level.",
        )
    if family == "hvac":
        return (
            "inspect_physical_source_before_fill",
            "HVAC variables should not be blindly interpolated or set to zero.",
        )
    return (
        "manual_source_review",
        "Source family is ambiguous or derived from multiple upstream signals.",
    )


def _discover_existing_audit_tables(inference_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in inference_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        name = path.name.lower()
        if "missing" in name or "audit" in name or "root_cause" in name:
            candidates.append(path)
    return sorted(candidates)


def _extract_audit_events(path: Path) -> pd.DataFrame:
    try:
        frame = _read_table(path)
    except Exception:
        return pd.DataFrame()

    if frame.empty:
        return frame

    time_col = _find_time_column(frame)
    if time_col is None:
        return pd.DataFrame()

    frame = frame.copy()
    frame["__timestamp"] = _parse_time(frame[time_col])
    frame = frame[frame["__timestamp"].notna()].copy()
    if frame.empty:
        return frame

    # Keep rows that explicitly indicate missing values when such columns exist.
    missing_indicator_cols = [
        c for c in frame.columns
        if any(token in str(c).lower() for token in ("missing", "is_nan", "nan_mask"))
    ]
    if missing_indicator_cols:
        mask = pd.Series(False, index=frame.index)
        for col in missing_indicator_cols:
            values = frame[col]
            if values.dtype == bool:
                mask |= values.fillna(False)
            else:
                text = values.astype(str).str.lower()
                mask |= text.isin({"true", "1", "yes", "missing", "nan"})
                numeric = pd.to_numeric(values, errors="coerce")
                mask |= numeric.fillna(0).gt(0)
        if mask.any():
            frame = frame[mask].copy()

    frame["__source_table"] = str(path)
    return frame


def _discover_inference_tables(inference_root: Path) -> list[Path]:
    candidates = []
    for path in inference_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        name = path.name.lower()
        if any(t in name for t in ("prediction", "inference", "full_year", "annual")):
            if "validation" not in name and "summary" not in name:
                candidates.append(path)
    return sorted(candidates)


def _infer_zone_component(path: Path, frame: pd.DataFrame) -> tuple[str, str]:
    zone = ""
    component = ""

    for col in ("aggregate_zone_id", "zone_id", "zone"):
        if col in frame.columns and frame[col].notna().any():
            zone = str(frame[col].dropna().iloc[0])
            break

    for col in ("component_id", "model_id", "component", "target_id"):
        if col in frame.columns and frame[col].notna().any():
            component = str(frame[col].dropna().iloc[0])
            break

    parts = list(path.parts)
    known_components = {
        "QAC", "QSol1", "QSol2", "QZic_P", "QZir_P", "QZic_L", "QZir_L",
        "QZic_EE", "QZir_EE", "QZir_GE", "QZivr_L",
    }
    if not component:
        for part in reversed(parts):
            if part in known_components:
                component = part
                break
    if not zone:
        for part in reversed(parts):
            if part in {"RestaurantFastFood_All", "Dining", "Kitchen"}:
                zone = part
                break

    return zone, component


def _events_from_inference_table(path: Path) -> list[dict[str, Any]]:
    try:
        frame = _read_table(path)
    except Exception:
        return []
    if frame.empty:
        return []

    time_col = _find_time_column(frame)
    if time_col is None:
        return []

    timestamps = _parse_time(frame[time_col])
    zone, component = _infer_zone_component(path, frame)

    numeric_cols = [
        str(c) for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and str(c) not in META_COLUMNS
    ]
    pred_cols = [
        c for c in numeric_cols
        if any(token in c.lower() for token in PREDICTION_TOKENS)
    ]
    if not pred_cols:
        # Fall back to columns whose names resemble components or predicted terms.
        pred_cols = [
            c for c in numeric_cols
            if c.lower().startswith("pred") or c in {
                "QAC", "QSol1", "QSol2", "QZic_P", "QZir_P", "QZic_L",
                "QZir_L", "QZic_EE", "QZir_EE", "QZir_GE", "QZivr_L",
            }
        ]

    predictor_cols = [
        c for c in numeric_cols
        if c not in pred_cols and any(token in c.lower() for token in PREDICTOR_TOKENS)
    ]

    events: list[dict[str, Any]] = []
    for pred_col in pred_cols:
        missing_mask = frame[pred_col].isna()
        for idx in frame.index[missing_mask]:
            missing_predictors = [
                c for c in predictor_cols if pd.isna(frame.at[idx, c])
            ]
            events.append({
                "aggregate_zone_id": zone,
                "component_id": component or pred_col,
                "timestamp": timestamps.loc[idx],
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "prediction_column": pred_col,
                "missing_predictor_columns": ";".join(missing_predictors),
                "missing_predictor_count": len(missing_predictors),
                "inference_table": str(path),
                "detection_method": "inference_table_prediction_nan",
            })
    return events


def _discover_feature_parquets(feature_root: Path) -> list[Path]:
    return sorted(feature_root.rglob("derived_heat_input_features.parquet"))


def _zone_from_feature_path(path: Path) -> str:
    return path.parent.name


def _build_feature_index(feature_root: Path) -> dict[str, tuple[Path, pd.DataFrame, str]]:
    index: dict[str, tuple[Path, pd.DataFrame, str]] = {}
    for path in _discover_feature_parquets(feature_root):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        time_col = _find_time_column(frame)
        if time_col is None:
            continue
        frame = frame.copy()
        frame["__timestamp"] = _parse_time(frame[time_col])
        index[_zone_from_feature_path(path)] = (path, frame, time_col)
    return index


def _neighbor_context(
    events: pd.DataFrame,
    feature_index: dict[str, tuple[Path, pd.DataFrame, str]],
    radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []

    for _, event in events.iterrows():
        zone = str(event.get("aggregate_zone_id", ""))
        timestamp = pd.to_datetime(event.get("timestamp"), errors="coerce")
        missing_cols = [
            c for c in str(event.get("missing_predictor_columns", "")).split(";") if c
        ]

        enriched = event.to_dict()
        enriched["feature_table"] = ""
        enriched["feature_row_found"] = False
        enriched["feature_missing_columns"] = ""
        enriched["source_family"] = "other"
        enriched["missing_run_length"] = 1
        enriched["bracketing_neighbors_available"] = False

        if zone not in feature_index or pd.isna(timestamp):
            enriched_rows.append(enriched)
            continue

        path, frame, _ = feature_index[zone]
        matches = frame.index[frame["__timestamp"] == timestamp].tolist()
        if not matches:
            enriched["feature_table"] = str(path)
            enriched_rows.append(enriched)
            continue

        pos = frame.index.get_loc(matches[0])
        if isinstance(pos, slice):
            pos = pos.start
        if not isinstance(pos, (int, np.integer)):
            pos = int(np.flatnonzero(frame.index == matches[0])[0])

        row = frame.iloc[pos]
        actual_missing = [
            str(c) for c in frame.columns
            if c != "__timestamp" and pd.isna(row[c])
        ]

        relevant_missing = missing_cols or actual_missing
        family_counts = Counter(_classify_source_family(c) for c in relevant_missing)
        source_family = (
            family_counts.most_common(1)[0][0] if family_counts else "other"
        )

        neighbor_positions = range(max(0, pos - radius), min(len(frame), pos + radius + 1))
        for npos in neighbor_positions:
            nrow = frame.iloc[npos]
            offset = npos - pos
            columns_to_report = relevant_missing or [
                c for c in frame.columns
                if c != "__timestamp" and pd.api.types.is_numeric_dtype(frame[c])
            ][:12]
            for col in columns_to_report:
                if col not in frame.columns:
                    continue
                context_rows.append({
                    "aggregate_zone_id": zone,
                    "component_id": event.get("component_id", ""),
                    "affected_timestamp": timestamp,
                    "neighbor_offset_steps": offset,
                    "neighbor_timestamp": nrow["__timestamp"],
                    "column": col,
                    "value": nrow[col],
                    "is_missing": pd.isna(nrow[col]),
                    "source_family": _classify_source_family(col),
                    "feature_table": str(path),
                })

        neighbors_available = False
        if relevant_missing and pos > 0 and pos < len(frame) - 1:
            neighbors_available = all(
                col in frame.columns
                and pd.notna(frame.iloc[pos - 1][col])
                and pd.notna(frame.iloc[pos + 1][col])
                for col in relevant_missing
            )

        enriched.update({
            "feature_table": str(path),
            "feature_row_found": True,
            "feature_missing_columns": ";".join(actual_missing),
            "source_family": source_family,
            "bracketing_neighbors_available": neighbors_available,
        })
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows), pd.DataFrame(context_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--neighbor-radius", type=int, default=2)
    args = parser.parse_args()

    inference_root = Path(args.inference_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not inference_root.is_dir():
        raise FileNotFoundError(inference_root)
    if not feature_root.is_dir():
        raise FileNotFoundError(feature_root)

    print("=" * 100)
    print("SCALEBRIDGE C8 RESIDUAL-GAP DEEP AUDIT")
    print("=" * 100)
    print(f"inference_root: {inference_root}")
    print(f"feature_root: {feature_root}")
    print(f"output_root: {output_root}")

    audit_tables = _discover_existing_audit_tables(inference_root)
    audit_frames = [_extract_audit_events(p) for p in audit_tables]
    audit_frames = [f for f in audit_frames if not f.empty]

    inference_tables = _discover_inference_tables(inference_root)
    inferred_events = []
    for path in inference_tables:
        inferred_events.extend(_events_from_inference_table(path))

    events = pd.DataFrame(inferred_events)

    # If the adaptive inference scan did not recover event rows, derive timestamp
    # and metadata rows from existing audit tables.
    if events.empty and audit_frames:
        combined = pd.concat(audit_frames, ignore_index=True, sort=False)
        events = pd.DataFrame({
            "aggregate_zone_id": combined.get(
                "aggregate_zone_id", combined.get("zone_id", "")
            ),
            "component_id": combined.get(
                "component_id", combined.get("model_id", "")
            ),
            "timestamp": combined["__timestamp"],
            "row_index": combined.get("row_index", ""),
            "prediction_column": combined.get("prediction_column", ""),
            "missing_predictor_columns": combined.get(
                "missing_predictor_columns",
                combined.get("predictor_column", ""),
            ),
            "missing_predictor_count": combined.get(
                "missing_predictor_count", 1
            ),
            "inference_table": combined["__source_table"],
            "detection_method": "existing_missing_value_audit",
        })

    if events.empty:
        raise RuntimeError(
            "No residual missing-value events could be recovered from the "
            "inference artifacts or existing C8 audit outputs."
        )

    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events = events[events["timestamp"].notna()].copy()
    events = events.drop_duplicates(
        subset=[
            "aggregate_zone_id",
            "component_id",
            "timestamp",
            "prediction_column",
        ]
    ).sort_values(
        ["timestamp", "aggregate_zone_id", "component_id"]
    )

    feature_index = _build_feature_index(feature_root)
    enriched, context = _neighbor_context(
        events,
        feature_index,
        radius=args.neighbor_radius,
    )

    recommendations = []
    for _, row in enriched.iterrows():
        action, rationale = _recommend(
            str(row.get("source_family", "other")),
            int(row.get("missing_run_length", 1)),
            bool(row.get("bracketing_neighbors_available", False)),
        )
        recommendations.append({
            "aggregate_zone_id": row.get("aggregate_zone_id", ""),
            "component_id": row.get("component_id", ""),
            "timestamp": row.get("timestamp"),
            "source_family": row.get("source_family", ""),
            "recommended_action": action,
            "rationale": rationale,
            "feature_missing_columns": row.get("feature_missing_columns", ""),
            "bracketing_neighbors_available": row.get(
                "bracketing_neighbors_available", False
            ),
        })
    recommendations_df = pd.DataFrame(recommendations)

    component_summary = (
        enriched.groupby(
            ["aggregate_zone_id", "component_id", "source_family"],
            dropna=False,
        )
        .agg(
            missing_event_count=("timestamp", "size"),
            unique_affected_timestamp_count=("timestamp", "nunique"),
            feature_rows_found=("feature_row_found", "sum"),
        )
        .reset_index()
    )

    timestamp_summary = (
        enriched.groupby("timestamp", dropna=False)
        .agg(
            affected_zone_count=("aggregate_zone_id", "nunique"),
            affected_component_count=("component_id", "nunique"),
            event_count=("timestamp", "size"),
            source_families=(
                "source_family",
                lambda s: ";".join(sorted(set(map(str, s)))),
            ),
            feature_missing_columns=(
                "feature_missing_columns",
                lambda s: ";".join(
                    sorted({
                        token
                        for value in s.astype(str)
                        for token in value.split(";")
                        if token
                    })
                ),
            ),
        )
        .reset_index()
    )

    source_summary = (
        enriched.groupby("source_family", dropna=False)
        .agg(
            missing_event_count=("timestamp", "size"),
            unique_affected_timestamp_count=("timestamp", "nunique"),
            affected_zone_count=("aggregate_zone_id", "nunique"),
            affected_component_count=("component_id", "nunique"),
        )
        .reset_index()
    )

    outputs = {
        "events": output_root / "residual_gap_events.csv",
        "component_summary": output_root / "residual_gap_component_summary.csv",
        "timestamp_summary": output_root / "residual_gap_timestamp_summary.csv",
        "source_family_summary": output_root / "residual_gap_source_family_summary.csv",
        "neighbor_context": output_root / "residual_gap_neighbor_context.csv",
        "recommendations": output_root / "residual_gap_recommendations.csv",
        "manifest": output_root / "residual_gap_audit_manifest.json",
        "report": output_root / "residual_gap_audit_report.txt",
    }

    enriched.to_csv(outputs["events"], index=False)
    component_summary.to_csv(outputs["component_summary"], index=False)
    timestamp_summary.to_csv(outputs["timestamp_summary"], index=False)
    source_summary.to_csv(outputs["source_family_summary"], index=False)
    context.to_csv(outputs["neighbor_context"], index=False)
    recommendations_df.to_csv(outputs["recommendations"], index=False)

    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "inference_root": inference_root,
        "feature_root": feature_root,
        "output_root": output_root,
        "existing_audit_table_count": len(audit_tables),
        "inference_table_count": len(inference_tables),
        "feature_zone_count": len(feature_index),
        "residual_event_count": len(enriched),
        "unique_affected_timestamp_count": int(enriched["timestamp"].nunique()),
        "affected_zone_count": int(enriched["aggregate_zone_id"].nunique()),
        "affected_component_count": int(enriched["component_id"].nunique()),
        "feature_row_found_count": int(enriched["feature_row_found"].sum()),
        "outputs": outputs,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, default=_safe_json),
        encoding="utf-8",
    )

    report_lines = [
        "SCALEBRIDGE C8 RESIDUAL-GAP DEEP AUDIT",
        "=" * 100,
        f"inference_root: {inference_root}",
        f"feature_root: {feature_root}",
        f"output_root: {output_root}",
        "",
        f"residual_event_count: {len(enriched)}",
        f"unique_affected_timestamp_count: {enriched['timestamp'].nunique()}",
        f"affected_zone_count: {enriched['aggregate_zone_id'].nunique()}",
        f"affected_component_count: {enriched['component_id'].nunique()}",
        f"feature_row_found_count: {int(enriched['feature_row_found'].sum())}",
        "",
        "TIMESTAMP SUMMARY",
        "-" * 100,
        timestamp_summary.to_string(index=False),
        "",
        "SOURCE-FAMILY SUMMARY",
        "-" * 100,
        source_summary.to_string(index=False),
        "",
        "RECOMMENDATIONS",
        "-" * 100,
        recommendations_df.to_string(index=False),
        "",
        "Interpretation rule:",
        "Do not apply a blanket fill. Apply treatment by source family after reviewing neighbor context.",
    ]
    outputs["report"].write_text("\n".join(report_lines), encoding="utf-8")

    print()
    print("=" * 100)
    print("RESIDUAL-GAP AUDIT SUMMARY")
    print("=" * 100)
    print(f"residual_event_count: {len(enriched)}")
    print(f"unique_affected_timestamp_count: {enriched['timestamp'].nunique()}")
    print(f"affected_zone_count: {enriched['aggregate_zone_id'].nunique()}")
    print(f"affected_component_count: {enriched['component_id'].nunique()}")
    print(f"feature_row_found_count: {int(enriched['feature_row_found'].sum())}")
    print(f"report: {outputs['report']}")
    print(f"manifest: {outputs['manifest']}")
    print("status: completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
