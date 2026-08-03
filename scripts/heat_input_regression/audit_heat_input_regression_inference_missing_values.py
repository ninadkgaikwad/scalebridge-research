# -*- coding: utf-8 -*-
"""Audit the root causes of missing C8 full-year heat-input predictions.

The audit starts from each C8 zone manifest and traces intentional prediction
NaNs to the C2 predictor columns. It compares missing masks across all derived
features, identifies common timestamp gaps, and attempts to inspect the raw
source columns listed in the C2 derived-feature catalog under the aggregation
zone root.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw: Any, base: Path) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def split_source_columns(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def classify_pattern(
    predictor_missing: np.ndarray,
    feature_missing_masks: dict[str, np.ndarray],
) -> tuple[str, int, int]:
    if not predictor_missing.any():
        return "no_missing_values", 0, 0
    if not feature_missing_masks:
        return "predictor_missing_source_features_unavailable", int(predictor_missing.sum()), 0
    all_missing = np.logical_and.reduce(list(feature_missing_masks.values()))
    common_count = int((predictor_missing & all_missing).sum())
    specific_count = int((predictor_missing & ~all_missing).sum())
    if common_count == int(predictor_missing.sum()):
        return "common_full_feature_timestamp_gap", common_count, specific_count
    if common_count > 0:
        return "common_gap_plus_predictor_specific_gap", common_count, specific_count
    return "predictor_specific_gap", common_count, specific_count


def inspect_source_columns(
    zone_root: Path,
    source_columns: list[str],
    missing_timestamps: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    file_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    if not zone_root.is_dir() or not source_columns:
        return file_rows, value_rows
    candidates = [p for p in zone_root.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".csv"}]
    for path in candidates:
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path, nrows=0)
                matching = [c for c in source_columns if c in frame.columns]
                if not matching:
                    continue
                frame = pd.read_csv(path)
            matching = [c for c in source_columns if c in frame.columns]
            if not matching:
                continue
            timestamp_candidates = [c for c in ("timestamp_raw", "timestamp", "Date/Time", "datetime") if c in frame.columns]
            file_rows.append({
                "source_file": str(path),
                "matching_source_columns": " | ".join(matching),
                "timestamp_columns": " | ".join(timestamp_candidates),
                "row_count": len(frame),
            })
            if not timestamp_candidates:
                continue
            ts_col = timestamp_candidates[0]
            ts = frame[ts_col].astype(str)
            subset = frame.loc[ts.isin(missing_timestamps), [ts_col, *matching]].copy()
            for _, row in subset.iterrows():
                for col in matching:
                    value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                    value_rows.append({
                        "source_file": str(path),
                        "timestamp_column": ts_col,
                        "timestamp_raw": str(row[ts_col]),
                        "source_column": col,
                        "source_value": value if pd.notna(value) else None,
                        "source_value_is_finite": bool(pd.notna(value) and np.isfinite(float(value))),
                    })
        except Exception as exc:
            file_rows.append({
                "source_file": str(path),
                "matching_source_columns": "",
                "timestamp_columns": "",
                "row_count": "",
                "inspection_error": f"{type(exc).__name__}: {exc}",
            })
    return file_rows, value_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--inspect-source-files", action="store_true")
    args = parser.parse_args()

    root = Path(args.inference_root).resolve()
    manifests = sorted(root.rglob("annual_component_predictions_manifest.json"))
    component_rows: list[dict[str, Any]] = []
    timestamp_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    source_file_rows: list[dict[str, Any]] = []
    source_value_rows: list[dict[str, Any]] = []

    print("=" * 100)
    print("SCALEBRIDGE C8 MISSING-VALUE ROOT-CAUSE AUDIT")
    print("=" * 100)
    print(f"inference_root: {root}")
    print(f"zone_artifact_count: {len(manifests)}")

    for index, manifest_path in enumerate(manifests, 1):
        manifest = read_json(manifest_path)
        outputs = manifest.get("outputs", {}) or {}
        pred_path = resolve_path(outputs.get("annual_component_predictions"), manifest_path.parent)
        registry_path = resolve_path(outputs.get("component_prediction_registry"), manifest_path.parent)
        feature_path = resolve_path(manifest.get("source_feature_parquet"), manifest_path.parent)
        feature_manifest_path = resolve_path(manifest.get("source_feature_manifest"), manifest_path.parent)
        feature_manifest = read_json(feature_manifest_path)
        feature_catalog_path = resolve_path((feature_manifest.get("outputs", {}) or {}).get("derived_feature_catalog"), feature_manifest_path.parent)
        features = pd.read_parquet(feature_path)
        predictions = pd.read_parquet(pred_path)
        registry = pd.read_csv(registry_path)
        feature_catalog = pd.read_csv(feature_catalog_path) if feature_catalog_path.is_file() else pd.DataFrame()
        derived_columns = [c for c in feature_manifest.get("derived_feature_columns", []) if c in features.columns]
        feature_missing_masks = {
            c: ~np.isfinite(pd.to_numeric(features[c], errors="coerce").to_numpy(dtype=float))
            for c in derived_columns
        }
        common_all_feature_missing = np.logical_and.reduce(list(feature_missing_masks.values())) if feature_missing_masks else np.zeros(len(features), dtype=bool)
        zone = str(manifest.get("aggregate_zone_id", ""))
        print(f"[{index}/{len(manifests)}] {zone} | components={len(registry)}")

        for _, reg in registry.iterrows():
            model_id = str(reg["model_id"])
            predictor = str(reg["predictor_column"])
            output_col = str(reg["output_prediction_column"])
            x = pd.to_numeric(features[predictor], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(predictions[output_col], errors="coerce").to_numpy(dtype=float)
            predictor_missing = ~np.isfinite(x)
            prediction_missing = ~np.isfinite(y)
            classification, common_count, specific_count = classify_pattern(predictor_missing, feature_missing_masks)
            catalog_match = feature_catalog.loc[feature_catalog.get("feature_name", pd.Series(dtype=str)).astype(str) == predictor] if not feature_catalog.empty and "feature_name" in feature_catalog.columns else pd.DataFrame()
            source_columns = split_source_columns(catalog_match.iloc[0].get("source_columns", "")) if len(catalog_match) else []
            missing_indices = np.flatnonzero(predictor_missing)
            missing_timestamps = set(features.iloc[missing_indices]["timestamp_raw"].astype(str))
            component_rows.append({
                "case_id": manifest.get("case_id", ""),
                "aggregation_id": manifest.get("aggregation_id", ""),
                "weight_mode": manifest.get("weight_mode", ""),
                "aggregate_zone_id": zone,
                "model_id": model_id,
                "predictor_column": predictor,
                "output_prediction_column": output_col,
                "row_count": len(features),
                "predictor_missing_count": int(predictor_missing.sum()),
                "prediction_missing_count": int(prediction_missing.sum()),
                "missing_masks_match": bool(np.array_equal(predictor_missing, prediction_missing)),
                "common_all_feature_gap_count": common_count,
                "predictor_specific_gap_count": specific_count,
                "root_cause_classification": classification,
                "source_columns": " | ".join(source_columns),
                "source_feature_parquet": str(feature_path),
                "feature_manifest": str(feature_manifest_path),
            })
            for idx in missing_indices:
                missing_features = [name for name, mask in feature_missing_masks.items() if mask[idx]]
                timestamp_rows.append({
                    "case_id": manifest.get("case_id", ""),
                    "aggregation_id": manifest.get("aggregation_id", ""),
                    "weight_mode": manifest.get("weight_mode", ""),
                    "aggregate_zone_id": zone,
                    "row_index": int(idx),
                    "timestamp_raw": str(features.iloc[idx]["timestamp_raw"]),
                    "timestamp": str(features.iloc[idx]["timestamp"]),
                    "model_id": model_id,
                    "predictor_column": predictor,
                    "output_prediction_column": output_col,
                    "missing_derived_feature_count": len(missing_features),
                    "missing_derived_features": " | ".join(missing_features),
                    "is_common_all_feature_gap": bool(common_all_feature_missing[idx]),
                    "root_cause_classification": "common_full_feature_timestamp_gap" if common_all_feature_missing[idx] else "feature_or_source_specific_gap",
                })
            if args.inspect_source_files and missing_timestamps and source_columns:
                zone_root = Path(str(feature_manifest.get("zone_root", ""))).expanduser()
                files, values = inspect_source_columns(zone_root, source_columns, missing_timestamps)
                for row in files:
                    source_file_rows.append({"aggregate_zone_id": zone, "model_id": model_id, "predictor_column": predictor, **row})
                for row in values:
                    source_value_rows.append({"aggregate_zone_id": zone, "model_id": model_id, "predictor_column": predictor, **row})

        # Timestamp-level overlap across predictors in this zone.
        missing_matrix = pd.DataFrame({
            str(reg["model_id"]): ~np.isfinite(pd.to_numeric(features[str(reg["predictor_column"])], errors="coerce").to_numpy(dtype=float))
            for _, reg in registry.iterrows()
        })
        any_missing = missing_matrix.any(axis=1)
        for idx in np.flatnonzero(any_missing.to_numpy()):
            missing_models = missing_matrix.columns[missing_matrix.iloc[idx].to_numpy(dtype=bool)].tolist()
            overlap_rows.append({
                "aggregate_zone_id": zone,
                "row_index": int(idx),
                "timestamp_raw": str(features.iloc[idx]["timestamp_raw"]),
                "timestamp": str(features.iloc[idx]["timestamp"]),
                "missing_model_count": len(missing_models),
                "missing_models": " | ".join(missing_models),
                "all_selected_models_missing": len(missing_models) == len(missing_matrix.columns),
            })

    component_df = pd.DataFrame(component_rows)
    timestamp_df = pd.DataFrame(timestamp_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    source_files_df = pd.DataFrame(source_file_rows)
    source_values_df = pd.DataFrame(source_value_rows)
    component_df.to_csv(root / "missing_value_root_cause_by_component.csv", index=False)
    timestamp_df.to_csv(root / "missing_value_root_cause_by_timestamp.csv", index=False)
    overlap_df.to_csv(root / "missing_value_overlap_by_timestamp.csv", index=False)
    source_files_df.to_csv(root / "missing_value_source_file_inventory.csv", index=False)
    source_values_df.to_csv(root / "missing_value_source_values.csv", index=False)

    summary = {
        "schema_version": "0.1.0",
        "stage": "C8_missing_value_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_root": str(root),
        "zone_artifact_count": len(manifests),
        "component_count": len(component_df),
        "components_with_missing_values": int((component_df.get("predictor_missing_count", pd.Series(dtype=int)) > 0).sum()) if len(component_df) else 0,
        "total_missing_component_values": int(component_df.get("predictor_missing_count", pd.Series(dtype=int)).sum()) if len(component_df) else 0,
        "unique_affected_timestamps": int(timestamp_df["timestamp_raw"].nunique()) if len(timestamp_df) else 0,
        "all_prediction_masks_match_predictors": bool(component_df.get("missing_masks_match", pd.Series(dtype=bool)).all()) if len(component_df) else True,
        "source_file_inspection_enabled": bool(args.inspect_source_files),
        "outputs": {
            "component_root_cause": str(root / "missing_value_root_cause_by_component.csv"),
            "timestamp_root_cause": str(root / "missing_value_root_cause_by_timestamp.csv"),
            "timestamp_overlap": str(root / "missing_value_overlap_by_timestamp.csv"),
            "source_file_inventory": str(root / "missing_value_source_file_inventory.csv"),
            "source_values": str(root / "missing_value_source_values.csv"),
        },
    }
    (root / "missing_value_root_cause_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"components_with_missing_values: {summary['components_with_missing_values']}")
    print(f"total_missing_component_values: {summary['total_missing_component_values']}")
    print(f"unique_affected_timestamps: {summary['unique_affected_timestamps']}")
    print(f"all_prediction_masks_match_predictors: {summary['all_prediction_masks_match_predictors']}")


if __name__ == "__main__":
    main()
