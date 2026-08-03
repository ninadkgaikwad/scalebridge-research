# -*- coding: utf-8 -*-
"""Numerical and provenance validation for Stage C1 readiness audits."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalebridge.data.heat_input_regression.signal_catalog import SIGNAL_DEFINITIONS, resolve_present_column


def audit_signal_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = set(str(c) for c in frame.columns)
    rows: list[dict[str, Any]] = []
    for definition in SIGNAL_DEFINITIONS:
        physical = resolve_present_column(definition.semantic_name, columns)
        row: dict[str, Any] = {
            "semantic_name": definition.semantic_name, "canonical_column": definition.canonical_column,
            "physical_column_name": physical or "", "signal_category": definition.category,
            "expected_units": definition.expected_units, "present": physical is not None,
        }
        if physical is None:
            row.update({"dtype": "", "row_count": len(frame), "non_null_count": 0, "nan_count": len(frame), "nan_fraction": 1.0 if len(frame) else "", "zero_count": 0, "zero_fraction": "", "minimum": "", "maximum": "", "mean": "", "standard_deviation": "", "unique_non_null_count": 0, "is_constant": False, "is_all_zero": False})
        else:
            numeric = pd.to_numeric(frame[physical], errors="coerce")
            nonnull = numeric.dropna(); n = len(numeric); nz = len(nonnull)
            zero_count = int((nonnull == 0.0).sum())
            row.update({
                "dtype": str(frame[physical].dtype), "row_count": n, "non_null_count": nz, "nan_count": n-nz,
                "nan_fraction": (n-nz)/n if n else "", "zero_count": zero_count, "zero_fraction": zero_count/nz if nz else "",
                "minimum": float(nonnull.min()) if nz else "", "maximum": float(nonnull.max()) if nz else "",
                "mean": float(nonnull.mean()) if nz else "", "standard_deviation": float(nonnull.std(ddof=0)) if nz else "",
                "unique_non_null_count": int(nonnull.nunique()), "is_constant": bool(nz > 0 and nonnull.nunique() <= 1),
                "is_all_zero": bool(nz > 0 and (nonnull == 0.0).all()),
            })
        rows.append(row)
    return rows


def load_provenance_sets(*, rdd_path: Path | None, source_manifest_path: Path | None, variable_manifest_path: Path | None, loaded_variables_path: Path | None = None) -> dict[str, set[str]]:
    available, unavailable, requested, generated, loaded = set(), set(), set(), set(), set()
    if rdd_path and rdd_path.is_file():
        payload = json.loads(rdd_path.read_text(encoding="utf-8"))
        available.update(str(x) for x in payload.get("available_variables", [])); unavailable.update(str(x) for x in payload.get("unavailable_variables", []))
    if source_manifest_path and source_manifest_path.is_file():
        payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        spec = payload.get("case_spec", {})
        for item in spec.get("output_variables", []) if isinstance(spec, dict) else []:
            if isinstance(item, dict) and item.get("variable_name"):
                requested.add(str(item["variable_name"]))
    if variable_manifest_path and variable_manifest_path.is_file():
        payload = json.loads(variable_manifest_path.read_text(encoding="utf-8"))
        candidates = payload.get("variables") or payload.get("records") or payload.get("artifacts") or []
        if isinstance(candidates, dict): candidates = candidates.values()
        for item in candidates:
            if isinstance(item, dict):
                name = item.get("variable_name") or item.get("name")
                status = str(item.get("status", "completed")).casefold()
                if name and status not in {"failed", "missing"}: generated.add(str(name))
    if loaded_variables_path and loaded_variables_path.is_file():
        try:
            with loaded_variables_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for item in csv.DictReader(stream):
                    name = str(item.get("variable_name", "")).strip()
                    status = str(item.get("load_status", "")).strip().casefold()
                    if name and status not in {"failed", "missing", "skipped"}:
                        loaded.add(name)
        except Exception:
            pass
    return {"rdd_available": available, "rdd_unavailable": unavailable, "requested": requested, "generated": generated, "loaded": loaded}


def validate_series_pair(predictor: pd.Series, target: pd.Series, minimum_sample_count: int) -> tuple[str, str, int]:
    pair = pd.DataFrame({"x": pd.to_numeric(predictor, errors="coerce"), "y": pd.to_numeric(target, errors="coerce")}).dropna()
    n = len(pair)
    if n < minimum_sample_count: return "invalid_insufficient_samples", f"aligned valid samples {n} < {minimum_sample_count}", n
    if pair["x"].nunique() <= 1:
        return ("invalid_all_zero_predictor" if (pair["x"] == 0).all() else "invalid_constant_predictor"), "predictor is constant", n
    if pair["y"].nunique() <= 1:
        return ("invalid_all_zero_target" if (pair["y"] == 0).all() else "invalid_constant_target"), "target is constant", n
    return "applicable", "predictor and target contain sufficient varying data", n


def evaluate_node_mapping_quality(*, summary_path: Path | None, mapping_path: Path | None, zone_mapping_path: Path, variable_label: str) -> dict[str, Any]:
    source_zone_count = 0
    try:
        zones = pd.read_csv(zone_mapping_path)
        source_zone_count = int(zones["source_zone"].dropna().astype(str).nunique())
    except Exception:
        pass
    summary = {}
    if summary_path and summary_path.is_file():
        records = pd.read_csv(summary_path).to_dict("records")
        if records: summary = records[0]
    mapped_source_zone_count = 0
    if mapping_path and mapping_path.is_file():
        mapping = pd.read_csv(mapping_path)
        mapped = mapping[mapping.get("match_status", "").astype(str) == "mapped"] if "match_status" in mapping else mapping
        if "source_zone" in mapped: mapped_source_zone_count = int(mapped["source_zone"].dropna().astype(str).nunique())
    source_keys = _int(summary.get("source_key_count")); mapped_keys = _int(summary.get("mapped_key_count")); unmapped_keys = _int(summary.get("unmapped_key_count"))
    return {
        "variable": variable_label, "source_key_count": source_keys, "mapped_key_count": mapped_keys, "unmapped_key_count": unmapped_keys,
        "raw_key_mapping_ratio": mapped_keys/source_keys if source_keys else "",
        "source_zone_count": source_zone_count, "mapped_source_zone_count": mapped_source_zone_count,
        "thermal_zone_coverage_ratio": mapped_source_zone_count/source_zone_count if source_zone_count else "",
        "mapped_row_count": _int(summary.get("mapped_row_count")), "skipped_row_count": _int(summary.get("skipped_row_count")),
    }


def _int(value: Any) -> int:
    try: return int(float(value))
    except (TypeError, ValueError): return 0
