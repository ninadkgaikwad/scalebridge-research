# -*- coding: utf-8 -*-
"""Validate a completed Stage C2 derived-feature run against Stage B inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalebridge.data.aggregation.writers import make_safe_name, write_csv, write_json
from scalebridge.data.heat_input_regression.discovery import (
    discover_from_matrix_run,
    resolve_inputs,
)
from scalebridge.data.heat_input_regression.alignment import (
    canonicalize_wide_frame,
)
from scalebridge.data.heat_input_regression.feature_engineering import (
    build_zone_derived_features,
    validate_zone_derived_features,
)

DEFAULT_CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    campaign_root = resolve_inputs(
        campaign_id=args.campaign_id,
        campaign_root=args.campaign_root,
        generated_data_root=args.generated_data_root,
    )
    audit_root = campaign_root / "heat_input_regression" / "audit_runs" / args.audit_run_id
    feature_root = campaign_root / "heat_input_regression" / "feature_runs" / args.feature_run_id
    if not audit_root.is_dir():
        raise SystemExit(f"Audit run not found: {audit_root}")
    if not feature_root.is_dir():
        raise SystemExit(f"Feature run not found: {feature_root}")

    refs, discovery_issues = discover_from_matrix_run(
        campaign_root=campaign_root,
        matrix_run_id=args.matrix_run_id,
        case_id=args.case_id,
        aggregation_id=args.aggregation_id,
        weight_mode=args.weight_mode,
        aggregate_zone_id=args.aggregate_zone_id,
    )
    if discovery_issues:
        write_csv(feature_root / "feature_validation_discovery_issues.csv", discovery_issues)

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION FEATURE VALIDATION")
    print("=" * 100)
    print(f"feature_root: {feature_root}")
    print(f"selected_aggregation_zone_count: {len(refs)}")
    print()

    rows: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        print(
            f"[{index}/{len(refs)}] {ref.case_id} | {ref.aggregation_id} | "
            f"{ref.aggregate_zone_id}"
        )
        rows.append(
            validate_one_zone(
                ref=ref,
                audit_root=audit_root,
                feature_root=feature_root,
                args=args,
            )
        )

    write_csv(feature_root / "feature_validation_results.csv", rows)
    passed_count = sum(row["validation_status"] == "passed" for row in rows)
    failed_count = len(rows) - passed_count
    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if failed_count == 0 else "completed_with_failures",
        "feature_run_id": args.feature_run_id,
        "audit_run_id": args.audit_run_id,
        "matrix_run_id": args.matrix_run_id,
        "campaign_id": args.campaign_id,
        "selected_zone_count": len(rows),
        "passed_zone_count": passed_count,
        "failed_zone_count": failed_count,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "runtime_seconds": time.perf_counter() - started,
        "feature_root": str(feature_root),
    }
    write_json(feature_root / "feature_validation_manifest.json", manifest)

    print()
    print("=" * 100)
    print("FEATURE VALIDATION SUMMARY")
    print("=" * 100)
    print(f"passed_zone_count: {passed_count}")
    print(f"failed_zone_count: {failed_count}")
    print(f"feature_root: {feature_root}")
    return 0 if failed_count == 0 else 1


def validate_one_zone(*, ref, audit_root: Path, feature_root: Path, args) -> dict[str, Any]:
    base = ref.identity_dict()
    zone_feature_root = (
        feature_root
        / "cases"
        / make_safe_name(ref.case_id)
        / make_safe_name(ref.aggregation_run_id)
        / make_safe_name(ref.aggregate_zone_id)
    )
    try:
        manifest_path = zone_feature_root / "zone_feature_manifest.json"
        feature_path = zone_feature_root / "derived_heat_input_features.parquet"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Zone feature manifest not found: {manifest_path}")
        if not feature_path.is_file():
            raise FileNotFoundError(f"Derived feature parquet not found: {feature_path}")
        zone_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        applicable_model_ids = [str(item) for item in zone_manifest.get("applicable_model_ids", [])]
        audit_zone_root = (
            audit_root / "cases" / make_safe_name(ref.case_id)
            / make_safe_name(ref.aggregation_run_id) / make_safe_name(ref.aggregate_zone_id)
        )
        audit_applicable_path = audit_zone_root / "applicable_models.csv"
        if not audit_applicable_path.is_file():
            raise FileNotFoundError(f"C1 applicable-model table not found: {audit_applicable_path}")
        with audit_applicable_path.open("r", encoding="utf-8-sig", newline="") as stream:
            audit_applicable_model_ids = [
                str(row.get("model_id", "")).strip()
                for row in csv.DictReader(stream)
                if str(row.get("model_id", "")).strip()
            ]

        source_wide = pd.read_parquet(ref.wide_parquet_path)
        source_wide, recomputed_canonical_metadata = canonicalize_wide_frame(
            source_wide
        )
        static_frame = pd.read_csv(ref.static_equipment_path)
        contribution_frame = pd.read_csv(ref.equipment_contributions_path)
        stored = pd.read_parquet(feature_path)

        manifest_zone_count = zone_manifest.get("aggregate_zone_count")
        reference_zone_count = getattr(ref, "aggregate_zone_count", None)
        if manifest_zone_count is None and reference_zone_count is None:
            raise ValueError(
                "aggregate_zone_count is absent from both zone manifest and "
                "aggregation reference"
            )
        aggregate_zone_count = int(
            manifest_zone_count
            if manifest_zone_count is not None
            else reference_zone_count
        )

        recomputed, _, recomputed_manifest = build_zone_derived_features(
            wide_frame=source_wide,
            static_equipment_frame=static_frame,
            contribution_frame=contribution_frame,
            applicable_model_ids=applicable_model_ids,
            internal_gain_predictor_method=str(
                zone_manifest.get(
                    "internal_gain_predictor_method",
                    "aggregate_average",
                )
            ),
            hvac_target_method=str(
                zone_manifest.get(
                    "hvac_target_method",
                    "signed_zone_sensible",
                )
            ),
            aggregate_zone_count=aggregate_zone_count,
        )

        errors: list[str] = []
        if applicable_model_ids != audit_applicable_model_ids:
            errors.append(
                "C2 applicable_model_ids differ from C1 applicable_models.csv: "
                f"c2={applicable_model_ids}, c1={audit_applicable_model_ids}"
            )
        if int(zone_manifest.get("applicable_model_count", len(applicable_model_ids))) != len(applicable_model_ids):
            errors.append("C2 applicable_model_count is inconsistent with applicable_model_ids")
        if bool(zone_manifest.get("zero_applicable_models", False)) != (len(applicable_model_ids) == 0):
            errors.append("C2 zero_applicable_models flag is inconsistent")

        stored_canonical_metadata = zone_manifest.get(
            "canonical_timestamp_metadata",
            {},
        )
        if int(zone_manifest.get("aggregate_zone_count", aggregate_zone_count)) != aggregate_zone_count:
            errors.append(
                "aggregate_zone_count differs between saved manifest and recomputation"
            )
        if int(recomputed_manifest.get("aggregate_zone_count", -1)) != aggregate_zone_count:
            errors.append(
                "recomputed manifest aggregate_zone_count is inconsistent"
            )
        if int(stored_canonical_metadata.get("output_row_count", -1)) != len(stored):
            errors.append(
                "saved canonical output_row_count does not match stored parquet"
            )
        if int(recomputed_canonical_metadata.get("output_row_count", -1)) != len(recomputed):
            errors.append(
                "recomputed canonical output_row_count does not match recomputed frame"
            )

        if list(stored.columns) != list(recomputed.columns):
            errors.append(
                f"column mismatch: stored={list(stored.columns)}, recomputed={list(recomputed.columns)}"
            )
        if len(stored) != len(recomputed):
            errors.append(f"row-count mismatch: stored={len(stored)}, recomputed={len(recomputed)}")

        compared_columns = 0
        maximum_absolute_error = 0.0
        maximum_relative_error = 0.0
        if not errors:
            if not stored["timestamp_raw"].astype(str).equals(
                recomputed["timestamp_raw"].astype(str)
            ):
                errors.append("timestamp_raw values differ from recomputation")
            for column in stored.columns:
                if column in {"timestamp_raw", "timestamp"}:
                    continue
                left = pd.to_numeric(stored[column], errors="coerce")
                right = pd.to_numeric(recomputed[column], errors="coerce")
                same_nan = left.isna().equals(right.isna())
                if not same_nan:
                    errors.append(f"NaN mask differs for {column}")
                    continue
                valid = left.notna() & right.notna()
                absolute_error = (left[valid] - right[valid]).abs()
                denominator = right[valid].abs().clip(lower=1.0e-12)
                relative_error = absolute_error / denominator
                column_abs = float(absolute_error.max()) if not absolute_error.empty else 0.0
                column_rel = float(relative_error.max()) if not relative_error.empty else 0.0
                maximum_absolute_error = max(maximum_absolute_error, column_abs)
                maximum_relative_error = max(maximum_relative_error, column_rel)
                compared_columns += 1
                if not np.allclose(
                    left[valid],
                    right[valid],
                    atol=args.absolute_tolerance,
                    rtol=args.relative_tolerance,
                    equal_nan=True,
                ):
                    errors.append(
                        f"formula mismatch for {column}: max_abs={column_abs}, max_rel={column_rel}"
                    )

        statistical_rows = validate_zone_derived_features(
            stored,
            minimum_sample_count=args.minimum_sample_count,
        )
        invalid_stats = [
            row for row in statistical_rows if row["validation_status"] != "valid"
        ]
        if invalid_stats:
            errors.extend(
                f"{row['feature_column']}: {row['validation_reason']}"
                for row in invalid_stats
            )

        return {
            **base,
            "validation_status": "passed" if not errors else "failed",
            "reason": "stored features exactly match deterministic recomputation" if not errors else " | ".join(errors),
            "row_count": len(stored),
            "derived_feature_count": max(0, len(stored.columns) - 2),
            "compared_numeric_column_count": compared_columns,
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
            "feature_path": str(feature_path),
        }
    except Exception as exc:
        return {
            **base,
            "validation_status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "feature_path": str(zone_feature_root / "derived_heat_input_features.parquet"),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--generated-data-root", type=Path, default=None)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--aggregation-id", default=None)
    parser.add_argument("--weight-mode", default=None)
    parser.add_argument("--aggregate-zone-id", default=None)
    parser.add_argument("--minimum-sample-count", type=int, default=1000)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-9)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
