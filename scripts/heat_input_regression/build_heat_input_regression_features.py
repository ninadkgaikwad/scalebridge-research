# -*- coding: utf-8 -*-
"""Build deterministic Stage C2 heat-input regression features.

This script consumes a validated Stage C1 audit run and writes one reusable
zone-level derived-feature dataset per selected aggregate zone. It does not
create regression splits or model-specific train/test pair files.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalebridge.data.aggregation.writers import (
    make_safe_name,
    write_csv,
    write_dataframe_csv,
    write_dataframe_parquet,
    write_json,
)
from scalebridge.data.heat_input_regression.discovery import (
    discover_from_matrix_run,
    resolve_inputs,
)
from scalebridge.data.heat_input_regression.alignment import canonicalize_wide_frame
from scalebridge.data.heat_input_regression.feature_engineering import (
    build_zone_derived_features,
    validate_zone_derived_features,
)

DEFAULT_CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"


def _json_safe(value: Any) -> Any:
    """Recursively convert scientific-Python values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    campaign_root = resolve_inputs(
        campaign_id=args.campaign_id,
        campaign_root=args.campaign_root,
        generated_data_root=args.generated_data_root,
    )
    audit_root = campaign_root / "heat_input_regression" / "audit_runs" / args.audit_run_id
    if not audit_root.is_dir():
        raise SystemExit(f"Stage C1 audit run not found: {audit_root}")

    refs, discovery_issues = discover_from_matrix_run(
        campaign_root=campaign_root,
        matrix_run_id=args.matrix_run_id,
        case_id=args.case_id,
        aggregation_id=args.aggregation_id,
        weight_mode=args.weight_mode,
        aggregate_zone_id=args.aggregate_zone_id,
    )
    if not refs:
        raise SystemExit("No aggregation-zone outputs were selected")

    feature_run_id = args.feature_run_id or build_feature_run_id()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else campaign_root / "heat_input_regression" / "feature_runs" / feature_run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_root / "selected_aggregation_zone_outputs.csv",
        [ref.identity_dict() for ref in refs],
    )
    write_csv(output_root / "source_discovery_issues.csv", discovery_issues)

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION FEATURE BUILDER")
    print("=" * 100)
    print(f"campaign_root: {campaign_root}")
    print(f"audit_run_id: {args.audit_run_id}")
    print(f"matrix_run_id: {args.matrix_run_id}")
    print(f"feature_run_id: {feature_run_id}")
    print(f"selected_aggregation_zone_count: {len(refs)}")
    print(f"internal_gain_predictor_method: {args.internal_gain_predictor_method}")
    print(f"hvac_target_method: {args.hvac_target_method}")
    print()

    result_rows: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        print(
            f"[{index}/{len(refs)}] {ref.case_id} | {ref.aggregation_id} | "
            f"{ref.weight_mode} | {ref.aggregate_zone_id}"
        )
        result_rows.append(
            build_one_zone(
                ref=ref,
                audit_root=audit_root,
                output_root=output_root,
                args=args,
            )
        )
        if result_rows[-1]["status"] != "completed" and not args.continue_on_error:
            raise RuntimeError(result_rows[-1]["error_message"])

    write_csv(output_root / "feature_zone_results.csv", result_rows)

    successful_zone_count = sum(row["status"] == "completed" for row in result_rows)
    failed_zone_count = len(result_rows) - successful_zone_count
    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if failed_zone_count == 0 else "completed_with_failures",
        "feature_run_id": feature_run_id,
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "source_audit_run_id": args.audit_run_id,
        "source_matrix_run_id": args.matrix_run_id,
        "selected_aggregation_run_count": len({ref.aggregation_run_id for ref in refs}),
        "selected_aggregation_zone_count": len(refs),
        "successful_zone_count": successful_zone_count,
        "failed_zone_count": failed_zone_count,
        "internal_gain_predictor_method": args.internal_gain_predictor_method,
        "hvac_target_method": args.hvac_target_method,
        "minimum_sample_count": args.minimum_sample_count,
        "preview_rows": args.preview_rows,
        "runtime_seconds": time.perf_counter() - started,
        "output_root": str(output_root),
    }
    write_json(output_root / "heat_input_feature_run_manifest.json", _json_safe(manifest))

    print()
    print("=" * 100)
    print("FEATURE BUILD SUMMARY")
    print("=" * 100)
    print(f"successful_zone_count: {successful_zone_count}")
    print(f"failed_zone_count: {failed_zone_count}")
    print(f"output_root: {output_root}")
    return 0 if failed_zone_count == 0 else 1


def build_one_zone(*, ref, audit_root: Path, output_root: Path, args) -> dict[str, Any]:
    zone_started = time.perf_counter()
    zone_output_root = (
        output_root
        / "cases"
        / make_safe_name(ref.case_id)
        / make_safe_name(ref.aggregation_run_id)
        / make_safe_name(ref.aggregate_zone_id)
    )
    zone_output_root.mkdir(parents=True, exist_ok=True)

    try:
        audit_zone_root = (
            audit_root
            / "cases"
            / make_safe_name(ref.case_id)
            / make_safe_name(ref.aggregation_run_id)
            / make_safe_name(ref.aggregate_zone_id)
        )
        applicable_path = audit_zone_root / "applicable_models.csv"
        if not applicable_path.is_file():
            raise FileNotFoundError(
                f"Stage C1 applicable-model table not found: {applicable_path}"
            )
        applicable_rows = read_csv_dicts(applicable_path)
        applicability_path = audit_zone_root / "model_applicability.csv"
        if not applicability_path.is_file():
            applicability_path = audit_zone_root / "candidate_models.csv"
        applicability_rows = (
            read_csv_dicts(applicability_path)
            if applicability_path.is_file()
            else list(applicable_rows)
        )
        applicable_model_ids = [
            str(row.get("model_id", "")).strip()
            for row in applicable_rows
            if str(row.get("model_id", "")).strip()
        ]
        if args.model_id:
            requested = set(args.model_id)
            applicable_model_ids = [
                model_id for model_id in applicable_model_ids if model_id in requested
            ]
            applicability_rows = [
                row for row in applicability_rows
                if str(row.get("model_id", "")).strip() in requested
            ]

        selected_applicable_rows = [
            row for row in applicable_rows
            if str(row.get("model_id", "")).strip() in set(applicable_model_ids)
        ]
        inapplicable_rows = [
            row for row in applicability_rows
            if str(row.get("applicable", "")).strip().lower() not in {"true", "1", "yes"}
        ]
        write_csv(zone_output_root / "model_applicability_snapshot.csv", applicability_rows)
        write_csv(zone_output_root / "applicable_models_snapshot.csv", selected_applicable_rows)
        write_csv(zone_output_root / "inapplicable_models_snapshot.csv", inapplicable_rows)

        try:
            wide_frame = pd.read_parquet(ref.wide_parquet_path)
        except ImportError as exc:
            raise RuntimeError(
                "Reading Stage B Parquet requires pyarrow or fastparquet in the active "
                "ScaleBridge environment."
            ) from exc
        wide_frame, canonical_timestamp_metadata = canonicalize_wide_frame(wide_frame)
        static_frame = pd.read_csv(ref.static_equipment_path)
        contribution_frame = pd.read_csv(ref.equipment_contributions_path)

        features, feature_catalog, feature_manifest = build_zone_derived_features(
            wide_frame=wide_frame,
            static_equipment_frame=static_frame,
            contribution_frame=contribution_frame,
            applicable_model_ids=applicable_model_ids,
            internal_gain_predictor_method=args.internal_gain_predictor_method,
            hvac_target_method=args.hvac_target_method,
            aggregate_zone_count=ref.aggregate_zone_count,
        )
        validation_rows = validate_zone_derived_features(
            features,
            minimum_sample_count=args.minimum_sample_count,
        )
        invalid_features = [
            row for row in validation_rows if row["validation_status"] != "valid"
        ]
        if invalid_features:
            reasons = " | ".join(
                f"{row['feature_column']}: {row['validation_reason']}"
                for row in invalid_features
            )
            raise ValueError(f"Derived-feature validation failed: {reasons}")

        parquet_path = zone_output_root / "derived_heat_input_features.parquet"
        preview_path = zone_output_root / "derived_heat_input_features_preview.csv"
        write_dataframe_parquet(parquet_path, features)
        write_dataframe_csv(preview_path, features.head(args.preview_rows))
        write_csv(zone_output_root / "derived_feature_catalog.csv", feature_catalog)
        write_csv(zone_output_root / "derived_feature_validation.csv", validation_rows)
        write_json(
            zone_output_root / "zone_feature_manifest.json",
            _json_safe({
                **ref.identity_dict(),
                **feature_manifest,
                "candidate_model_count": len(applicability_rows),
                "applicable_model_count": len(applicable_model_ids),
                "inapplicable_model_count": len(inapplicable_rows),
                "zero_applicable_models": len(applicable_model_ids) == 0,
                "source_model_applicability_path": str(applicability_path),
                "canonical_timestamp_metadata": canonical_timestamp_metadata,
                "schema_version": "0.1.0",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "source_audit_run_id": args.audit_run_id,
                "source_matrix_run_id": args.matrix_run_id,
                "output_root": str(zone_output_root),
                "outputs": {
                    "derived_features_parquet": str(parquet_path),
                    "derived_features_preview": str(preview_path),
                    "derived_feature_catalog": str(
                        zone_output_root / "derived_feature_catalog.csv"
                    ),
                    "derived_feature_validation": str(
                        zone_output_root / "derived_feature_validation.csv"
                    ),
                },
            }),
        )
        return {
            **ref.identity_dict(),
            "status": "completed",
            "candidate_model_count": len(applicability_rows),
            "applicable_model_count": len(applicable_model_ids),
            "inapplicable_model_count": len(inapplicable_rows),
            "zero_applicable_models": len(applicable_model_ids) == 0,
            "derived_feature_count": len(feature_catalog),
            "row_count": len(features),
            "input_row_count": canonical_timestamp_metadata["input_row_count"],
            "removed_duplicate_row_count": canonical_timestamp_metadata["removed_duplicate_row_count"],
            "column_count": len(features.columns),
            "output_root": str(zone_output_root),
            "runtime_seconds": time.perf_counter() - zone_started,
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return {
            **ref.identity_dict(),
            "status": "failed",
            "applicable_model_count": "",
            "derived_feature_count": "",
            "row_count": "",
            "column_count": "",
            "output_root": str(zone_output_root),
            "runtime_seconds": time.perf_counter() - zone_started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--feature-run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--aggregation-id", default=None)
    parser.add_argument("--weight-mode", default=None)
    parser.add_argument("--aggregate-zone-id", default=None)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--minimum-sample-count", type=int, default=1000)
    parser.add_argument(
        "--internal-gain-predictor-method",
        choices=["aggregate_average", "contribution_sum"],
        default="aggregate_average",
    )
    parser.add_argument(
        "--hvac-target-method",
        choices=[
            "signed_zone_sensible",
            "absolute_zone_sensible",
        ],
        default="signed_zone_sensible",
    )
    parser.add_argument("--preview-rows", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build_feature_run_id() -> str:
    return f"heat_input_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
