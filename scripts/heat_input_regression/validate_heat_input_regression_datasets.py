# -*- coding: utf-8 -*-
"""Independently validate saved Stage C4 regression-pair datasets."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scalebridge.data.aggregation.writers import make_safe_name, write_csv, write_json
from scalebridge.data.heat_input_regression.dataset_validation import (
    compare_saved_to_recomputed,
    validate_regression_pair_dataset,
    validation_passed,
)
from scalebridge.data.heat_input_regression.datasets import (
    RegressionPairDataset,
    build_regression_pair_dataset,
)
from scalebridge.data.heat_input_regression.discovery import (
    discover_from_matrix_run,
    resolve_inputs,
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
    split_root = campaign_root / "heat_input_regression" / "split_runs" / args.split_run_id
    dataset_root = campaign_root / "heat_input_regression" / "dataset_runs" / args.dataset_run_id
    if not dataset_root.is_dir():
        raise SystemExit(f"C4 dataset run not found: {dataset_root}")

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

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION DATASET VALIDATION")
    print("=" * 100)
    print(f"dataset_root: {dataset_root}")
    print(f"selected_aggregation_zone_count: {len(refs)}")
    print()

    zone_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        print(
            f"[{index}/{len(refs)}] {ref.case_id} | {ref.aggregation_id} | "
            f"{ref.aggregate_zone_id}"
        )
        relative_parts = (
            "cases", make_safe_name(ref.case_id),
            make_safe_name(ref.aggregation_run_id), make_safe_name(ref.aggregate_zone_id),
        )
        audit_zone_root = audit_root.joinpath(*relative_parts)
        feature_zone_root = feature_root.joinpath(*relative_parts)
        split_zone_root = split_root.joinpath(*relative_parts)
        dataset_zone_root = dataset_root.joinpath(*relative_parts)

        applicable_path = audit_zone_root / "applicable_models.csv"
        model_ids = [
            str(row.get("model_id", "")).strip()
            for row in read_csv_dicts(applicable_path)
            if str(row.get("model_id", "")).strip()
        ]
        if args.model_id:
            requested = set(args.model_id)
            model_ids = [model_id for model_id in model_ids if model_id in requested]

        zone_manifest_path = dataset_zone_root / "zone_dataset_manifest.json"
        model_index_path = dataset_zone_root / "model_dataset_index.csv"
        if not zone_manifest_path.is_file():
            raise FileNotFoundError(f"C4 zone dataset manifest not found: {zone_manifest_path}")
        import json
        zone_manifest = json.loads(zone_manifest_path.read_text(encoding="utf-8"))
        indexed_rows = read_csv_dicts(model_index_path) if model_index_path.is_file() else []
        indexed_model_ids = [
            str(row.get("model_id", "")).strip()
            for row in indexed_rows
            if str(row.get("model_id", "")).strip()
        ]
        model_set_errors = []
        if indexed_model_ids != model_ids:
            model_set_errors.append(f"C4 model index differs from C1 applicable set: c4={indexed_model_ids}, c1={model_ids}")
        if int(zone_manifest.get("applicable_model_count", len(model_ids))) != len(model_ids):
            model_set_errors.append("zone manifest applicable_model_count is inconsistent")
        if bool(zone_manifest.get("zero_applicable_models", False)) != (len(model_ids) == 0):
            model_set_errors.append("zone manifest zero_applicable_models flag is inconsistent")
        zone_rows.append({
            **ref.identity_dict(),
            "status": "passed" if not model_set_errors else "failed",
            "expected_applicable_model_count": len(model_ids),
            "indexed_model_count": len(indexed_model_ids),
            "zero_applicable_models": len(model_ids) == 0,
            "reason": "C4 model set matches C1 applicability" if not model_set_errors else " | ".join(model_set_errors),
        })

        feature_frame = pd.read_parquet(feature_zone_root / "derived_heat_input_features.parquet")
        split_frame = pd.read_parquet(split_zone_root / "split_assignments.parquet")
        stage_b_frame = pd.read_parquet(ref.wide_parquet_path)

        for model_id in model_ids:
            print(f"    {model_id}")
            result, current_diagnostics = validate_one_model(
                ref=ref,
                model_id=model_id,
                dataset_zone_root=dataset_zone_root,
                feature_frame=feature_frame,
                split_frame=split_frame,
                stage_b_frame=stage_b_frame,
                args=args,
            )
            model_rows.append(result)
            diagnostic_rows.extend(current_diagnostics)

    write_csv(dataset_root / "dataset_zone_validation_results.csv", zone_rows)
    write_csv(dataset_root / "dataset_validation_results.csv", model_rows)
    write_csv(dataset_root / "dataset_validation_diagnostics.csv", diagnostic_rows)
    write_csv(dataset_root / "dataset_validation_discovery_issues.csv", discovery_issues)

    passed_zone_count = sum(row["status"] == "passed" for row in zone_rows)
    failed_zone_count = len(zone_rows) - passed_zone_count
    passed_model_count = sum(row["status"] == "passed" for row in model_rows)
    failed_model_count = len(model_rows) - passed_model_count
    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if failed_zone_count == 0 and failed_model_count == 0 else "failed",
        "campaign_id": args.campaign_id,
        "source_matrix_run_id": args.matrix_run_id,
        "source_audit_run_id": args.audit_run_id,
        "source_feature_run_id": args.feature_run_id,
        "source_split_run_id": args.split_run_id,
        "dataset_run_id": args.dataset_run_id,
        "selected_zone_count": len(refs),
        "passed_zone_count": passed_zone_count,
        "failed_zone_count": failed_zone_count,
        "validated_model_count": len(model_rows),
        "passed_model_count": passed_model_count,
        "failed_model_count": failed_model_count,
        "minimum_split_samples": args.minimum_split_samples,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "runtime_seconds": time.perf_counter() - started,
        "dataset_root": str(dataset_root),
    }
    write_json(dataset_root / "dataset_validation_manifest.json", manifest)

    print()
    print("=" * 100)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 100)
    print(f"passed_zone_count: {passed_zone_count}")
    print(f"failed_zone_count: {failed_zone_count}")
    print(f"passed_model_count: {passed_model_count}")
    print(f"failed_model_count: {failed_model_count}")
    print(f"dataset_root: {dataset_root}")
    return 0 if failed_zone_count == 0 and failed_model_count == 0 else 1


def validate_one_model(*, ref, model_id: str, dataset_zone_root: Path, feature_frame: pd.DataFrame, split_frame: pd.DataFrame, stage_b_frame: pd.DataFrame, args):
    started = time.perf_counter()
    model_root = dataset_zone_root / "models" / make_safe_name(model_id)
    diagnostics: list[dict[str, Any]] = []
    try:
        full_path = model_root / "regression_pairs_full.parquet"
        if not full_path.is_file():
            raise FileNotFoundError(f"Saved C4 dataset not found: {full_path}")
        saved = pd.read_parquet(full_path)
        recomputed_dataset = build_regression_pair_dataset(
            model_id=model_id,
            feature_frame=feature_frame,
            split_frame=split_frame,
            stage_b_frame=stage_b_frame,
        )
        saved_dataset = RegressionPairDataset(
            model_id=model_id,
            predictor_column=recomputed_dataset.predictor_column,
            target_column=recomputed_dataset.target_column,
            predictor_units=recomputed_dataset.predictor_units,
            target_units=recomputed_dataset.target_units,
            output_prediction_column=recomputed_dataset.output_prediction_column,
            frame=saved,
            metadata=recomputed_dataset.metadata,
        )
        diagnostics.extend(validate_regression_pair_dataset(
            saved_dataset,
            minimum_split_samples=args.minimum_split_samples,
        ))
        diagnostics.extend(compare_saved_to_recomputed(
            saved_frame=saved,
            model_id=model_id,
            feature_frame=feature_frame,
            split_frame=split_frame,
            stage_b_frame=stage_b_frame,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        ))

        for row in diagnostics:
            row.update({
                "case_id": ref.case_id,
                "aggregation_run_id": ref.aggregation_run_id,
                "aggregate_zone_id": ref.aggregate_zone_id,
                "model_id": model_id,
            })
        passed = validation_passed(diagnostics)
        return ({
            **ref.identity_dict(),
            "model_id": model_id,
            "status": "passed" if passed else "failed",
            "check_count": len(diagnostics),
            "failed_check_count": sum(row["status"] != "passed" for row in diagnostics),
            "runtime_seconds": time.perf_counter() - started,
            "error_type": "",
            "error_message": "",
        }, diagnostics)
    except Exception as exc:
        diagnostics.append({
            "case_id": ref.case_id,
            "aggregation_run_id": ref.aggregation_run_id,
            "aggregate_zone_id": ref.aggregate_zone_id,
            "model_id": model_id,
            "check_name": "validation_execution",
            "status": "failed",
            "observed_value": type(exc).__name__,
            "expected_value": "successful validation",
            "message": str(exc),
        })
        return ({
            **ref.identity_dict(),
            "model_id": model_id,
            "status": "failed",
            "check_count": len(diagnostics),
            "failed_check_count": 1,
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }, diagnostics)


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--split-run-id", required=True)
    parser.add_argument("--dataset-run-id", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--aggregation-id", default=None)
    parser.add_argument("--weight-mode", default=None)
    parser.add_argument("--aggregate-zone-id", default=None)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--minimum-split-samples", type=int, default=1000)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
