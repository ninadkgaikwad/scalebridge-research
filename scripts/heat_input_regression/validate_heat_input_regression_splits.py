# -*- coding: utf-8 -*-
"""Independently validate saved Stage C3 split assignments."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scalebridge.data.aggregation.writers import make_safe_name, write_csv, write_json
from scalebridge.data.heat_input_regression.discovery import (
    discover_from_matrix_run,
    resolve_inputs,
)
from scalebridge.data.heat_input_regression.splitting import SplitConfig
from scalebridge.data.heat_input_regression.split_validation import (
    validate_reproducibility,
    validate_split_assignments,
    validate_timestamp_coverage,
    validation_passed,
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
    feature_root = campaign_root / "heat_input_regression" / "feature_runs" / args.feature_run_id
    split_root = campaign_root / "heat_input_regression" / "split_runs" / args.split_run_id
    if not feature_root.is_dir():
        raise SystemExit(f"Feature run not found: {feature_root}")
    if not split_root.is_dir():
        raise SystemExit(f"Split run not found: {split_root}")

    run_manifest_path = split_root / "split_run_manifest.json"
    if not run_manifest_path.is_file():
        raise SystemExit(f"Split run manifest not found: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    config = SplitConfig(
        strategy=str(run_manifest["split_strategy"]),
        train_fraction=float(run_manifest["train_fraction"]),
        validation_fraction=float(run_manifest["validation_fraction"]),
        test_fraction=float(run_manifest["test_fraction"]),
        random_seed=int(run_manifest.get("random_seed", 42)),
    )

    refs, discovery_issues = discover_from_matrix_run(
        campaign_root=campaign_root,
        matrix_run_id=args.matrix_run_id,
        case_id=args.case_id,
        aggregation_id=args.aggregation_id,
        weight_mode=args.weight_mode,
        aggregate_zone_id=args.aggregate_zone_id,
    )
    write_csv(split_root / "split_validation_discovery_issues.csv", discovery_issues)

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION SPLIT VALIDATION")
    print("=" * 100)
    print(f"feature_root: {feature_root}")
    print(f"split_root: {split_root}")
    print(f"selected_aggregation_zone_count: {len(refs)}")
    print()

    results: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        print(
            f"[{index}/{len(refs)}] {ref.case_id} | {ref.aggregation_id} | "
            f"{ref.aggregate_zone_id}"
        )
        result, diagnostics = validate_one_zone(
            ref=ref,
            feature_root=feature_root,
            split_root=split_root,
            config=config,
            args=args,
        )
        results.append(result)
        all_diagnostics.extend(diagnostics)

    write_csv(split_root / "split_validation_results.csv", results)
    write_csv(split_root / "split_validation_diagnostics.csv", all_diagnostics)
    passed_zone_count = sum(row["status"] == "passed" for row in results)
    failed_zone_count = len(results) - passed_zone_count
    write_json(
        split_root / "split_validation_manifest.json",
        {
            "schema_version": "0.1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if failed_zone_count == 0 else "failed",
            "campaign_id": args.campaign_id,
            "source_matrix_run_id": args.matrix_run_id,
            "source_feature_run_id": args.feature_run_id,
            "source_split_run_id": args.split_run_id,
            "selected_aggregation_zone_count": len(refs),
            "passed_zone_count": passed_zone_count,
            "failed_zone_count": failed_zone_count,
            "minimum_split_samples": args.minimum_split_samples,
            "fraction_tolerance": args.fraction_tolerance,
            "runtime_seconds": time.perf_counter() - started,
            "split_root": str(split_root),
        },
    )

    print()
    print("=" * 100)
    print("SPLIT VALIDATION SUMMARY")
    print("=" * 100)
    print(f"passed_zone_count: {passed_zone_count}")
    print(f"failed_zone_count: {failed_zone_count}")
    print(f"split_root: {split_root}")
    return 0 if failed_zone_count == 0 else 1


def validate_one_zone(*, ref, feature_root: Path, split_root: Path, config: SplitConfig, args):
    started = time.perf_counter()
    relative_parts = (
        "cases",
        make_safe_name(ref.case_id),
        make_safe_name(ref.aggregation_run_id),
        make_safe_name(ref.aggregate_zone_id),
    )
    feature_path = feature_root.joinpath(*relative_parts) / "derived_heat_input_features.parquet"
    assignment_path = split_root.joinpath(*relative_parts) / "split_assignments.parquet"
    try:
        features = pd.read_parquet(feature_path)
        stage_b = pd.read_parquet(ref.wide_parquet_path)
        assignments = pd.read_parquet(assignment_path)
        diagnostics = validate_split_assignments(
            assignments,
            config=config,
            minimum_split_samples=args.minimum_split_samples,
            fraction_tolerance=args.fraction_tolerance,
        )
        diagnostics.extend(
            validate_timestamp_coverage(
                feature_frame=features,
                stage_b_frame=stage_b,
                assignments=assignments,
            )
        )
        diagnostics.extend(
            validate_reproducibility(
                feature_frame=features,
                saved_assignments=assignments,
                config=config,
            )
        )
        for row in diagnostics:
            row.update(
                {
                    "case_id": ref.case_id,
                    "aggregation_run_id": ref.aggregation_run_id,
                    "aggregation_id": ref.aggregation_id,
                    "aggregate_zone_id": ref.aggregate_zone_id,
                }
            )
        passed = validation_passed(diagnostics)
        return (
            {
                **ref.identity_dict(),
                "status": "passed" if passed else "failed",
                "diagnostic_count": len(diagnostics),
                "failed_check_count": sum(row["status"] != "passed" for row in diagnostics),
                "runtime_seconds": time.perf_counter() - started,
                "error_type": "",
                "error_message": "",
            },
            diagnostics,
        )
    except Exception as exc:
        return (
            {
                **ref.identity_dict(),
                "status": "failed",
                "diagnostic_count": 0,
                "failed_check_count": 1,
                "runtime_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
            [
                {
                    "case_id": ref.case_id,
                    "aggregation_run_id": ref.aggregation_run_id,
                    "aggregation_id": ref.aggregation_id,
                    "aggregate_zone_id": ref.aggregate_zone_id,
                    "check_name": "validation_execution",
                    "status": "failed",
                    "observed_value": type(exc).__name__,
                    "expected_value": "successful execution",
                    "message": str(exc),
                }
            ],
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--split-run-id", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--aggregation-id", default=None)
    parser.add_argument("--weight-mode", default=None)
    parser.add_argument("--aggregate-zone-id", default=None)
    parser.add_argument("--minimum-split-samples", type=int, default=1000)
    parser.add_argument("--fraction-tolerance", type=float, default=0.01)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
