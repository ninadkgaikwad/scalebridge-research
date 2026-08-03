# -*- coding: utf-8 -*-
"""Build deterministic Stage C3 train/validation/test split assignments."""

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
from scalebridge.data.heat_input_regression.splitting import (
    SUPPORTED_SPLIT_STRATEGIES,
    SplitConfig,
    build_split_assignments,
    build_split_summary,
    split_counts,
)
from scalebridge.data.heat_input_regression.split_validation import (
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
    feature_root = (
        campaign_root
        / "heat_input_regression"
        / "feature_runs"
        / args.feature_run_id
    )
    if not feature_root.is_dir():
        raise SystemExit(f"Stage C2 feature run not found: {feature_root}")

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

    split_config = SplitConfig(
        strategy=args.split_strategy,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )
    split_config.validate()

    split_run_id = args.split_run_id or build_split_run_id()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else campaign_root / "heat_input_regression" / "split_runs" / split_run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_root / "selected_aggregation_zone_outputs.csv",
        [ref.identity_dict() for ref in refs],
    )
    write_csv(output_root / "source_discovery_issues.csv", discovery_issues)

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION SPLIT BUILDER")
    print("=" * 100)
    print(f"campaign_root: {campaign_root}")
    print(f"matrix_run_id: {args.matrix_run_id}")
    print(f"feature_run_id: {args.feature_run_id}")
    print(f"split_run_id: {split_run_id}")
    print(f"selected_aggregation_zone_count: {len(refs)}")
    print(f"split_strategy: {args.split_strategy}")
    print(
        "fractions: "
        f"train={args.train_fraction}, "
        f"validation={args.validation_fraction}, test={args.test_fraction}"
    )
    print()

    result_rows: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        print(
            f"[{index}/{len(refs)}] {ref.case_id} | {ref.aggregation_id} | "
            f"{ref.weight_mode} | {ref.aggregate_zone_id}"
        )
        result = build_one_zone(
            ref=ref,
            feature_root=feature_root,
            output_root=output_root,
            split_config=split_config,
            args=args,
        )
        result_rows.append(result)
        if result["status"] != "completed" and not args.continue_on_error:
            raise RuntimeError(result["error_message"])

    write_csv(output_root / "split_zone_results.csv", result_rows)
    successful_zone_count = sum(row["status"] == "completed" for row in result_rows)
    failed_zone_count = len(result_rows) - successful_zone_count

    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if failed_zone_count == 0 else "completed_with_failures",
        "split_run_id": split_run_id,
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "source_matrix_run_id": args.matrix_run_id,
        "source_audit_run_id": args.audit_run_id,
        "source_feature_run_id": args.feature_run_id,
        "selected_aggregation_run_count": len({ref.aggregation_run_id for ref in refs}),
        "selected_aggregation_zone_count": len(refs),
        "successful_zone_count": successful_zone_count,
        "failed_zone_count": failed_zone_count,
        "split_strategy": args.split_strategy,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "minimum_split_samples": args.minimum_split_samples,
        "fraction_tolerance": args.fraction_tolerance,
        "random_seed": args.random_seed,
        "preview_rows": args.preview_rows,
        "runtime_seconds": time.perf_counter() - started,
        "output_root": str(output_root),
    }
    write_json(output_root / "split_run_manifest.json", manifest)

    print()
    print("=" * 100)
    print("SPLIT BUILD SUMMARY")
    print("=" * 100)
    print(f"successful_zone_count: {successful_zone_count}")
    print(f"failed_zone_count: {failed_zone_count}")
    print(f"output_root: {output_root}")
    return 0 if failed_zone_count == 0 else 1


def build_one_zone(*, ref, feature_root: Path, output_root: Path, split_config: SplitConfig, args) -> dict[str, Any]:
    zone_started = time.perf_counter()
    relative_parts = (
        "cases",
        make_safe_name(ref.case_id),
        make_safe_name(ref.aggregation_run_id),
        make_safe_name(ref.aggregate_zone_id),
    )
    feature_zone_root = feature_root.joinpath(*relative_parts)
    zone_output_root = output_root.joinpath(*relative_parts)
    zone_output_root.mkdir(parents=True, exist_ok=True)

    try:
        feature_path = feature_zone_root / "derived_heat_input_features.parquet"
        if not feature_path.is_file():
            raise FileNotFoundError(f"C2 feature parquet not found: {feature_path}")
        feature_manifest_path = feature_zone_root / "zone_feature_manifest.json"
        if not feature_manifest_path.is_file():
            raise FileNotFoundError(f"C2 zone feature manifest not found: {feature_manifest_path}")
        feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        applicable_model_ids = [
            str(item) for item in feature_manifest.get("applicable_model_ids", [])
        ]
        candidate_model_count = int(feature_manifest.get("candidate_model_count", len(applicable_model_ids)))
        inapplicable_model_count = int(feature_manifest.get("inapplicable_model_count", max(0, candidate_model_count - len(applicable_model_ids))))

        feature_frame = pd.read_parquet(feature_path)
        stage_b_frame = pd.read_parquet(ref.wide_parquet_path)

        assignments = build_split_assignments(feature_frame, config=split_config)
        diagnostics = validate_split_assignments(
            assignments,
            config=split_config,
            minimum_split_samples=args.minimum_split_samples,
            fraction_tolerance=args.fraction_tolerance,
        )
        diagnostics.extend(
            validate_timestamp_coverage(
                feature_frame=feature_frame,
                stage_b_frame=stage_b_frame,
                assignments=assignments,
            )
        )
        if not validation_passed(diagnostics):
            failed = [row["check_name"] for row in diagnostics if row["status"] != "passed"]
            raise ValueError("Split validation failed: " + " | ".join(failed))

        summary_rows = build_split_summary(assignments)
        counts = split_counts(assignments)
        parquet_path = zone_output_root / "split_assignments.parquet"
        preview_path = zone_output_root / "split_assignments_preview.csv"
        write_dataframe_parquet(parquet_path, assignments)
        write_dataframe_csv(preview_path, assignments.head(args.preview_rows))
        write_csv(zone_output_root / "split_summary.csv", summary_rows)
        write_csv(zone_output_root / "split_diagnostics.csv", diagnostics)

        manifest = {
            **ref.identity_dict(),
            "schema_version": "0.1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "source_audit_run_id": args.audit_run_id,
            "source_matrix_run_id": args.matrix_run_id,
            "source_feature_run_id": args.feature_run_id,
            "candidate_model_count": candidate_model_count,
            "applicable_model_count": len(applicable_model_ids),
            "inapplicable_model_count": inapplicable_model_count,
            "zero_applicable_models": len(applicable_model_ids) == 0,
            "applicable_model_ids": applicable_model_ids,
            "split_strategy": split_config.strategy,
            "train_fraction": split_config.train_fraction,
            "validation_fraction": split_config.validation_fraction,
            "test_fraction": split_config.test_fraction,
            "random_seed": split_config.random_seed,
            "timestamp_column": split_config.timestamp_column,
            "raw_timestamp_column": split_config.raw_timestamp_column,
            "source_row_count": int(len(feature_frame)),
            "assignment_row_count": int(len(assignments)),
            "train_row_count": counts["train"],
            "validation_row_count": counts["validation"],
            "test_row_count": counts["test"],
            "excluded_row_count": counts["excluded"],
            "output_root": str(zone_output_root),
            "outputs": {
                "split_assignments": str(parquet_path),
                "split_assignments_preview": str(preview_path),
                "split_summary": str(zone_output_root / "split_summary.csv"),
                "split_diagnostics": str(zone_output_root / "split_diagnostics.csv"),
            },
        }
        write_json(zone_output_root / "zone_split_manifest.json", manifest)
        return {
            **ref.identity_dict(),
            "status": "completed",
            "candidate_model_count": candidate_model_count,
            "applicable_model_count": len(applicable_model_ids),
            "inapplicable_model_count": inapplicable_model_count,
            "zero_applicable_models": len(applicable_model_ids) == 0,
            "source_row_count": len(feature_frame),
            "assignment_row_count": len(assignments),
            "train_row_count": counts["train"],
            "validation_row_count": counts["validation"],
            "test_row_count": counts["test"],
            "excluded_row_count": counts["excluded"],
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
            "source_row_count": "",
            "assignment_row_count": "",
            "train_row_count": "",
            "validation_row_count": "",
            "test_row_count": "",
            "excluded_row_count": "",
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
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--split-run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--aggregation-id", default=None)
    parser.add_argument("--weight-mode", default=None)
    parser.add_argument("--aggregate-zone-id", default=None)
    parser.add_argument(
        "--split-strategy",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default="monthly_distributed_holdout",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-split-samples", type=int, default=1000)
    parser.add_argument("--fraction-tolerance", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--preview-rows", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def build_split_run_id() -> str:
    return f"heat_input_splits_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
