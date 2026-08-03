# -*- coding: utf-8 -*-
"""Build Stage C4 model-specific heat-input regression-pair datasets."""

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

from scalebridge.data.aggregation.writers import (
    make_safe_name,
    write_csv,
    write_dataframe_csv,
    write_dataframe_parquet,
    write_json,
)
from scalebridge.data.heat_input_regression.dataset_validation import (
    validate_regression_pair_dataset,
    validation_passed,
)
from scalebridge.data.heat_input_regression.datasets import (
    build_dataset_summary,
    build_exclusion_summary,
    build_regression_pair_dataset,
    split_valid_pairs,
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
    for label, root in (
        ("C1 audit", audit_root),
        ("C2 feature", feature_root),
        ("C3 split", split_root),
    ):
        if not root.is_dir():
            raise SystemExit(f"{label} run not found: {root}")

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

    dataset_run_id = args.dataset_run_id or build_dataset_run_id()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else campaign_root / "heat_input_regression" / "dataset_runs" / dataset_run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_root / "selected_aggregation_zone_outputs.csv",
        [ref.identity_dict() for ref in refs],
    )
    write_csv(output_root / "source_discovery_issues.csv", discovery_issues)

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION DATASET BUILDER")
    print("=" * 100)
    print(f"campaign_root: {campaign_root}")
    print(f"matrix_run_id: {args.matrix_run_id}")
    print(f"audit_run_id: {args.audit_run_id}")
    print(f"feature_run_id: {args.feature_run_id}")
    print(f"split_run_id: {args.split_run_id}")
    print(f"dataset_run_id: {dataset_run_id}")
    print(f"selected_aggregation_zone_count: {len(refs)}")
    print()

    model_result_rows: list[dict[str, Any]] = []
    zone_result_rows: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, start=1):
        print(
            f"[{index}/{len(refs)}] {ref.case_id} | {ref.aggregation_id} | "
            f"{ref.weight_mode} | {ref.aggregate_zone_id}"
        )
        zone_result, current_model_rows = build_one_zone(
            ref=ref,
            audit_root=audit_root,
            feature_root=feature_root,
            split_root=split_root,
            output_root=output_root,
            args=args,
        )
        zone_result_rows.append(zone_result)
        model_result_rows.extend(current_model_rows)
        if zone_result["status"] != "completed" and not args.continue_on_error:
            raise RuntimeError(zone_result["error_message"])

    write_csv(output_root / "dataset_zone_results.csv", zone_result_rows)
    write_csv(output_root / "dataset_model_results.csv", model_result_rows)

    successful_zone_count = sum(row["status"] == "completed" for row in zone_result_rows)
    failed_zone_count = len(zone_result_rows) - successful_zone_count
    successful_model_count = sum(row["status"] == "completed" for row in model_result_rows)
    failed_model_count = len(model_result_rows) - successful_model_count

    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if failed_zone_count == 0 and failed_model_count == 0 else "completed_with_failures",
        "dataset_run_id": dataset_run_id,
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "source_matrix_run_id": args.matrix_run_id,
        "source_audit_run_id": args.audit_run_id,
        "source_feature_run_id": args.feature_run_id,
        "source_split_run_id": args.split_run_id,
        "selected_aggregation_run_count": len({ref.aggregation_run_id for ref in refs}),
        "selected_aggregation_zone_count": len(refs),
        "successful_zone_count": successful_zone_count,
        "failed_zone_count": failed_zone_count,
        "selected_model_count": len(model_result_rows),
        "successful_model_count": successful_model_count,
        "failed_model_count": failed_model_count,
        "minimum_split_samples": args.minimum_split_samples,
        "preview_rows": args.preview_rows,
        "runtime_seconds": time.perf_counter() - started,
        "output_root": str(output_root),
    }
    write_json(output_root / "dataset_run_manifest.json", manifest)

    print()
    print("=" * 100)
    print("DATASET BUILD SUMMARY")
    print("=" * 100)
    print(f"successful_zone_count: {successful_zone_count}")
    print(f"failed_zone_count: {failed_zone_count}")
    print(f"successful_model_count: {successful_model_count}")
    print(f"failed_model_count: {failed_model_count}")
    print(f"output_root: {output_root}")
    return 0 if failed_zone_count == 0 and failed_model_count == 0 else 1


def build_one_zone(*, ref, audit_root: Path, feature_root: Path, split_root: Path, output_root: Path, args):
    zone_started = time.perf_counter()
    relative_parts = (
        "cases",
        make_safe_name(ref.case_id),
        make_safe_name(ref.aggregation_run_id),
        make_safe_name(ref.aggregate_zone_id),
    )
    audit_zone_root = audit_root.joinpath(*relative_parts)
    feature_zone_root = feature_root.joinpath(*relative_parts)
    split_zone_root = split_root.joinpath(*relative_parts)
    zone_output_root = output_root.joinpath(*relative_parts)
    zone_output_root.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, Any]] = []
    try:
        applicable_path = audit_zone_root / "applicable_models.csv"
        feature_path = feature_zone_root / "derived_heat_input_features.parquet"
        split_path = split_zone_root / "split_assignments.parquet"
        for path in (applicable_path, feature_path, split_path, ref.wide_parquet_path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Required source file not found: {path}")

        applicable_rows = read_csv_dicts(applicable_path)
        applicability_path = audit_zone_root / "model_applicability.csv"
        if not applicability_path.is_file():
            applicability_path = audit_zone_root / "candidate_models.csv"
        applicability_rows = (
            read_csv_dicts(applicability_path)
            if applicability_path.is_file()
            else list(applicable_rows)
        )
        model_ids = [
            str(row.get("model_id", "")).strip()
            for row in applicable_rows
            if str(row.get("model_id", "")).strip()
        ]
        if args.model_id:
            requested = set(args.model_id)
            model_ids = [model_id for model_id in model_ids if model_id in requested]
            applicability_rows = [
                row for row in applicability_rows
                if str(row.get("model_id", "")).strip() in requested
            ]
        selected_applicable_rows = [
            row for row in applicable_rows
            if str(row.get("model_id", "")).strip() in set(model_ids)
        ]
        inapplicable_rows = [
            row for row in applicability_rows
            if str(row.get("applicable", "")).strip().lower() not in {"true", "1", "yes"}
        ]
        write_csv(zone_output_root / "model_applicability_snapshot.csv", applicability_rows)
        write_csv(zone_output_root / "applicable_models_snapshot.csv", selected_applicable_rows)
        write_csv(zone_output_root / "inapplicable_models_snapshot.csv", inapplicable_rows)

        feature_frame = pd.read_parquet(feature_path)
        split_frame = pd.read_parquet(split_path)
        stage_b_frame = pd.read_parquet(ref.wide_parquet_path)

        for model_index, model_id in enumerate(model_ids, start=1):
            print(f"    [{model_index}/{len(model_ids)}] {model_id}")
            model_rows.append(
                build_one_model(
                    ref=ref,
                    model_id=model_id,
                    feature_frame=feature_frame,
                    split_frame=split_frame,
                    stage_b_frame=stage_b_frame,
                    zone_output_root=zone_output_root,
                    args=args,
                )
            )
            if model_rows[-1]["status"] != "completed" and not args.continue_on_error:
                raise RuntimeError(model_rows[-1]["error_message"])

        write_csv(zone_output_root / "model_dataset_index.csv", model_rows)
        successful = sum(row["status"] == "completed" for row in model_rows)
        failed = len(model_rows) - successful
        zone_manifest = {
            **ref.identity_dict(),
            "schema_version": "0.1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed" if failed == 0 else "completed_with_failures",
            "source_audit_run_id": args.audit_run_id,
            "source_matrix_run_id": args.matrix_run_id,
            "source_feature_run_id": args.feature_run_id,
            "source_split_run_id": args.split_run_id,
            "candidate_model_count": len(applicability_rows),
            "applicable_model_count": len(model_ids),
            "inapplicable_model_count": len(inapplicable_rows),
            "zero_applicable_models": len(model_ids) == 0,
            "applicable_model_ids": list(model_ids),
            "selected_model_count": len(model_rows),
            "successful_model_count": successful,
            "failed_model_count": failed,
            "output_root": str(zone_output_root),
        }
        write_json(zone_output_root / "zone_dataset_manifest.json", zone_manifest)
        status = "completed" if failed == 0 else "failed"
        return ({
            **ref.identity_dict(),
            "status": status,
            "candidate_model_count": len(applicability_rows),
            "applicable_model_count": len(model_ids),
            "inapplicable_model_count": len(inapplicable_rows),
            "zero_applicable_models": len(model_ids) == 0,
            "selected_model_count": len(model_rows),
            "successful_model_count": successful,
            "failed_model_count": failed,
            "output_root": str(zone_output_root),
            "runtime_seconds": time.perf_counter() - zone_started,
            "error_type": "" if failed == 0 else "ModelDatasetFailure",
            "error_message": "" if failed == 0 else "One or more model datasets failed",
        }, model_rows)
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return ({
            **ref.identity_dict(),
            "status": "failed",
            "selected_model_count": len(model_rows),
            "successful_model_count": sum(row.get("status") == "completed" for row in model_rows),
            "failed_model_count": sum(row.get("status") != "completed" for row in model_rows),
            "output_root": str(zone_output_root),
            "runtime_seconds": time.perf_counter() - zone_started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }, model_rows)


def build_one_model(*, ref, model_id: str, feature_frame: pd.DataFrame, split_frame: pd.DataFrame, stage_b_frame: pd.DataFrame, zone_output_root: Path, args) -> dict[str, Any]:
    started = time.perf_counter()
    model_output_root = zone_output_root / "models" / make_safe_name(model_id)
    model_output_root.mkdir(parents=True, exist_ok=True)
    try:
        dataset = build_regression_pair_dataset(
            model_id=model_id,
            feature_frame=feature_frame,
            split_frame=split_frame,
            stage_b_frame=stage_b_frame,
        )
        validation_rows = validate_regression_pair_dataset(
            dataset,
            minimum_split_samples=args.minimum_split_samples,
        )
        if not validation_passed(validation_rows):
            failed = [row["check_name"] for row in validation_rows if row["status"] != "passed"]
            raise ValueError("Dataset validation failed: " + " | ".join(failed))

        split_frames = split_valid_pairs(dataset)
        summary_rows = build_dataset_summary(dataset)
        exclusion_rows = build_exclusion_summary(dataset)

        full_path = model_output_root / "regression_pairs_full.parquet"
        valid_path = model_output_root / "regression_pairs_valid.parquet"
        preview_path = model_output_root / "regression_pairs_preview.csv"
        write_dataframe_parquet(full_path, dataset.frame)
        write_dataframe_parquet(valid_path, dataset.frame[dataset.frame["pair_valid"]].reset_index(drop=True))
        write_dataframe_csv(preview_path, dataset.frame.head(args.preview_rows))
        for split, frame in split_frames.items():
            write_dataframe_parquet(model_output_root / f"{split}.parquet", frame)
        write_csv(model_output_root / "dataset_summary.csv", summary_rows)
        write_csv(model_output_root / "exclusion_summary.csv", exclusion_rows)
        write_csv(model_output_root / "dataset_validation.csv", validation_rows)

        counts = {split: int(len(frame)) for split, frame in split_frames.items()}
        manifest = {
            **ref.identity_dict(),
            **dataset.metadata,
            "schema_version": "0.1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "source_audit_run_id": args.audit_run_id,
            "source_matrix_run_id": args.matrix_run_id,
            "source_feature_run_id": args.feature_run_id,
            "source_split_run_id": args.split_run_id,
            "train_row_count": counts["train"],
            "validation_row_count": counts["validation"],
            "test_row_count": counts["test"],
            "output_root": str(model_output_root),
            "outputs": {
                "regression_pairs_full": str(full_path),
                "regression_pairs_valid": str(valid_path),
                "train": str(model_output_root / "train.parquet"),
                "validation": str(model_output_root / "validation.parquet"),
                "test": str(model_output_root / "test.parquet"),
                "dataset_summary": str(model_output_root / "dataset_summary.csv"),
                "exclusion_summary": str(model_output_root / "exclusion_summary.csv"),
                "dataset_validation": str(model_output_root / "dataset_validation.csv"),
            },
        }
        write_json(model_output_root / "model_dataset_manifest.json", manifest)
        return {
            **ref.identity_dict(),
            "model_id": model_id,
            "status": "completed",
            "predictor_column": dataset.predictor_column,
            "target_column": dataset.target_column,
            "source_row_count": len(dataset.frame),
            "valid_pair_count": int(dataset.frame["pair_valid"].sum()),
            "invalid_pair_count": int((~dataset.frame["pair_valid"]).sum()),
            "train_row_count": counts["train"],
            "validation_row_count": counts["validation"],
            "test_row_count": counts["test"],
            "output_root": str(model_output_root),
            "runtime_seconds": time.perf_counter() - started,
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:
        print(f"        ERROR: {exc}")
        return {
            **ref.identity_dict(),
            "model_id": model_id,
            "status": "failed",
            "predictor_column": "",
            "target_column": "",
            "source_row_count": "",
            "valid_pair_count": "",
            "invalid_pair_count": "",
            "train_row_count": "",
            "validation_row_count": "",
            "test_row_count": "",
            "output_root": str(model_output_root),
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


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
    parser.add_argument("--dataset-run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--aggregation-id", default=None)
    parser.add_argument("--weight-mode", default=None)
    parser.add_argument("--aggregate-zone-id", default=None)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--minimum-split-samples", type=int, default=1000)
    parser.add_argument("--preview-rows", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def build_dataset_run_id() -> str:
    return f"heat_input_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
