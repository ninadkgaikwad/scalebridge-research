# -*- coding: utf-8 -*-
"""Train and persist Stage C6 heat-input regression models from a C4 dataset run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.training.heat_input_regression import (  # noqa: E402
    EstimatorTrainingConfig,
    discover_model_datasets,
    train_model_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Validated C4 dataset run root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Parent directory for the C6 training run.")
    parser.add_argument("--training-run-id", default=None)
    parser.add_argument("--estimator-type", action="append", choices=["closed_form_linear", "pytorch_linear"], default=None)
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--aggregate-zone-id", action="append", default=None)
    parser.add_argument("--max-model-datasets", type=int, default=None)
    parser.add_argument("--fit-intercept", action=argparse.BooleanOptionalAction, default=None, help="Optional global override. Default: read fit_intercept from each C4 model_dataset_manifest.json.")
    parser.add_argument("--ridge-alpha", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pytorch-device",
        action="append",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Repeat to train PyTorch separately on CPU and CUDA.",
    )
    parser.add_argument("--reload-atol", type=float, default=1e-12)
    parser.add_argument("--reload-rtol", type=float, default=1e-12)
    parser.add_argument("--prediction-preview-rows", type=int, default=100)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_id = args.training_run_id or f"heat_input_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_root = args.output_root.resolve() / run_id
    training_root.mkdir(parents=True, exist_ok=True)
    estimators = args.estimator_type or ["closed_form_linear", "pytorch_linear"]

    references = discover_model_datasets(
        args.dataset_root,
        model_ids=set(args.model_id or []),
        aggregate_zone_ids=set(args.aggregate_zone_id or []),
        max_model_datasets=args.max_model_datasets,
    )
    # A validated C4 run may legitimately contain zero model datasets when no
    # Phase C relationship is applicable to the selected zone set. This is a
    # successful zero-task training run, not a campaign failure.

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION TRAINER")
    print("=" * 100)
    print(f"dataset_root: {args.dataset_root.resolve()}")
    print(f"training_root: {training_root}")
    print(f"training_run_id: {run_id}")
    print(f"selected_model_dataset_count: {len(references)}")
    print(f"estimators: {', '.join(estimators)}")

    selected_rows = [reference.identity_dict() | {"manifest_path": str(reference.manifest_path)} for reference in references]
    pd.DataFrame(selected_rows).to_csv(training_root / "selected_model_datasets.csv", index=False)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pytorch_devices = args.pytorch_device or ["cpu"]
    tasks = [
        (reference, estimator, device)
        for reference in references
        for estimator in estimators
        for device in (pytorch_devices if estimator == "pytorch_linear" else ["cpu"])
    ]
    task_count = len(tasks)
    task_index = 0
    started = time.perf_counter()
    for reference, estimator, device in tasks:
        task_index += 1
        print(
            f"[{task_index}/{task_count}] {reference.case_id} | {reference.aggregation_id} | "
            f"{reference.aggregate_zone_id} | {reference.model_id} | {estimator} | {device}"
        )
        config = EstimatorTrainingConfig(
            estimator_type=estimator,
            fit_intercept=args.fit_intercept,
            ridge_alpha=args.ridge_alpha,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            tolerance=args.tolerance,
            patience=args.patience,
            seed=args.seed,
            device=device,
        )
        try:
            result = train_model_dataset(
                reference,
                config,
                training_root=training_root,
                training_run_id=run_id,
                overwrite_existing=args.overwrite_existing,
                reload_atol=args.reload_atol,
                reload_rtol=args.reload_rtol,
                prediction_preview_rows=args.prediction_preview_rows,
            )
            results.append({**result.row, "status": "completed", "error_type": "", "error_message": ""})
        except Exception as exc:
            row = {
                **reference.identity_dict(),
                "estimator_type": estimator,
                "requested_device": device,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            results.append(row)
            failures.append(row)
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                break
        if failures and not args.continue_on_error:
            break

    result_columns = [
        "campaign_id", "case_id", "aggregation_id", "weight_mode",
        "aggregate_zone_id", "model_id", "estimator_type",
        "requested_device", "resolved_device", "status", "error_type",
        "error_message",
    ]
    frame = pd.DataFrame(results)
    if frame.empty:
        frame = pd.DataFrame(columns=result_columns)
    frame.to_csv(training_root / "training_results.csv", index=False)
    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=result_columns)
    failure_frame.to_csv(training_root / "training_failures.csv", index=False)
    successful = int((frame["status"] == "completed").sum()) if len(frame) else 0
    failed = int((frame["status"] == "failed").sum()) if len(frame) else 0
    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run_id": run_id,
        "status": "completed" if failed == 0 else "completed_with_failures",
        "dataset_root": str(args.dataset_root.resolve()),
        "training_root": str(training_root),
        "estimator_types": estimators,
        "pytorch_devices": pytorch_devices,
        "selected_model_dataset_count": len(references),
        "zero_selected_model_datasets": len(references) == 0,
        "requested_training_task_count": task_count,
        "completed_training_task_count": successful,
        "failed_training_task_count": failed,
        "runtime_seconds": time.perf_counter() - started,
        "configuration": vars(args) | {"dataset_root": str(args.dataset_root), "output_root": str(args.output_root)},
        "outputs": {
            "selected_model_datasets": str(training_root / "selected_model_datasets.csv"),
            "training_results": str(training_root / "training_results.csv"),
            "training_failures": str(training_root / "training_failures.csv"),
        },
    }
    write_json(training_root / "training_run_manifest.json", manifest)

    print("\n" + "=" * 100)
    print("TRAINING SUMMARY")
    print("=" * 100)
    print(f"completed_training_task_count: {successful}")
    print(f"failed_training_task_count: {failed}")
    print(f"training_root: {training_root}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
