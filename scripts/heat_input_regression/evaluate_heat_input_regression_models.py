# -*- coding: utf-8 -*-
"""Evaluate persisted Stage C6 heat-input regression artifacts without refitting."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.evaluation.heat_input_regression import (  # noqa: E402
    discover_training_artifacts,
    evaluate_training_artifact,
    build_phvac_building_reconstruction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--evaluation-run-id", required=True)
    p.add_argument("--model-id", action="append", default=[])
    p.add_argument("--aggregate-zone-id", action="append", default=[])
    p.add_argument("--estimator-type", action="append", default=[])
    p.add_argument("--requested-device", action="append", default=[])
    p.add_argument("--max-artifacts", type=int, default=None)
    p.add_argument("--prediction-preview-rows", type=int, default=200)
    p.add_argument("--no-full-predictions", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    evaluation_root = args.output_root / args.evaluation_run_id
    evaluation_root.mkdir(parents=True, exist_ok=True)
    refs = discover_training_artifacts(
        args.training_root,
        model_ids=args.model_id,
        aggregate_zone_ids=args.aggregate_zone_id,
        estimator_types=args.estimator_type,
        requested_devices=args.requested_device,
        max_artifacts=args.max_artifacts,
    )
    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION EVALUATOR")
    print("=" * 100)
    print(f"training_root: {args.training_root}")
    print(f"evaluation_root: {evaluation_root}")
    print(f"selected_training_artifact_count: {len(refs)}")

    selected_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for index, ref in enumerate(refs, 1):
        print(f"[{index}/{len(refs)}] {ref.aggregate_zone_id} | {ref.model_id} | {ref.estimator_type} | {ref.requested_device}")
        selected_rows.append({
            "case_id": ref.case_id,
            "aggregation_id": ref.aggregation_id,
            "weight_mode": ref.weight_mode,
            "aggregate_zone_id": ref.aggregate_zone_id,
            "model_id": ref.model_id,
            "estimator_type": ref.estimator_type,
            "requested_device": ref.requested_device,
            "resolved_device": ref.resolved_device,
            "training_manifest_path": str(ref.training_manifest_path),
        })
        out_dir = evaluation_root / "cases" / ref.case_id / ref.aggregation_id / ref.weight_mode / ref.aggregate_zone_id / ref.model_id / f"{ref.estimator_type}_{ref.requested_device}"
        try:
            result = evaluate_training_artifact(
                ref,
                out_dir,
                prediction_rows=args.prediction_preview_rows,
                write_full_predictions=not args.no_full_predictions,
            )
            result_rows.append({
                **selected_rows[-1],
                "status": result.status,
                "evaluation_dir": str(result.evaluation_dir),
                "metrics_path": str(result.metrics_path),
                "manifest_path": str(result.manifest_path),
                "error_type": "",
                "error_message": "",
            })
        except Exception as exc:
            result_rows.append({
                **selected_rows[-1],
                "status": "failed",
                "evaluation_dir": str(out_dir),
                "metrics_path": "",
                "manifest_path": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                break

    selected_columns = ["case_id", "aggregation_id", "weight_mode", "aggregate_zone_id", "model_id", "estimator_type", "requested_device", "resolved_device", "training_manifest_path"]
    result_columns = selected_columns + ["status", "evaluation_dir", "metrics_path", "manifest_path", "error_type", "error_message"]
    selected = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(columns=selected_columns)
    results = pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=result_columns)
    selected.to_csv(evaluation_root / "selected_training_artifacts.csv", index=False)
    results.to_csv(evaluation_root / "evaluation_results.csv", index=False)
    failures = results.loc[results["status"] != "completed"] if len(results) else pd.DataFrame(columns=result_columns)
    failures.to_csv(evaluation_root / "evaluation_failures.csv", index=False)

    building_phvac_metrics_path = build_phvac_building_reconstruction(evaluation_root)

    manifest = {
        "evaluation_run_id": args.evaluation_run_id,
        "training_root": str(args.training_root),
        "evaluation_root": str(evaluation_root),
        "selected_training_artifact_count": int(len(refs)),
        "zero_selected_training_artifacts": len(refs) == 0,
        "completed_evaluation_count": int((results["status"] == "completed").sum()) if not results.empty else 0,
        "failed_evaluation_count": int((results["status"] != "completed").sum()) if not results.empty else 0,
        "write_full_predictions": not args.no_full_predictions,
        "runtime_seconds": time.perf_counter() - started,
        "building_phvac_metrics_path": str(building_phvac_metrics_path or ""),
    }
    (evaluation_root / "evaluation_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    print(f"completed_evaluation_count: {manifest['completed_evaluation_count']}")
    print(f"failed_evaluation_count: {manifest['failed_evaluation_count']}")
    print(f"evaluation_root: {evaluation_root}")
    return 1 if manifest["failed_evaluation_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
