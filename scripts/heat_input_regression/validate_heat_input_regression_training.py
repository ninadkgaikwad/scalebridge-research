# -*- coding: utf-8 -*-
"""Independently validate every completed artifact in a Stage C6 training run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.training.heat_input_regression import validate_training_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True, help="One C6 training run directory.")
    parser.add_argument("--coefficient-atol", type=float, default=0.0)
    parser.add_argument("--prediction-atol", type=float, default=1e-12)
    parser.add_argument("--prediction-rtol", type=float, default=1e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.training_root.resolve()
    run_manifest_path = root / "training_run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Training run manifest not found: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifests = sorted(root.rglob("training_manifest.json"))
    requested_task_count = int(run_manifest.get("requested_training_task_count", 0) or 0)
    if not manifests and requested_task_count > 0:
        raise ValueError(f"No completed training manifests found under {root} for {requested_task_count} requested tasks")

    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION TRAINING VALIDATION")
    print("=" * 100)
    print(f"training_root: {root}")
    print(f"training_artifact_count: {len(manifests)}")

    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for index, manifest_path in enumerate(manifests, start=1):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = " | ".join(
            str(payload.get(key, ""))
            for key in ("aggregate_zone_id", "model_id", "estimator_type")
        )
        print(f"[{index}/{len(manifests)}] {label}")
        try:
            checks = validate_training_artifact(
                manifest_path,
                coefficient_atol=args.coefficient_atol,
                prediction_atol=args.prediction_atol,
                prediction_rtol=args.prediction_rtol,
            )
            for check in checks:
                diagnostic_rows.append(
                    {
                        "training_manifest": str(manifest_path),
                        "case_id": payload.get("case_id", ""),
                        "aggregation_id": payload.get("aggregation_id", ""),
                        "aggregate_zone_id": payload.get("aggregate_zone_id", ""),
                        "model_id": payload.get("model_id", ""),
                        "estimator_type": payload.get("estimator_type", ""),
                        **check,
                    }
                )
            failed = [check for check in checks if check["status"] != "passed"]
            result_rows.append(
                {
                    "training_manifest": str(manifest_path),
                    "case_id": payload.get("case_id", ""),
                    "aggregation_id": payload.get("aggregation_id", ""),
                    "aggregate_zone_id": payload.get("aggregate_zone_id", ""),
                    "model_id": payload.get("model_id", ""),
                    "estimator_type": payload.get("estimator_type", ""),
                    "status": "passed" if not failed else "failed",
                    "check_count": len(checks),
                    "failed_check_count": len(failed),
                    "error_message": "" if not failed else " | ".join(check["check_name"] for check in failed),
                }
            )
        except Exception as exc:
            result_rows.append(
                {
                    "training_manifest": str(manifest_path),
                    "case_id": payload.get("case_id", ""),
                    "aggregation_id": payload.get("aggregation_id", ""),
                    "aggregate_zone_id": payload.get("aggregate_zone_id", ""),
                    "model_id": payload.get("model_id", ""),
                    "estimator_type": payload.get("estimator_type", ""),
                    "status": "failed",
                    "check_count": 0,
                    "failed_check_count": 1,
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if results.empty:
        results = pd.DataFrame(columns=["training_manifest", "case_id", "aggregation_id", "aggregate_zone_id", "model_id", "estimator_type", "status", "check_count", "failed_check_count", "error_message"])
    if diagnostics.empty:
        diagnostics = pd.DataFrame(columns=["training_manifest", "case_id", "aggregation_id", "aggregate_zone_id", "model_id", "estimator_type", "check_name", "status"])
    results.to_csv(root / "training_validation_results.csv", index=False)
    diagnostics.to_csv(root / "training_validation_diagnostics.csv", index=False)
    failed_count = int((results["status"] == "failed").sum()) if len(results) else 0
    passed_count = int((results["status"] == "passed").sum()) if len(results) else 0
    manifest = {
        "schema_version": "0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "passed" if failed_count == 0 else "failed",
        "training_root": str(root),
        "artifact_count": len(manifests),
        "passed_artifact_count": passed_count,
        "failed_artifact_count": failed_count,
        "diagnostic_check_count": int(len(diagnostics)),
        "failed_diagnostic_check_count": int((diagnostics["status"] == "failed").sum()) if not diagnostics.empty else 0,
        "outputs": {
            "results": str(root / "training_validation_results.csv"),
            "diagnostics": str(root / "training_validation_diagnostics.csv"),
        },
    }
    (root / "training_validation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 100)
    print("TRAINING VALIDATION SUMMARY")
    print("=" * 100)
    print(f"passed_artifact_count: {passed_count}")
    print(f"failed_artifact_count: {failed_count}")
    print(f"validation_status: {manifest['validation_status']}")
    print(f"training_root: {root}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
