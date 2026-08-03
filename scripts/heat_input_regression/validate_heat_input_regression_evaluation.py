# -*- coding: utf-8 -*-
"""Independently validate a Stage C7 heat-input regression evaluation run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.evaluation.heat_input_regression import validate_evaluation_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root", type=Path, required=True)
    p.add_argument("--metric-atol", type=float, default=1e-12)
    p.add_argument("--metric-rtol", type=float, default=1e-12)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifests = sorted(args.evaluation_root.rglob("evaluation_manifest.json"))
    print("=" * 100)
    print("SCALEBRIDGE HEAT-INPUT REGRESSION EVALUATION VALIDATION")
    print("=" * 100)
    print(f"evaluation_root: {args.evaluation_root}")
    print(f"evaluation_artifact_count: {len(manifests)}")

    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for i, path in enumerate(manifests, 1):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        print(f"[{i}/{len(manifests)}] {manifest.get('aggregate_zone_id')} | {manifest.get('model_id')} | {manifest.get('estimator_type')} | {manifest.get('requested_device')}")
        try:
            checks = validate_evaluation_artifact(path, args.metric_atol, args.metric_rtol)
            failed = [c for c in checks if c["status"] == "failed"]
            status = "passed" if not failed else "failed"
        except Exception as exc:
            checks = [{
                "check_name": "validator_exception",
                "status": "failed",
                "observed_value": type(exc).__name__,
                "expected_value": "no exception",
                "message": str(exc),
            }]
            failed = checks
            status = "failed"
        result_rows.append({
            "aggregate_zone_id": manifest.get("aggregate_zone_id", ""),
            "model_id": manifest.get("model_id", ""),
            "estimator_type": manifest.get("estimator_type", ""),
            "requested_device": manifest.get("requested_device", ""),
            "resolved_device": manifest.get("resolved_device", ""),
            "status": status,
            "check_count": len(checks),
            "failed_check_count": len(failed),
            "evaluation_manifest_path": str(path),
        })
        for check in checks:
            diagnostic_rows.append({
                **result_rows[-1],
                **check,
            })

    results = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    results.to_csv(args.evaluation_root / "evaluation_validation_results.csv", index=False)
    diagnostics.to_csv(args.evaluation_root / "evaluation_validation_diagnostics.csv", index=False)
    failed_count = int((results["status"] == "failed").sum()) if not results.empty else 0
    summary = {
        "validation_status": "passed" if failed_count == 0 else "failed",
        "evaluation_artifact_count": len(manifests),
        "passed_artifact_count": int((results["status"] == "passed").sum()) if not results.empty else 0,
        "failed_artifact_count": failed_count,
        "diagnostic_check_count": int(len(diagnostics)),
        "failed_diagnostic_check_count": int((diagnostics["status"] == "failed").sum()) if not diagnostics.empty else 0,
    }
    (args.evaluation_root / "evaluation_validation_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print("EVALUATION VALIDATION SUMMARY")
    print("=" * 100)
    print(f"passed_artifact_count: {summary['passed_artifact_count']}")
    print(f"failed_artifact_count: {summary['failed_artifact_count']}")
    print(f"validation_status: {summary['validation_status']}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
