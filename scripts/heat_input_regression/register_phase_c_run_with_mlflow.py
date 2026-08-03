# -*- coding: utf-8 -*-
"""Register a completed ScaleBridge Phase C C1-C8 run into MLflow.

Normal usage requires only --campaign-id and --phase-c-run-id.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.tracking.mlflow.heat_input_regression import (
    PhaseCTrackingConfig,
    discover_phase_c_run,
    register_phase_c_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--phase-c-run-id", required=True)
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--validation-mode", choices=["full", "lightweight", "none"], default="full")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--no-compact-artifacts", action="store_true")
    parser.add_argument("--log-model-artifacts", action="store_true")
    parser.add_argument("--max-artifact-bytes", type=int, default=20_000_000)
    parser.add_argument("--registration-output-dir", type=Path, default=None)
    for stage in range(1, 9):
        parser.add_argument(f"--c{stage}-manifest", type=Path, default=None)
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--inference-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_overrides = {
        f"C{stage}": getattr(args, f"c{stage}_manifest")
        for stage in range(1, 9)
        if getattr(args, f"c{stage}_manifest") is not None
    }
    discovered = discover_phase_c_run(
        campaign_id=args.campaign_id,
        phase_c_run_id=args.phase_c_run_id,
        campaign_root=args.campaign_root,
        stage_manifest_overrides=stage_overrides,
        training_root_override=args.training_root,
        evaluation_root_override=args.evaluation_root,
        inference_root_override=args.inference_root,
    )
    registration_output_dir = (
        args.registration_output_dir.resolve()
        if args.registration_output_dir is not None
        else discovered["registration_output_dir"]
    )
    print("=" * 100)
    print("SCALEBRIDGE C9 PHASE C DISCOVERY")
    print("=" * 100)
    print(f"campaign_id: {args.campaign_id}")
    print(f"phase_c_run_id: {args.phase_c_run_id}")
    print(f"run_suffix: {discovered['run_suffix']}")
    print(f"phase_c_root: {discovered['phase_c_root']}")
    for stage in sorted(discovered["stage_manifests"]):
        print(f"{stage}_manifest: {discovered['stage_manifests'][stage]}")
    print(f"training_root: {discovered['training_root']}")
    print(f"evaluation_root: {discovered['evaluation_root']}")
    print(f"inference_root: {discovered['inference_root']}")
    print(f"registration_output_dir: {registration_output_dir}")
    print("")
    result = register_phase_c_run(
        config=PhaseCTrackingConfig(
            campaign_id=args.campaign_id,
            phase_c_run_id=args.phase_c_run_id,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            validation_mode=args.validation_mode,
            strict=not args.non_strict,
            log_compact_artifacts=not args.no_compact_artifacts,
            log_model_artifacts=args.log_model_artifacts,
            max_artifact_bytes=args.max_artifact_bytes,
        ),
        stage_manifests=discovered["stage_manifests"],
        training_root=discovered["training_root"],
        evaluation_root=discovered["evaluation_root"],
        inference_root=discovered["inference_root"],
        registration_output_dir=registration_output_dir,
    )
    print("=" * 100)
    print("SCALEBRIDGE C9 PHASE C MLFLOW REGISTRATION")
    print("=" * 100)
    for key, value in result.to_dict().items():
        print(f"{key}: {value}")
    print(f"registration_manifest: {registration_output_dir / 'phase_c_mlflow_registration_manifest.json'}")
    return 1 if result.failed_registration_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
