# -*- coding: utf-8 -*-
"""Run the complete ScaleBridge Phase C heat-input-regression campaign.

Pipeline
--------
C1 aggregation/readiness audit
C2 canonical feature construction
C3 train/validation/test split construction
C4 model-dataset construction
C5 model API validation
C6 model training
C7 persisted-model evaluation
C8 full-year component inference
C9 MLflow registration and optional MLflow validation

The runner accepts an explicit campaign folder path. A single timestamp suffix
is shared across all C1-C9 run identifiers so that artifacts remain discoverable
and provenance links remain deterministic.

Validation profiles
-------------------
full:
    Run all available stage validators, final inference validation, and C9
    MLflow validation.

some:
    Run the high-value validators for C2, C4, C6, C7, C8, and C9.

none:
    Run C1-C8 and register to MLflow, but skip explicit validation scripts.

The runner executes the existing stage scripts as subprocesses. It inspects
each script's ``--help`` output and resolves supported option aliases, which
keeps the orchestrator compatible with small CLI naming differences.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "heat_input_regression"

STAGE_SCRIPT_NAMES = {
    "C1": "audit_aggregation_for_heat_input_regression.py",
    "C2": "build_heat_input_regression_features.py",
    "C3": "build_heat_input_regression_splits.py",
    "C4": "build_heat_input_regression_datasets.py",
    "C5": "validate_heat_input_regression_model_api.py",
    "C6": "train_heat_input_regression_models.py",
    "C7": "evaluate_heat_input_regression_models.py",
    "C8": "run_heat_input_regression_full_year_inference.py",
    "C9": "register_phase_c_run_with_mlflow.py",
}

VALIDATOR_SCRIPT_NAMES = {
    "source": "validate_python_source_syntax.py",
    "C2_features": "validate_heat_input_regression_features_canonical_aware.py",
    "C2_timestamps": "validate_heat_input_regression_canonical_timestamps.py",
    "C2_coalescence": "validate_heat_input_regression_timestamp_coalescence.py",
    "C3": "validate_heat_input_regression_splits.py",
    "C4": "validate_heat_input_regression_datasets.py",
    "C6": "validate_heat_input_regression_training.py",
    "C7": "validate_heat_input_regression_evaluation.py",
    "C8": "validate_heat_input_regression_full_year_inference.py",
    "C9": "validate_phase_c_mlflow_tracking.py",
}


@dataclass(frozen=True)
class RunLayout:
    campaign_root: Path
    campaign_id: str
    matrix_run_id: str
    suffix: str
    phase_c_run_id: str
    audit_run_id: str
    feature_run_id: str
    split_run_id: str
    dataset_run_id: str
    c5_run_id: str
    training_run_id: str
    evaluation_run_id: str
    inference_run_id: str

    @property
    def phase_root(self) -> Path:
        return self.campaign_root / "heat_input_regression"

    @property
    def audit_root(self) -> Path:
        return self.phase_root / "audit_runs" / self.audit_run_id

    @property
    def feature_root(self) -> Path:
        return self.phase_root / "feature_runs" / self.feature_run_id

    @property
    def split_root(self) -> Path:
        return self.phase_root / "split_runs" / self.split_run_id

    @property
    def dataset_root(self) -> Path:
        return self.phase_root / "dataset_runs" / self.dataset_run_id

    @property
    def c5_root(self) -> Path:
        return self.phase_root / "model_api_validation" / self.c5_run_id

    @property
    def training_parent(self) -> Path:
        return self.phase_root / "training_runs"

    @property
    def training_root(self) -> Path:
        return self.training_parent / self.training_run_id

    @property
    def evaluation_parent(self) -> Path:
        return self.phase_root / "evaluation_runs"

    @property
    def evaluation_root(self) -> Path:
        return self.evaluation_parent / self.evaluation_run_id

    @property
    def inference_parent(self) -> Path:
        return self.phase_root / "inference_runs"

    @property
    def inference_root(self) -> Path:
        return self.inference_parent / self.inference_run_id

    @property
    def campaign_run_root(self) -> Path:
        return self.phase_root / "campaign_runs" / self.phase_c_run_id

    @property
    def mlflow_registration_manifest(self) -> Path:
        return (
            self.phase_root
            / "mlflow_registration_runs"
            / self.phase_c_run_id
            / "phase_c_mlflow_registration_manifest.json"
        )


@dataclass
class CommandResult:
    name: str
    script: str
    command: list[str]
    started_at_utc: str
    runtime_seconds: float
    return_code: int
    status: str
    log_path: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Existing campaign folder containing aggregation outputs.",
    )
    p.add_argument(
        "--matrix-run-id",
        default=None,
        help=(
            "Aggregation matrix run consumed by Phase C. If omitted, the latest "
            "successful aggregation/matrix_runs/* manifest is selected."
        ),
    )
    p.add_argument(
        "--phase-c-run-id",
        default=None,
        help=(
            "Optional complete Phase C run ID. It must end with YYYYMMDD_HHMMSS. "
            "Default: phase_c_<timestamp>."
        ),
    )
    p.add_argument(
        "--validation",
        choices=("full", "some", "none"),
        default="full",
        help="Validation profile. Default: full.",
    )
    p.add_argument(
        "--estimator-type",
        action="append",
        choices=("closed_form_linear", "pytorch_linear"),
        default=None,
        help="Estimator to train. Repeatable. Default: pytorch_linear.",
    )
    p.add_argument(
        "--pytorch-device",
        action="append",
        choices=("cpu", "cuda", "auto"),
        default=None,
        help="PyTorch device. Repeatable. Default: auto.",
    )
    p.add_argument("--max-model-datasets", type=int, default=None)
    p.add_argument("--max-artifacts", type=int, default=None)
    p.add_argument("--max-zones", type=int, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument("--disable-mlflow", action="store_true")
    p.add_argument(
        "--mlflow-validation-mode",
        choices=("full", "lightweight", "none"),
        default="full",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--start-stage",
        choices=tuple(STAGE_SCRIPT_NAMES),
        default="C1",
        help="Resume from this stage. Default: C1.",
    )
    p.add_argument(
        "--stop-stage",
        choices=tuple(STAGE_SCRIPT_NAMES),
        default="C9",
        help="Stop after this stage. Default: C9.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    campaign_root = args.campaign_root.expanduser().resolve()
    if not campaign_root.is_dir():
        raise SystemExit(f"Campaign root does not exist: {campaign_root}")

    layout = build_layout(
        campaign_root=campaign_root,
        matrix_run_id=args.matrix_run_id,
        phase_c_run_id=args.phase_c_run_id,
    )
    layout.campaign_run_root.mkdir(parents=True, exist_ok=True)
    log_root = layout.campaign_run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    write_json(
        layout.campaign_run_root / "phase_c_campaign_plan.json",
        {
            "schema_version": "0.1.0",
            "created_at_utc": utc_now(),
            "validation": args.validation,
            "start_stage": args.start_stage,
            "stop_stage": args.stop_stage,
            "layout": layout_to_json(layout),
            "configuration": serializable_args(args),
        },
    )

    print_plan(layout, args)
    commands = build_pipeline_commands(layout, args)

    if args.dry_run:
        print("\nDRY RUN COMMANDS")
        for name, command in commands:
            print(f"\n{name}")
            print(format_command(command))
        return 0

    results: list[CommandResult] = []
    failed = False
    started = time.perf_counter()

    for sequence, (name, command) in enumerate(commands, 1):
        print("\n" + "=" * 100)
        print(f"[{sequence}/{len(commands)}] {name}")
        print("=" * 100)
        print(format_command(command))

        log_path = log_root / f"{sequence:02d}_{safe_token(name)}.log"
        result = run_command(name=name, command=command, log_path=log_path)
        results.append(result)

        print(f"return_code: {result.return_code}")
        print(f"runtime_seconds: {result.runtime_seconds:.3f}")
        print(f"log_path: {result.log_path}")

        if result.return_code != 0:
            failed = True
            if not args.continue_on_error:
                break

    summary = {
        "schema_version": "0.1.0",
        "created_at_utc": utc_now(),
        "campaign_id": layout.campaign_id,
        "campaign_root": str(layout.campaign_root),
        "matrix_run_id": layout.matrix_run_id,
        "phase_c_run_id": layout.phase_c_run_id,
        "validation": args.validation,
        "status": "failed" if failed else "completed",
        "runtime_seconds": time.perf_counter() - started,
        "command_count": len(results),
        "passed_command_count": sum(r.return_code == 0 for r in results),
        "failed_command_count": sum(r.return_code != 0 for r in results),
        "availability_summary": collect_availability_summary(layout),
        "results": [asdict(r) for r in results],
        "mlflow_registration_manifest": (
            str(layout.mlflow_registration_manifest)
            if layout.mlflow_registration_manifest.is_file()
            else None
        ),
    }
    summary_path = layout.campaign_run_root / "phase_c_campaign_run_manifest.json"
    write_json(summary_path, summary)

    print("\n" + "=" * 100)
    print("PHASE C CAMPAIGN SUMMARY")
    print("=" * 100)
    print(f"phase_c_run_id: {layout.phase_c_run_id}")
    print(f"status: {summary['status']}")
    print(f"passed_command_count: {summary['passed_command_count']}")
    print(f"failed_command_count: {summary['failed_command_count']}")
    print(f"availability_summary: {summary['availability_summary']}")
    print(f"campaign_run_manifest: {summary_path}")
    return 1 if failed else 0


def build_layout(
    *,
    campaign_root: Path,
    matrix_run_id: str | None,
    phase_c_run_id: str | None,
) -> RunLayout:
    campaign_id = campaign_root.name
    suffix = extract_or_create_suffix(phase_c_run_id)
    resolved_phase_id = phase_c_run_id or f"phase_c_{suffix}"
    resolved_matrix_id = matrix_run_id or discover_latest_matrix_run_id(campaign_root)

    return RunLayout(
        campaign_root=campaign_root,
        campaign_id=campaign_id,
        matrix_run_id=resolved_matrix_id,
        suffix=suffix,
        phase_c_run_id=resolved_phase_id,
        audit_run_id=f"heat_input_audit_{suffix}",
        feature_run_id=f"heat_input_features_{suffix}",
        split_run_id=f"heat_input_splits_{suffix}",
        dataset_run_id=f"heat_input_datasets_{suffix}",
        c5_run_id=f"c5_{suffix}",
        training_run_id=f"c6_pytorch_{suffix}",
        evaluation_run_id=f"c7_pytorch_{suffix}",
        inference_run_id=f"c8_pytorch_{suffix}",
    )


def discover_latest_matrix_run_id(campaign_root: Path) -> str:
    matrix_root = campaign_root / "aggregation" / "matrix_runs"
    manifests = sorted(
        matrix_root.glob("*/aggregation_matrix_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifests:
        payload = read_json(manifest_path)
        failed = int(payload.get("failed_plan_count", 0) or 0)
        successful = int(payload.get("successful_plan_count", 0) or 0)
        if failed == 0 and successful > 0:
            return str(payload.get("matrix_run_id") or manifest_path.parent.name)
    raise RuntimeError(
        f"No successful aggregation matrix run found under {matrix_root}. "
        "Provide --matrix-run-id explicitly."
    )


def build_pipeline_commands(
    layout: RunLayout,
    args: argparse.Namespace,
) -> list[tuple[str, list[str]]]:
    stage_order = list(STAGE_SCRIPT_NAMES)
    start = stage_order.index(args.start_stage)
    stop = stage_order.index(args.stop_stage)
    if start > stop:
        raise ValueError("--start-stage must not come after --stop-stage")

    commands: list[tuple[str, list[str]]] = []
    estimators = args.estimator_type or ["pytorch_linear"]
    devices = args.pytorch_device or ["auto"]

    stage_semantics = {
        "C1": {
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "max_zones": args.max_zones,
            "continue_on_error": args.continue_on_error,
        },
        "C2": {
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "max_zones": args.max_zones,
            "continue_on_error": args.continue_on_error,
            "overwrite_existing": args.overwrite_existing,
        },
        "C3": {
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
            "max_zones": args.max_zones,
            "continue_on_error": args.continue_on_error,
            "overwrite_existing": args.overwrite_existing,
        },
        "C4": {
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
            "dataset_run_id": layout.dataset_run_id,
            "max_zones": args.max_zones,
            "continue_on_error": args.continue_on_error,
            "overwrite_existing": args.overwrite_existing,
        },
        "C5": {
            "output_root": layout.c5_root,
            "dataset_root": layout.dataset_root,
        },
        "C6": {
            "dataset_root": layout.dataset_root,
            "output_root": layout.training_parent,
            "training_run_id": layout.training_run_id,
            "estimator_type": estimators,
            "pytorch_device": devices,
            "max_model_datasets": args.max_model_datasets,
            "continue_on_error": args.continue_on_error,
            "overwrite_existing": args.overwrite_existing,
        },
        "C7": {
            "training_root": layout.training_root,
            "output_root": layout.evaluation_parent,
            "evaluation_run_id": layout.evaluation_run_id,
            "estimator_type": estimators,
            "requested_device": devices,
            "max_artifacts": args.max_artifacts,
            "continue_on_error": args.continue_on_error,
        },
        "C8": {
            "evaluation_root": layout.evaluation_root,
            "feature_root": layout.feature_root,
            "dataset_root": layout.dataset_root,
            "output_root": layout.inference_parent,
            "inference_run_id": layout.inference_run_id,
            "estimator_type": estimators,
            "requested_device": devices,
            "max_artifacts": args.max_artifacts,
            "continue_on_error": args.continue_on_error,
            "overwrite_existing": args.overwrite_existing,
        },
        "C9": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "phase_c_run_id": layout.phase_c_run_id,
            "validation_mode": args.mlflow_validation_mode,
        },
    }

    for stage in stage_order[start : stop + 1]:
        if stage == "C9" and args.disable_mlflow:
            continue
        script = require_script(STAGE_SCRIPT_NAMES[stage])
        command = build_adaptive_command(script, stage_semantics[stage])
        commands.append((f"{stage} {script.stem}", command))

        if stage != "C9":
            commands.extend(validation_commands_for_stage(stage, layout, args))

    if (
        not args.disable_mlflow
        and "C9" in stage_order[start : stop + 1]
        and args.validation != "none"
    ):
        commands.extend(validation_commands_for_stage("C9", layout, args))

    return commands


def validation_commands_for_stage(
    stage: str,
    layout: RunLayout,
    args: argparse.Namespace,
) -> list[tuple[str, list[str]]]:
    if args.validation == "none":
        return []

    full_keys = {
        "C1": ["source"],
        "C2": ["C2_features", "C2_timestamps", "C2_coalescence"],
        "C3": ["C3"],
        "C4": ["C4"],
        "C5": [],
        "C6": ["C6"],
        "C7": ["C7"],
        "C8": ["C8"],
        "C9": ["C9"],
    }
    some_keys = {
        "C1": [],
        "C2": ["C2_features", "C2_timestamps"],
        "C3": [],
        "C4": ["C4"],
        "C5": [],
        "C6": ["C6"],
        "C7": ["C7"],
        "C8": ["C8"],
        "C9": ["C9"],
    }
    keys = full_keys[stage] if args.validation == "full" else some_keys[stage]

    semantics = {
        "source": {
            "paths": [
                REPO_ROOT / "src" / "scalebridge" / "data" / "heat_input_regression",
                REPO_ROOT / "src" / "scalebridge" / "models" / "heat_input_regression",
                REPO_ROOT / "src" / "scalebridge" / "training" / "heat_input_regression",
                REPO_ROOT / "src" / "scalebridge" / "evaluation" / "heat_input_regression",
                REPO_ROOT / "src" / "scalebridge" / "inference" / "heat_input_regression",
                REPO_ROOT / "src" / "scalebridge" / "tracking" / "mlflow",
                SCRIPT_ROOT,
            ],
        },

        "C2_features": {
            "feature_root": layout.feature_root,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
        },
        "C2_timestamps": {
            "feature_root": layout.feature_root,
        },
        "C2_coalescence": {
            "feature_root": layout.feature_root,
        },

        "C3": {
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
        },

        "C4": {
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
            "dataset_run_id": layout.dataset_run_id,
        },

        "C6": {
            "training_root": layout.training_root,
        },

        "C7": {
            "evaluation_root": layout.evaluation_root,
        },

        "C8": {
            "inference_root": layout.inference_root,
        },

        "C9": {
            "registration_manifest": layout.mlflow_registration_manifest,
        },
    }

    # C9 validation runs after registration. Do not freeze C6-C8 task counts
    # while the pipeline command list is being planned, because those artifacts
    # do not exist yet. The C9 validator resolves omitted task expectations from
    # the completed registration manifest's availability_summary at runtime.
    semantics["C9"].update({"expected_stage_runs": 8})

    commands: list[tuple[str, list[str]]] = []
    for key in keys:
        script = require_script(VALIDATOR_SCRIPT_NAMES[key])
        command = build_adaptive_command(script, semantics[key])
        commands.append((f"VALIDATE {key}", command))
    return commands


OPTION_ALIASES = {
    "paths": ("--paths",),
    "campaign_root": ("--campaign-root",),
    "campaign_id": ("--campaign-id",),
    "matrix_run_id": ("--matrix-run-id",),
    "audit_run_id": ("--audit-run-id",),
    "feature_run_id": ("--feature-run-id",),
    "split_run_id": ("--split-run-id",),
    "dataset_run_id": ("--dataset-run-id",),
    "training_run_id": ("--training-run-id",),
    "evaluation_run_id": ("--evaluation-run-id",),
    "inference_run_id": ("--inference-run-id",),
    "phase_c_run_id": ("--phase-c-run-id",),
    "audit_root": ("--audit-root",),
    "feature_root": ("--feature-root",),
    "split_root": ("--split-root",),
    "dataset_root": ("--dataset-root",),
    "training_root": ("--training-root",),
    "evaluation_root": ("--evaluation-root",),
    "inference_root": ("--inference-root",),
    "output_root": ("--output-root",),
    "validation_mode": ("--validation-mode",),
    "registration_manifest": ("--registration-manifest",),
    "estimator_type": ("--estimator-type",),
    "pytorch_device": ("--pytorch-device",),
    "requested_device": ("--requested-device",),
    "max_model_datasets": ("--max-model-datasets",),
    "max_artifacts": ("--max-artifacts",),
    "max_zones": (
        "--max-zones",
        "--max-aggregation-zones",
        "--max-selected-zones",
    ),
    "continue_on_error": ("--continue-on-error",),
    "overwrite_existing": ("--overwrite-existing",),
    "expected_stage_runs": ("--expected-stage-runs",),
    "expected_training_task_runs": ("--expected-training-task-runs",),
    "expected_evaluation_task_runs": ("--expected-evaluation-task-runs",),
    "expected_inference_task_runs": ("--expected-inference-task-runs",),
}


def build_adaptive_command(
    script: Path,
    semantic_values: dict[str, object],
) -> list[str]:
    help_text = get_help_text(script)
    command = [sys.executable, str(script)]

    for semantic, value in semantic_values.items():
        if value is None or value is False:
            continue
        option = first_supported_option(help_text, OPTION_ALIASES[semantic])
        if option is None:
            # Optional controls may not exist on every historical stage script.
            if semantic in {
                "max_zones",
                "continue_on_error",
                "overwrite_existing",
                "validation_mode",
                "expected_stage_runs",
                "expected_training_task_runs",
                "expected_evaluation_task_runs",
                "expected_inference_task_runs",
            }:
                continue
            raise RuntimeError(
                f"{script.name} does not expose a supported option for "
                f"{semantic!r}. Tried: {OPTION_ALIASES[semantic]}"
            )

        if semantic == "paths" and isinstance(value, (list, tuple)):
            command.append(option)
            command.extend(str(item) for item in value)
        elif value is True:
            command.append(option)
        elif isinstance(value, (list, tuple)):
            for item in value:
                command.extend([option, str(item)])
        else:
            command.extend([option, str(value)])

    return command


def get_help_text(script: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Could not inspect CLI for {script}: {completed.stdout}"
        )
    return completed.stdout


def first_supported_option(help_text: str, aliases: Iterable[str]) -> str | None:
    return next((item for item in aliases if item in help_text), None)


def require_script(name: str) -> Path:
    path = SCRIPT_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase C script is missing: {path}")
    return path


def run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
) -> CommandResult:
    started_at = utc_now()
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    with log_path.open("w", encoding="utf-8", newline="") as stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
            stream.flush()
        return_code = process.wait()

    runtime = time.perf_counter() - started
    return CommandResult(
        name=name,
        script=Path(command[1]).name,
        command=command,
        started_at_utc=started_at,
        runtime_seconds=runtime,
        return_code=return_code,
        status="passed" if return_code == 0 else "failed",
        log_path=str(log_path),
    )


def infer_expected_counts(layout: RunLayout) -> dict[str, int]:
    return {
        "expected_stage_runs": 8,
        "expected_training_task_runs": count_json(
            layout.training_root, "training_manifest.json"
        ),
        "expected_evaluation_task_runs": count_json(
            layout.evaluation_root, "evaluation_manifest.json"
        ),
        "expected_inference_task_runs": count_json(
            layout.inference_root, "annual_component_predictions_manifest.json"
        ),
    }


def count_json(root: Path, filename: str) -> int:
    return len(list(root.rglob(filename))) if root.is_dir() else 0


def collect_availability_summary(layout: RunLayout) -> dict[str, object]:
    """Collect availability-aware counts from completed stage manifests."""
    manifest_candidates = {
        "C1": [layout.audit_root / "heat_input_regression_audit_manifest.json"],
        "C2": [layout.feature_root / "heat_input_feature_run_manifest.json"],
        "C3": [layout.split_root / "split_run_manifest.json"],
        "C4": [layout.dataset_root / "dataset_run_manifest.json"],
        "C6": [layout.training_root / "training_run_manifest.json"],
        "C7": [layout.evaluation_root / "evaluation_run_manifest.json"],
        "C8": [layout.inference_root / "inference_run_manifest.json"],
    }
    payloads: dict[str, dict[str, object]] = {}
    for stage, candidates in manifest_candidates.items():
        for path in candidates:
            if path.is_file():
                payloads[stage] = read_json(path)
                break
        if stage not in payloads:
            run_root = {
                "C1": layout.audit_root, "C2": layout.feature_root,
                "C3": layout.split_root, "C4": layout.dataset_root,
                "C6": layout.training_root, "C7": layout.evaluation_root,
                "C8": layout.inference_root,
            }[stage]
            for path in sorted(run_root.glob("*.json")) if run_root.is_dir() else []:
                try:
                    candidate = read_json(path)
                except Exception:
                    continue
                if isinstance(candidate, dict):
                    payloads[stage] = candidate
                    break

    c1 = payloads.get("C1", {})
    c4 = payloads.get("C4", {})
    c6 = payloads.get("C6", {})
    c7 = payloads.get("C7", {})
    c8 = payloads.get("C8", {})
    return {
        "candidate_model_count": c1.get("candidate_model_count"),
        "applicable_model_count": c1.get("applicable_model_count"),
        "structurally_inapplicable_model_count": c1.get(
            "structurally_inapplicable_model_count"
        ),
        "invalid_model_count": c1.get("invalid_model_count"),
        "missing_expected_data_model_count": c1.get(
            "missing_expected_data_model_count"
        ),
        "created_dataset_count": c4.get(
            "successful_model_count", c4.get("selected_model_count")
        ),
        "trained_model_count": c6.get("completed_training_task_count"),
        "evaluated_model_count": c7.get("completed_evaluation_count"),
        "inference_zone_count": c8.get("completed_zone_count"),
        "zero_component_zone_count": c8.get("zero_component_zone_count"),
        "inferred_component_count": c8.get(
            "total_component_count", c8.get("selected_evaluation_artifact_count")
        ),
    }


def extract_or_create_suffix(phase_c_run_id: str | None) -> str:
    if phase_c_run_id is None:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    import re

    match = re.search(r"(\d{8}_\d{6})$", phase_c_run_id)
    if not match:
        raise ValueError(
            "--phase-c-run-id must end in YYYYMMDD_HHMMSS."
        )
    return match.group(1)


def print_plan(layout: RunLayout, args: argparse.Namespace) -> None:
    print("=" * 100)
    print("SCALEBRIDGE PHASE C CAMPAIGN PLAN")
    print("=" * 100)
    print(f"campaign_id: {layout.campaign_id}")
    print(f"campaign_root: {layout.campaign_root}")
    print(f"matrix_run_id: {layout.matrix_run_id}")
    print(f"phase_c_run_id: {layout.phase_c_run_id}")
    print(f"validation: {args.validation}")
    print(f"start_stage: {args.start_stage}")
    print(f"stop_stage: {args.stop_stage}")
    print(f"mlflow: {not args.disable_mlflow}")
    print(f"campaign_run_root: {layout.campaign_run_root}")


def layout_to_json(layout: RunLayout) -> dict[str, str]:
    payload = asdict(layout)
    return {key: str(value) for key, value in payload.items()} | {
        "phase_root": str(layout.phase_root),
        "audit_root": str(layout.audit_root),
        "feature_root": str(layout.feature_root),
        "split_root": str(layout.split_root),
        "dataset_root": str(layout.dataset_root),
        "c5_root": str(layout.c5_root),
        "training_root": str(layout.training_root),
        "evaluation_root": str(layout.evaluation_root),
        "inference_root": str(layout.inference_root),
    }


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_token(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in value
    ).strip("_")


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
