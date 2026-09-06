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

The runner exposes the complete public configuration supported by the current
C1-C9 stage CLIs and validators. Configuration may be supplied as JSON and/or
CLI overrides. A single timestamp suffix is shared across C1-C9 run identifiers
so artifacts remain discoverable and provenance links remain deterministic.

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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig
from scalebridge.data.heat_input_regression.discovery import resolve_inputs

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

DIAGNOSTIC_SCRIPT_NAMES = {
    "C8_missing_values": "audit_heat_input_regression_inference_missing_values.py",
    "C8_residual_gaps": "audit_heat_input_regression_residual_gaps.py",
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    suppress = argparse.SUPPRESS

    # Machine-readable contract / configuration-file controls.
    p.add_argument("--config", type=Path, default=None, help="PhaseCCampaignConfig JSON file. Explicit CLI options override it.")
    p.add_argument("--print-capabilities", action="store_true", help="Print the machine-readable Phase C capability manifest and exit.")
    p.add_argument("--write-default-config", type=Path, default=None, help="Write a default PhaseCCampaignConfig JSON template and exit.")
    p.add_argument("--print-effective-config", action="store_true", help="Resolve config + CLI overrides, print normalized JSON, and exit.")
    p.add_argument("--write-effective-config", type=Path, default=None, help="Write normalized effective configuration before execution.")
    p.add_argument("--dry-run", action="store_true", help="Build and print the exact C1-C9 command plan without executing it.")

    # Identity / parent resolution.
    p.add_argument("--campaign-root", default=suppress)
    p.add_argument("--campaign-id", default=suppress)
    p.add_argument("--generated-data-root", default=suppress)
    p.add_argument("--matrix-run-id", default=suppress)
    p.add_argument("--c1-aggregation-run-root", default=suppress)
    p.add_argument("--phase-c-run-id", default=suppress)

    # Scope.
    p.add_argument("--case-id", default=suppress)
    p.add_argument("--aggregation-id", default=suppress)
    p.add_argument("--weight-mode", default=suppress)
    p.add_argument("--aggregate-zone-id", default=suppress)
    p.add_argument("--model-id", dest="model_ids", action="append", default=suppress)
    p.add_argument("--downstream-aggregate-zone-id", dest="downstream_aggregate_zone_ids", action="append", default=suppress)

    # C1/C2 features/targets.
    p.add_argument("--minimum-sample-count", type=int, default=suppress)
    p.add_argument("--internal-gain-predictor-method", choices=("aggregate_average", "contribution_sum"), default=suppress)
    p.add_argument("--hvac-target-method", choices=("signed_zone_sensible", "absolute_zone_sensible"), default=suppress)
    p.add_argument("--feature-preview-rows", type=int, default=suppress)

    # C3 splitting.
    p.add_argument("--split-strategy", choices=("monthly_distributed_holdout", "chronological_fraction"), default=suppress)
    p.add_argument("--train-fraction", type=float, default=suppress)
    p.add_argument("--validation-fraction", type=float, default=suppress)
    p.add_argument("--test-fraction", type=float, default=suppress)
    p.add_argument("--minimum-split-samples", type=int, default=suppress)
    p.add_argument("--fraction-tolerance", type=float, default=suppress)
    p.add_argument("--split-random-seed", "--random-seed", dest="split_random_seed", type=int, default=suppress)
    p.add_argument("--split-preview-rows", type=int, default=suppress)

    # C4 dataset construction.
    p.add_argument("--dataset-minimum-split-samples", type=int, default=suppress)
    p.add_argument("--dataset-preview-rows", type=int, default=suppress)

    # C5 API validation.
    p.add_argument("--c5-max-c4-models", type=int, default=suppress)
    p.add_argument("--c5-skip-pytorch", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--c5-pytorch-device", dest="c5_pytorch_devices", action="append", choices=("cpu", "cuda", "auto"), default=suppress)

    # C6 training.
    p.add_argument("--estimator-type", dest="estimator_types", action="append", choices=("closed_form_linear", "pytorch_linear"), default=suppress)
    p.add_argument("--pytorch-device", dest="pytorch_devices", action="append", choices=("cpu", "cuda", "auto"), default=suppress)
    p.add_argument("--fit-intercept", dest="fit_intercept_override", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--ridge-alpha", type=float, default=suppress)
    p.add_argument("--learning-rate", type=float, default=suppress)
    p.add_argument("--max-epochs", type=int, default=suppress)
    p.add_argument("--tolerance", type=float, default=suppress)
    p.add_argument("--patience", type=int, default=suppress)
    p.add_argument("--training-seed", "--seed", dest="training_seed", type=int, default=suppress)
    p.add_argument("--reload-atol", type=float, default=suppress)
    p.add_argument("--reload-rtol", type=float, default=suppress)
    p.add_argument("--training-prediction-preview-rows", type=int, default=suppress)

    # C7/C8 output controls.
    p.add_argument("--evaluation-prediction-preview-rows", type=int, default=suppress)
    p.add_argument("--full-predictions", dest="write_full_predictions", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--evaluation-requested-device", dest="evaluation_requested_devices", action="append", choices=("cpu", "cuda", "auto"), default=suppress)
    p.add_argument("--inference-preview-rows", type=int, default=suppress)
    p.add_argument("--inference-requested-device", dest="inference_requested_devices", action="append", choices=("cpu", "cuda", "auto"), default=suppress)

    # Validator thresholds.
    p.add_argument("--feature-validation-absolute-tolerance", type=float, default=suppress)
    p.add_argument("--feature-validation-relative-tolerance", type=float, default=suppress)
    p.add_argument("--expected-canonical-row-count", type=int, default=suppress)
    p.add_argument("--canonical-timestamp-expected-row-count", type=int, default=suppress)
    p.add_argument("--expected-cadence-seconds", type=float, default=suppress)
    p.add_argument("--fail-on-conflicting-source-values", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--dataset-validation-absolute-tolerance", type=float, default=suppress)
    p.add_argument("--dataset-validation-relative-tolerance", type=float, default=suppress)
    p.add_argument("--training-validation-coefficient-atol", type=float, default=suppress)
    p.add_argument("--training-validation-prediction-atol", type=float, default=suppress)
    p.add_argument("--training-validation-prediction-rtol", type=float, default=suppress)
    p.add_argument("--evaluation-validation-metric-atol", type=float, default=suppress)
    p.add_argument("--evaluation-validation-metric-rtol", type=float, default=suppress)
    p.add_argument("--inference-validation-prediction-atol", type=float, default=suppress)
    p.add_argument("--inference-validation-prediction-rtol", type=float, default=suppress)

    # C9 MLflow.
    p.add_argument("--mlflow-enabled", dest="mlflow_enabled", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--disable-mlflow", dest="mlflow_enabled", action="store_false", default=suppress)
    p.add_argument("--mlflow-validation-mode", choices=("full", "lightweight", "none"), default=suppress)
    p.add_argument("--mlflow-experiment-name", default=suppress)
    p.add_argument("--mlflow-run-name", default=suppress)
    p.add_argument("--mlflow-strict", dest="mlflow_strict", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--mlflow-non-strict", dest="mlflow_strict", action="store_false", default=suppress)
    p.add_argument("--mlflow-compact-artifacts", dest="mlflow_log_compact_artifacts", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--mlflow-log-model-artifacts", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--mlflow-max-artifact-bytes", type=int, default=suppress)

    # Optional C8 diagnostics already present in the repository.
    p.add_argument("--run-inference-missing-value-audit", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--inspect-source-files", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--run-residual-gap-audit", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--residual-gap-neighbor-radius", type=int, default=suppress)

    # Execution/recovery/truncation. --validation is retained for backwards compatibility.
    p.add_argument("--validation", "--validation-profile", dest="validation_profile", choices=("full", "some", "none"), default=suppress)
    p.add_argument("--start-stage", choices=tuple(STAGE_SCRIPT_NAMES), default=suppress)
    p.add_argument("--stop-stage", choices=tuple(STAGE_SCRIPT_NAMES), default=suppress)
    p.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--overwrite-existing", action=argparse.BooleanOptionalAction, default=suppress)
    p.add_argument("--max-zones", type=int, default=suppress)
    p.add_argument("--max-model-datasets", type=int, default=suppress)
    p.add_argument("--max-artifacts", type=int, default=suppress)
    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_configuration(args: argparse.Namespace) -> PhaseCCampaignConfig:
    payload: dict[str, object] = {}
    if args.config is not None:
        raw = json.loads(args.config.expanduser().read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object in {args.config}")
        payload.update(raw)

    control_names = {
        "config", "print_capabilities", "write_default_config",
        "print_effective_config", "write_effective_config", "dry_run",
    }
    for key, value in vars(args).items():
        if key not in control_names:
            payload[key] = value
    return PhaseCCampaignConfig.model_validate(payload)


def resolve_campaign_identity(config: PhaseCCampaignConfig) -> tuple[Path, str]:
    if config.campaign_root:
        root = Path(config.campaign_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Campaign root does not exist: {root}")
        campaign_id = config.campaign_id or root.name
        if config.campaign_id and config.campaign_id != root.name:
            raise ValueError(
                f"campaign_id={config.campaign_id!r} does not match campaign_root name {root.name!r}"
            )
        return root, campaign_id

    if not config.campaign_id:
        raise ValueError("Provide campaign_root or campaign_id + generated_data_root")
    root = resolve_inputs(
        campaign_id=config.campaign_id,
        campaign_root=None,
        generated_data_root=config.generated_data_root,
    )
    return root.resolve(), config.campaign_id

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.print_capabilities:
        print(json.dumps(PhaseCCampaignConfig.capability_manifest(), indent=2, default=str))
        return 0
    if args.write_default_config is not None:
        write_json(args.write_default_config.expanduser().resolve(), PhaseCCampaignConfig().to_dict())
        print(args.write_default_config.expanduser().resolve())
        return 0

    config = resolve_configuration(args)
    campaign_root, campaign_id = resolve_campaign_identity(config)
    config = config.model_copy(update={"campaign_root": str(campaign_root), "campaign_id": campaign_id})

    if args.print_effective_config:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.write_effective_config is not None:
        write_json(args.write_effective_config.expanduser().resolve(), config.to_dict())

    layout = build_layout(
        campaign_root=campaign_root,
        matrix_run_id=config.matrix_run_id,
        phase_c_run_id=config.phase_c_run_id,
        direct_aggregation_run_root=config.c1_aggregation_run_root,
    )
    layout.campaign_run_root.mkdir(parents=True, exist_ok=True)
    log_root = layout.campaign_run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    write_json(
        layout.campaign_run_root / "phase_c_campaign_plan.json",
        {
            "schema_version": "0.2.0",
            "created_at_utc": utc_now(),
            "validation": config.validation_profile,
            "start_stage": config.start_stage,
            "stop_stage": config.stop_stage,
            "layout": layout_to_json(layout),
            "configuration": config.to_dict(),
        },
    )

    print_plan(layout, config)
    commands = build_pipeline_commands(layout, config)

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
            if not config.continue_on_error:
                break

    summary = {
        "schema_version": "0.2.0",
        "created_at_utc": utc_now(),
        "campaign_id": layout.campaign_id,
        "campaign_root": str(layout.campaign_root),
        "matrix_run_id": layout.matrix_run_id,
        "phase_c_run_id": layout.phase_c_run_id,
        "validation": config.validation_profile,
        "status": "failed" if failed else "completed",
        "runtime_seconds": time.perf_counter() - started,
        "command_count": len(results),
        "passed_command_count": sum(r.return_code == 0 for r in results),
        "failed_command_count": sum(r.return_code != 0 for r in results),
        "configuration": config.to_dict(),
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
    direct_aggregation_run_root: str | None = None,
) -> RunLayout:
    campaign_id = campaign_root.name
    suffix = extract_or_create_suffix(phase_c_run_id)
    resolved_phase_id = phase_c_run_id or f"phase_c_{suffix}"
    resolved_matrix_id = (
        matrix_run_id
        or ("direct_aggregation_run" if direct_aggregation_run_root else None)
        or discover_latest_matrix_run_id(campaign_root)
    )

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
        training_run_id=f"c6_models_{suffix}",
        evaluation_run_id=f"c7_models_{suffix}",
        inference_run_id=f"c8_models_{suffix}",
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
    config: PhaseCCampaignConfig,
) -> list[tuple[str, list[str]]]:
    stage_order = list(STAGE_SCRIPT_NAMES)
    start = stage_order.index(config.start_stage)
    stop = stage_order.index(config.stop_stage)

    zones = (
        config.downstream_aggregate_zone_ids
        if config.downstream_aggregate_zone_ids
        else ((config.aggregate_zone_id,) if config.aggregate_zone_id else ())
    )
    stage_semantics = {
        "C1": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": None if config.c1_aggregation_run_root else layout.matrix_run_id,
            "aggregation_run_root": config.c1_aggregation_run_root,
            "audit_run_id": layout.audit_run_id,
            "case_id": config.case_id,
            "aggregation_id": config.aggregation_id,
            "weight_mode": config.weight_mode,
            "aggregate_zone_id": config.aggregate_zone_id,
            "max_zones": config.max_zones,
            "minimum_sample_count": config.minimum_sample_count,
            "internal_gain_predictor_method": config.internal_gain_predictor_method,
            "hvac_target_method": config.hvac_target_method,
            "continue_on_error": config.continue_on_error,
        },
        "C2": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "case_id": config.case_id,
            "aggregation_id": config.aggregation_id,
            "weight_mode": config.weight_mode,
            "aggregate_zone_id": config.aggregate_zone_id,
            "model_id": config.model_ids,
            "minimum_sample_count": config.minimum_sample_count,
            "internal_gain_predictor_method": config.internal_gain_predictor_method,
            "hvac_target_method": config.hvac_target_method,
            "preview_rows": config.feature_preview_rows,
            "continue_on_error": config.continue_on_error,
        },
        "C3": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
            "case_id": config.case_id,
            "aggregation_id": config.aggregation_id,
            "weight_mode": config.weight_mode,
            "aggregate_zone_id": config.aggregate_zone_id,
            "split_strategy": config.split_strategy,
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
            "minimum_split_samples": config.minimum_split_samples,
            "fraction_tolerance": config.fraction_tolerance,
            "random_seed": config.split_random_seed,
            "preview_rows": config.split_preview_rows,
            "continue_on_error": config.continue_on_error,
        },
        "C4": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
            "dataset_run_id": layout.dataset_run_id,
            "case_id": config.case_id,
            "aggregation_id": config.aggregation_id,
            "weight_mode": config.weight_mode,
            "aggregate_zone_id": config.aggregate_zone_id,
            "model_id": config.model_ids,
            "minimum_split_samples": config.dataset_minimum_split_samples,
            "preview_rows": config.dataset_preview_rows,
            "continue_on_error": config.continue_on_error,
        },
        "C5": {
            "output_root": layout.c5_root,
            "dataset_root": layout.dataset_root,
            "max_c4_models": config.c5_max_c4_models,
            "skip_pytorch": config.c5_skip_pytorch,
            "pytorch_device": config.c5_pytorch_devices,
        },
        "C6": {
            "dataset_root": layout.dataset_root,
            "output_root": layout.training_parent,
            "training_run_id": layout.training_run_id,
            "estimator_type": config.estimator_types,
            "model_id": config.model_ids,
            "aggregate_zone_id": zones,
            "max_model_datasets": config.max_model_datasets,
            "fit_intercept_override": config.fit_intercept_override,
            "ridge_alpha": config.ridge_alpha,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "tolerance": config.tolerance,
            "patience": config.patience,
            "seed": config.training_seed,
            "pytorch_device": config.pytorch_devices,
            "reload_atol": config.reload_atol,
            "reload_rtol": config.reload_rtol,
            "prediction_preview_rows": config.training_prediction_preview_rows,
            "continue_on_error": config.continue_on_error,
            "overwrite_existing": config.overwrite_existing,
        },
        "C7": {
            "training_root": layout.training_root,
            "output_root": layout.evaluation_parent,
            "evaluation_run_id": layout.evaluation_run_id,
            "model_id": config.model_ids,
            "aggregate_zone_id": zones,
            "estimator_type": config.estimator_types,
            "requested_device": config.evaluation_requested_devices,
            "max_artifacts": config.max_artifacts,
            "prediction_preview_rows": config.evaluation_prediction_preview_rows,
            "no_full_predictions": not config.write_full_predictions,
            "continue_on_error": config.continue_on_error,
        },
        "C8": {
            "evaluation_root": layout.evaluation_root,
            "feature_root": layout.feature_root,
            "dataset_root": layout.dataset_root,
            "output_root": layout.inference_parent,
            "inference_run_id": layout.inference_run_id,
            "model_id": config.model_ids,
            "aggregate_zone_id": zones,
            "estimator_type": config.estimator_types,
            "requested_device": config.inference_requested_devices,
            "max_artifacts": config.max_artifacts,
            "preview_rows": config.inference_preview_rows,
            "continue_on_error": config.continue_on_error,
            "overwrite_existing": config.overwrite_existing,
        },
        "C9": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "phase_c_run_id": layout.phase_c_run_id,
            "experiment_name": config.mlflow_experiment_name,
            "run_name": config.mlflow_run_name,
            "validation_mode": config.mlflow_validation_mode,
            "non_strict": not config.mlflow_strict,
            "no_compact_artifacts": not config.mlflow_log_compact_artifacts,
            "log_model_artifacts": config.mlflow_log_model_artifacts,
            "max_artifact_bytes": config.mlflow_max_artifact_bytes,
            "training_root": layout.training_root,
            "evaluation_root": layout.evaluation_root,
            "inference_root": layout.inference_root,
        },
    }

    commands: list[tuple[str, list[str]]] = []
    selected_stages = stage_order[start : stop + 1]
    for stage in selected_stages:
        if stage == "C9" and not config.mlflow_enabled:
            continue
        script = require_script(STAGE_SCRIPT_NAMES[stage])
        commands.append((f"{stage} {script.stem}", build_adaptive_command(script, stage_semantics[stage])))
        if stage != "C9":
            commands.extend(validation_commands_for_stage(stage, layout, config))
        if stage == "C8":
            commands.extend(diagnostic_commands_for_c8(layout, config))

    if config.mlflow_enabled and "C9" in selected_stages and config.validation_profile != "none":
        commands.extend(validation_commands_for_stage("C9", layout, config))
    return commands


def diagnostic_commands_for_c8(layout: RunLayout, config: PhaseCCampaignConfig) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if config.run_inference_missing_value_audit:
        script = require_script(DIAGNOSTIC_SCRIPT_NAMES["C8_missing_values"])
        commands.append((
            "DIAGNOSTIC C8 missing values",
            build_adaptive_command(script, {
                "inference_root": layout.inference_root,
                "inspect_source_files": config.inspect_source_files,
            }),
        ))
    if config.run_residual_gap_audit:
        script = require_script(DIAGNOSTIC_SCRIPT_NAMES["C8_residual_gaps"])
        commands.append((
            "DIAGNOSTIC C8 residual gaps",
            build_adaptive_command(script, {
                "inference_root": layout.inference_root,
                "feature_root": layout.feature_root,
                "output_root": layout.campaign_run_root / "diagnostics" / "c8_residual_gap_audit",
                "neighbor_radius": config.residual_gap_neighbor_radius,
            }),
        ))
    return commands


def validation_commands_for_stage(
    stage: str,
    layout: RunLayout,
    config: PhaseCCampaignConfig,
) -> list[tuple[str, list[str]]]:
    if config.validation_profile == "none":
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
        "C1": [], "C2": ["C2_features", "C2_timestamps"], "C3": [],
        "C4": ["C4"], "C5": [], "C6": ["C6"], "C7": ["C7"],
        "C8": ["C8"], "C9": ["C9"],
    }
    keys = full_keys[stage] if config.validation_profile == "full" else some_keys[stage]

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
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "case_id": config.case_id,
            "aggregation_id": config.aggregation_id,
            "weight_mode": config.weight_mode,
            "aggregate_zone_id": config.aggregate_zone_id,
            "minimum_sample_count": config.minimum_sample_count,
            "absolute_tolerance": config.feature_validation_absolute_tolerance,
            "relative_tolerance": config.feature_validation_relative_tolerance,
            "expected_canonical_row_count": config.expected_canonical_row_count,
        },
        "C2_timestamps": {
            "feature_root": layout.feature_root,
            "expected_row_count": config.canonical_timestamp_expected_row_count,
            "expected_cadence_seconds": config.expected_cadence_seconds,
        },
        "C2_coalescence": {
            "feature_root": layout.feature_root,
            "expected_row_count": config.expected_canonical_row_count,
            "fail_on_conflicting_source_values": config.fail_on_conflicting_source_values,
        },
        "C3": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
            "case_id": config.case_id,
            "aggregation_id": config.aggregation_id,
            "weight_mode": config.weight_mode,
            "aggregate_zone_id": config.aggregate_zone_id,
            "minimum_split_samples": config.minimum_split_samples,
            "fraction_tolerance": config.fraction_tolerance,
        },
        "C4": {
            "campaign_id": layout.campaign_id,
            "campaign_root": layout.campaign_root,
            "matrix_run_id": layout.matrix_run_id,
            "audit_run_id": layout.audit_run_id,
            "feature_run_id": layout.feature_run_id,
            "split_run_id": layout.split_run_id,
            "dataset_run_id": layout.dataset_run_id,
            "case_id": config.case_id,
            "aggregation_id": config.aggregation_id,
            "weight_mode": config.weight_mode,
            "aggregate_zone_id": config.aggregate_zone_id,
            "model_id": config.model_ids,
            "minimum_split_samples": config.dataset_minimum_split_samples,
            "absolute_tolerance": config.dataset_validation_absolute_tolerance,
            "relative_tolerance": config.dataset_validation_relative_tolerance,
        },
        "C6": {
            "training_root": layout.training_root,
            "coefficient_atol": config.training_validation_coefficient_atol,
            "prediction_atol": config.training_validation_prediction_atol,
            "prediction_rtol": config.training_validation_prediction_rtol,
        },
        "C7": {
            "evaluation_root": layout.evaluation_root,
            "metric_atol": config.evaluation_validation_metric_atol,
            "metric_rtol": config.evaluation_validation_metric_rtol,
        },
        "C8": {
            "inference_root": layout.inference_root,
            "prediction_atol": config.inference_validation_prediction_atol,
            "prediction_rtol": config.inference_validation_prediction_rtol,
        },
        "C9": {
            "registration_manifest": layout.mlflow_registration_manifest,
            "expected_stage_runs": 8,
        },
    }

    commands: list[tuple[str, list[str]]] = []
    for key in keys:
        script = require_script(VALIDATOR_SCRIPT_NAMES[key])
        commands.append((f"VALIDATE {key}", build_adaptive_command(script, semantics[key])))
    return commands


OPTION_ALIASES = {
    "paths": ("--paths",),
    "campaign_root": ("--campaign-root",), "campaign_id": ("--campaign-id",),
    "matrix_run_id": ("--matrix-run-id",), "aggregation_run_root": ("--aggregation-run-root",),
    "audit_run_id": ("--audit-run-id",),
    "feature_run_id": ("--feature-run-id",), "split_run_id": ("--split-run-id",),
    "dataset_run_id": ("--dataset-run-id",), "training_run_id": ("--training-run-id",),
    "evaluation_run_id": ("--evaluation-run-id",), "inference_run_id": ("--inference-run-id",),
    "phase_c_run_id": ("--phase-c-run-id",), "feature_root": ("--feature-root",),
    "dataset_root": ("--dataset-root",), "training_root": ("--training-root",),
    "evaluation_root": ("--evaluation-root",), "inference_root": ("--inference-root",),
    "output_root": ("--output-root",), "registration_manifest": ("--registration-manifest",),
    "case_id": ("--case-id",), "aggregation_id": ("--aggregation-id",),
    "weight_mode": ("--weight-mode",), "aggregate_zone_id": ("--aggregate-zone-id",),
    "model_id": ("--model-id",), "minimum_sample_count": ("--minimum-sample-count",),
    "internal_gain_predictor_method": ("--internal-gain-predictor-method",),
    "hvac_target_method": ("--hvac-target-method",), "preview_rows": ("--preview-rows",),
    "split_strategy": ("--split-strategy",), "train_fraction": ("--train-fraction",),
    "validation_fraction": ("--validation-fraction",), "test_fraction": ("--test-fraction",),
    "minimum_split_samples": ("--minimum-split-samples",), "fraction_tolerance": ("--fraction-tolerance",),
    "random_seed": ("--random-seed",), "max_c4_models": ("--max-c4-models",),
    "skip_pytorch": ("--skip-pytorch",), "estimator_type": ("--estimator-type",),
    "pytorch_device": ("--pytorch-device",), "requested_device": ("--requested-device",),
    "max_model_datasets": ("--max-model-datasets",), "max_artifacts": ("--max-artifacts",),
    "max_zones": ("--max-zones",), "ridge_alpha": ("--ridge-alpha",),
    "learning_rate": ("--learning-rate",), "max_epochs": ("--max-epochs",),
    "tolerance": ("--tolerance",), "patience": ("--patience",), "seed": ("--seed",),
    "reload_atol": ("--reload-atol",), "reload_rtol": ("--reload-rtol",),
    "prediction_preview_rows": ("--prediction-preview-rows",),
    "no_full_predictions": ("--no-full-predictions",),
    "continue_on_error": ("--continue-on-error",), "overwrite_existing": ("--overwrite-existing",),
    "absolute_tolerance": ("--absolute-tolerance",), "relative_tolerance": ("--relative-tolerance",),
    "expected_canonical_row_count": ("--expected-canonical-row-count",),
    "expected_row_count": ("--expected-row-count",), "expected_cadence_seconds": ("--expected-cadence-seconds",),
    "fail_on_conflicting_source_values": ("--fail-on-conflicting-source-values",),
    "coefficient_atol": ("--coefficient-atol",), "prediction_atol": ("--prediction-atol",),
    "prediction_rtol": ("--prediction-rtol",), "metric_atol": ("--metric-atol",), "metric_rtol": ("--metric-rtol",),
    "experiment_name": ("--experiment-name",), "run_name": ("--run-name",),
    "validation_mode": ("--validation-mode",), "non_strict": ("--non-strict",),
    "no_compact_artifacts": ("--no-compact-artifacts",), "log_model_artifacts": ("--log-model-artifacts",),
    "max_artifact_bytes": ("--max-artifact-bytes",), "inspect_source_files": ("--inspect-source-files",),
    "neighbor_radius": ("--neighbor-radius",), "expected_stage_runs": ("--expected-stage-runs",),
}

def build_adaptive_command(
    script: Path,
    semantic_values: dict[str, object],
) -> list[str]:
    help_text = get_help_text(script)
    command = [sys.executable, str(script)]

    for semantic, value in semantic_values.items():
        if semantic == "fit_intercept_override":
            if value is not None:
                command.append("--fit-intercept" if value else "--no-fit-intercept")
            continue
        if value is None or value is False or value == () or value == []:
            continue
        option = first_supported_option(help_text, OPTION_ALIASES[semantic])
        if option is None:
            # Optional controls may not exist on every historical stage script.
            if semantic in {"continue_on_error", "overwrite_existing"}:
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


def print_plan(layout: RunLayout, config: PhaseCCampaignConfig) -> None:
    print("=" * 100)
    print("SCALEBRIDGE PHASE C CAMPAIGN PLAN")
    print("=" * 100)
    print(f"campaign_id: {layout.campaign_id}")
    print(f"campaign_root: {layout.campaign_root}")
    print(f"matrix_run_id: {layout.matrix_run_id}")
    print(f"phase_c_run_id: {layout.phase_c_run_id}")
    print(f"validation: {config.validation_profile}")
    print(f"start_stage: {config.start_stage}")
    print(f"stop_stage: {config.stop_stage}")
    print(f"estimators: {list(config.estimator_types)}")
    print(f"pytorch_devices: {list(config.pytorch_devices)}")
    print(f"mlflow: {config.mlflow_enabled}")
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
