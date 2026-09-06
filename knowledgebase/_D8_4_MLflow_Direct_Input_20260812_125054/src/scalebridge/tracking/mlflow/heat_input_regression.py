# -*- coding: utf-8 -*-
"""Manifest-driven MLflow registration for ScaleBridge Phase C.

C9 does not rerun C1-C8 and does not replace their filesystem manifests.
It registers an already completed Phase C run into semantic MLflow tracking.

Run hierarchy
-------------
Phase C parent
    C1-C8 stage children
    C6 stage
        one nested child per training_manifest.json
    C7 stage
        one nested child per evaluation_manifest.json
    C8 stage
        one nested child per annual_component_predictions_manifest.json
"""
from __future__ import annotations

import os
import sys


def _configure_windows_mlflow_console() -> None:
    """Prevent MLflow decorative Unicode URL output from breaking Windows pipes."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    try:
        from mlflow.tracking._tracking_service.client import TrackingServiceClient

        def _silent_log_url(self, run_id: str) -> None:
            return None

        TrackingServiceClient._log_url = _silent_log_url
    except Exception:
        pass


_configure_windows_mlflow_console()

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
from typing import Any, Iterable, Iterator

from scalebridge.tracking.mlflow.semantic import (
    configure_mlflow_tracking,
    get_machine_id,
    get_or_create_semantic_experiment,
)


@dataclass(frozen=True)
class PhaseCTrackingConfig:
    """Configuration for one Phase C MLflow registration."""

    campaign_id: str
    phase_c_run_id: str
    experiment_name: str | None = None
    run_name: str | None = None
    artifact_subdir: str | None = None
    validation_mode: str = "full"
    strict: bool = True
    log_compact_artifacts: bool = True
    log_model_artifacts: bool = False
    max_artifact_bytes: int = 20_000_000

    def resolved_experiment_name(self) -> str:
        return (
            self.experiment_name
            or f"{self.campaign_id}_phase_c_heat_input_regression"
        )

    def resolved_run_name(self) -> str:
        return self.run_name or self.phase_c_run_id

    def resolved_artifact_subdir(self) -> str:
        return self.artifact_subdir or self.resolved_experiment_name()


@dataclass(frozen=True)
class PhaseCRegistrationResult:
    """Identifiers and counts produced by C9 registration."""

    experiment_name: str
    experiment_id: str
    parent_run_id: str
    tracking_uri: str
    stage_run_count: int
    task_run_count: int
    failed_registration_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "parent_run_id": self.parent_run_id,
            "tracking_uri": self.tracking_uri,
            "stage_run_count": self.stage_run_count,
            "task_run_count": self.task_run_count,
            "failed_registration_count": self.failed_registration_count,
        }


STAGE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "C1": {
        "run_id_keys": ["audit_run_id"],
        "metric_keys": [
            "selected_aggregation_run_count",
            "selected_aggregation_zone_count",
            "successful_zone_count",
            "failed_zone_count",
            "runtime_seconds",
            "candidate_model_count",
            "applicable_model_count",
            "structurally_inapplicable_model_count",
            "invalid_model_count",
            "missing_expected_data_model_count",
        ],
        "param_keys": [
            "source_matrix_run_id",
            "internal_gain_predictor_method",
            "hvac_target_method",
            "minimum_sample_count",
        ],
    },
    "C2": {
        "run_id_keys": ["feature_run_id"],
        "metric_keys": [
            "selected_aggregation_run_count",
            "selected_aggregation_zone_count",
            "successful_zone_count",
            "failed_zone_count",
            "selected_zone_count",
            "passed_zone_count",
            "runtime_seconds",
            "candidate_model_count",
            "applicable_model_count",
            "inapplicable_model_count",
            "zero_applicable_model_zone_count",
        ],
        "param_keys": [
            "source_matrix_run_id",
            "source_audit_run_id",
            "internal_gain_predictor_method",
            "hvac_target_method",
            "minimum_sample_count",
            "preview_rows",
        ],
    },
    "C3": {
        "run_id_keys": ["source_split_run_id", "split_run_id"],
        "metric_keys": [
            "selected_aggregation_zone_count",
            "passed_zone_count",
            "failed_zone_count",
            "runtime_seconds",
            "candidate_model_count",
            "applicable_model_count",
            "inapplicable_model_count",
            "zero_applicable_model_zone_count",
        ],
        "param_keys": [
            "source_matrix_run_id",
            "source_feature_run_id",
            "minimum_split_samples",
            "fraction_tolerance",
        ],
    },
    "C4": {
        "run_id_keys": ["dataset_run_id"],
        "metric_keys": [
            "selected_aggregation_run_count",
            "selected_aggregation_zone_count",
            "selected_model_count",
            "successful_zone_count",
            "failed_zone_count",
            "successful_model_count",
            "failed_model_count",
            "runtime_seconds",
            "candidate_model_count",
            "applicable_model_count",
            "inapplicable_model_count",
            "zero_applicable_model_zone_count",
        ],
        "param_keys": [
            "source_matrix_run_id",
            "source_audit_run_id",
            "source_feature_run_id",
            "source_split_run_id",
            "minimum_split_samples",
            "preview_rows",
        ],
    },
    "C5": {
        "run_id_keys": [],
        "metric_keys": [
            "check_count",
            "passed_check_count",
            "failed_check_count",
        ],
        "param_keys": ["validation_status"],
    },
    "C6": {
        "run_id_keys": ["training_run_id"],
        "metric_keys": [
            "selected_model_dataset_count",
            "requested_training_task_count",
            "completed_training_task_count",
            "failed_training_task_count",
            "runtime_seconds",
            "zero_selected_model_datasets",
        ],
        "param_keys": ["estimator_types", "pytorch_devices"],
    },
    "C7": {
        "run_id_keys": ["evaluation_run_id"],
        "metric_keys": [
            "selected_training_artifact_count",
            "completed_evaluation_count",
            "failed_evaluation_count",
            "runtime_seconds",
            "zero_selected_training_artifacts",
        ],
        "param_keys": ["write_full_predictions"],
    },
    "C8": {
        "run_id_keys": ["inference_run_id"],
        "metric_keys": [
            "selected_evaluation_artifact_count",
            "selected_zone_count",
            "completed_zone_count",
            "failed_zone_count",
            "runtime_seconds",
            "zero_component_zone_count",
            "total_component_count",
        ],
        "param_keys": [],
    },
}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stringify(value: Any, *, max_length: int = 500) -> str:
    if isinstance(value, dict):
        text = json.dumps(value, sort_keys=True, default=str)
    elif isinstance(value, (list, tuple, set)):
        text = ",".join(str(item) for item in value)
    else:
        text = str(value)
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _status_to_mlflow(status: Any) -> str:
    text = str(status or "").strip().lower()
    return "FAILED" if text in {
        "failed",
        "invalid",
        "cancelled",
        "completed_with_failures",
    } else "FINISHED"


def _extract_run_id(stage: str, manifest: dict[str, Any]) -> str:
    for key in STAGE_DEFINITIONS[stage]["run_id_keys"]:
        value = str(manifest.get(key, "")).strip()
        if value:
            return value
    return f"{stage.lower()}_manifest_registration"


def _safe_log_params(mlflow: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        try:
            mlflow.log_param(str(key), _stringify(value))
        except Exception:
            pass


def _safe_log_metrics(mlflow: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        number = _finite_float(value)
        if number is None:
            continue
        try:
            mlflow.log_metric(str(key), number)
        except Exception:
            pass


def _safe_set_tags(mlflow: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        try:
            mlflow.set_tag(str(key), _stringify(value, max_length=5000))
        except Exception:
            pass


def _log_file_if_compact(
    mlflow: Any,
    path: Path,
    *,
    artifact_path: str,
    max_bytes: int,
) -> bool:
    if not path.is_file() or path.stat().st_size > max_bytes:
        return False
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
    return True


def _log_compact_directory_files(
    mlflow: Any,
    directory: Path,
    *,
    artifact_path: str,
    max_bytes: int,
    allowed_suffixes: set[str] | None = None,
) -> int:
    if not directory.is_dir():
        return 0
    suffixes = allowed_suffixes or {".json", ".csv", ".txt", ".log"}
    count = 0
    for path in sorted(directory.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
            and path.stat().st_size <= max_bytes
        ):
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            count += 1
    return count


def _standard_tags(
    *,
    config: PhaseCTrackingConfig,
    stage: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tags: dict[str, Any] = {
        "campaign_id": config.campaign_id,
        "phase_c_run_id": config.phase_c_run_id,
        "pipeline": "heat_input_regression",
        "pipeline_phase": "C",
        "machine_id": get_machine_id(),
        "hostname": socket.gethostname(),
        "environment_name": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "tracking_uri": configure_mlflow_tracking(),
        "validation_mode": config.validation_mode,
        "filesystem_manifests_authoritative": "true",
    }
    if stage:
        tags["phase_c_stage"] = stage
    if extra:
        tags.update(extra)
    return tags


def _discover(root: str | Path | None, filename: str) -> list[Path]:
    if root is None:
        return []
    path = Path(root)
    if not path.is_dir():
        return []
    return sorted(path.rglob(filename))



def get_phase_c_root(
    *,
    campaign_id: str,
    campaign_root: str | Path | None = None,
) -> Path:
    """Resolve the standard Phase C root for one campaign."""
    if campaign_root is not None:
        resolved_campaign_root = Path(campaign_root).resolve()
    else:
        generated_root = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
        if not generated_root:
            raise RuntimeError(
                "SCALEBRIDGE_GENERATED_DATA_ROOT is not configured and "
                "--campaign-root was not provided."
            )
        resolved_campaign_root = (
            Path(generated_root).resolve() / "campaigns" / campaign_id
        )

    phase_c_root = resolved_campaign_root / "heat_input_regression"
    if not phase_c_root.is_dir():
        raise FileNotFoundError(f"Phase C root does not exist: {phase_c_root}")
    return phase_c_root


def extract_phase_c_run_suffix(phase_c_run_id: str) -> str:
    """Extract the YYYYMMDD_HHMMSS suffix from a Phase C run identifier."""
    import re
    match = re.search(r"(\d{8}_\d{6})$", str(phase_c_run_id).strip())
    if not match:
        raise ValueError(
            "phase_c_run_id must end with YYYYMMDD_HHMMSS, for example "
            "phase_c_c2fix_20260722_205232"
        )
    return match.group(1)


def _stage_run_directory_candidates(
    *,
    phase_c_root: Path,
    stage: str,
    run_suffix: str,
) -> list[Path]:
    roots_by_stage = {
        "C1": ["audit_runs"],
        "C2": ["feature_runs"],
        "C3": ["split_runs"],
        "C4": ["dataset_runs"],
        "C5": ["model_api_validation", "model_api_validation_runs"],
        "C6": ["training_runs"],
        "C7": ["evaluation_runs"],
        "C8": ["inference_runs"],
    }
    candidates: list[Path] = []
    for root_name in roots_by_stage[stage]:
        stage_root = phase_c_root / root_name
        if not stage_root.is_dir():
            continue
        candidates.extend(
            path.resolve()
            for path in stage_root.iterdir()
            if path.is_dir() and path.name.endswith(run_suffix)
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(candidates):
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _manifest_schema_score(
    *,
    stage: str,
    payload: dict[str, Any],
    run_suffix: str,
) -> int:
    """Score a JSON payload as the authoritative run-level manifest for a stage."""
    required_any: dict[str, list[set[str]]] = {
        "C1": [
            {"audit_run_id", "selected_aggregation_zone_count"},
            {"audit_run_id", "successful_zone_count"},
        ],
        "C2": [
            {"feature_run_id", "selected_aggregation_zone_count"},
            {"feature_run_id", "successful_zone_count"},
        ],
        "C3": [
            {"split_run_id", "selected_aggregation_zone_count"},
            {"source_split_run_id", "passed_zone_count"},
        ],
        "C4": [
            {"dataset_run_id", "selected_model_count"},
            {"dataset_run_id", "successful_model_count"},
        ],
        "C5": [
            {"validation_status", "check_count", "passed_check_count"},
        ],
        "C6": [
            {"training_run_id", "completed_training_task_count"},
            {"training_run_id", "selected_model_dataset_count"},
        ],
        "C7": [
            {"evaluation_run_id", "completed_evaluation_count"},
        ],
        "C8": [
            {"inference_run_id", "completed_zone_count"},
        ],
    }

    keys = set(payload)
    score = 0
    for required in required_any[stage]:
        if required.issubset(keys):
            score = max(score, 100 + len(required))

    if score == 0:
        return 0

    stage_run_keys = {
        "C1": ["audit_run_id"],
        "C2": ["feature_run_id"],
        "C3": ["split_run_id", "source_split_run_id"],
        "C4": ["dataset_run_id"],
        "C5": [],
        "C6": ["training_run_id"],
        "C7": ["evaluation_run_id"],
        "C8": ["inference_run_id"],
    }

    for key in stage_run_keys[stage]:
        value = str(payload.get(key, ""))
        if value.endswith(run_suffix):
            score += 50
        elif value:
            score -= 25

    if str(payload.get("stage", "")).upper() == stage:
        score += 10

    # Run-level manifests normally contain root/count/status metadata.
    for key in ["status", "output_root", "runtime_seconds", "schema_version"]:
        if key in payload:
            score += 1

    # Penalize task-level manifests.
    task_markers = {
        "aggregate_zone_id",
        "model_id",
        "estimator_type",
        "component_count",
        "row_count",
    }
    score -= 5 * len(task_markers.intersection(keys))
    return score


def _find_manifest_in_run_dir(
    *,
    stage: str,
    run_dir: Path,
    run_suffix: str,
) -> Path:
    """Find the authoritative stage manifest using JSON schema and run identity."""
    candidates: list[tuple[int, Path]] = []

    for path in sorted(run_dir.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        score = _manifest_schema_score(
            stage=stage,
            payload=payload,
            run_suffix=run_suffix,
        )
        if score > 0:
            candidates.append((score, path.resolve()))

    if not candidates:
        json_names = [
            str(path.relative_to(run_dir))
            for path in sorted(run_dir.rglob("*.json"))
            if path.is_file()
        ]
        raise FileNotFoundError(
            f"No authoritative {stage} manifest could be identified under "
            f"{run_dir}. JSON files found: {json_names}"
        )

    candidates.sort(key=lambda item: (-item[0], str(item[1]).lower()))
    best_score = candidates[0][0]
    best = [path for score, path in candidates if score == best_score]

    if len(best) != 1:
        raise RuntimeError(
            f"Ambiguous {stage} manifest discovery under {run_dir}. "
            f"Best score {best_score}; candidates: {[str(path) for path in best]}"
        )

    return best[0]

def discover_phase_c_run(
    *,
    campaign_id: str,
    phase_c_run_id: str,
    campaign_root: str | Path | None = None,
    stage_manifest_overrides: dict[str, str | Path] | None = None,
    training_root_override: str | Path | None = None,
    evaluation_root_override: str | Path | None = None,
    inference_root_override: str | Path | None = None,
) -> dict[str, Any]:
    """Discover C1-C8 manifests and task roots from campaign/run identity."""
    phase_c_root = get_phase_c_root(campaign_id=campaign_id, campaign_root=campaign_root)
    run_suffix = extract_phase_c_run_suffix(phase_c_run_id)
    overrides = {
        str(key).upper(): Path(value).resolve()
        for key, value in (stage_manifest_overrides or {}).items()
        if value is not None
    }
    stage_manifests: dict[str, Path] = {}
    stage_run_dirs: dict[str, Path] = {}
    for stage in STAGE_DEFINITIONS:
        if stage in overrides:
            path = overrides[stage]
            if not path.is_file():
                raise FileNotFoundError(f"{stage} manifest override does not exist: {path}")
            stage_manifests[stage] = path
            stage_run_dirs[stage] = path.parent
            continue
        candidates = _stage_run_directory_candidates(
            phase_c_root=phase_c_root,
            stage=stage,
            run_suffix=run_suffix,
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one {stage} run directory ending in {run_suffix} "
                f"under {phase_c_root}, found {len(candidates)}: "
                f"{[str(path) for path in candidates]}"
            )
        stage_run_dirs[stage] = candidates[0]
        stage_manifests[stage] = _find_manifest_in_run_dir(
            stage=stage,
            run_dir=candidates[0],
            run_suffix=run_suffix,
        )
    training_root = Path(training_root_override).resolve() if training_root_override else stage_run_dirs["C6"]
    evaluation_root = Path(evaluation_root_override).resolve() if evaluation_root_override else stage_run_dirs["C7"]
    inference_root = Path(inference_root_override).resolve() if inference_root_override else stage_run_dirs["C8"]
    for label, path in [("training_root", training_root), ("evaluation_root", evaluation_root), ("inference_root", inference_root)]:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return {
        "campaign_id": campaign_id,
        "phase_c_run_id": phase_c_run_id,
        "run_suffix": run_suffix,
        "phase_c_root": phase_c_root,
        "stage_manifests": stage_manifests,
        "stage_run_dirs": stage_run_dirs,
        "training_root": training_root,
        "evaluation_root": evaluation_root,
        "inference_root": inference_root,
        "registration_output_dir": phase_c_root / "mlflow_registration_runs" / phase_c_run_id,
    }


def _split_metrics(path: Path) -> dict[str, float]:
    import pandas as pd

    frame = pd.read_csv(path)
    out: dict[str, float] = {}
    for row in frame.to_dict(orient="records"):
        split = str(row.get("split", "")).strip()
        if not split:
            continue
        for key in [
            "row_count",
            "rmse",
            "mae",
            "r2",
            "mean_bias_error",
            "max_absolute_error",
            "nrmse_by_range",
            "nrmse_by_mean_abs_target",
        ]:
            value = _finite_float(row.get(key))
            if value is not None:
                out[f"{split}_{key}"] = value
    return out


def _log_stage_manifest(
    *,
    mlflow: Any,
    config: PhaseCTrackingConfig,
    stage: str,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    definition = STAGE_DEFINITIONS[stage]
    _safe_log_params(
        mlflow,
        {
            key: manifest.get(key)
            for key in definition["param_keys"]
            if key in manifest
        },
    )
    _safe_log_metrics(
        mlflow,
        {
            key: manifest.get(key)
            for key in definition["metric_keys"]
            if key in manifest
        },
    )
    _safe_set_tags(
        mlflow,
        _standard_tags(
            config=config,
            stage=stage,
            extra={
                "stage_status": manifest.get(
                    "validation_status",
                    manifest.get("status", "unknown"),
                ),
                "stage_run_id": _extract_run_id(stage, manifest),
                "manifest_path": str(manifest_path),
            },
        ),
    )
    if config.log_compact_artifacts:
        _log_file_if_compact(
            mlflow,
            manifest_path,
            artifact_path=f"{stage.lower()}/manifests",
            max_bytes=config.max_artifact_bytes,
        )


def _register_c6_tasks(
    *,
    mlflow: Any,
    config: PhaseCTrackingConfig,
    training_root: str | Path | None,
) -> tuple[int, int]:
    completed = failed = 0
    for manifest_path in _discover(training_root, "training_manifest.json"):
        try:
            manifest = read_json(manifest_path)
            run_name = (
                f"C6_{manifest.get('aggregate_zone_id', 'zone')}_"
                f"{manifest.get('model_id', 'model')}_"
                f"{manifest.get('estimator_type', 'estimator')}_"
                f"{manifest.get('requested_device', 'device')}"
            )
            with mlflow.start_run(run_name=run_name, nested=True):
                _safe_log_params(
                    mlflow,
                    {
                        "case_id": manifest.get("case_id"),
                        "aggregation_id": manifest.get("aggregation_id"),
                        "weight_mode": manifest.get("weight_mode"),
                        "aggregate_zone_id": manifest.get("aggregate_zone_id"),
                        "model_id": manifest.get("model_id"),
                        "estimator_type": manifest.get("estimator_type"),
                        "requested_device": manifest.get("requested_device"),
                        "resolved_device": manifest.get(
                            "resolved_device", manifest.get("device")
                        ),
                        **(manifest.get("estimator_config", {}) or {}),
                    },
                )
                _safe_log_metrics(
                    mlflow,
                    {
                        key: manifest.get(key)
                        for key in [
                            "sample_count",
                            "coefficient",
                            "intercept",
                            "training_rmse",
                            "training_loss",
                            "epochs_completed",
                            "reload_max_absolute_difference",
                            "runtime_seconds",
                        ]
                    },
                )
                _safe_set_tags(
                    mlflow,
                    _standard_tags(
                        config=config,
                        stage="C6",
                        extra={
                            "run_kind": "model_training",
                            "task_status": manifest.get("status", "completed"),
                            "converged": manifest.get("converged", ""),
                            "reload_predictions_match": manifest.get(
                                "reload_predictions_match", ""
                            ),
                            "manifest_path": str(manifest_path),
                        },
                    ),
                )
                if config.log_compact_artifacts:
                    _log_compact_directory_files(
                        mlflow,
                        manifest_path.parent,
                        artifact_path="c6/task",
                        max_bytes=config.max_artifact_bytes,
                    )
                if config.log_model_artifacts:
                    artifact_dir = Path(str(manifest.get("artifact_dir", "")))
                    if artifact_dir.is_dir():
                        mlflow.log_artifacts(
                            str(artifact_dir),
                            artifact_path="c6/model_artifact",
                        )
            completed += 1
        except Exception:
            failed += 1
            if config.strict:
                raise
    return completed, failed


def _register_c7_tasks(
    *,
    mlflow: Any,
    config: PhaseCTrackingConfig,
    evaluation_root: str | Path | None,
) -> tuple[int, int]:
    completed = failed = 0
    for manifest_path in _discover(evaluation_root, "evaluation_manifest.json"):
        try:
            manifest = read_json(manifest_path)
            run_name = (
                f"C7_{manifest.get('aggregate_zone_id', 'zone')}_"
                f"{manifest.get('model_id', 'model')}_"
                f"{manifest.get('estimator_type', 'estimator')}_"
                f"{manifest.get('requested_device', 'device')}"
            )
            with mlflow.start_run(run_name=run_name, nested=True):
                _safe_log_params(
                    mlflow,
                    {
                        key: manifest.get(key)
                        for key in [
                            "case_id",
                            "aggregation_id",
                            "weight_mode",
                            "aggregate_zone_id",
                            "model_id",
                            "estimator_type",
                            "requested_device",
                            "resolved_device",
                            "coefficient",
                            "intercept",
                            "write_full_predictions",
                            "fit_intercept",
                            "model_role",
                            "input_transform",
                            "dependency_model_id",
                            "target_allocation",
                            "evaluation_modes",
                        ]
                    },
                )
                metrics_path = Path(
                    str(
                        manifest.get("split_metrics_path")
                        or (manifest.get("outputs", {}) or {}).get("split_metrics")
                        or ""
                    )
                )
                metrics: dict[str, Any] = {
                    "runtime_seconds": manifest.get("runtime_seconds")
                }
                if metrics_path.is_file():
                    metrics.update(_split_metrics(metrics_path))
                _safe_log_metrics(mlflow, metrics)
                _safe_set_tags(
                    mlflow,
                    _standard_tags(
                        config=config,
                        stage="C7",
                        extra={
                            "run_kind": "model_evaluation",
                            "task_status": manifest.get("status", "completed"),
                            "zero_applicable_components": manifest.get(
                                "zero_applicable_components", False
                            ),
                            "component_availability_recorded": str(
                                (manifest_path.parent / "component_applicability.csv").is_file()
                            ).lower(),
                            "manifest_path": str(manifest_path),
                        },
                    ),
                )
                if config.log_compact_artifacts:
                    _log_compact_directory_files(
                        mlflow,
                        manifest_path.parent,
                        artifact_path="c7/task",
                        max_bytes=config.max_artifact_bytes,
                    )
            completed += 1
        except Exception:
            failed += 1
            if config.strict:
                raise
    return completed, failed


def _register_c8_tasks(
    *,
    mlflow: Any,
    config: PhaseCTrackingConfig,
    inference_root: str | Path | None,
) -> tuple[int, int]:
    completed = failed = 0
    for manifest_path in _discover(
        inference_root, "annual_component_predictions_manifest.json"
    ):
        try:
            manifest = read_json(manifest_path)
            run_name = (
                f"C8_{manifest.get('aggregate_zone_id', 'zone')}_"
                f"{manifest.get('aggregation_id', 'aggregation')}_"
                f"{manifest.get('weight_mode', 'weight')}"
            )
            with mlflow.start_run(run_name=run_name, nested=True):
                _safe_log_params(
                    mlflow,
                    {
                        key: manifest.get(key)
                        for key in [
                            "case_id",
                            "aggregation_id",
                            "weight_mode",
                            "aggregate_zone_id",
                            "inference_run_id",
                            "missing_value_policy",
                            "timestamp_start",
                            "timestamp_end",
                        ]
                    },
                )
                _safe_log_metrics(
                    mlflow,
                    {
                        key: manifest.get(key)
                        for key in [
                            "row_count",
                            "component_count",
                            "duplicate_timestamp_count",
                            "total_unavailable_component_values",
                            "runtime_seconds",
                            "candidate_model_count",
                            "applicable_model_count",
                            "inapplicable_model_count",
                        ]
                    },
                )
                _safe_set_tags(
                    mlflow,
                    _standard_tags(
                        config=config,
                        stage="C8",
                        extra={
                            "run_kind": "zone_inference",
                            "task_status": manifest.get("status", "completed"),
                            "manifest_path": str(manifest_path),
                        },
                    ),
                )
                if config.log_compact_artifacts:
                    _log_compact_directory_files(
                        mlflow,
                        manifest_path.parent,
                        artifact_path="c8/zone",
                        max_bytes=config.max_artifact_bytes,
                    )
            completed += 1
        except Exception:
            failed += 1
            if config.strict:
                raise
    return completed, failed


def _aggregate_phase_c_metrics(
    stage_payloads: dict[str, tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Collect availability-aware campaign metrics from authoritative manifests."""
    out: dict[str, Any] = {}
    c1 = stage_payloads.get("C1", (None, {}))[1]
    c4 = stage_payloads.get("C4", (None, {}))[1]
    c6 = stage_payloads.get("C6", (None, {}))[1]
    c7 = stage_payloads.get("C7", (None, {}))[1]
    c8 = stage_payloads.get("C8", (None, {}))[1]
    mappings = {
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
    for key, value in mappings.items():
        if _finite_float(value) is not None:
            out[key] = value
    return out


def register_phase_c_run(
    *,
    config: PhaseCTrackingConfig,
    stage_manifests: dict[str, str | Path],
    training_root: str | Path | None = None,
    evaluation_root: str | Path | None = None,
    inference_root: str | Path | None = None,
    registration_output_dir: str | Path | None = None,
) -> PhaseCRegistrationResult:
    """Register one completed Phase C filesystem run into MLflow."""

    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError(
            "C9 requires MLflow, but mlflow could not be imported."
        ) from exc

    tracking_uri = configure_mlflow_tracking()
    experiment_name = config.resolved_experiment_name()
    experiment_id = get_or_create_semantic_experiment(
        experiment_name=experiment_name,
        artifact_subdir=config.resolved_artifact_subdir(),
    )

    stage_payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    for stage, raw_path in stage_manifests.items():
        normalized_stage = stage.upper()
        if normalized_stage not in STAGE_DEFINITIONS:
            raise ValueError(f"Unsupported Phase C stage: {stage}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{normalized_stage} manifest does not exist: {path}"
            )
        stage_payloads[normalized_stage] = (path, read_json(path))

    parent = mlflow.start_run(run_name=config.resolved_run_name())
    stage_run_count = 0
    task_run_count = 0
    failed_count = 0

    try:
        _safe_log_params(
            mlflow,
            {
                "campaign_id": config.campaign_id,
                "phase_c_run_id": config.phase_c_run_id,
                "validation_mode": config.validation_mode,
                "registered_stage_count": len(stage_payloads),
                "log_compact_artifacts": config.log_compact_artifacts,
                "log_model_artifacts": config.log_model_artifacts,
            },
        )
        _safe_set_tags(
            mlflow,
            _standard_tags(
                config=config,
                extra={
                    "run_kind": "phase_c_campaign",
                    "c9_registration_mode": "posthoc_manifest_registration",
                },
            ),
        )

        for stage in sorted(stage_payloads):
            manifest_path, manifest = stage_payloads[stage]
            with mlflow.start_run(
                run_name=f"{stage}_{_extract_run_id(stage, manifest)}",
                nested=True,
            ):
                _log_stage_manifest(
                    mlflow=mlflow,
                    config=config,
                    stage=stage,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )

                if stage == "C6":
                    done, failed = _register_c6_tasks(
                        mlflow=mlflow,
                        config=config,
                        training_root=training_root,
                    )
                    task_run_count += done
                    failed_count += failed
                elif stage == "C7":
                    done, failed = _register_c7_tasks(
                        mlflow=mlflow,
                        config=config,
                        evaluation_root=evaluation_root,
                    )
                    task_run_count += done
                    failed_count += failed
                elif stage == "C8":
                    done, failed = _register_c8_tasks(
                        mlflow=mlflow,
                        config=config,
                        inference_root=inference_root,
                    )
                    task_run_count += done
                    failed_count += failed

            stage_run_count += 1

        _safe_log_metrics(
            mlflow,
            {
                "registered_stage_run_count": stage_run_count,
                "registered_task_run_count": task_run_count,
                "failed_registration_count": failed_count,
                **_aggregate_phase_c_metrics(stage_payloads),
            },
        )

        result = PhaseCRegistrationResult(
            experiment_name=experiment_name,
            experiment_id=str(experiment_id),
            parent_run_id=parent.info.run_id,
            tracking_uri=str(tracking_uri),
            stage_run_count=stage_run_count,
            task_run_count=task_run_count,
            failed_registration_count=failed_count,
        )

        if registration_output_dir is not None:
            output_dir = Path(registration_output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / "phase_c_mlflow_registration_manifest.json"
            result_path.write_text(
                json.dumps(
                    {
                        **result.to_dict(),
                        "campaign_id": config.campaign_id,
                        "phase_c_run_id": config.phase_c_run_id,
                        "stage_manifests": {
                            key: str(value[0])
                            for key, value in stage_payloads.items()
                        },
                        "training_root": str(training_root or ""),
                        "evaluation_root": str(evaluation_root or ""),
                        "inference_root": str(inference_root or ""),
                        "availability_summary": _aggregate_phase_c_metrics(
                            stage_payloads
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            if config.log_compact_artifacts:
                mlflow.log_artifact(
                    str(result_path),
                    artifact_path="c9_registration",
                )

        mlflow.end_run(status="FINISHED" if failed_count == 0 else "FAILED")
        return result

    except Exception:
        try:
            mlflow.set_tag("registration_status", "failed")
            mlflow.end_run(status="FAILED")
        finally:
            raise
