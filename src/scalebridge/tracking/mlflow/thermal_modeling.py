# -*- coding: utf-8 -*-
"""Optional MLflow tracking for ScaleBridge Phase D campaign dataset builds.

D8.4 deliberately uses one MLflow run per Phase D campaign execution. Detailed
per-aggregation and per-dataset provenance remains authoritative in the Phase D
CSV/JSON registries; those compact registries are logged as MLflow artifacts.
The final Phase D Parquets and per-aggregation logs are not uploaded to MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass
import atexit
from pathlib import Path
from typing import Any


def _stringify(value: Any, max_length: int = 500) -> str:
    if isinstance(value, (list, tuple, set)):
        text = ",".join(str(x) for x in value)
    elif isinstance(value, dict):
        import json
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
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


@dataclass
class PhaseDMLflowTracker:
    enabled: bool = False
    experiment_name: str | None = None
    run_name: str | None = None
    strict: bool = False
    _active: bool = False

    def _handle(self, exc: Exception, context: str) -> None:
        if self.strict:
            raise RuntimeError(f"Phase D MLflow {context} failed: {exc}") from exc
        print(f"WARNING: Phase D MLflow {context} failed: {type(exc).__name__}: {exc}")

    def start(
        self,
        *,
        campaign_id: str,
        matrix_run_id: str,
        phase_c_campaign_run_id: str,
        phase_d_run_id: str,
        selected_aggregation_run_count: int,
        configuration: dict[str, Any],
        resume: bool,
        overwrite_existing: bool,
        continue_on_error: bool,
        dry_run: bool,
    ) -> str | None:
        if not self.enabled:
            return None
        try:
            import mlflow
            from scalebridge.tracking.mlflow.semantic import (
                get_or_create_semantic_experiment,
                set_standard_tags,
            )

            effective_experiment = self.experiment_name or f"ScaleBridge_PhaseD_{campaign_id}"
            effective_run_name = self.run_name or phase_d_run_id
            get_or_create_semantic_experiment(
                effective_experiment,
                artifact_subdir=f"phase_d/{campaign_id}",
            )
            run = mlflow.start_run(run_name=effective_run_name)
            self._active = True
            atexit.register(self._atexit_finish)
            set_standard_tags(
                campaign_id=campaign_id,
                model_family="phase_d_thermal_model_data",
                extra_tags={
                    "pipeline_stage": "phase_d",
                    "matrix_run_id": matrix_run_id,
                    "phase_c_campaign_run_id": phase_c_campaign_run_id,
                    "phase_d_run_id": phase_d_run_id,
                },
            )
            params = {
                "campaign_id": campaign_id,
                "matrix_run_id": matrix_run_id,
                "phase_c_campaign_run_id": phase_c_campaign_run_id,
                "phase_d_run_id": phase_d_run_id,
                "selected_aggregation_run_count": selected_aggregation_run_count,
                "resume": resume,
                "overwrite_existing": overwrite_existing,
                "continue_on_error": continue_on_error,
                "dry_run": dry_run,
                **configuration,
            }
            for key, value in params.items():
                if value is not None:
                    mlflow.log_param(str(key), _stringify(value))
            return run.info.run_id
        except Exception as exc:
            self._handle(exc, "start")
            return None

    def _atexit_finish(self) -> None:
        if not self._active:
            return
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        self._active = False

    def log_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled or not self._active:
            return
        try:
            import mlflow
            metric_keys = (
                "selected_aggregation_run_count",
                "completed_aggregation_run_count",
                "skipped_completed_aggregation_run_count",
                "failed_aggregation_run_count",
                "dataset_count",
                "ml_dataset_count",
                "opt_bayes_dataset_count",
                "ind_dataset_count",
                "dep1_dataset_count",
                "dep2_dataset_count",
                "runtime_seconds",
            )
            for key in metric_keys:
                number = _finite_float(summary.get(key))
                if number is not None:
                    mlflow.log_metric(key, number)
            mlflow.set_tag("phase_d_status", str(summary.get("status", "unknown")))
        except Exception as exc:
            self._handle(exc, "summary logging")

    def log_campaign_artifacts(self, campaign_run_root: Path) -> None:
        if not self.enabled or not self._active:
            return
        try:
            import mlflow
            names = (
                "phase_d_campaign_plan.json",
                "phase_d_campaign_run_manifest.json",
                "aggregation_run_registry.csv",
                "dataset_registry.csv",
                "failures.csv",
            )
            for name in names:
                path = Path(campaign_run_root) / name
                if path.is_file():
                    mlflow.log_artifact(str(path), artifact_path="phase_d_campaign")
        except Exception as exc:
            self._handle(exc, "artifact logging")

    def finish(self, *, failed: bool = False) -> None:
        if not self.enabled or not self._active:
            return
        try:
            import mlflow
            mlflow.end_run(status="FAILED" if failed else "FINISHED")
            self._active = False
            try:
                atexit.unregister(self._atexit_finish)
            except Exception:
                pass
        except Exception as exc:
            self._handle(exc, "run finalization")
