"""Semantic MLflow logging for EnergyPlus generation attempts.

This module logs compact EnergyPlus generation metadata to the configured
local MLflow backend while keeping local run manifests authoritative.

Tracking policy:
    - Windows machines normally use http://127.0.0.1:5000.
    - Kamiak normally uses a SQLite MLflow tracking URI.
    - Artifact organization is semantic:
        Data/ScaleBridge/mlflow_artifacts/<campaign_or_experiment_name>/
    - Machine identity is metadata, not an artifact folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from scalebridge.integration.energyplus.manifests.models import (
    CaseSpec,
    GenerationResult,
    TrackingMetadata,
)
from scalebridge.tracking.mlflow.semantic import (
    configure_mlflow_tracking,
    get_machine_id,
    get_or_create_semantic_experiment,
    set_standard_tags,
)


@dataclass(frozen=True)
class MLflowTrackingHandle:
    """Identifiers for one active or successfully created MLflow run."""

    experiment_name: str
    run_id: str
    tracking_uri: str

    def to_metadata(self) -> TrackingMetadata:
        """Convert the handle to persisted manifest tracking metadata."""
        return TrackingMetadata(
            mlflow_experiment=self.experiment_name,
            mlflow_run_id=self.run_id,
            mlflow_tracking_uri=self.tracking_uri,
        )


class MLflowGenerationTracker:
    """Log compact EnergyPlus generation metadata to semantic MLflow."""

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        artifact_subdir: str | None = None,
        enabled: bool = True,
        strict: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.artifact_subdir = artifact_subdir
        self._enabled = enabled
        self.strict = strict

    @property
    def enabled(self) -> bool:
        """Return whether generation tracking is enabled."""
        return self._enabled

    def _resolve_experiment_name(self, campaign_id: str | None) -> str:
        if self.experiment_name:
            return self.experiment_name
        if campaign_id:
            return f"{campaign_id}_generation"
        return "energyplus_generation"

    def _resolve_artifact_subdir(
        self,
        *,
        experiment_name: str,
        campaign_id: str | None,
    ) -> str:
        if self.artifact_subdir:
            return self.artifact_subdir
        if campaign_id:
            return campaign_id
        return experiment_name

    def start(
        self,
        *,
        case_spec: CaseSpec,
        run_id: str,
        campaign_id: str | None,
        machine_id: str | None = None,
    ) -> MLflowTrackingHandle | None:
        """Create one MLflow run and log immutable case parameters."""
        if not self.enabled:
            return None

        experiment_name = self._resolve_experiment_name(campaign_id)
        artifact_subdir = self._resolve_artifact_subdir(
            experiment_name=experiment_name,
            campaign_id=campaign_id,
        )

        try:
            tracking_uri = configure_mlflow_tracking()
            get_or_create_semantic_experiment(
                experiment_name=experiment_name,
                artifact_subdir=artifact_subdir,
            )

            active = mlflow.start_run(run_name=run_id)

            resolved_machine_id = machine_id or get_machine_id()

            mlflow.log_params(
                {
                    "case_id": case_spec.case_id,
                    "campaign_id": campaign_id or "",
                    "building_type": case_spec.building_type or "",
                    "weather_location": case_spec.weather_location or "",
                    "climate_zone": case_spec.climate_zone or "",
                    "prototype_standard": case_spec.prototype_standard or "",
                    "prototype_year": case_spec.prototype_year or "",
                    "timestep_minutes": case_spec.timestep_minutes,
                    "calendar_year": case_spec.run_period.calendar_year or "",
                    "start_date": (
                        f"{case_spec.run_period.start_month:02d}-"
                        f"{case_spec.run_period.start_day:02d}"
                    ),
                    "end_date": (
                        f"{case_spec.run_period.end_month:02d}-"
                        f"{case_spec.run_period.end_day:02d}"
                    ),
                    "requested_variable_count": len(case_spec.output_variables),
                    "machine_id": resolved_machine_id,
                    "idf_sha256": case_spec.idf_sha256,
                    "epw_sha256": case_spec.epw_sha256,
                    "energyplus_version_requested": case_spec.energyplus_version or "",
                }
            )

            set_standard_tags(
                campaign_id=campaign_id,
                case_id=case_spec.case_id,
                model_family="energyplus_generation",
                extra_tags={
                    "run_id": run_id,
                    "paper": case_spec.tags.get("paper", ""),
                    "generation_stage": "energyplus",
                    "artifact_subdir": artifact_subdir,
                },
            )

            return MLflowTrackingHandle(
                experiment_name=experiment_name,
                run_id=active.info.run_id,
                tracking_uri=str(tracking_uri),
            )

        except Exception:
            if self.strict:
                raise
            try:
                if mlflow.active_run() is not None:
                    mlflow.end_run(status="FAILED")
            except Exception:
                pass
            return None

    def finish(
        self,
        *,
        handle: MLflowTrackingHandle | None,
        result: GenerationResult,
        manifest_path: Path,
    ) -> None:
        """Log final metrics and the compact run manifest, then close the run."""
        if handle is None:
            return

        try:
            mlflow.log_metrics(
                {
                    "runtime_seconds": float(result.runtime_seconds),
                    "warning_count": float(result.warning_count),
                    "severe_count": float(result.severe_count),
                    "fatal_count": float(result.fatal_count),
                    "requested_signal_count": float(result.requested_signal_count),
                    "produced_signal_count": float(result.produced_signal_count),
                    "missing_required_signal_count": float(
                        len(result.missing_required_signals)
                    ),
                    "timestep_count": float(result.timestep_count or 0),
                    "energyplus_exit_code": float(result.energyplus_exit_code or 0),
                }
            )

            mlflow.set_tags(
                {
                    "generation_status": result.status.value,
                    "manifest_path": str(manifest_path),
                    "artifact_root": str(result.artifact_root),
                }
            )

            mlflow.log_artifact(str(manifest_path), artifact_path="manifests")
            mlflow_status = (
                "FAILED"
                if result.status.value in {"failed", "invalid", "cancelled"}
                else "FINISHED"
            )
            mlflow.end_run(status=mlflow_status)

        except Exception:
            if self.strict:
                try:
                    mlflow.end_run(status="FAILED")
                finally:
                    raise
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass

    def fail(
        self,
        *,
        handle: MLflowTrackingHandle | None,
        error: BaseException,
    ) -> None:
        """Mark an active MLflow generation run as failed."""
        if handle is None:
            return

        try:
            mlflow.set_tags(
                {
                    "generation_status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            mlflow.end_run(status="FAILED")
        except Exception:
            if self.strict:
                raise