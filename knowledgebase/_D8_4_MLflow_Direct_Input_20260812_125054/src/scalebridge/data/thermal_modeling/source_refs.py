# -*- coding: utf-8 -*-
"""Immutable source-reference contracts for Phase D D2 discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AggregationRunRef:
    """Resolved Phase B aggregation run."""

    campaign_id: str
    case_id: str
    matrix_run_id: str
    aggregation_run_id: str
    aggregation_id: str
    weight_mode: str
    strategy: str
    run_root: Path
    manifest_path: Path
    plan_path: Path
    zone_mapping_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "case_id": self.case_id,
            "matrix_run_id": self.matrix_run_id,
            "aggregation_run_id": self.aggregation_run_id,
            "aggregation_id": self.aggregation_id,
            "weight_mode": self.weight_mode,
            "strategy": self.strategy,
            "run_root": str(self.run_root),
            "manifest_path": str(self.manifest_path),
            "plan_path": str(self.plan_path),
            "zone_mapping_path": str(self.zone_mapping_path),
        }


@dataclass(frozen=True)
class AggregationZoneRef:
    """Resolved Phase B aggregate-zone output."""

    aggregation_run: AggregationRunRef
    aggregate_zone_id: str
    zone_root: Path
    wide_parquet_path: Path
    wide_preview_path: Path | None = None
    long_parquet_path: Path | None = None
    zone_mapping_path: Path | None = None
    static_equipment_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregation_run": self.aggregation_run.to_dict(),
            "aggregate_zone_id": self.aggregate_zone_id,
            "zone_root": str(self.zone_root),
            "wide_parquet_path": str(self.wide_parquet_path),
            "wide_preview_path": (
                str(self.wide_preview_path) if self.wide_preview_path else None
            ),
            "long_parquet_path": (
                str(self.long_parquet_path) if self.long_parquet_path else None
            ),
            "zone_mapping_path": (
                str(self.zone_mapping_path) if self.zone_mapping_path else None
            ),
            "static_equipment_path": (
                str(self.static_equipment_path) if self.static_equipment_path else None
            ),
        }


@dataclass(frozen=True)
class PhaseCChildRunRefs:
    """Resolved child-run lineage from one authoritative Phase C campaign."""

    campaign_run_id: str
    campaign_run_root: Path
    campaign_plan_path: Path
    campaign_manifest_path: Path
    audit_run_id: str
    feature_run_id: str
    split_run_id: str
    dataset_run_id: str
    training_run_id: str
    evaluation_run_id: str
    inference_run_id: str
    mlflow_registration_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_run_id": self.campaign_run_id,
            "campaign_run_root": str(self.campaign_run_root),
            "campaign_plan_path": str(self.campaign_plan_path),
            "campaign_manifest_path": str(self.campaign_manifest_path),
            "audit_run_id": self.audit_run_id,
            "feature_run_id": self.feature_run_id,
            "split_run_id": self.split_run_id,
            "dataset_run_id": self.dataset_run_id,
            "training_run_id": self.training_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "inference_run_id": self.inference_run_id,
            "mlflow_registration_run_id": self.mlflow_registration_run_id,
        }


@dataclass(frozen=True)
class PhaseCZoneRef:
    """Resolved Phase C applicability, inference, and split references."""

    case_id: str
    aggregation_id: str
    weight_mode: str
    aggregate_zone_id: str
    applicable_models_path: Path
    unavailable_models_path: Path
    signal_catalog_path: Path
    inference_zone_root: Path
    predictions_parquet_path: Path
    predictions_preview_path: Path | None
    component_prediction_summary_path: Path | None
    timestamp_component_availability_path: Path | None
    split_zone_root: Path | None
    split_assignments_parquet_path: Path | None
    split_assignments_preview_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "aggregation_id": self.aggregation_id,
            "weight_mode": self.weight_mode,
            "aggregate_zone_id": self.aggregate_zone_id,
            "applicable_models_path": str(self.applicable_models_path),
            "unavailable_models_path": str(self.unavailable_models_path),
            "signal_catalog_path": str(self.signal_catalog_path),
            "inference_zone_root": str(self.inference_zone_root),
            "predictions_parquet_path": str(self.predictions_parquet_path),
            "predictions_preview_path": (
                str(self.predictions_preview_path)
                if self.predictions_preview_path else None
            ),
            "component_prediction_summary_path": (
                str(self.component_prediction_summary_path)
                if self.component_prediction_summary_path else None
            ),
            "timestamp_component_availability_path": (
                str(self.timestamp_component_availability_path)
                if self.timestamp_component_availability_path else None
            ),
            "split_zone_root": str(self.split_zone_root) if self.split_zone_root else None,
            "split_assignments_parquet_path": (
                str(self.split_assignments_parquet_path)
                if self.split_assignments_parquet_path else None
            ),
            "split_assignments_preview_path": (
                str(self.split_assignments_preview_path)
                if self.split_assignments_preview_path else None
            ),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PhaseDDiscoveryResult:
    """Complete D2 discovery result for one Phase D source selection."""

    campaign_root: Path
    aggregation_zone: AggregationZoneRef
    phase_c_runs: PhaseCChildRunRefs
    phase_c_zone: PhaseCZoneRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_root": str(self.campaign_root),
            "aggregation_zone": self.aggregation_zone.to_dict(),
            "phase_c_runs": self.phase_c_runs.to_dict(),
            "phase_c_zone": self.phase_c_zone.to_dict(),
        }
