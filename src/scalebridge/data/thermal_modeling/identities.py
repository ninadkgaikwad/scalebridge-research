# -*- coding: utf-8 -*-
"""Stable identity models for Phase D datasets and zone products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import ModelingSilo, PhaseDMode


@dataclass(frozen=True)
class PhaseDSourceLineage:
    """Authoritative Phase B and Phase C lineage for one Phase D product."""

    campaign_id: str
    case_id: str
    aggregation_matrix_run_id: str
    aggregation_run_id: str
    aggregation_id: str
    phase_c_campaign_run_id: str
    phase_c_inference_run_id: str
    phase_c_split_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "case_id": self.case_id,
            "aggregation_matrix_run_id": self.aggregation_matrix_run_id,
            "aggregation_run_id": self.aggregation_run_id,
            "aggregation_id": self.aggregation_id,
            "phase_c_campaign_run_id": self.phase_c_campaign_run_id,
            "phase_c_inference_run_id": self.phase_c_inference_run_id,
            "phase_c_split_run_id": self.phase_c_split_run_id,
        }


@dataclass(frozen=True)
class PhaseDDatasetIdentity:
    """Stable logical identity for a canonical Phase D dataset."""

    phase_d_run_id: str
    dataset_id: str
    mode: PhaseDMode
    silo: ModelingSilo
    building_type: str
    climate_zone: str
    weather_location: str
    aggregate_zone_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.aggregate_zone_ids:
            raise ValueError("aggregate_zone_ids must contain at least one zone")
        if len(set(self.aggregate_zone_ids)) != len(self.aggregate_zone_ids):
            raise ValueError("aggregate_zone_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_d_run_id": self.phase_d_run_id,
            "dataset_id": self.dataset_id,
            "mode": self.mode.value,
            "silo": self.silo.value,
            "building_type": self.building_type,
            "climate_zone": self.climate_zone,
            "weather_location": self.weather_location,
            "aggregate_zone_ids": list(self.aggregate_zone_ids),
        }
