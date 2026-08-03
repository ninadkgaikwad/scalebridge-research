# -*- coding: utf-8 -*-
"""Data references and audit result models for Stage C1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AggregationZoneOutputRef:
    campaign_id: str
    case_id: str
    building_type: str
    climate_zone: str
    weather_location: str
    source_generation_run_id: str
    aggregation_run_id: str
    aggregation_id: str
    aggregation_level: str
    aggregation_family: str
    strategy: str
    rule_set: str
    weight_mode: str
    aggregate_zone_id: str
    aggregate_zone_count: int
    campaign_root: Path
    aggregation_run_root: Path
    aggregation_manifest_path: Path
    aggregation_plan_path: Path
    zone_root: Path
    wide_parquet_path: Path
    static_equipment_path: Path
    equipment_contributions_path: Path
    zone_mapping_path: Path
    rule_summary_path: Path | None
    rule_diagnostics_path: Path | None
    loaded_variables_path: Path | None
    schedule_mapping_path: Path | None
    node_temperature_summary_path: Path | None
    node_temperature_mapping_path: Path | None
    node_mass_flow_summary_path: Path | None
    node_mass_flow_mapping_path: Path | None
    source_run_manifest_path: Path | None
    rdd_intersection_path: Path | None
    variable_manifest_path: Path | None

    def identity_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id, "case_id": self.case_id,
            "building_type": self.building_type, "climate_zone": self.climate_zone,
            "weather_location": self.weather_location,
            "source_generation_run_id": self.source_generation_run_id,
            "aggregation_run_id": self.aggregation_run_id, "aggregation_id": self.aggregation_id,
            "aggregation_level": self.aggregation_level, "aggregation_family": self.aggregation_family,
            "strategy": self.strategy, "rule_set": self.rule_set, "weight_mode": self.weight_mode,
            "aggregate_zone_id": self.aggregate_zone_id,
            "aggregate_zone_count": self.aggregate_zone_count,
            "aggregation_run_root": str(self.aggregation_run_root), "zone_root": str(self.zone_root),
        }
