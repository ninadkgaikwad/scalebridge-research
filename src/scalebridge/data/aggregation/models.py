# -*- coding: utf-8 -*-
"""Data models for ScaleBridge aggregation discovery and audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from enum import Enum
from typing import Any



SUCCESS_STATUSES = {
    "completed",
    "completed_with_warnings",
}


@dataclass(frozen=True)
class GenerationRunRef:
    """Resolved generation run reference."""

    case_id: str
    run_id: str
    status: str
    case_root: Path
    run_root: Path
    manifest_path: Path


@dataclass(frozen=True)
class RddVariableIntersection:
    """Optional RDD probe variable-availability summary."""

    case_id: str
    status: str
    path: Path | None
    requested_variable_count: int | None
    rdd_available_variable_count: int | None
    rdd_unavailable_variable_count: int | None
    available_variables: tuple[str, ...]
    unavailable_variables: tuple[str, ...]

    @property
    def available_set(self) -> set[str]:
        """Return available variable names as a set."""
        return set(self.available_variables)

    @property
    def unavailable_set(self) -> set[str]:
        """Return unavailable variable names as a set."""
        return set(self.unavailable_variables)
    

class AggregationStrategy(str, Enum):
    """Supported source-zone grouping strategies."""

    ALL_THERMAL_ZONES_TO_ONE = "all_thermal_zones_to_one"
    CUSTOM_GROUPS = "custom_groups"
    IDENTITY = "identity"


class AggregationWeightMode(str, Enum):
    """Supported legacy aggregation weighting modes."""

    EQUAL = "equal"
    FLOOR_AREA = "floor_area"
    VOLUME = "volume"


class AggregationRuleSet(str, Enum):
    """Supported aggregation rule sets."""

    LEGACY_V1 = "legacy_v1"


@dataclass(frozen=True)
class AggregateZoneGroup:
    """One aggregate zone and its source thermal zones."""

    aggregate_zone_id: str
    source_zones: tuple[str, ...]


@dataclass(frozen=True)
class AggregationPlan:
    """Expanded aggregation plan."""

    schema_version: str
    aggregation_id: str
    strategy: AggregationStrategy
    rule_set: AggregationRuleSet
    weight_mode: AggregationWeightMode
    aggregate_zone_name_stem: str
    system_node_name_pattern: str
    source_case_id: str
    source_generation_run_id: str
    campaign_id: str
    building_type: str
    weather_location: str
    climate_zone: str
    thermal_zone_filter: dict[str, Any]
    aggregate_zones: tuple[AggregateZoneGroup, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "aggregation_id": self.aggregation_id,
            "strategy": self.strategy.value,
            "rule_set": self.rule_set.value,
            "weight_mode": self.weight_mode.value,
            "aggregate_zone_name_stem": self.aggregate_zone_name_stem,
            "system_node_name_pattern": self.system_node_name_pattern,
            "source_case_id": self.source_case_id,
            "source_generation_run_id": self.source_generation_run_id,
            "campaign_id": self.campaign_id,
            "building_type": self.building_type,
            "weather_location": self.weather_location,
            "climate_zone": self.climate_zone,
            "thermal_zone_filter": self.thermal_zone_filter,
            "aggregate_zones": [
                {
                    "aggregate_zone_id": group.aggregate_zone_id,
                    "source_zones": list(group.source_zones),
                }
                for group in self.aggregate_zones
            ],
        }