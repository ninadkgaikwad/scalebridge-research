# -*- coding: utf-8 -*-
"""Portable definitions for general ScaleBridge Aggregation campaigns.

The definition layer describes *what* Phase B should run. Scientific zone
selection, grouping validation, weighting, rule application, and output writing
remain in the existing aggregation modules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scalebridge.data.aggregation.models import (
    AggregationRuleSet,
    AggregationStrategy,
    AggregationWeightMode,
)
from scalebridge.data.aggregation.plans import (
    DEFAULT_AGGREGATE_ZONE_NAME_STEM,
    DEFAULT_SYSTEM_NODE_NAME_PATTERN,
)


_CAMPAIGN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$"


class AggregationPlanRequest(BaseModel):
    """One general plan-building request inside an Aggregation campaign.

    One request is applied to every selected Generation case. For built-in
    strategies it normally produces one plan per case. For ``custom_groups`` it
    may produce multiple plans per case when ``custom_aggregation_ids`` is empty
    and the grouping CSV contains multiple aggregation_id blocks.

    The optional level/family fields are presentation/analysis metadata. They do
    not alter scientific grouping behavior. When omitted, the campaign runner
    uses aggregation_id and strategy-derived generic labels.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: AggregationStrategy
    weight_mode: AggregationWeightMode = AggregationWeightMode.EQUAL
    rule_set: AggregationRuleSet = AggregationRuleSet.LEGACY_V1
    custom_aggregation_ids: tuple[str, ...] = ()

    aggregation_level: str | None = None
    aggregation_level_index: int | None = Field(default=None, ge=0)
    aggregation_family: str | None = None

    @field_validator("custom_aggregation_ids")
    @classmethod
    def validate_custom_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(str(value).strip() for value in values if str(value).strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("custom_aggregation_ids must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def validate_strategy_specific_fields(self):
        if (
            self.strategy != AggregationStrategy.CUSTOM_GROUPS
            and self.custom_aggregation_ids
        ):
            raise ValueError(
                "custom_aggregation_ids are only valid for strategy=custom_groups"
            )
        return self

    @property
    def request_key(self) -> tuple[str, str, str, tuple[str, ...]]:
        """Stable identity used to reject duplicate requests."""
        return (
            self.strategy.value,
            self.rule_set.value,
            self.weight_mode.value,
            tuple(sorted(self.custom_aggregation_ids)),
        )


class AggregationCampaignDefinition(BaseModel):
    """Portable definition for a general Phase B Aggregation campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    aggregation_campaign_id: str = Field(pattern=_CAMPAIGN_ID_PATTERN)
    parent_generation_campaign_id: str = Field(pattern=_CAMPAIGN_ID_PATTERN)
    machine_id: str = Field(min_length=1)

    # Empty means all successful latest Generation cases in the parent campaign.
    case_ids: tuple[str, ...] = ()
    case_limit: int | None = Field(default=None, ge=1)

    # Explicit plan requests make the campaign general. BGIRS may build these
    # from checkboxes/multi-selects without hard-coding P1 L01-L05 semantics.
    plan_requests: tuple[AggregationPlanRequest, ...] = Field(min_length=1)

    # Required at runtime when any request uses custom_groups. Relative paths
    # are resolved relative to the campaign-definition JSON file.
    custom_zone_groups_path: str | None = None

    aggregate_zone_name_stem: str = DEFAULT_AGGREGATE_ZONE_NAME_STEM
    system_node_name_pattern: str = DEFAULT_SYSTEM_NODE_NAME_PATTERN

    # Parent data resolution. parent_generation_campaign_root takes precedence.
    parent_generation_campaign_root: str | None = None
    generated_data_root: str | None = None

    # Scientific execution options already supported by engine.py.
    max_variables: int | None = Field(default=None, ge=1)
    preview_rows: int = Field(default=100, ge=0)
    write_legacy_pickle: bool = False
    continue_on_error: bool = False

    # Optional MLflow tracking through the existing Aggregation helper.
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = "http://127.0.0.1:5000"
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    mlflow_strict: bool = False

    @field_validator("case_ids")
    @classmethod
    def validate_case_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(str(value).strip() for value in values if str(value).strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("case_ids must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def validate_campaign(self):
        if len({request.request_key for request in self.plan_requests}) != len(
            self.plan_requests
        ):
            raise ValueError("plan_requests must not contain duplicate requests")

        uses_custom_groups = any(
            request.strategy == AggregationStrategy.CUSTOM_GROUPS
            for request in self.plan_requests
        )
        if uses_custom_groups and not self.custom_zone_groups_path:
            raise ValueError(
                "custom_zone_groups_path is required when any plan request uses "
                "strategy=custom_groups"
            )
        return self

    @property
    def requested_strategy_values(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(request.strategy.value for request in self.plan_requests))

    @property
    def requested_weight_mode_values(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(request.weight_mode.value for request in self.plan_requests)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation."""
        return self.model_dump(mode="json")


def load_aggregation_campaign_definition(
    path: str | Path,
) -> AggregationCampaignDefinition:
    """Load and validate one Aggregation campaign definition JSON file."""
    definition_path = Path(path).expanduser().resolve()
    payload = json.loads(definition_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {definition_path}")
    return AggregationCampaignDefinition.model_validate(payload)


def write_aggregation_campaign_definition(
    path: str | Path,
    definition: AggregationCampaignDefinition,
) -> Path:
    """Write a deterministic Aggregation campaign definition JSON file."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(definition.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
