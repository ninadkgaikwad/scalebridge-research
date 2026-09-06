# -*- coding: utf-8 -*-
"""Portable BGIRS definition envelope for Phase C Heat-Input Regression.

The scientific configuration itself remains authoritative in
``PhaseCCampaignConfig``.  This Dash-facing envelope adds only persistent UI
identity/lineage metadata that is not part of the scientific runner contract.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig


_CAMPAIGN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$"
_UPSTREAM_AGGREGATION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,255}$"


class HeatInputCampaignDefinition(BaseModel):
    """Saved BGIRS Phase C campaign definition.

    ``phase_c_campaign_id`` identifies the reusable BGIRS definition.
    ``runner_config`` is the complete typed scientific/execution configuration
    consumed later by the authoritative Phase C runner.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    phase_c_campaign_id: str = Field(pattern=_CAMPAIGN_ID_PATTERN)
    parent_aggregation_campaign_id: str = Field(pattern=_UPSTREAM_AGGREGATION_ID_PATTERN)
    parent_generation_campaign_id: str = Field(pattern=_CAMPAIGN_ID_PATTERN)
    machine_id: str = Field(min_length=1)
    display_name: str | None = None
    notes: str | None = None
    runner_config: PhaseCCampaignConfig

    @model_validator(mode="after")
    def _validate_lineage(self):
        if (
            self.runner_config.campaign_id
            and self.runner_config.campaign_id != self.parent_generation_campaign_id
        ):
            raise ValueError(
                "runner_config.campaign_id must match parent_generation_campaign_id"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return self.model_dump(mode="json")
