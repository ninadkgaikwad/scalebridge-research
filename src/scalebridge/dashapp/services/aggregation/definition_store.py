"""Persistence helpers for BGIRS Phase B Aggregation campaign definitions.

This module deliberately mirrors the Phase A Generation definition-store
contract.  It stores only definition JSON; scientific plan construction and
execution remain in ``scalebridge.data.aggregation``.
"""
from __future__ import annotations

from pathlib import Path

from scalebridge.data.aggregation.campaign_definition import (
    AggregationCampaignDefinition,
    load_aggregation_campaign_definition,
    write_aggregation_campaign_definition,
)
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


def definition_root() -> Path:
    """Return the persistent BGIRS Aggregation-definition directory."""
    path = resolve_generated_data_root() / "campaign_definitions" / "aggregation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def definition_path(campaign_id: str) -> Path:
    """Return the JSON path for one Aggregation campaign definition."""
    return definition_root() / f"{campaign_id}.json"


def list_definitions() -> list[dict[str, object]]:
    """Return compact metadata for all readable saved definitions.

    Corrupt/obsolete JSON files are skipped, matching the established
    Generation definition-store behavior so one bad file cannot break the
    Execution dropdown.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(definition_root().glob("*.json")):
        try:
            definition = load_aggregation_campaign_definition(path)
            rows.append(
                {
                    # Keep the generic ``campaign_id`` key used by the
                    # Generation service/UI pattern while also exposing the
                    # authoritative Phase B field name.
                    "campaign_id": definition.aggregation_campaign_id,
                    "aggregation_campaign_id": definition.aggregation_campaign_id,
                    "parent_generation_campaign_id": (
                        definition.parent_generation_campaign_id
                    ),
                    "plan_request_count": len(definition.plan_requests),
                    "strategies": definition.requested_strategy_values,
                    "weight_modes": definition.requested_weight_mode_values,
                    "machine_id": definition.machine_id,
                    "path": str(path),
                }
            )
        except Exception:
            continue
    return rows


def load_definition(campaign_id: str) -> AggregationCampaignDefinition:
    """Load and validate one saved Aggregation campaign definition."""
    return load_aggregation_campaign_definition(definition_path(campaign_id))


def save_definition(definition: AggregationCampaignDefinition) -> Path:
    """Persist one validated Aggregation campaign definition deterministically."""
    return write_aggregation_campaign_definition(
        definition_path(definition.aggregation_campaign_id),
        definition,
    )
