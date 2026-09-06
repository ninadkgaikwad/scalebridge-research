from __future__ import annotations

import pytest
from pydantic import ValidationError

from scalebridge.data.aggregation.campaign_definition import (
    AggregationCampaignDefinition,
    AggregationPlanRequest,
)
from scalebridge.data.aggregation.models import (
    AggregationStrategy,
    AggregationWeightMode,
)


def _base_payload():
    return {
        "aggregation_campaign_id": "bgirs_phase_b_test_v1",
        "parent_generation_campaign_id": "generation_parent_v1",
        "machine_id": "labpc",
        "plan_requests": [
            {
                "strategy": "all_thermal_zones_to_one",
                "weight_mode": "equal",
            }
        ],
        "mlflow_enabled": False,
    }


def test_general_campaign_accepts_builtin_strategy():
    definition = AggregationCampaignDefinition.model_validate(_base_payload())
    assert definition.aggregation_campaign_id == "bgirs_phase_b_test_v1"
    assert definition.plan_requests[0].strategy == (
        AggregationStrategy.ALL_THERMAL_ZONES_TO_ONE
    )
    assert definition.plan_requests[0].weight_mode == AggregationWeightMode.EQUAL


def test_custom_groups_requires_custom_grouping_path():
    payload = _base_payload()
    payload["plan_requests"] = [
        {"strategy": "custom_groups", "weight_mode": "floor_area"}
    ]
    with pytest.raises(ValidationError):
        AggregationCampaignDefinition.model_validate(payload)


def test_non_custom_request_rejects_custom_ids():
    with pytest.raises(ValidationError):
        AggregationPlanRequest.model_validate(
            {
                "strategy": "identity",
                "weight_mode": "equal",
                "custom_aggregation_ids": ["not_allowed"],
            }
        )


def test_duplicate_plan_requests_are_rejected():
    payload = _base_payload()
    payload["plan_requests"] = [
        {"strategy": "identity", "weight_mode": "equal"},
        {"strategy": "identity", "weight_mode": "equal"},
    ]
    with pytest.raises(ValidationError):
        AggregationCampaignDefinition.model_validate(payload)


def test_all_repo_weight_modes_can_be_requested():
    payload = _base_payload()
    payload["plan_requests"] = [
        {"strategy": "identity", "weight_mode": "equal"},
        {"strategy": "identity", "weight_mode": "floor_area"},
        {"strategy": "identity", "weight_mode": "volume"},
    ]
    definition = AggregationCampaignDefinition.model_validate(payload)
    assert definition.requested_weight_mode_values == (
        "equal",
        "floor_area",
        "volume",
    )
