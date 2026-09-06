from __future__ import annotations

import pytest

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig
from scalebridge.dashapp.schemas.pipeline.heat_input import HeatInputCampaignDefinition


def _definition():
    return HeatInputCampaignDefinition(
        phase_c_campaign_id="phase_c_test_v1",
        parent_aggregation_campaign_id="Agg_BGIRS_e2e_V1",
        parent_generation_campaign_id="bgirs_e2e_dropdown_2b2w_v1",
        machine_id="laptop",
        runner_config=PhaseCCampaignConfig(
            campaign_id="bgirs_e2e_dropdown_2b2w_v1",
            campaign_root="C:/tmp/bgirs_e2e_dropdown_2b2w_v1",
            matrix_run_id="aggregation_matrix_test",
        ),
    )


def test_definition_embeds_authoritative_phase_c_config():
    definition = _definition()
    assert isinstance(definition.runner_config, PhaseCCampaignConfig)
    assert definition.runner_config.split_strategy == "monthly_distributed_holdout"


def test_definition_keeps_bgirs_id_separate_from_scientific_parent_id():
    definition = _definition()
    assert definition.phase_c_campaign_id == "phase_c_test_v1"
    assert definition.parent_generation_campaign_id == (
        definition.runner_config.campaign_id
    )


def test_definition_rejects_mismatched_generation_lineage():
    with pytest.raises(ValueError, match="campaign_id must match"):
        HeatInputCampaignDefinition(
            phase_c_campaign_id="phase_c_test_v1",
            parent_aggregation_campaign_id="Agg_BGIRS_e2e_V1",
            parent_generation_campaign_id="different_parent",
            machine_id="laptop",
            runner_config=PhaseCCampaignConfig(
                campaign_id="bgirs_e2e_dropdown_2b2w_v1",
                campaign_root="C:/tmp/bgirs_e2e_dropdown_2b2w_v1",
                matrix_run_id="aggregation_matrix_test",
            ),
        )


def test_definition_accepts_established_legacy_aggregation_artifact_id():
    definition = HeatInputCampaignDefinition(
        phase_c_campaign_id="phase_c_legacy_v1",
        parent_aggregation_campaign_id="legacy::generation_parent_v1",
        parent_generation_campaign_id="generation_parent_v1",
        machine_id="laptop",
        runner_config=PhaseCCampaignConfig(
            campaign_id="generation_parent_v1",
            campaign_root="C:/tmp/generation_parent_v1",
            matrix_run_id="aggregation_matrix_legacy",
        ),
    )
    assert definition.parent_aggregation_campaign_id == "legacy::generation_parent_v1"
