from __future__ import annotations

import json
from pathlib import Path

import pytest

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig
from scalebridge.dashapp.schemas.pipeline.heat_input import HeatInputCampaignDefinition
from scalebridge.dashapp.services.heat_input import definition_store


def _definition(campaign_id: str = "phase_c_test_v1"):
    return HeatInputCampaignDefinition(
        phase_c_campaign_id=campaign_id,
        parent_aggregation_campaign_id="Agg_BGIRS_e2e_V1",
        parent_generation_campaign_id="bgirs_e2e_dropdown_2b2w_v1",
        machine_id="laptop",
        runner_config=PhaseCCampaignConfig(
            campaign_id="bgirs_e2e_dropdown_2b2w_v1",
            campaign_root="C:/tmp/bgirs_e2e_dropdown_2b2w_v1",
            matrix_run_id="aggregation_matrix_test",
        ),
    )


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        definition_store,
        "resolve_generated_data_root",
        lambda: tmp_path,
    )
    return tmp_path


def test_definition_root_uses_heat_input_namespace(isolated_root):
    assert definition_store.definition_root() == (
        isolated_root / "campaign_definitions" / "heat_input"
    )


def test_round_trip_and_deterministic_json(isolated_root):
    definition = _definition()
    path = definition_store.save_definition(definition)
    assert definition_store.load_definition(definition.phase_c_campaign_id) == definition

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase_c_campaign_id"] == "phase_c_test_v1"
    assert payload["runner_config"]["matrix_run_id"] == "aggregation_matrix_test"


def test_save_refuses_silent_replacement(isolated_root):
    definition = _definition()
    definition_store.save_definition(definition)
    with pytest.raises(FileExistsError):
        definition_store.save_definition(definition)

    replaced = definition_store.save_definition(definition, replace=True)
    assert replaced.is_file()


def test_list_definitions_skips_bad_json(isolated_root):
    definition_store.save_definition(_definition("phase_c_b_v1"))
    definition_store.save_definition(_definition("phase_c_a_v1"))
    (definition_store.definition_root() / "bad.json").write_text(
        "{bad",
        encoding="utf-8",
    )

    rows = definition_store.list_definitions()
    assert [row["campaign_id"] for row in rows] == [
        "phase_c_a_v1",
        "phase_c_b_v1",
    ]
