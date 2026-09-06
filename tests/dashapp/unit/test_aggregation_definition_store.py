from __future__ import annotations

import json
from pathlib import Path

import pytest

from scalebridge.data.aggregation.campaign_definition import (
    AggregationCampaignDefinition,
    AggregationPlanRequest,
)
from scalebridge.data.aggregation.models import (
    AggregationStrategy,
    AggregationWeightMode,
)
from scalebridge.dashapp.services.aggregation import definition_store


def _definition(campaign_id: str = "bgirs_aggregation_test_v1") -> AggregationCampaignDefinition:
    return AggregationCampaignDefinition(
        aggregation_campaign_id=campaign_id,
        parent_generation_campaign_id="bgirs_generation_parent_v1",
        machine_id="laptop",
        plan_requests=(
            AggregationPlanRequest(
                strategy=AggregationStrategy.ALL_THERMAL_ZONES_TO_ONE,
                weight_mode=AggregationWeightMode.EQUAL,
            ),
            AggregationPlanRequest(
                strategy=AggregationStrategy.IDENTITY,
                weight_mode=AggregationWeightMode.VOLUME,
            ),
        ),
        mlflow_enabled=False,
    )


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        definition_store,
        "resolve_generated_data_root",
        lambda: tmp_path,
    )
    return tmp_path


def test_definition_root_mirrors_generation_storage_layout(isolated_root: Path) -> None:
    root = definition_store.definition_root()
    assert root == isolated_root / "campaign_definitions" / "aggregation"
    assert root.is_dir()


def test_save_and_load_round_trip(isolated_root: Path) -> None:
    definition = _definition()
    path = definition_store.save_definition(definition)

    assert path == (
        isolated_root
        / "campaign_definitions"
        / "aggregation"
        / "bgirs_aggregation_test_v1.json"
    )
    assert path.is_file()
    assert definition_store.load_definition(definition.aggregation_campaign_id) == definition


def test_saved_json_is_deterministic_and_uses_phase_b_schema(isolated_root: Path) -> None:
    definition = _definition()
    path = definition_store.save_definition(definition)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["aggregation_campaign_id"] == "bgirs_aggregation_test_v1"
    assert payload["parent_generation_campaign_id"] == "bgirs_generation_parent_v1"
    assert [row["strategy"] for row in payload["plan_requests"]] == [
        "all_thermal_zones_to_one",
        "identity",
    ]


def test_list_definitions_returns_generation_style_summary_rows(isolated_root: Path) -> None:
    definition_store.save_definition(_definition("bgirs_aggregation_b_v1"))
    definition_store.save_definition(_definition("bgirs_aggregation_a_v1"))

    rows = definition_store.list_definitions()

    assert [row["campaign_id"] for row in rows] == [
        "bgirs_aggregation_a_v1",
        "bgirs_aggregation_b_v1",
    ]
    assert rows[0]["aggregation_campaign_id"] == "bgirs_aggregation_a_v1"
    assert rows[0]["parent_generation_campaign_id"] == "bgirs_generation_parent_v1"
    assert rows[0]["plan_request_count"] == 2
    assert rows[0]["strategies"] == (
        "all_thermal_zones_to_one",
        "identity",
    )
    assert rows[0]["weight_modes"] == ("equal", "volume")
    assert rows[0]["machine_id"] == "laptop"
    assert rows[0]["path"].endswith("bgirs_aggregation_a_v1.json")


def test_list_definitions_skips_invalid_json_without_breaking_dropdown(isolated_root: Path) -> None:
    definition_store.save_definition(_definition("bgirs_aggregation_good_v1"))
    bad = definition_store.definition_root() / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")

    rows = definition_store.list_definitions()

    assert [row["campaign_id"] for row in rows] == ["bgirs_aggregation_good_v1"]


def test_load_definition_accepts_bom_via_authoritative_b1_loader(isolated_root: Path) -> None:
    definition = _definition("bgirs_aggregation_bom_v1")
    path = definition_store.definition_path(definition.aggregation_campaign_id)
    text = json.dumps(definition.model_dump(mode="json"), indent=2)
    path.write_text("\ufeff" + text, encoding="utf-8")

    assert definition_store.load_definition(definition.aggregation_campaign_id) == definition
