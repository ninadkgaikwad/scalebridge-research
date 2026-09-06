from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scalebridge.dashapp.services.heat_input import upstream_aggregation


@pytest.fixture()
def aggregation_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    generated = tmp_path / "ScaleBridge"
    campaign_root = generated / "campaigns" / "generation_parent_v1"

    modern_root = (
        campaign_root / "aggregation" / "matrix_runs" / "aggregation_matrix_modern"
    )
    legacy_root = (
        campaign_root / "aggregation" / "matrix_runs" / "aggregation_matrix_legacy"
    )
    modern_root.mkdir(parents=True)
    legacy_root.mkdir(parents=True)

    def write_matrix(root: Path, *, owner: str | None, aggregation_id: str):
        manifest = {
            "matrix_run_id": root.name,
            "parent_generation_campaign_id": "generation_parent_v1",
            "selected_plan_count": 2,
            "successful_plan_count": 2,
            "failed_plan_count": 0,
            "building_types": ["RestaurantFastFood"],
            "weather_locations": ["Buffalo"],
            "aggregation_ids": [aggregation_id],
            "weight_modes": ["equal"],
        }
        if owner:
            manifest["aggregation_campaign_id"] = owner
        (root / "aggregation_matrix_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        with (root / "aggregation_matrix_case_runs.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["case_id", "aggregation_id", "weight_mode", "status"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "case_id": "case_1",
                    "aggregation_id": aggregation_id,
                    "weight_mode": "equal",
                    "status": "completed",
                }
            )

    write_matrix(
        modern_root,
        owner="aggregation_parent_v1",
        aggregation_id="identity",
    )
    write_matrix(legacy_root, owner=None, aggregation_id="all_to_one")

    rows = [
        {
            "aggregation_campaign_id": "aggregation_parent_v1",
            "parent_generation_campaign_id": "generation_parent_v1",
            "matrix_run_id": "aggregation_matrix_modern",
            "status": "completed",
            "created_at_utc": "2026-08-20T12:00:00+00:00",
            "selected_plan_count": 2,
            "successful_plan_count": 2,
            "failed_plan_count": 0,
            "matrix_root": str(modern_root),
            "manifest_path": str(modern_root / "aggregation_matrix_manifest.json"),
        },
        {
            "aggregation_campaign_id": "legacy::generation_parent_v1",
            "parent_generation_campaign_id": "generation_parent_v1",
            "matrix_run_id": "aggregation_matrix_legacy",
            "status": "completed",
            "created_at_utc": "2026-07-15T12:00:00+00:00",
            "selected_plan_count": 2,
            "successful_plan_count": 2,
            "failed_plan_count": 0,
            "matrix_root": str(legacy_root),
            "manifest_path": str(legacy_root / "aggregation_matrix_manifest.json"),
        },
    ]
    monkeypatch.setattr(upstream_aggregation, "discover_all_matrix_runs", lambda: rows)
    monkeypatch.setattr(
        upstream_aggregation,
        "resolve_generated_data_root",
        lambda: generated,
    )

    definition = SimpleNamespace(
        parent_generation_campaign_id="generation_parent_v1",
        machine_id="laptop",
        parent_generation_campaign_root=None,
        generated_data_root=str(generated),
        requested_strategy_values=("identity",),
        requested_weight_mode_values=("equal",),
    )

    def load_definition(campaign_id):
        if campaign_id == "aggregation_parent_v1":
            return definition
        raise FileNotFoundError(campaign_id)

    monkeypatch.setattr(upstream_aggregation, "load_aggregation_definition", load_definition)
    return generated, campaign_root


def test_parent_options_include_modern_and_legacy_artifact_campaigns(aggregation_artifacts):
    options = upstream_aggregation.parent_aggregation_options()
    values = {row["value"] for row in options}
    assert values == {
        "aggregation_parent_v1",
        "legacy::generation_parent_v1",
    }
    legacy = next(row for row in options if row["value"].startswith("legacy::"))
    assert "legacy Aggregation outputs" in legacy["label"]


def test_modern_context_uses_definition_when_available(aggregation_artifacts):
    _generated, campaign_root = aggregation_artifacts
    context = upstream_aggregation.resolve_parent_context("aggregation_parent_v1")
    assert context["parent_generation_campaign_id"] == "generation_parent_v1"
    assert Path(context["campaign_root"]) == campaign_root
    assert context["definition_available"] is True
    assert context["legacy_artifact_only"] is False


def test_legacy_context_resolves_directly_from_matrix_artifacts(aggregation_artifacts):
    _generated, campaign_root = aggregation_artifacts
    context = upstream_aggregation.resolve_parent_context(
        "legacy::generation_parent_v1"
    )
    assert context["parent_generation_campaign_id"] == "generation_parent_v1"
    assert Path(context["campaign_root"]) == campaign_root
    assert context["definition_available"] is False
    assert context["legacy_artifact_only"] is True
    assert context["root_resolution_method"] == "phase_b_matrix_artifact_discovery"


def test_matrix_discovery_and_scope_facets_work_for_both_sources(aggregation_artifacts):
    modern = upstream_aggregation.discover_matrix_runs("aggregation_parent_v1")
    legacy = upstream_aggregation.discover_matrix_runs(
        "legacy::generation_parent_v1"
    )
    assert modern[0]["ownership_status"] == "scoped"
    assert legacy[0]["ownership_status"] == "legacy_unscoped"

    summary = upstream_aggregation.matrix_summary(
        "legacy::generation_parent_v1",
        "aggregation_matrix_legacy",
    )
    assert summary["case_ids"] == ["case_1"]
    assert summary["aggregation_ids"] == ["all_to_one"]
    assert summary["weight_modes"] == ["equal"]
    assert summary["readiness"] == "ready"



def test_normalized_scope_separates_strategy_custom_group_weight_and_rule():
    rows = [
        {
            "case_id": "case_office",
            "building_type": "OfficeSmall",
            "weather_location": "Seattle",
            "aggregation_id": "custom_v1",
            "plan_strategy": "custom_groups",
            "weight_mode": "equal",
            "rule_set": "legacy_v1",
        },
        {
            "case_id": "case_office",
            "building_type": "OfficeSmall",
            "weather_location": "Seattle",
            "aggregation_id": "custom_v1",
            "plan_strategy": "custom_groups",
            "weight_mode": "floor_area",
            "rule_set": "legacy_v1",
        },
        {
            "case_id": "case_office",
            "building_type": "OfficeSmall",
            "weather_location": "Seattle",
            "aggregation_id": "identity_legacy_v1_equal_v1",
            "plan_strategy": "identity",
            "weight_mode": "equal",
            "rule_set": "legacy_v1",
        },
    ]
    scope = [upstream_aggregation._normalize_scope_row(row) for row in rows]
    summary = {"scope_rows": scope, "case_options": upstream_aggregation._case_options(scope)}

    assert [row["value"] for row in upstream_aggregation.scope_options(summary, "strategy")] == [
        "custom_groups",
        "identity",
    ]
    assert [row["value"] for row in upstream_aggregation.scope_options(summary, "custom_grouping_id")] == [
        "custom_v1"
    ]
    assert [row["value"] for row in upstream_aggregation.scope_options(summary, "weight_mode")] == [
        "equal",
        "floor_area",
    ]
    assert [row["value"] for row in upstream_aggregation.scope_options(summary, "rule_set")] == [
        "legacy_v1"
    ]

    equal = upstream_aggregation.resolve_scope_selection(
        summary,
        strategy="custom_groups",
        custom_grouping_id="custom_v1",
        weight_mode="equal",
        rule_set="legacy_v1",
    )
    floor = upstream_aggregation.resolve_scope_selection(
        summary,
        strategy="custom_groups",
        custom_grouping_id="custom_v1",
        weight_mode="floor_area",
        rule_set="legacy_v1",
    )
    assert equal == {
        "case_id": None,
        "aggregation_id": "custom_v1",
        "weight_mode": "equal",
    }
    assert floor == {
        "case_id": None,
        "aggregation_id": "custom_v1",
        "weight_mode": "floor_area",
    }


def test_noncustom_strategy_requires_unambiguous_internal_plan_id():
    scope = [
        upstream_aggregation._normalize_scope_row(
            {
                "aggregation_id": "identity_legacy_v1_equal_v1",
                "plan_strategy": "identity",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
            }
        ),
        upstream_aggregation._normalize_scope_row(
            {
                "aggregation_id": "identity_legacy_v1_floor_area_v1",
                "plan_strategy": "identity",
                "weight_mode": "floor_area",
                "rule_set": "legacy_v1",
            }
        ),
    ]
    summary = {"scope_rows": scope}

    with pytest.raises(ValueError, match="multiple Phase B plan IDs"):
        upstream_aggregation.resolve_scope_selection(summary, strategy="identity")

    resolved = upstream_aggregation.resolve_scope_selection(
        summary,
        strategy="identity",
        weight_mode="floor_area",
    )
    assert resolved["aggregation_id"] == "identity_legacy_v1_floor_area_v1"
    assert resolved["weight_mode"] == "floor_area"


def test_case_options_use_building_and_weather_labels_not_only_opaque_id():
    scope = [
        upstream_aggregation._normalize_scope_row(
            {
                "case_id": "epcase_123",
                "building_type": "RestaurantFastFood",
                "weather_location": "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3",
                "aggregation_id": "custom_v1",
                "plan_strategy": "custom_groups",
                "weight_mode": "equal",
                "rule_set": "legacy_v1",
            }
        )
    ]
    options = upstream_aggregation._case_options(scope)
    assert options[0]["value"] == "epcase_123"
    assert "RestaurantFastFood" in options[0]["label"]
    assert "Seattle" in options[0]["label"]



def test_accepted_phase_b_six_plan_matrix_is_exposed_without_conflating_custom_id_and_weight():
    raw = []
    for strategy, aggregation_id, weight in (
        ("all_thermal_zones_to_one", "all_thermal_zones_to_one_legacy_v1_equal_v1", "equal"),
        ("all_thermal_zones_to_one", "all_thermal_zones_to_one_legacy_v1_floor_area_v1", "floor_area"),
        ("custom_groups", "custom_v1", "equal"),
        ("custom_groups", "custom_v1", "floor_area"),
        ("identity", "identity_legacy_v1_equal_v1", "equal"),
        ("identity", "identity_legacy_v1_floor_area_v1", "floor_area"),
    ):
        raw.append(
            {
                "case_id": "epcase_test",
                "building_type": "OfficeSmall",
                "weather_location": "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3",
                "aggregation_id": aggregation_id,
                "plan_strategy": strategy,
                "weight_mode": weight,
                "rule_set": "legacy_v1",
            }
        )

    scope = [upstream_aggregation._normalize_scope_row(row) for row in raw]
    summary = {"scope_rows": scope}
    assert upstream_aggregation._scope_values(scope, "strategy") == [
        "all_thermal_zones_to_one",
        "custom_groups",
        "identity",
    ]
    assert upstream_aggregation._scope_values(scope, "custom_grouping_id") == ["custom_v1"]
    assert upstream_aggregation._scope_values(scope, "weight_mode") == ["equal", "floor_area"]
    assert upstream_aggregation._scope_values(scope, "rule_set") == ["legacy_v1"]

    for weight in ("equal", "floor_area"):
        resolved = upstream_aggregation.resolve_scope_selection(
            summary,
            strategy="custom_groups",
            custom_grouping_id="custom_v1",
            weight_mode=weight,
            rule_set="legacy_v1",
        )
        assert resolved["aggregation_id"] == "custom_v1"
        assert resolved["weight_mode"] == weight
