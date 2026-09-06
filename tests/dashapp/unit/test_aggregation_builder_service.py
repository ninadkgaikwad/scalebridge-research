from pathlib import Path

import pytest

from scalebridge.dashapp.services.aggregation import builder


def test_plan_requests_are_strategy_weight_cross_product():
    requests = builder.build_plan_requests(
        strategies=["all_thermal_zones_to_one", "identity"],
        weight_modes=["equal", "volume"],
        rule_set="legacy_v1",
    )
    assert [
        (item.strategy.value, item.weight_mode.value, item.rule_set.value)
        for item in requests
    ] == [
        ("all_thermal_zones_to_one", "equal", "legacy_v1"),
        ("all_thermal_zones_to_one", "volume", "legacy_v1"),
        ("identity", "equal", "legacy_v1"),
        ("identity", "volume", "legacy_v1"),
    ]


def test_custom_strategy_carries_requested_custom_aggregation_id():
    requests = builder.build_plan_requests(
        strategies=["custom_groups"],
        weight_modes=["floor_area"],
        rule_set="legacy_v1",
        custom_aggregation_id="custom_floorplan_v1",
    )
    assert requests[0].custom_aggregation_ids == ("custom_floorplan_v1",)


def test_custom_group_rows_use_existing_scientific_partition_validation():
    selected = [
        {
            "case_id": "case_a",
            "zone_inventory_status": "available",
            "thermal_zone_names": ["DINING", "KITCHEN"],
        }
    ]
    rows = [
        {
            "case_id": "case_a",
            "source_zone_name": "DINING",
            "aggregate_zone_name": "Front",
        },
        {
            "case_id": "case_a",
            "source_zone_name": "KITCHEN",
            "aggregate_zone_name": "Back",
        },
    ]
    validated = builder.validate_custom_group_rows(
        rows=rows,
        selected_case_rows=selected,
        aggregation_id="custom_v1",
    )
    assert validated == [
        {
            "case_id": "case_a",
            "aggregation_id": "custom_v1",
            "source_zone_name": "DINING",
            "aggregate_zone_name": "Front",
        },
        {
            "case_id": "case_a",
            "aggregation_id": "custom_v1",
            "source_zone_name": "KITCHEN",
            "aggregate_zone_name": "Back",
        },
    ]


def test_custom_group_rows_reject_incomplete_partition():
    selected = [
        {
            "case_id": "case_a",
            "zone_inventory_status": "available",
            "thermal_zone_names": ["DINING", "KITCHEN"],
        }
    ]
    with pytest.raises(ValueError, match="Invalid custom grouping"):
        builder.validate_custom_group_rows(
            rows=[
                {
                    "case_id": "case_a",
                    "source_zone_name": "DINING",
                    "aggregate_zone_name": "All",
                }
            ],
            selected_case_rows=selected,
            aggregation_id="custom_v1",
        )


def test_builder_definition_is_authoritative_b1_model():
    requests = builder.build_plan_requests(
        strategies=["identity"],
        weight_modes=["equal"],
        rule_set="legacy_v1",
    )
    definition = builder.build_definition(
        aggregation_campaign_id="agg_builder_v1",
        parent_generation_campaign_id="generation_parent_v1",
        machine_id="laptop",
        case_ids=["case_b", "case_a"],
        plan_requests=requests,
        custom_zone_groups_path=None,
    )
    assert definition.aggregation_campaign_id == "agg_builder_v1"
    assert definition.parent_generation_campaign_id == "generation_parent_v1"
    assert definition.case_ids == ("case_b", "case_a")
    assert definition.plan_requests == requests


def test_relative_custom_group_path_is_definition_relative():
    assert builder.relative_custom_grouping_path("agg_builder_v1") == str(
        Path("custom_groups") / "agg_builder_v1.csv"
    )
