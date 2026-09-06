from __future__ import annotations

from scalebridge.dashapp.pages.data_pipeline.phase_b_aggregation.campaign_builder.page import (
    build_layout,
)


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _components_by_string_id(layout):
    return {
        component.id: component
        for component in _walk(layout)
        if isinstance(getattr(component, "id", None), str)
    }


def _option_values(component):
    values = []
    for option in getattr(component, "options", []) or []:
        if isinstance(option, dict):
            values.append(option.get("value"))
        else:
            values.append(option)
    return values


def test_builder_starts_with_parent_generation_campaign_selector():
    ids = _components_by_string_id(build_layout())

    parent = ids["aggregation-builder-parent-campaign"]
    assert getattr(parent, "placeholder", None) == "Select a Generation campaign"


def test_builder_exposes_generation_filters_and_case_selection():
    ids = _components_by_string_id(build_layout())

    assert "aggregation-builder-building-filter" in ids
    assert "aggregation-builder-weather-filter" in ids
    assert "aggregation-builder-climate-filter" not in ids
    assert "aggregation-builder-cases" in ids


def test_builder_exposes_all_authoritative_strategy_and_weight_values():
    ids = _components_by_string_id(build_layout())

    assert set(_option_values(ids["aggregation-builder-strategies"])) == {
        "all_thermal_zones_to_one",
        "custom_groups",
        "identity",
    }
    assert set(_option_values(ids["aggregation-builder-weight-modes"])) == {
        "equal",
        "floor_area",
        "volume",
    }
    assert set(_option_values(ids["aggregation-builder-rule-set"])) == {
        "legacy_v1",
    }


def test_custom_grouping_ui_is_present_but_hidden_by_default():
    ids = _components_by_string_id(build_layout())

    panel = ids["aggregation-builder-custom-panel"]
    assert getattr(panel, "style", {}).get("display") == "none"

    table = ids["aggregation-builder-custom-table"]
    assert [column["id"] for column in table.columns] == [
        "case_id",
        "source_zone_name",
        "aggregate_zone_name",
    ]


def test_builder_has_campaign_id_and_save_definition_controls():
    ids = _components_by_string_id(build_layout())

    assert "aggregation-builder-campaign-id" in ids
    assert "aggregation-builder-save" in ids
    assert "aggregation-builder-save-status" in ids


def test_builder_has_refresh_and_upstream_status_controls():
    ids = _components_by_string_id(build_layout())

    assert "aggregation-builder-refresh-campaigns" in ids
    assert "aggregation-builder-parent-status" in ids


def test_custom_grouping_only_allows_aggregate_zone_editing():
    ids = _components_by_string_id(build_layout())
    table = ids["aggregation-builder-custom-table"]
    editable = {column["id"]: column.get("editable") for column in table.columns}
    assert editable == {
        "case_id": False,
        "source_zone_name": False,
        "aggregate_zone_name": True,
    }
    assert "aggregation-builder-custom-status" in ids


def test_builder_save_supports_custom_validity_disable_state():
    ids = _components_by_string_id(build_layout())
    assert "aggregation-builder-save" in ids
    # Default strategy is non-custom, so the button starts usable; callback
    # disables it only while a selected custom grouping is incomplete.
    assert getattr(ids["aggregation-builder-save"], "disabled", False) is False


def test_builder_save_alert_uses_wrapping_path_contract():
    from pathlib import Path

    callbacks = (
        Path(__file__).parents[3]
        / "src"
        / "scalebridge"
        / "dashapp"
        / "pages"
        / "data_pipeline"
        / "phase_b_aggregation"
        / "callbacks.py"
    ).read_text(encoding="utf-8")

    assert 'className="aggregation-wrap-alert"' in callbacks
    assert 'className="aggregation-path-text"' in callbacks
    assert 'html.Strong("Saved Aggregation definition:")' in callbacks
    assert 'html.Strong("Custom grouping:")' in callbacks
