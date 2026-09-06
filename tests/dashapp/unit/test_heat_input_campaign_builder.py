from __future__ import annotations

from scalebridge.dashapp.pages.data_pipeline.phase_c_heat_input.campaign_builder import page


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


def _ids(layout):
    return {
        component.id: component
        for component in _walk(layout)
        if isinstance(getattr(component, "id", None), str)
    }


def test_builder_exposes_curated_campaign_level_controls(monkeypatch):
    monkeypatch.setattr(page, "parent_aggregation_options", lambda: [])
    ids = _ids(page.build_layout())
    required = {
        "phase-c-builder-campaign-id",
        "phase-c-builder-display-name",
        "phase-c-builder-machine-id",
        "phase-c-builder-parent-aggregation",
        "phase-c-builder-refresh-parents",
        "phase-c-builder-matrix-run",
        "phase-c-builder-case",
        "phase-c-builder-strategy",
        "phase-c-builder-custom-grouping",
        "phase-c-builder-weight",
        "phase-c-builder-rule-set",
        "phase-c-builder-model-ids",
        "phase-c-builder-internal-gain-method",
        "phase-c-builder-hvac-target-method",
        "phase-c-builder-split-strategy",
        "phase-c-builder-train-fraction",
        "phase-c-builder-validation-fraction",
        "phase-c-builder-test-fraction",
        "phase-c-builder-estimators",
        "phase-c-builder-devices",
        "phase-c-builder-validation-enabled",
        "phase-c-builder-mlflow-enabled",
        "phase-c-builder-preview",
        "phase-c-builder-save",
        "phase-c-builder-replace",
    }
    assert required.issubset(ids)


def test_builder_does_not_render_the_81_field_runner_form(monkeypatch):
    monkeypatch.setattr(page, "parent_aggregation_options", lambda: [])
    layout = page.build_layout()
    pattern_fields = [
        component
        for component in _walk(layout)
        if isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == "phase-c-config-field"
    ]
    assert pattern_fields == []

    ids = _ids(layout)
    assert "phase-c-builder-start-stage" not in ids
    assert "phase-c-builder-stop-stage" not in ids
    assert "phase-c-builder-overwrite" not in ids
    assert "phase-c-builder-model-registry" not in ids


def test_builder_uses_checkboxes_for_boolean_choices(monkeypatch):
    monkeypatch.setattr(page, "parent_aggregation_options", lambda: [])
    layout = page.build_layout()
    ids = _ids(layout)
    for component_id in (
        "phase-c-builder-validation-enabled",
        "phase-c-builder-mlflow-enabled",
        "phase-c-builder-replace",
    ):
        assert ids[component_id].__class__.__name__ == "Checkbox"
    assert not any(component.__class__.__name__ == "Switch" for component in _walk(layout))


def test_builder_model_selector_contains_all_19_relationships(monkeypatch):
    monkeypatch.setattr(page, "parent_aggregation_options", lambda: [])
    ids = _ids(page.build_layout())
    values = [row["value"] for row in ids["phase-c-builder-model-ids"].options]
    assert len(values) == 19
    assert values[-2:] == ["QAC", "PHVAC"]


def test_builder_separates_phase_b_lineage_dimensions(monkeypatch):
    monkeypatch.setattr(page, "parent_aggregation_options", lambda: [])
    ids = _ids(page.build_layout())
    assert "phase-c-builder-aggregation" not in ids
    assert ids["phase-c-builder-custom-grouping"].disabled is True
    assert ids["phase-c-builder-strategy"].placeholder == "All aggregation strategies"
    assert ids["phase-c-builder-weight"].placeholder == "All weight modes"
    assert ids["phase-c-builder-rule-set"].placeholder == "All rule sets"
