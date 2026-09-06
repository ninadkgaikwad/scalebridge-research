from __future__ import annotations

from scalebridge.dashapp.pages.data_pipeline.phase_d_thermal_model_data.campaign_builder import page


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


def test_phase_d_builder_exposes_general_runner_science_options(monkeypatch):
    monkeypatch.setattr(page, "phase_c_run_options", lambda: [])
    ids = _ids(page.build_layout())
    required = {
        "phase-d-builder-campaign-id",
        "phase-d-builder-phase-c-run",
        "phase-d-builder-case-ids",
        "phase-d-builder-aggregation-ids",
        "phase-d-builder-weight-modes",
        "phase-d-builder-max-runs",
        "phase-d-builder-heat-representation",
        "phase-d-builder-qzivr-separate",
        "phase-d-builder-ml-policies",
        "phase-d-builder-ml-input-lags",
        "phase-d-builder-ml-target-horizons",
        "phase-d-builder-ml-train-fraction",
        "phase-d-builder-ml-test-fraction",
        "phase-d-builder-ml-validation-fraction",
        "phase-d-builder-ml-sh-train",
        "phase-d-builder-ml-sh-test",
        "phase-d-builder-ml-sh-validation",
        "phase-d-builder-ob-policies",
        "phase-d-builder-ml-fraction-options",
        "phase-d-builder-ml-seasonal-holdout-options",
        "phase-d-builder-ob-sd-options",
        "phase-d-builder-ob-sbh-options",
        "phase-d-builder-ob-ci-options",
        "phase-d-builder-ob-cdr-options",
        "phase-d-builder-sd-offset",
        "phase-d-builder-sd-train-days",
        "phase-d-builder-sd-test-days",
        "phase-d-builder-sbh-train",
        "phase-d-builder-sbh-test",
        "phase-d-builder-ci-start",
        "phase-d-builder-ci-train-days",
        "phase-d-builder-ci-test-days",
        "phase-d-builder-cdr-train",
        "phase-d-builder-cdr-test",
        "phase-d-builder-calendar-year",
        "phase-d-builder-parquet-compression",
        "phase-d-builder-output-root",
        "phase-d-builder-mlflow-enabled",
        "phase-d-builder-mlflow-experiment",
        "phase-d-builder-mlflow-run-name",
        "phase-d-builder-mlflow-strict",
        "phase-d-builder-preview",
        "phase-d-builder-save",
    }
    assert required.issubset(ids)


def test_phase_d_builder_does_not_expose_scientific_internal_stages(monkeypatch):
    monkeypatch.setattr(page, "phase_c_run_options", lambda: [])
    ids = _ids(page.build_layout())
    for forbidden in (
        "phase-d-builder-d2",
        "phase-d-builder-d3",
        "phase-d-builder-d4",
        "phase-d-builder-d5",
        "phase-d-builder-d6",
        "phase-d-builder-d7",
        "phase-d-builder-d8",
        "phase-d-builder-resample",
        "phase-d-builder-imputation",
    ):
        assert forbidden not in ids


def test_phase_d_builder_uses_checkboxes_not_switches(monkeypatch):
    monkeypatch.setattr(page, "phase_c_run_options", lambda: [])
    layout = page.build_layout()
    ids = _ids(layout)
    for component_id in (
        "phase-d-builder-qzivr-separate",
        "phase-d-builder-mlflow-enabled",
        "phase-d-builder-mlflow-strict",
        "phase-d-builder-replace",
    ):
        assert ids[component_id].__class__.__name__ == "Checkbox"
    assert not any(component.__class__.__name__ == "Switch" for component in _walk(layout))
