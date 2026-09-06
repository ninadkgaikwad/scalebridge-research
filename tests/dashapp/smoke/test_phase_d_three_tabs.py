from scalebridge.dashapp.pages.data_pipeline.phase_d_thermal_model_data.page import build_layout


def test_phase_d_has_exactly_three_tabs_and_lazy_content(monkeypatch):
    # Avoid filesystem discovery while building initial Campaign Builder layout.
    from scalebridge.dashapp.pages.data_pipeline.phase_d_thermal_model_data.campaign_builder import page
    monkeypatch.setattr(page, "phase_c_run_options", lambda: [])
    layout = build_layout()
    tabs = layout.children[1]
    assert [item.value for item in tabs.children] == ["campaign_builder", "execution", "results"]
    assert [item.label for item in tabs.children] == ["Campaign Builder", "Execution", "Results"]
    assert layout.children[2].id == "phase-d-workspace-content"
