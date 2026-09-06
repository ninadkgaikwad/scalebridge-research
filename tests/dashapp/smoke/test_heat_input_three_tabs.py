from scalebridge.dashapp.pages.data_pipeline.phase_c_heat_input.page import build_layout


def test_heat_input_has_exactly_three_tabs_and_lazy_content():
    layout = build_layout()
    tabs = layout.children[1]
    assert [item.value for item in tabs.children] == [
        "campaign_builder",
        "execution",
        "results",
    ]
    assert [item.label for item in tabs.children] == [
        "Campaign Builder",
        "Execution",
        "Results",
    ]
    assert layout.children[2].id == "phase-c-workspace-content"
