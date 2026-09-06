from scalebridge.dashapp.pages.data_pipeline.phase_b_aggregation.page import build_layout


def test_aggregation_has_three_tabs_and_lazy_content():
    layout = build_layout()
    tabs = layout.children[1]
    assert [item.value for item in tabs.children] == [
        "campaign_builder",
        "execution",
        "results",
    ]
    assert layout.children[2].id == "aggregation-workspace-content"
