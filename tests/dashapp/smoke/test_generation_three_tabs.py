from scalebridge.dashapp.pages.data_pipeline.phase_a_generation.page import build_layout
def test_generation_has_three_tabs_and_lazy_content():
 layout=build_layout(); tabs=layout.children[1]
 assert [x.value for x in tabs.children]==['campaign_builder','execution','results']
 assert layout.children[2].id=='generation-workspace-content'
