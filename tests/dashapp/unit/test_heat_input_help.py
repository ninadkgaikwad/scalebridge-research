from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig
from scalebridge.dashapp.help.registry.heat_input_help import HEAT_INPUT_HELP_ENTRIES


def test_every_public_runner_field_has_contextual_help():
    for field in PhaseCCampaignConfig.capability_manifest()["fields"]:
        help_id = f"heat_input.field.{field['name']}"
        assert help_id in HEAT_INPUT_HELP_ENTRIES
        assert HEAT_INPUT_HELP_ENTRIES[help_id]["summary"]


def test_phase_c_workspace_and_simplified_builder_sections_have_help():
    required = {
        "subpage.data_pipeline.phase_c_heat_input",
        "heat_input.page.campaign_builder",
        "heat_input.page.execution",
        "heat_input.page.results",
        "heat_input.section.identity",
        "heat_input.section.upstream",
        "heat_input.section.scope",
        "heat_input.section.scientific",
        "heat_input.section.preview_save",
    }
    assert required.issubset(HEAT_INPUT_HELP_ENTRIES)
