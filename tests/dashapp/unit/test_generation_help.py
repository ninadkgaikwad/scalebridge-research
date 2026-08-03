"""Generation contextual-help coverage tests."""
from scalebridge.dashapp.help.registry.generation_help import GENERATION_HELP_ENTRIES
REQUIRED=("generation.page.setup","generation.page.validation","generation.page.execution","generation.page.results","generation.page.diagnostics","generation.input.dataset_role","generation.input.visualization_campaign","generation.table.known_profiles","generation.table.validation","generation.table.discovered_campaigns","generation.plot.generated_signal")
def test_required_generation_help_entries_exist():
    for help_id in REQUIRED:
        assert help_id in GENERATION_HELP_ENTRIES
        entry=GENERATION_HELP_ENTRIES[help_id]
        assert entry.get("title") and entry.get("summary") and entry.get("details")
