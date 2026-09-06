from scalebridge.dashapp.help.registry.heat_input_help import HEAT_INPUT_HELP_ENTRIES


def test_results_help_covers_model_trajectory_metrics_validation_and_annual_sections():
    expected = {
        "heat_input.page.results",
        "heat_input.results.run",
        "heat_input.results.filters",
        "heat_input.results.dataset_trajectory",
        "heat_input.results.inventory",
        "heat_input.results.evaluation",
        "heat_input.results.phvac_modes",
        "heat_input.results.validation",
        "heat_input.results.annual",
        "heat_input.results.plot_download",
        "heat_input.results.artifact_downloads",
        "heat_input.results.summary_download",
        "heat_input.results.model_download",
    }
    assert expected.issubset(HEAT_INPUT_HELP_ENTRIES)
