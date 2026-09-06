from scalebridge.dashapp.help.registry.phase_d_help import PHASE_D_HELP_ENTRIES


def test_phase_d_help_covers_builder_sections_and_key_runner_inputs():
    required = {
        "subpage.data_pipeline.phase_d_thermal_model_data",
        "phase_d.page.campaign_builder",
        "phase_d.page.execution",
        "phase_d.page.results",
        "phase_d.section.upstream",
        "phase_d.section.scope",
        "phase_d.section.ml",
        "phase_d.section.ob",
        "phase_d.input.phase_c_run",
        "phase_d.input.heat_representation",
        "phase_d.input.ml_policies",
        "phase_d.input.ml_target_horizons",
        "phase_d.input.ci",
        "phase_d.input.cdr",
        "phase_d.input.ob_policies",
        "phase_d.input.mlflow",
    }
    assert required.issubset(PHASE_D_HELP_ENTRIES)
    assert all(PHASE_D_HELP_ENTRIES[key]["summary"] for key in required)
    assert all(PHASE_D_HELP_ENTRIES[key]["details"] for key in required)
