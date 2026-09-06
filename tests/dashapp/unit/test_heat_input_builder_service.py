from __future__ import annotations

import pytest

from scalebridge.dashapp.services.heat_input import builder


def test_builder_represents_every_public_runner_field():
    fields = builder.runner_fields()
    direct = builder.editable_runner_fields()

    assert len(fields) == 81
    assert builder.LINEAGE_MANAGED_FIELDS == {
        "campaign_root",
        "campaign_id",
        "generated_data_root",
        "matrix_run_id",
    }
    assert len(direct) == 77
    assert len(direct) + len(builder.LINEAGE_MANAGED_FIELDS) == len(fields)


def test_visibility_metadata_partitions_all_direct_fields():
    grouped_names = []
    for visibility in ("basic", "advanced", "expert"):
        for _group, fields in builder.grouped_runner_fields(visibility):
            grouped_names.extend(field["name"] for field in fields)

    assert len(grouped_names) == 77
    assert len(set(grouped_names)) == 77


def test_model_registry_remains_authoritative_and_read_only():
    rows = builder.model_registry_rows()
    assert len(rows) == 19
    assert rows[0]["model_id"] == "QSol1"
    assert rows[-2]["model_id"] == "QAC"
    assert rows[-1]["model_id"] == "PHVAC"
    assert rows[-1]["dependency_model_id"] == "QAC"
    assert rows[-1]["fit_intercept"] is True


def test_control_spec_handles_tri_state_and_repeatable_fields():
    fields = {field["name"]: field for field in builder.runner_fields()}
    assert builder.control_spec(fields["fit_intercept_override"])["kind"] == "tri_bool"
    assert builder.control_spec(fields["pytorch_devices"])["kind"] == "multi_choice"
    assert builder.control_spec(fields["model_ids"])["kind"] == "model_ids"
    assert builder.control_spec(fields["downstream_aggregate_zone_ids"])["kind"] == "list_text"


def test_collect_config_values_normalizes_lists_and_inherit():
    ids = [
        {"type": "phase-c-config-field", "field": "model_ids"},
        {"type": "phase-c-config-field", "field": "downstream_aggregate_zone_ids"},
        {"type": "phase-c-config-field", "field": "fit_intercept_override"},
    ]
    values = [["QAC", "PHVAC"], "Dining, Kitchen", "inherit"]
    normalized = builder.collect_config_values(ids, values)

    assert normalized["model_ids"] == ["QAC", "PHVAC"]
    assert normalized["downstream_aggregate_zone_ids"] == ["Dining", "Kitchen"]
    assert normalized["fit_intercept_override"] is None


def test_build_definition_injects_parent_lineage(monkeypatch):
    monkeypatch.setattr(
        builder,
        "resolve_parent_context",
        lambda _campaign_id: {
            "parent_generation_campaign_id": "generation_parent_v1",
            "campaign_root": "C:/data/ScaleBridge/campaigns/generation_parent_v1",
            "generated_data_root": "C:/data/ScaleBridge",
        },
    )
    monkeypatch.setattr(
        builder,
        "validate_matrix_selection",
        lambda _campaign_id, matrix_run_id: {"matrix_run_id": matrix_run_id},
    )

    definition = builder.build_definition(
        phase_c_campaign_id="phase_c_builder_v1",
        parent_aggregation_campaign_id="aggregation_parent_v1",
        matrix_run_id="aggregation_matrix_123",
        machine_id="laptop",
        config_values={
            "split_strategy": "chronological_fraction",
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "estimator_types": ["closed_form_linear", "pytorch_linear"],
            "pytorch_devices": ["auto"],
        },
    )

    config = definition.runner_config
    assert definition.parent_generation_campaign_id == "generation_parent_v1"
    assert config.campaign_id == "generation_parent_v1"
    assert config.matrix_run_id == "aggregation_matrix_123"
    assert config.split_strategy == "chronological_fraction"
    assert config.estimator_types == (
        "closed_form_linear",
        "pytorch_linear",
    )


def test_build_definition_rejects_unknown_model_id(monkeypatch):
    monkeypatch.setattr(
        builder,
        "resolve_parent_context",
        lambda _campaign_id: {
            "parent_generation_campaign_id": "generation_parent_v1",
            "campaign_root": "C:/data/ScaleBridge/campaigns/generation_parent_v1",
            "generated_data_root": "C:/data/ScaleBridge",
        },
    )
    monkeypatch.setattr(
        builder,
        "validate_matrix_selection",
        lambda *_args: {},
    )

    with pytest.raises(ValueError, match="Unsupported model_ids"):
        builder.build_definition(
            phase_c_campaign_id="phase_c_builder_v1",
            parent_aggregation_campaign_id="aggregation_parent_v1",
            matrix_run_id="aggregation_matrix_123",
            machine_id="laptop",
            config_values={"model_ids": ["NOT_A_MODEL"]},
        )


def test_curated_builder_fields_are_small_and_scientifically_meaningful():
    assert builder.CURATED_CONFIG_FIELDS == {
        "case_id",
        "aggregation_id",
        "weight_mode",
        "model_ids",
        "internal_gain_predictor_method",
        "hvac_target_method",
        "split_strategy",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "estimator_types",
        "pytorch_devices",
        "validation_profile",
        "mlflow_enabled",
    }
    assert "start_stage" not in builder.CURATED_CONFIG_FIELDS
    assert "stop_stage" not in builder.CURATED_CONFIG_FIELDS
    assert "overwrite_existing" not in builder.CURATED_CONFIG_FIELDS


def test_builder_forces_complete_non_overwriting_phase_c(monkeypatch):
    monkeypatch.setattr(
        builder,
        "resolve_parent_context",
        lambda _campaign_id: {
            "parent_generation_campaign_id": "generation_parent_v1",
            "campaign_root": "C:/data/ScaleBridge/campaigns/generation_parent_v1",
            "generated_data_root": "C:/data/ScaleBridge",
        },
    )
    monkeypatch.setattr(builder, "validate_matrix_selection", lambda *_args: {})

    definition = builder.build_definition(
        phase_c_campaign_id="phase_c_complete_v1",
        parent_aggregation_campaign_id="aggregation_parent_v1",
        matrix_run_id="aggregation_matrix_123",
        machine_id="laptop",
        config_values={"validation_profile": "none", "mlflow_enabled": False},
    )
    config = definition.runner_config
    assert config.start_stage == "C1"
    assert config.stop_stage == "C9"
    assert config.continue_on_error is False
    assert config.overwrite_existing is False
    assert config.c1_aggregation_run_root is None


def test_builder_rejects_non_curated_runner_field(monkeypatch):
    monkeypatch.setattr(
        builder,
        "resolve_parent_context",
        lambda _campaign_id: {
            "parent_generation_campaign_id": "generation_parent_v1",
            "campaign_root": "C:/data/ScaleBridge/campaigns/generation_parent_v1",
            "generated_data_root": "C:/data/ScaleBridge",
        },
    )
    monkeypatch.setattr(builder, "validate_matrix_selection", lambda *_args: {})

    with pytest.raises(ValueError, match="Unsupported Campaign Builder field"):
        builder.build_definition(
            phase_c_campaign_id="phase_c_reject_v1",
            parent_aggregation_campaign_id="aggregation_parent_v1",
            matrix_run_id="aggregation_matrix_123",
            machine_id="laptop",
            config_values={"start_stage": "C4"},
        )
