"""Tests for deterministic, cross-machine EnergyPlus case identity."""

from __future__ import annotations

from pathlib import Path

from scalebridge.integration.energyplus import CaseSpec, OutputVariableRequest


def test_case_id_ignores_paths_tags_and_compatibility_settings(case_spec: CaseSpec) -> None:
    """Execution location and storage preferences must not alter case identity."""
    changed = case_spec.model_copy(
        update={
            "idf_path": Path(r"D:\different\model.idf"),
            "epw_path": Path("/another/weather.epw"),
            "tags": {"machine": "kamiak"},
            "write_legacy_pickles": False,
            "preserve_raw_outputs": False,
        }
    )

    assert changed.case_id == case_spec.case_id


def test_case_id_ignores_output_request_order(case_spec: CaseSpec) -> None:
    """Equivalent output request sets must produce the same identifier."""
    changed = case_spec.model_copy(update={"output_variables": case_spec.output_variables[::-1]})

    assert changed.case_id == case_spec.case_id


def test_case_id_ignores_output_request_annotations(case_spec: CaseSpec) -> None:
    """Processing annotations must not change EnergyPlus simulation identity."""
    changed_requests = tuple(
        request.model_copy(
            update={
                "required": not request.required,
                "semantic_role": f"changed_{index}",
            }
        )
        for index, request in enumerate(case_spec.output_variables)
    )
    changed = case_spec.model_copy(update={"output_variables": changed_requests})

    assert changed.case_id == case_spec.case_id


def test_case_id_changes_with_scientific_configuration(case_spec: CaseSpec) -> None:
    """Adding an EnergyPlus output request must produce a different identifier."""
    changed = case_spec.model_copy(
        update={
            "output_variables": case_spec.output_variables
            + (OutputVariableRequest(variable_name="Zone People Convective Heating Rate"),)
        }
    )

    assert changed.case_id != case_spec.case_id
