"""Tests for invalid EnergyPlus manifest states and unsafe artifact paths."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from scalebridge.integration.energyplus import (
    ArtifactRecord,
    CaseSpec,
    ExecutionMetadata,
    OutputVariableRequest,
    RunManifest,
    RunPeriod,
    RunStatus,
    SoftwareMetadata,
)


def test_timestep_must_divide_hour() -> None:
    """EnergyPlus timestep minutes must represent a whole number per hour."""
    with pytest.raises(ValidationError, match="divide evenly into 60"):
        CaseSpec(
            case_name="invalid",
            idf_path="model.idf",
            epw_path="weather.epw",
            idf_sha256="a" * 64,
            epw_sha256="b" * 64,
            run_period=RunPeriod(
                start_month=1,
                start_day=1,
                end_month=1,
                end_day=2,
            ),
            timestep_minutes=7,
            output_variables=(OutputVariableRequest(variable_name="Zone Air Temperature"),),
        )


def test_artifact_path_must_be_relative() -> None:
    """POSIX absolute paths must not be persisted as portable artifact paths."""
    with pytest.raises(ValidationError, match="relative"):
        ArtifactRecord(
            role="raw_csv",
            relative_path="/tmp/eplusout.csv",
            media_type="text/csv",
        )


def test_windows_artifact_path_must_be_relative_on_all_platforms() -> None:
    """Windows absolute paths must be rejected even on non-Windows systems."""
    with pytest.raises(ValidationError, match="relative"):
        ArtifactRecord(
            role="raw_csv",
            relative_path=r"D:\runs\eplusout.csv",
            media_type="text/csv",
        )


def test_failed_manifest_requires_error(case_spec: CaseSpec) -> None:
    """Failed runs must preserve actionable error information."""
    started_at = datetime(2026, 6, 23, tzinfo=timezone.utc)

    with pytest.raises(ValidationError, match="require an error record"):
        RunManifest(
            case_id=case_spec.case_id,
            run_id="eprun_failed",
            status=RunStatus.FAILED,
            case_spec=case_spec,
            execution=ExecutionMetadata(
                machine_id="lab_pc",
                hostname="LAB-PC",
                platform="windows",
                started_at=started_at,
            ),
            software=SoftwareMetadata(scalebridge_version="0.1.0"),
        )


def test_manifest_rejects_mismatched_case_id(case_spec: CaseSpec) -> None:
    """A manifest must not claim an identity inconsistent with its CaseSpec."""
    started_at = datetime(2026, 6, 23, tzinfo=timezone.utc)

    with pytest.raises(ValidationError, match="does not match"):
        RunManifest(
            case_id="epcase_wrong",
            run_id="eprun_test",
            status=RunStatus.PREPARED,
            case_spec=case_spec,
            execution=ExecutionMetadata(
                machine_id="dev_laptop",
                hostname="NKG-WIN-LAPTOP",
                platform="windows",
                started_at=started_at,
            ),
            software=SoftwareMetadata(scalebridge_version="0.1.0"),
        )
