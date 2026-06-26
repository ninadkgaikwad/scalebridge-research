"""Tests for lossless and validated manifest JSON persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from scalebridge.integration.energyplus import (
    ArtifactRecord,
    CaseSpec,
    ExecutionMetadata,
    RunManifest,
    RunStatus,
    SoftwareMetadata,
    ValidationSummary,
    load_case_spec,
    load_run_manifest,
    write_case_spec,
    write_run_manifest,
)


def test_case_spec_json_round_trip(tmp_path, case_spec: CaseSpec) -> None:
    """A persisted case specification must survive a lossless round trip."""
    path = write_case_spec(case_spec, tmp_path / "case.json")

    assert load_case_spec(path) == case_spec


def test_run_manifest_json_round_trip(tmp_path, case_spec: CaseSpec) -> None:
    """A complete run manifest must survive a lossless round trip."""
    started_at = datetime(2026, 6, 23, 18, 0, tzinfo=timezone.utc)
    manifest = RunManifest(
        case_id=case_spec.case_id,
        run_id="eprun_test",
        campaign_id="p1_generation_2013",
        status=RunStatus.COMPLETED_WITH_WARNINGS,
        case_spec=case_spec,
        execution=ExecutionMetadata(
            machine_id="dev_laptop",
            hostname="NKG-WIN-LAPTOP",
            platform="windows",
            started_at=started_at,
            completed_at=started_at,
            runtime_seconds=0,
        ),
        software=SoftwareMetadata(
            scalebridge_version="0.1.0",
            energyplus_version="9.6.0",
            python_version="3.10",
        ),
        validation=ValidationSummary(
            exit_code=0,
            warnings=1,
            requested_signals=2,
            produced_signals=2,
            timestep_count=105120,
        ),
        artifacts=(
            ArtifactRecord(
                role="canonical_timeseries",
                relative_path="canonical/timeseries.parquet",
                media_type="application/vnd.apache.parquet",
            ),
        ),
    )

    path = write_run_manifest(manifest, tmp_path / "manifest.json")

    assert load_run_manifest(path) == manifest
