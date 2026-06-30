"""Orchestrate one complete, isolated EnergyPlus data-generation attempt.

This module composes the existing ScaleBridge boundaries without embedding
their implementation details:

1. persist immutable case intent;
2. prepare a new IDF;
3. run EnergyPlus through opyplus;
4. validate and extract canonical outputs;
5. optionally write legacy compatibility pickles;
6. inventory generated artifacts;
7. persist an authoritative run manifest; and
8. return a compact ``GenerationResult`` suitable for local, multiprocessing,
   Ray, SLURM, or other loop-based execution.

Expected per-case failures are contained and returned as failed or invalid
results. This allows a campaign worker to continue processing later cases.
Every attempt receives a unique run directory, so retries never mix outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from scalebridge.integration.energyplus.idf import IdfPreparer
from scalebridge.integration.energyplus.manifests.models import (
    ArtifactRecord,
    CaseSpec,
    ErrorRecord,
    ExecutionMetadata,
    GenerationResult,
    RunManifest,
    RunStatus,
    SoftwareMetadata,
    TrackingMetadata,
    ValidationSummary,
)
from scalebridge.integration.energyplus.manifests.serialization import (
    write_case_spec,
    write_run_manifest,
)
from scalebridge.integration.energyplus.outputs import (
    CanonicalExtractionError,
    EnergyPlusOutputExtractor,
)
from scalebridge.integration.energyplus.prototypes import (
    resolve_generated_data_root,
)
from scalebridge.integration.energyplus.simulation import EnergyPlusRunner


class EnergyPlusGenerationOrchestrator:
    """Execute complete generation attempts with isolated artifacts.

    Parameters
    ----------
    generated_data_root:
        Optional ScaleBridge generated-data root override.
    case_collection_name:
        Directory grouping beneath the generated-data root.
    machine_id:
        Stable machine label. Defaults to ``SCALEBRIDGE_MACHINE_ID`` and then
        the local hostname.
    idf_preparer, runner, output_extractor:
        Optional injected services used for tests or alternate execution
        policies while retaining the same orchestration contract.
    """

    def __init__(
        self,
        *,
        generated_data_root: str | Path | None = None,
        case_collection_name: str = "generation",
        machine_id: str | None = None,
        idf_preparer: Any | None = None,
        runner: Any | None = None,
        output_extractor: Any | None = None,
        mlflow_tracker: Any | None = None,
    ) -> None:
        self._generated_root = resolve_generated_data_root(generated_data_root)
        self._collection_name = case_collection_name
        self._machine_id = (
            machine_id
            or os.environ.get("SCALEBRIDGE_MACHINE_ID")
            or socket.gethostname()
        )
        self._idf_preparer = idf_preparer or IdfPreparer()
        self._runner = runner or EnergyPlusRunner(beat_frequency_seconds=5)
        self._output_extractor = output_extractor or EnergyPlusOutputExtractor()
        if mlflow_tracker is None:
            from scalebridge.tracking.mlflow import MLflowGenerationTracker

            mlflow_tracker = MLflowGenerationTracker()
        self._mlflow_tracker = mlflow_tracker

    def generate(
        self,
        case_spec: CaseSpec,
        *,
        campaign_id: str | None = None,
        run_id: str | None = None,
        case_root: str | Path | None = None,
        tracking: TrackingMetadata | None = None,
    ) -> GenerationResult:
        """Execute one case and always return a persisted attempt result."""
        started_at = datetime.now(timezone.utc)
        started_counter = time.perf_counter()
        attempt_id = run_id or f"eprun_{uuid4().hex[:12]}"
        root = (
            Path(case_root).expanduser().resolve()
            if case_root is not None
            else self._generated_root
            / self._collection_name
            / "cases"
            / case_spec.case_id
        )
        run_root = root / "runs" / attempt_id
        inputs_root = run_root / "inputs"
        raw_root = run_root / "raw"
        canonical_root = run_root / "canonical"
        legacy_root = run_root / "legacy"
        manifest_path = run_root / "run_manifest.json"
        case_spec_path = inputs_root / "case_spec.json"
        prepared_idf_path = inputs_root / "prepared.idf"
        traceback_path = run_root / "traceback.txt"

        inputs_root.mkdir(parents=True, exist_ok=False)
        raw_root.mkdir(parents=True, exist_ok=False)
        write_case_spec(case_spec, case_spec_path)

        run_result: Any | None = None
        extraction: Any | None = None
        status = RunStatus.FAILED
        error: ErrorRecord | None = None
        tracking_metadata = tracking or TrackingMetadata()
        tracking_handle = None

        if tracking is None and self._mlflow_tracker is not None:
            try:
                tracking_handle = self._mlflow_tracker.start(
                    case_spec=case_spec,
                    run_id=attempt_id,
                    campaign_id=campaign_id,
                    machine_id=self._machine_id,
                )
                if tracking_handle is not None:
                    tracking_metadata = tracking_handle.to_metadata()
            except Exception:
                tracking_handle = None

        try:
            preparation = self._idf_preparer.prepare(
                case_spec,
                prepared_idf_path,
            )
            run_result = self._runner.run(
                idf_path=preparation.prepared_idf_path,
                epw_path=case_spec.epw_path,
                output_directory=raw_root,
            )

            if not run_result.completed_successfully:
                raise RuntimeError(
                    run_result.failure_message
                    or "EnergyPlus simulation did not complete successfully"
                )

            try:
                extraction = self._output_extractor.extract(
                    case_spec=case_spec,
                    simulation_directory=run_result.output_directory,
                    canonical_directory=canonical_root,
                    legacy_directory=legacy_root,
                )
            except CanonicalExtractionError:
                status = RunStatus.INVALID
                raise

            status = (
                RunStatus.COMPLETED_WITH_WARNINGS
                if run_result.warning_count
                else RunStatus.COMPLETED
            )
        except Exception as exc:
            if status is not RunStatus.INVALID:
                status = RunStatus.FAILED
            traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
            error = ErrorRecord(
                error_type=type(exc).__name__,
                message=str(exc),
                traceback_path=traceback_path.relative_to(run_root),
            )

        completed_at = datetime.now(timezone.utc)
        runtime_seconds = time.perf_counter() - started_counter
        artifacts = _inventory_artifacts(run_root, manifest_path=manifest_path)
        validation = _build_validation(case_spec, run_result, extraction)
        manifest = RunManifest(
            case_id=case_spec.case_id,
            run_id=attempt_id,
            campaign_id=campaign_id,
            status=status,
            case_spec=case_spec,
            execution=ExecutionMetadata(
                machine_id=self._machine_id,
                hostname=socket.gethostname(),
                platform=platform.platform(),
                worker_id=os.environ.get("SCALEBRIDGE_WORKER_ID"),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                started_at=started_at,
                completed_at=completed_at,
                runtime_seconds=runtime_seconds,
            ),
            software=_software_metadata(case_spec),
            validation=validation,
            artifacts=artifacts,
            tracking=tracking_metadata,
            error=error,
        )
        write_run_manifest(manifest, manifest_path)
        _write_latest_pointer(
            root=root,
            case_id=case_spec.case_id,
            run_id=attempt_id,
            status=status,
            manifest_path=manifest_path,
        )

        raw_paths = _relative_role_paths(run_root, raw_root)
        canonical_paths = _relative_role_paths(run_root, canonical_root)
        compatibility_paths = _relative_role_paths(run_root, legacy_root)
        result = GenerationResult(
            case_id=case_spec.case_id,
            run_id=attempt_id,
            status=status,
            artifact_root=run_root,
            manifest_path=manifest_path,
            raw_output_paths=raw_paths,
            canonical_output_paths=canonical_paths,
            compatibility_output_paths=compatibility_paths,
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
            energyplus_exit_code=validation.exit_code,
            warning_count=validation.warnings,
            severe_count=validation.severe_errors,
            fatal_count=validation.fatal_errors,
            requested_signal_count=validation.requested_signals,
            produced_signal_count=validation.produced_signals,
            missing_required_signals=validation.missing_required_signals,
            timestep_count=validation.timestep_count,
            error_type=error.error_type if error else None,
            error_message=error.message if error else None,
        )
        if self._mlflow_tracker is not None and tracking_handle is not None:
            try:
                self._mlflow_tracker.finish(
                    handle=tracking_handle,
                    result=result,
                    manifest_path=manifest_path,
                )
            except Exception as exc:
                try:
                    self._mlflow_tracker.fail(
                        handle=tracking_handle,
                        error=exc,
                    )
                except Exception:
                    pass

                if getattr(self._mlflow_tracker, "strict", False):
                    raise
        return result


def generate_energyplus_case(
    case_spec: CaseSpec,
    *,
    generated_data_root: str | Path | None = None,
    campaign_id: str | None = None,
    case_root: str | Path | None = None,
) -> GenerationResult:
    """Generate one EnergyPlus case through the default orchestrator."""
    return EnergyPlusGenerationOrchestrator(
        generated_data_root=generated_data_root,
    ).generate(
        case_spec,
        campaign_id=campaign_id,
        case_root=case_root,
    )


def _build_validation(
    case_spec: CaseSpec,
    run_result: Any | None,
    extraction: Any | None,
) -> ValidationSummary:
    """Build validation counts from available execution stages."""
    return ValidationSummary(
        exit_code=(
            0
            if run_result is not None and run_result.completed_successfully
            else None
        ),
        warnings=getattr(run_result, "warning_count", 0),
        severe_errors=getattr(run_result, "severe_count", 0),
        fatal_errors=getattr(run_result, "fatal_count", 0),
        requested_signals=len(case_spec.output_variables),
        produced_signals=getattr(extraction, "produced_signal_count", 0),
        missing_required_signals=getattr(
            extraction,
            "missing_required_signals",
            (),
        ),
        timestep_count=getattr(extraction, "timestep_count", None),
    )


def _inventory_artifacts(
    run_root: Path,
    *,
    manifest_path: Path,
) -> tuple[ArtifactRecord, ...]:
    """Create deterministic artifact records for files written so far."""
    records: list[ArtifactRecord] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(run_root)
        records.append(
            ArtifactRecord(
                role=_artifact_role(relative),
                relative_path=relative,
                media_type=_media_type(path),
                sha256=_sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(records)


def _artifact_role(relative_path: Path) -> str:
    """Infer a stable artifact role from its run-relative path."""
    top_level = relative_path.parts[0]
    return f"{top_level}_{relative_path.stem}".casefold()


def _media_type(path: Path) -> str:
    """Return a compact media type for known generation artifacts."""
    return {
        ".json": "application/json",
        ".parquet": "application/vnd.apache.parquet",
        ".pickle": "application/python-pickle",
        ".idf": "text/plain",
        ".epw": "text/plain",
        ".err": "text/plain",
        ".eio": "text/plain",
        ".eso": "text/plain",
        ".csv": "text/csv",
    }.get(path.suffix.casefold(), "application/octet-stream")


def _sha256_file(path: Path) -> str:
    """Hash one generated artifact without loading it completely."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_role_paths(run_root: Path, directory: Path) -> dict[str, Path]:
    """Map filenames to concrete paths when an artifact directory exists."""
    if not directory.is_dir():
        return {}
    return {
        path.name: path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.is_relative_to(run_root)
    }


def _software_metadata(case_spec: CaseSpec) -> SoftwareMetadata:
    """Collect portable software versions without invoking external commands."""
    try:
        scalebridge_version = metadata.version("scalebridge")
    except metadata.PackageNotFoundError:
        scalebridge_version = "0.1.0"
    try:
        opyplus_version = metadata.version("opyplus")
    except metadata.PackageNotFoundError:
        opyplus_version = None
    return SoftwareMetadata(
        scalebridge_version=scalebridge_version,
        energyplus_version=case_spec.energyplus_version,
        opyplus_version=opyplus_version,
        python_version=sys.version.split()[0],
    )


def _write_latest_pointer(
    *,
    root: Path,
    case_id: str,
    run_id: str,
    status: RunStatus,
    manifest_path: Path,
) -> None:
    """Write a small portable pointer to the latest completed attempt."""
    payload = {
        "case_id": case_id,
        "run_id": run_id,
        "status": status.value,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
    }
    (root / "latest_run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
