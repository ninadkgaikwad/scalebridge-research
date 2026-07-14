"""Variable-wise EnergyPlus generation strategy.

This module implements the scalable P1 output strategy:

    one EnergyPlus run per requested Output:Variable

The goal is to avoid one enormous all-variable eplusout.csv/eso artifact.
Each variable run produces a small raw CSV, which is immediately converted to
a canonical per-variable Parquet file. Raw variable CSV files can then be
deleted after successful canonical and compatibility export.

This module is intentionally introduced as a separate generation strategy.
The existing standard orchestrator remains available for small smoke tests and
ordinary all-variable EnergyPlus runs.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from uuid import uuid4

import tempfile

from concurrent.futures import ThreadPoolExecutor, as_completed

from scalebridge.integration.energyplus.manifests.models import (
    CaseSpec,
    OutputVariableRequest,
)
from scalebridge.integration.energyplus.idf import IdfPreparer
from scalebridge.integration.energyplus.manifests.models import (
    ArtifactRecord,
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
from scalebridge.integration.energyplus.outputs.eio import parse_eio
from scalebridge.integration.energyplus.simulation import EnergyPlusRunner

DATE_TIME_COLUMN = "Date/Time"


@dataclass(frozen=True)
class VariableWiseArtifact:
    """Artifacts produced for one EnergyPlus output variable."""

    parent_case_id: str
    variable_case_id: str
    variable_id: str
    variable_name: str
    reporting_frequency: str
    raw_csv_path: Path | None
    canonical_parquet_path: Path
    legacy_pickle_path: Path | None
    row_count: int
    column_count: int
    raw_csv_deleted: bool

@dataclass(frozen=True)
class VariableWiseTaskResult:
    """Result from one variable-wise EnergyPlus task."""

    variable_index: int
    artifact: VariableWiseArtifact
    warning_count: int
    severe_count: int
    fatal_count: int
    runtime_seconds: float

def safe_variable_id(request: OutputVariableRequest) -> str:
    """Create a stable filesystem-safe identifier for one output request."""
    base = f"{request.reporting_frequency}__{request.key_value}__{request.variable_name}"
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", base.strip()).strip("_").lower()
    return normalized or "output_variable"


def one_variable_case_spec(
    parent_case_spec: CaseSpec,
    request: OutputVariableRequest,
) -> CaseSpec:
    """Return a temporary one-variable CaseSpec used only for IDF preparation.

    The resulting ``case_id`` differs from the parent case because output
    variables are part of the scientific case identity. Variable-wise generation
    should store both identifiers:
        - parent_case_id: original P1 case identity
        - variable_case_id: one-variable preparation/execution identity
    """
    return parent_case_spec.model_copy(
        update={
            "output_variables": (request,),
            "preserve_raw_outputs": True,
        }
    )


def convert_variable_csv_to_parquet(
    *,
    csv_path: str | Path,
    parquet_path: str | Path,
    request: OutputVariableRequest,
    parent_case_spec: CaseSpec,
    variable_case_spec: CaseSpec,
    chunksize: int = 100_000,
) -> tuple[int, int]:
    """Convert one variable-wise EnergyPlus CSV to canonical long-form Parquet.

    The conversion is chunked so even a large variable CSV is not fully loaded
    into memory. The resulting Parquet schema is long-form:

        parent_case_id, variable_case_id, timestamp_raw, reporting_frequency,
        key_value, variable_name, units, semantic_role, value

    Returns
    -------
    tuple[int, int]
        ``(row_count, column_count)`` after canonical conversion.
    """
    source = Path(csv_path).expanduser().resolve()
    destination = Path(parquet_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        raise FileNotFoundError(f"variable CSV does not exist: {source}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for variable-wise Parquet output") from exc

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_columns = 0

    try:
        for chunk in pd.read_csv(source, chunksize=chunksize):
            if DATE_TIME_COLUMN not in chunk.columns:
                raise ValueError(
                    f"EnergyPlus CSV is missing required column {DATE_TIME_COLUMN!r}: "
                    f"{source}"
                )

            value_columns = [column for column in chunk.columns if column != DATE_TIME_COLUMN]
            total_columns = max(total_columns, len(value_columns))

            pieces: list[pd.DataFrame] = []
            for column in value_columns:
                key_value, variable_name, units = parse_energyplus_csv_column(
                    column,
                    fallback_variable_name=request.variable_name,
                )
                pieces.append(
                    pd.DataFrame(
                        {
                            "parent_case_id": parent_case_spec.case_id,
                            "variable_case_id": variable_case_spec.case_id,
                            "timestamp_raw": chunk[DATE_TIME_COLUMN].astype(str),
                            "reporting_frequency": request.reporting_frequency,
                            "key_value": key_value,
                            "variable_name": variable_name,
                            "units": units,
                            "semantic_role": request.semantic_role,
                            "value": pd.to_numeric(chunk[column], errors="coerce"),
                        }
                    )
                )

            if not pieces:
                continue

            long_chunk = pd.concat(pieces, ignore_index=True)
            table = pa.Table.from_pandas(long_chunk, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    destination,
                    table.schema,
                    compression="zstd",
                )
            writer.write_table(table)
            total_rows += len(long_chunk)

    finally:
        if writer is not None:
            writer.close()

    return total_rows, total_columns


def parse_energyplus_csv_column(
    column_name: str,
    *,
    fallback_variable_name: str,
) -> tuple[str, str, str]:
    """Parse one EnergyPlus CSV output column label.

    EnergyPlus labels commonly look like:

        ZONE ONE:Zone Air Temperature [C](TimeStep)

    This parser keeps the original key and variable names when available and
    falls back safely when a column does not follow the expected convention.
    """
    text = str(column_name).strip()

    units = ""
    unit_match = re.search(r"\[([^\]]+)\]", text)
    if unit_match:
        units = unit_match.group(1).strip()

    without_suffix = re.sub(r"\[[^\]]+\].*$", "", text).strip()

    if ":" in without_suffix:
        key_value, variable_name = without_suffix.split(":", 1)
        key_value = key_value.strip()
        variable_name = variable_name.strip() or fallback_variable_name
    else:
        key_value = "*"
        variable_name = without_suffix or fallback_variable_name

    return key_value, variable_name, units


def write_per_variable_legacy_pickle(
    *,
    parquet_path: str | Path,
    pickle_path: str | Path,
) -> Path:
    """Write one per-variable compatibility pickle.

    This intentionally writes one pickle per variable instead of one giant
    legacy dictionary. That preserves legacy-style DataFrame access while
    avoiding a full campaign/case dictionary in memory.
    """
    source = Path(parquet_path).expanduser().resolve()
    destination = Path(pickle_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source)

    if frame.empty:
        legacy_frame = pd.DataFrame()
        variable_name = source.stem
    else:
        variable_name = str(frame["variable_name"].iloc[0]).replace(" ", "_")
        legacy_frame = frame.pivot_table(
            index="timestamp_raw",
            columns="key_value",
            values="value",
            aggfunc="first",
        ).reset_index()

    payload = {
        "variable_name": variable_name,
        "data": legacy_frame,
    }

    with destination.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    return destination


def write_variable_manifest(
    *,
    artifacts: Iterable[VariableWiseArtifact],
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and CSV manifests for variable-wise generation artifacts."""
    artifact_rows = [
        {
            "parent_case_id": artifact.parent_case_id,
            "variable_case_id": artifact.variable_case_id,
            "variable_id": artifact.variable_id,
            "variable_name": artifact.variable_name,
            "reporting_frequency": artifact.reporting_frequency,
            "raw_csv_path": str(artifact.raw_csv_path) if artifact.raw_csv_path else "",
            "canonical_parquet_path": str(artifact.canonical_parquet_path),
            "legacy_pickle_path": (
                str(artifact.legacy_pickle_path)
                if artifact.legacy_pickle_path is not None
                else ""
            ),
            "row_count": artifact.row_count,
            "column_count": artifact.column_count,
            "raw_csv_deleted": artifact.raw_csv_deleted,
        }
        for artifact in artifacts
    ]

    json_destination = Path(json_path).expanduser().resolve()
    csv_destination = Path(csv_path).expanduser().resolve()
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)

    json_destination.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "artifact_count": len(artifact_rows),
                "artifacts": artifact_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(artifact_rows).to_csv(csv_destination, index=False)
    return json_destination, csv_destination


def delete_raw_csv_after_success(
    raw_csv_path: Path,
    *,
    retries: int = 5,
    delay_seconds: float = 1.0,
) -> bool:
    """
    Delete raw EnergyPlus CSV after canonical outputs were successfully written.

    This cleanup must never invalidate an otherwise successful variable artifact.
    On Windows/Dropbox machines, a recently written CSV can remain temporarily
    locked by the writer, Dropbox sync, antivirus, or indexing. In that case,
    retry a few times and then keep the raw CSV instead of raising.

    Returns
    -------
    bool
        True if the raw CSV is absent after cleanup.
        False if it still exists because deletion was blocked.
    """
    target = Path(raw_csv_path)

    for attempt in range(1, retries + 1):
        try:
            target.unlink()
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if attempt < retries:
                time.sleep(delay_seconds)
                continue
            return False

    return not target.exists()


def move_energyplus_variable_outputs(
    *,
    simulation_output_directory: str | Path,
    raw_variable_csv_path: str | Path,
    raw_eio_path: str | Path | None = None,
    raw_err_path: str | Path | None = None,
) -> None:
    """Move selected EnergyPlus output files from one variable run.

    This function moves the current run's ``eplusout.csv`` into the persistent
    variable-wise raw CSV folder. EIO and ERR files are optionally retained for
    diagnostics.
    """
    source_root = Path(simulation_output_directory).expanduser().resolve()
    csv_source = source_root / "eplusout.csv"

    if not csv_source.is_file():
        raise FileNotFoundError(f"EnergyPlus variable run did not produce {csv_source}")

    csv_destination = Path(raw_variable_csv_path).expanduser().resolve()
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(csv_source), str(csv_destination))

    if raw_eio_path is not None:
        eio_source = source_root / "eplusout.eio"
        if eio_source.is_file():
            eio_destination = Path(raw_eio_path).expanduser().resolve()
            eio_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(eio_source, eio_destination)

    if raw_err_path is not None:
        err_source = source_root / "eplusout.err"
        if err_source.is_file():
            err_destination = Path(raw_err_path).expanduser().resolve()
            err_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(err_source, err_destination)

def generate_variable_wise_case(
    case_spec: CaseSpec,
    *,
    generated_data_root: str | Path,
    case_collection_name: str,
    machine_id: str,
    campaign_id: str | None = None,
    selected_output_variables: tuple[OutputVariableRequest, ...] | None = None,
    run_id: str | None = None,
    delete_raw_csv: bool = True,
    mlflow_tracker=None,
    short_work_root: str | Path | None = None,
    parallel_variable_workers: int | None = None,
) -> GenerationResult:
    """Generate one case using one EnergyPlus run per output variable."""
    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    attempt_id = run_id or f"epvwr_{uuid4().hex[:12]}"

    root = (
        Path(generated_data_root).expanduser().resolve()
        / case_collection_name
        / "cases"
        / case_spec.case_id
    )
    run_root = root / "runs" / attempt_id

    inputs_root = run_root / "inputs"
    variable_idf_root = inputs_root / "variable_idfs"
    raw_variable_csv_root = run_root / "raw" / "variable_csv"
    raw_eio_root = run_root / "raw" / "eio"
    raw_err_root = run_root / "raw" / "err"
    diagnostic_variable_run_root = run_root / "variable_runs"
    short_work_base = _resolve_short_work_root(
        short_work_root=short_work_root,
        attempt_id=attempt_id,
    )
    canonical_root = run_root / "canonical"
    canonical_variable_root = canonical_root / "variables"
    legacy_root = run_root / "legacy"
    legacy_variable_root = legacy_root / "per_variable_pickle"

    manifest_path = run_root / "run_manifest.json"
    case_spec_path = inputs_root / "case_spec.json"
    traceback_path = run_root / "traceback.txt"

    inputs_root.mkdir(parents=True, exist_ok=False)
    variable_idf_root.mkdir(parents=True, exist_ok=True)
    raw_variable_csv_root.mkdir(parents=True, exist_ok=True)
    raw_eio_root.mkdir(parents=True, exist_ok=True)
    raw_err_root.mkdir(parents=True, exist_ok=True)
    diagnostic_variable_run_root.mkdir(parents=True, exist_ok=True)
    short_work_base.mkdir(parents=True, exist_ok=True)
    canonical_variable_root.mkdir(parents=True, exist_ok=True)

    write_case_spec(case_spec, case_spec_path)

    # requests = selected_output_variables or case_spec.output_variables
    # idf_preparer = IdfPreparer()
    # runner = EnergyPlusRunner(beat_frequency_seconds=5)

    requests = tuple(selected_output_variables or case_spec.output_variables)
    worker_count = resolve_parallel_variable_workers(
        machine_id=machine_id,
        requested=parallel_variable_workers,
        variable_count=len(requests),
    )

    artifacts: list[VariableWiseArtifact] = []
    warning_count = 0
    severe_count = 0
    fatal_count = 0
    error: ErrorRecord | None = None
    status = RunStatus.FAILED
    tracking_metadata = TrackingMetadata()
    tracking_handle = None

    if mlflow_tracker is not None:
        try:
            tracking_handle = mlflow_tracker.start(
                case_spec=case_spec,
                run_id=attempt_id,
                campaign_id=campaign_id,
                machine_id=machine_id,
            )
            if tracking_handle is not None:
                tracking_metadata = tracking_handle.to_metadata()
        except Exception:
            tracking_handle = None

    try:
        # for variable_index, request in enumerate(requests, start=1):
        #     variable_id = safe_variable_id(request)
        #     variable_case_spec = one_variable_case_spec(case_spec, request)

        #     variable_idf_path = variable_idf_root / f"{variable_id}.idf"
        #     variable_output_root = short_work_base / f"v{variable_index:03d}"
        #     diagnostic_output_root = diagnostic_variable_run_root / variable_id

        #     preparation = idf_preparer.prepare(
        #         variable_case_spec,
        #         variable_idf_path,
        #     )
        #     run_result = runner.run(
        #         idf_path=preparation.prepared_idf_path,
        #         epw_path=case_spec.epw_path,
        #         output_directory=variable_output_root,
        #     )

        #     _copy_variable_run_diagnostics(
        #         source_directory=run_result.output_directory,
        #         destination_directory=diagnostic_output_root,
        #     )

        #     warning_count += run_result.warning_count
        #     severe_count += run_result.severe_count
        #     fatal_count += run_result.fatal_count

        #     if not run_result.completed_successfully:
        #         raise RuntimeError(
        #             run_result.failure_message
        #             or f"EnergyPlus variable run failed for {variable_id}"
        #         )

        #     raw_csv_path = raw_variable_csv_root / f"{variable_id}.csv"
        #     raw_eio_path = raw_eio_root / f"{variable_id}.eio"
        #     raw_err_path = raw_err_root / f"{variable_id}.err"

        #     move_energyplus_variable_outputs(
        #         simulation_output_directory=run_result.output_directory,
        #         raw_variable_csv_path=raw_csv_path,
        #         raw_eio_path=raw_eio_path,
        #         raw_err_path=raw_err_path,
        #     )

        #     canonical_parquet_path = canonical_variable_root / f"{variable_id}.parquet"
        #     row_count, column_count = convert_variable_csv_to_parquet(
        #         csv_path=raw_csv_path,
        #         parquet_path=canonical_parquet_path,
        #         request=request,
        #         parent_case_spec=case_spec,
        #         variable_case_spec=variable_case_spec,
        #     )

        #     legacy_pickle_path: Path | None = None
        #     if case_spec.write_legacy_pickles:
        #         legacy_pickle_path = legacy_variable_root / f"{variable_id}.pickle"
        #         write_per_variable_legacy_pickle(
        #             parquet_path=canonical_parquet_path,
        #             pickle_path=legacy_pickle_path,
        #         )

        #     raw_csv_deleted = False
        #     stored_raw_csv_path: Path | None = raw_csv_path
        #     if delete_raw_csv:
        #         raw_csv_deleted = delete_raw_csv_after_success(raw_csv_path)
        #         if raw_csv_deleted:
        #             stored_raw_csv_path = None

        #     shutil.rmtree(variable_output_root, ignore_errors=True)

        #     artifacts.append(
        #         VariableWiseArtifact(
        #             parent_case_id=case_spec.case_id,
        #             variable_case_id=variable_case_spec.case_id,
        #             variable_id=variable_id,
        #             variable_name=request.variable_name,
        #             reporting_frequency=request.reporting_frequency,
        #             raw_csv_path=stored_raw_csv_path,
        #             canonical_parquet_path=canonical_parquet_path,
        #             legacy_pickle_path=legacy_pickle_path,
        #             row_count=row_count,
        #             column_count=column_count,
        #             raw_csv_deleted=raw_csv_deleted,
        #         )
        #     )

        task_specs = tuple(enumerate(requests, start=1))

        if worker_count == 1:
            task_results = [
                _generate_one_variable_artifact(
                    case_spec=case_spec,
                    request=request,
                    variable_index=variable_index,
                    variable_idf_root=variable_idf_root,
                    raw_variable_csv_root=raw_variable_csv_root,
                    raw_eio_root=raw_eio_root,
                    raw_err_root=raw_err_root,
                    diagnostic_variable_run_root=diagnostic_variable_run_root,
                    canonical_variable_root=canonical_variable_root,
                    legacy_variable_root=legacy_variable_root,
                    short_work_base=short_work_base,
                    delete_raw_csv=delete_raw_csv,
                )
                for variable_index, request in task_specs
            ]
        else:
            task_results = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _generate_one_variable_artifact,
                        case_spec=case_spec,
                        request=request,
                        variable_index=variable_index,
                        variable_idf_root=variable_idf_root,
                        raw_variable_csv_root=raw_variable_csv_root,
                        raw_eio_root=raw_eio_root,
                        raw_err_root=raw_err_root,
                        diagnostic_variable_run_root=diagnostic_variable_run_root,
                        canonical_variable_root=canonical_variable_root,
                        legacy_variable_root=legacy_variable_root,
                        short_work_base=short_work_base,
                        delete_raw_csv=delete_raw_csv,
                    )
                    for variable_index, request in task_specs
                ]

                for future in as_completed(futures):
                    task_results.append(future.result())

        ordered_task_results = sorted(task_results, key=lambda item: item.variable_index)

        for task_result in ordered_task_results:
            warning_count += task_result.warning_count
            severe_count += task_result.severe_count
            fatal_count += task_result.fatal_count
            artifacts.append(task_result.artifact)

        variable_timing_manifest = _write_variable_timing_manifest(
            task_results=ordered_task_results,
            destination=canonical_root / "variable_timing_manifest.json",
        )

        variable_manifest_json, variable_manifest_csv = write_variable_manifest(
            artifacts=artifacts,
            json_path=canonical_root / "variable_manifest.json",
            csv_path=canonical_root / "variable_manifest.csv",
        )

        _write_variable_wise_metadata(
            case_spec=case_spec,
            selected_output_variables=requests,
            artifacts=artifacts,
            metadata_path=canonical_root / "metadata.json",
            variable_manifest_json=variable_manifest_json,
            variable_manifest_csv=variable_manifest_csv,
            variable_timing_manifest=variable_timing_manifest,
            parallel_variable_workers=worker_count,
        )
        _write_variable_wise_eio_tables(
            eio_directory=raw_eio_root,
            destination=canonical_root / "eio_tables.json",
        )

        if case_spec.write_legacy_pickles:
            _write_variable_wise_legacy_manifest(
                artifacts=artifacts,
                destination=legacy_root / "legacy_manifest.json",
            )

        status = (
            RunStatus.COMPLETED_WITH_WARNINGS
            if warning_count
            else RunStatus.COMPLETED
        )

    except Exception as exc:
        status = RunStatus.FAILED
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        error = ErrorRecord(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback_path=traceback_path.relative_to(run_root),
        )

    completed_at = datetime.now(timezone.utc)
    runtime_seconds = time.perf_counter() - started_counter

    produced_signal_count = len(artifacts)
    missing_required = tuple(
        _request_identity_for_variable_wise(request)
        for request in requests
        if request.required
        and safe_variable_id(request) not in {artifact.variable_id for artifact in artifacts}
    )

    validation = ValidationSummary(
        exit_code=0 if status in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS} else None,
        warnings=warning_count,
        severe_errors=severe_count,
        fatal_errors=fatal_count,
        requested_signals=len(requests),
        produced_signals=produced_signal_count,
        missing_required_signals=missing_required,
        timestep_count=None,
    )

    manifest = RunManifest(
        case_id=case_spec.case_id,
        run_id=attempt_id,
        campaign_id=campaign_id,
        status=status,
        case_spec=case_spec,
        execution=ExecutionMetadata(
            machine_id=machine_id,
            hostname=socket.gethostname(),
            platform=platform.platform(),
            worker_id=os.environ.get("SCALEBRIDGE_WORKER_ID"),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
        ),
        software=_variable_wise_software_metadata(case_spec),
        validation=validation,
        artifacts=_inventory_variable_wise_artifacts(
            run_root,
            manifest_path=manifest_path,
        ),
        tracking=tracking_metadata,
        error=error,
    )
    write_run_manifest(manifest, manifest_path)
    _write_variable_wise_latest_pointer(
        root=root,
        case_id=case_spec.case_id,
        run_id=attempt_id,
        status=status,
        manifest_path=manifest_path,
    )
    _cleanup_short_work_base(short_work_base)

    result = GenerationResult(
        case_id=case_spec.case_id,
        run_id=attempt_id,
        status=status,
        artifact_root=run_root,
        manifest_path=manifest_path,
        raw_output_paths=_relative_file_paths(run_root / "raw"),
        canonical_output_paths=_relative_file_paths(canonical_root),
        compatibility_output_paths=_relative_file_paths(legacy_root),
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

    if mlflow_tracker is not None and tracking_handle is not None:
        try:
            mlflow_tracker.finish(
                handle=tracking_handle,
                result=result,
                manifest_path=manifest_path,
            )
        except Exception as exc:
            try:
                mlflow_tracker.fail(
                    handle=tracking_handle,
                    error=exc,
                )
            except Exception:
                pass

            if getattr(mlflow_tracker, "strict", False):
                raise

    return result


def _request_identity_for_variable_wise(request: OutputVariableRequest) -> str:
    """Return a compact requested-variable identity."""
    return (
        f"{request.key_value}|{request.variable_name}|"
        f"{request.reporting_frequency}"
    )


def _write_variable_wise_metadata(
    *,
    case_spec: CaseSpec,
    selected_output_variables: tuple[OutputVariableRequest, ...],
    artifacts: list[VariableWiseArtifact],
    metadata_path: Path,
    variable_manifest_json: Path,
    variable_manifest_csv: Path,
    variable_timing_manifest: Path,
    parallel_variable_workers: int,
) -> None:
    """Write canonical metadata for variable-wise generation."""
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "generation_mode": "variable-wise",
                "case_id": case_spec.case_id,
                "parent_output_variable_count": len(case_spec.output_variables),
                "selected_output_variable_count": len(selected_output_variables),
                "requested_signal_count": len(selected_output_variables),
                "produced_signal_count": len(artifacts),
                "parallel_variable_workers": parallel_variable_workers,
                "canonical_variable_parquet_count": len(artifacts),
                "raw_csv_deleted_count": sum(
                    1 for artifact in artifacts if artifact.raw_csv_deleted
                ),
                "variable_manifest_json": variable_manifest_json.name,
                "variable_manifest_csv": variable_manifest_csv.name,
                "variable_timing_manifest_json": variable_timing_manifest.name,
                "variables": [
                    {
                        "variable_id": artifact.variable_id,
                        "variable_name": artifact.variable_name,
                        "reporting_frequency": artifact.reporting_frequency,
                        "canonical_parquet_path": artifact.canonical_parquet_path.name,
                        "legacy_pickle_path": (
                            artifact.legacy_pickle_path.name
                            if artifact.legacy_pickle_path is not None
                            else None
                        ),
                        "row_count": artifact.row_count,
                        "column_count": artifact.column_count,
                        "raw_csv_deleted": artifact.raw_csv_deleted,
                    }
                    for artifact in artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_variable_wise_eio_tables(
    *,
    eio_directory: Path,
    destination: Path,
) -> None:
    """Write EIO metadata using the first available variable-run EIO file."""
    eio_files = sorted(eio_directory.glob("*.eio"))
    if not eio_files:
        destination.write_text(
            json.dumps(
                {
                    "schema_version": "0.1.0",
                    "source": None,
                    "table_count": 0,
                    "tables": {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return

    source = eio_files[0]
    tables = parse_eio(source)
    destination.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "source": str(source),
                "table_count": len(tables),
                "tables": {
                    name: table.to_serializable_dict()
                    for name, table in tables.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_variable_timing_manifest(
    *,
    task_results: list[VariableWiseTaskResult],
    destination: Path,
) -> Path:
    """Write per-variable runtime and warning diagnostics."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "variable_index": result.variable_index,
            "variable_id": result.artifact.variable_id,
            "variable_name": result.artifact.variable_name,
            "reporting_frequency": result.artifact.reporting_frequency,
            "runtime_seconds": result.runtime_seconds,
            "warning_count": result.warning_count,
            "severe_count": result.severe_count,
            "fatal_count": result.fatal_count,
            "row_count": result.artifact.row_count,
            "column_count": result.artifact.column_count,
            "raw_csv_deleted": result.artifact.raw_csv_deleted,
        }
        for result in task_results
    ]
    destination.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "variable_count": len(rows),
                "total_variable_runtime_seconds": sum(
                    row["runtime_seconds"] for row in rows
                ),
                "variables": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _write_variable_wise_legacy_manifest(
    *,
    artifacts: list[VariableWiseArtifact],
    destination: Path,
) -> None:
    """Write a manifest for per-variable compatibility pickles."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "legacy_layout": "per_variable_pickle",
                "pickle_count": sum(
                    1 for artifact in artifacts if artifact.legacy_pickle_path is not None
                ),
                "pickles": [
                    {
                        "variable_id": artifact.variable_id,
                        "variable_name": artifact.variable_name,
                        "path": (
                            str(artifact.legacy_pickle_path)
                            if artifact.legacy_pickle_path is not None
                            else None
                        ),
                    }
                    for artifact in artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _variable_wise_software_metadata(case_spec: CaseSpec) -> SoftwareMetadata:
    """Collect portable software versions for variable-wise generation."""
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


def _inventory_variable_wise_artifacts(
    run_root: Path,
    *,
    manifest_path: Path,
) -> tuple[ArtifactRecord, ...]:
    """Inventory variable-wise files without hashing large artifacts."""
    records: list[ArtifactRecord] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(run_root)
        records.append(
            ArtifactRecord(
                role=f"{relative.parts[0]}_{path.stem}".casefold(),
                relative_path=relative,
                media_type=_variable_wise_media_type(path),
                sha256=None,
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(records)


def _variable_wise_media_type(path: Path) -> str:
    """Return a compact media type for variable-wise artifacts."""
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".parquet": "application/vnd.apache.parquet",
        ".pickle": "application/python-pickle",
        ".idf": "text/plain",
        ".eio": "text/plain",
        ".err": "text/plain",
        ".log": "text/plain",
    }.get(path.suffix.casefold(), "application/octet-stream")


def _relative_file_paths(directory: Path) -> dict[str, Path]:
    """Return file paths beneath a directory, keyed by relative POSIX path."""
    if not directory.is_dir():
        return {}
    return {
        path.relative_to(directory).as_posix(): path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _write_variable_wise_latest_pointer(
    *,
    root: Path,
    case_id: str,
    run_id: str,
    status: RunStatus,
    manifest_path: Path,
) -> None:
    """Write latest-run pointer for the case directory."""
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

def _cleanup_short_work_base(short_work_base: Path) -> bool:
    """Remove the short work attempt directory when all variable dirs are gone.

    The function intentionally removes only an empty directory. If unexpected
    files remain, the directory is preserved for debugging.
    """
    try:
        short_work_base.rmdir()
        return True
    except OSError:
        return False


def _resolve_short_work_root(
    *,
    short_work_root: str | Path | None,
    attempt_id: str,
) -> Path:
    """Return a short local work directory for opyplus/EnergyPlus subprocesses."""
    configured = (
        short_work_root
        or os.environ.get("SCALEBRIDGE_EPLUS_WORK_ROOT")
        or Path(tempfile.gettempdir()) / "scalebridge_eplus"
    )
    root = Path(configured).expanduser().resolve() / attempt_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _copy_variable_run_diagnostics(
    *,
    source_directory: str | Path,
    destination_directory: str | Path,
) -> None:
    """Copy small diagnostic files from the short EnergyPlus work directory."""
    source = Path(source_directory).expanduser().resolve()
    destination = Path(destination_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if not source.is_dir():
        return

    for name in (
        "#opyplus.info",
        "opyplus.idf",
        "opyplus.epw",
        "eplusout.err",
        "eplusout.eio",
        "eplusout.rdd",
        "eplusout.mdd",
        "eplusout.audit",
        "opyplus_progress.log",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)

def resolve_parallel_variable_workers(
    *,
    machine_id: str | None,
    requested: int | None,
    variable_count: int | None = None,
) -> int:
    """Resolve variable-wise worker count from CLI/function/env/machine default."""
    if requested is not None:
        workers = requested
    else:
        env_value = os.environ.get("SCALEBRIDGE_EPLUS_PARALLEL_VARIABLE_WORKERS")
        if env_value:
            workers = int(env_value)
        else:
            normalized = (machine_id or get_machine_id_from_environment()).strip().lower()
            defaults = {
                "laptop": 2,
                "home-pc": 2,
                "home_pc": 2,
                "homepc": 2,
                "lab-pc": 2,
                "lab_pc": 2,
                "labpc": 2,
                "kamiak": 1,
            }
            workers = defaults.get(normalized, 1)

    workers = max(1, int(workers))
    if variable_count is not None and variable_count > 0:
        workers = min(workers, variable_count)
    return workers


def get_machine_id_from_environment() -> str:
    """Return machine id from common ScaleBridge environment variables."""
    return (
        os.environ.get("SCALEBRIDGE_MACHINE_ID")
        or os.environ.get("COMPUTERNAME")
        or socket.gethostname()
    )


def _generate_one_variable_artifact(
    *,
    case_spec: CaseSpec,
    request: OutputVariableRequest,
    variable_index: int,
    variable_idf_root: Path,
    raw_variable_csv_root: Path,
    raw_eio_root: Path,
    raw_err_root: Path,
    diagnostic_variable_run_root: Path,
    canonical_variable_root: Path,
    legacy_variable_root: Path,
    short_work_base: Path,
    delete_raw_csv: bool,
) -> VariableWiseTaskResult:
    """Run, canonicalize, and optionally legacy-export one variable."""
    started_counter = time.perf_counter()
    variable_id = safe_variable_id(request)
    variable_case_spec = one_variable_case_spec(case_spec, request)

    variable_idf_path = variable_idf_root / f"{variable_id}.idf"
    variable_output_root = short_work_base / f"v{variable_index:03d}"
    diagnostic_output_root = diagnostic_variable_run_root / variable_id

    idf_preparer = IdfPreparer()
    runner = EnergyPlusRunner(beat_frequency_seconds=5)

    preparation = idf_preparer.prepare(
        variable_case_spec,
        variable_idf_path,
    )
    run_result = runner.run(
        idf_path=preparation.prepared_idf_path,
        epw_path=case_spec.epw_path,
        output_directory=variable_output_root,
    )

    _copy_variable_run_diagnostics(
        source_directory=run_result.output_directory,
        destination_directory=diagnostic_output_root,
    )

    if not run_result.completed_successfully:
        raise RuntimeError(
            run_result.failure_message
            or f"EnergyPlus variable run failed for {variable_id}"
        )

    raw_csv_path = raw_variable_csv_root / f"{variable_id}.csv"
    raw_eio_path = raw_eio_root / f"{variable_id}.eio"
    raw_err_path = raw_err_root / f"{variable_id}.err"

    move_energyplus_variable_outputs(
        simulation_output_directory=run_result.output_directory,
        raw_variable_csv_path=raw_csv_path,
        raw_eio_path=raw_eio_path,
        raw_err_path=raw_err_path,
    )

    canonical_parquet_path = canonical_variable_root / f"{variable_id}.parquet"
    row_count, column_count = convert_variable_csv_to_parquet(
        csv_path=raw_csv_path,
        parquet_path=canonical_parquet_path,
        request=request,
        parent_case_spec=case_spec,
        variable_case_spec=variable_case_spec,
    )

    legacy_pickle_path: Path | None = None
    if case_spec.write_legacy_pickles:
        legacy_pickle_path = legacy_variable_root / f"{variable_id}.pickle"
        write_per_variable_legacy_pickle(
            parquet_path=canonical_parquet_path,
            pickle_path=legacy_pickle_path,
        )

    raw_csv_deleted = False
    stored_raw_csv_path: Path | None = raw_csv_path
    if delete_raw_csv:
        raw_csv_deleted = delete_raw_csv_after_success(raw_csv_path)
        if raw_csv_deleted:
            stored_raw_csv_path = None

    shutil.rmtree(variable_output_root, ignore_errors=True)

    artifact = VariableWiseArtifact(
        parent_case_id=case_spec.case_id,
        variable_case_id=variable_case_spec.case_id,
        variable_id=variable_id,
        variable_name=request.variable_name,
        reporting_frequency=request.reporting_frequency,
        raw_csv_path=stored_raw_csv_path,
        canonical_parquet_path=canonical_parquet_path,
        legacy_pickle_path=legacy_pickle_path,
        row_count=row_count,
        column_count=column_count,
        raw_csv_deleted=raw_csv_deleted,
    )

    return VariableWiseTaskResult(
        variable_index=variable_index,
        artifact=artifact,
        warning_count=run_result.warning_count,
        severe_count=run_result.severe_count,
        fatal_count=run_result.fatal_count,
        runtime_seconds=time.perf_counter() - started_counter,
    )
