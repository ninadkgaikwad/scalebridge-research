"""Validated contracts for EnergyPlus generation cases and execution records.

The models in this module define the boundary between campaign orchestration,
EnergyPlus execution, artifact storage, and downstream data processing.

The central distinction is:

``CaseSpec``
    Immutable scientific intent shared across machines and execution systems.

``RunManifest``
    Persistent provenance for one execution attempt of a case.

``GenerationResult``
    Compact in-memory result returned to loops, workers, or schedulers.

All models reject unknown fields and are immutable after validation. This
prevents configuration drift after a deterministic case identifier is created.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


SHA256_PATTERN = r"^[0-9a-f]{64}$"
REPORTING_FREQUENCIES = {
    "detailed",
    "timestep",
    "hourly",
    "daily",
    "monthly",
    "runperiod",
    "environment",
    "annual",
}


class FrozenModel(BaseModel):
    """Base class for immutable, strict manifest models."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RunPeriod(FrozenModel):
    """Calendar boundaries applied to the EnergyPlus ``RunPeriod`` object."""

    start_month: int = Field(ge=1, le=12)
    start_day: int = Field(ge=1, le=31)
    end_month: int = Field(ge=1, le=12)
    end_day: int = Field(ge=1, le=31)
    calendar_year: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_dates(self) -> RunPeriod:
        """Validate real calendar dates and ordered dated-year periods."""
        validation_year = self.calendar_year or 2000
        start = date(validation_year, self.start_month, self.start_day)
        end = date(validation_year, self.end_month, self.end_day)
        if self.calendar_year is not None and end < start:
            raise ValueError("end date must not precede start date when calendar_year is set")
        return self


class OutputVariableRequest(FrozenModel):
    """Declarative request for one EnergyPlus ``Output:Variable`` object."""

    variable_name: str = Field(min_length=1)
    key_value: str = Field(default="*", min_length=1)
    reporting_frequency: str = "timestep"
    required: bool = True
    semantic_role: str | None = None

    @field_validator("reporting_frequency")
    @classmethod
    def normalize_reporting_frequency(cls, value: str) -> str:
        """Normalize and validate an EnergyPlus reporting frequency."""
        normalized = value.casefold()
        if normalized not in REPORTING_FREQUENCIES:
            allowed = ", ".join(sorted(REPORTING_FREQUENCIES))
            raise ValueError(f"unsupported reporting frequency; expected one of: {allowed}")
        return normalized


class ScheduleOperation(FrozenModel):
    """Declarative mutation applied to a schedule while preparing an IDF."""

    operation: Literal["replace_fields", "add", "delete", "rename"]
    object_type: str = Field(default="Schedule:Compact", min_length=1)
    schedule_name: str = Field(min_length=1)
    fields: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_operation_fields(self) -> ScheduleOperation:
        """Ensure each operation contains the fields required by its action."""
        if self.operation in {"replace_fields", "add", "rename"} and not self.fields:
            raise ValueError(f"{self.operation} requires at least one field")
        if self.operation == "delete" and self.fields:
            raise ValueError("delete must not define fields")
        if self.operation == "rename" and set(self.fields) != {"name"}:
            raise ValueError("rename requires exactly one field named 'name'")
        return self


class CaseSpec(FrozenModel):
    """Immutable scientific specification for one EnergyPlus simulation case.

    Input paths locate files on the current machine but do not define case
    identity. Their SHA-256 values, simulation settings, output requests, and
    schedule operations define the deterministic ``case_id``.
    """

    schema_version: str = "0.1.0"
    case_name: str = Field(min_length=1)

    building_type: str | None = None
    prototype_standard: str | None = None
    prototype_year: str | None = None
    climate_zone: str | None = None
    weather_location: str | None = None

    idf_path: Path
    epw_path: Path
    idf_sha256: str = Field(pattern=SHA256_PATTERN)
    epw_sha256: str = Field(pattern=SHA256_PATTERN)

    run_period: RunPeriod
    timestep_minutes: int = Field(gt=0, le=60)
    output_variables: tuple[OutputVariableRequest, ...] = Field(min_length=1)
    schedule_operations: tuple[ScheduleOperation, ...] = ()
    request_variable_dictionary: bool = True
    energyplus_version: str | None = None

    write_legacy_pickles: bool = True
    preserve_raw_outputs: bool = True
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("idf_sha256", "epw_sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        """Normalize hexadecimal file digests to lowercase."""
        return value.casefold()

    @field_validator("timestep_minutes")
    @classmethod
    def validate_timestep(cls, value: int) -> int:
        """Require a whole number of simulation timesteps per hour."""
        if 60 % value != 0:
            raise ValueError("timestep_minutes must divide evenly into 60")
        return value

    @model_validator(mode="after")
    def validate_unique_output_requests(self) -> CaseSpec:
        """Reject duplicate variable, key, and frequency combinations."""
        identities = [
            (
                request.variable_name.casefold(),
                request.key_value.casefold(),
                request.reporting_frequency,
            )
            for request in self.output_variables
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("output_variables contains duplicate requests")
        return self

    @computed_field
    @property
    def case_id(self) -> str:
        """Return the deterministic identifier for this scientific case."""
        from scalebridge.integration.energyplus.manifests.identifiers import build_case_id

        return build_case_id(self)


class RunStatus(str, Enum):
    """Lifecycle state for one generation attempt."""

    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INVALID = "invalid"


class ArtifactRecord(FrozenModel):
    """Metadata for one artifact stored beneath a run directory."""

    role: str = Field(min_length=1)
    relative_path: Path
    media_type: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    size_bytes: int | None = Field(default=None, ge=0)
    required: bool = True

    @field_validator("sha256")
    @classmethod
    def normalize_optional_sha256(cls, value: str | None) -> str | None:
        """Normalize an optional artifact digest to lowercase."""
        return value.casefold() if value is not None else None

    @field_validator("relative_path")
    @classmethod
    def require_relative_path(cls, value: Path) -> Path:
        """Prevent artifact references from escaping the run directory."""
        raw = str(value)
        if value.is_absolute() or PureWindowsPath(raw).is_absolute() or raw.startswith(("/", "\\")):
            raise ValueError("artifact paths must be relative to the run directory")
        if ".." in value.parts:
            raise ValueError("artifact paths must not escape the run directory")
        return value


class ExecutionMetadata(FrozenModel):
    """Machine, scheduler, and timing information for one execution attempt."""

    machine_id: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    worker_id: str | None = None
    slurm_job_id: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    runtime_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_timing(self) -> ExecutionMetadata:
        """Reject execution completion times earlier than their start."""
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self


class SoftwareMetadata(FrozenModel):
    """Software versions associated with a generation attempt."""

    scalebridge_version: str
    git_commit: str | None = None
    energyplus_version: str | None = None
    opyplus_version: str | None = None
    python_version: str | None = None


class ValidationSummary(FrozenModel):
    """Post-run EnergyPlus, signal, and timestep validation summary."""

    exit_code: int | None = None
    warnings: int = Field(default=0, ge=0)
    severe_errors: int = Field(default=0, ge=0)
    fatal_errors: int = Field(default=0, ge=0)
    requested_signals: int = Field(default=0, ge=0)
    produced_signals: int = Field(default=0, ge=0)
    missing_required_signals: tuple[str, ...] = ()
    timestep_count: int | None = Field(default=None, ge=0)


class TrackingMetadata(FrozenModel):
    """Optional MLflow references associated with an execution attempt."""

    mlflow_experiment: str | None = None
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str | None = None


class ErrorRecord(FrozenModel):
    """Serializable failure details for failed or invalid runs."""

    error_type: str
    message: str
    traceback_path: Path | None = None


class RunManifest(FrozenModel):
    """Authoritative persisted provenance for one execution attempt."""

    schema_version: str = "0.1.0"
    case_id: str
    run_id: str = Field(min_length=1)
    campaign_id: str | None = None
    status: RunStatus
    case_spec: CaseSpec
    execution: ExecutionMetadata
    software: SoftwareMetadata
    validation: ValidationSummary = Field(default_factory=ValidationSummary)
    artifacts: tuple[ArtifactRecord, ...] = ()
    tracking: TrackingMetadata = Field(default_factory=TrackingMetadata)
    error: ErrorRecord | None = None

    @model_validator(mode="after")
    def validate_manifest_consistency(self) -> RunManifest:
        """Validate case identity and status-dependent error information."""
        if self.case_id != self.case_spec.case_id:
            raise ValueError("case_id does not match the embedded case_spec")

        failure_statuses = {RunStatus.FAILED, RunStatus.INVALID}
        if self.status in failure_statuses and self.error is None:
            raise ValueError(f"{self.status.value} manifests require an error record")
        if self.status not in failure_statuses and self.error is not None:
            raise ValueError("error records are only valid for failed or invalid manifests")
        return self


class GenerationResult(FrozenModel):
    """Compact in-memory outcome returned by a generation worker.

    Batch orchestration can inspect this object without loading the complete
    persisted manifest or any generated time-series artifacts.
    """

    case_id: str
    run_id: str
    status: RunStatus
    artifact_root: Path
    manifest_path: Path
    raw_output_paths: dict[str, Path] = Field(default_factory=dict)
    canonical_output_paths: dict[str, Path] = Field(default_factory=dict)
    compatibility_output_paths: dict[str, Path] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime
    runtime_seconds: float = Field(ge=0)
    energyplus_exit_code: int | None = None
    warning_count: int = Field(default=0, ge=0)
    severe_count: int = Field(default=0, ge=0)
    fatal_count: int = Field(default=0, ge=0)
    requested_signal_count: int = Field(default=0, ge=0)
    produced_signal_count: int = Field(default=0, ge=0)
    missing_required_signals: tuple[str, ...] = ()
    timestep_count: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> GenerationResult:
        """Validate timing and paired error fields."""
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if (self.error_type is None) != (self.error_message is None):
            raise ValueError("error_type and error_message must be provided together")
        return self


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for manifest creation."""
    return datetime.now(timezone.utc)
