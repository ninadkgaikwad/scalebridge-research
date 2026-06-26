"""EnergyPlus case identity and execution-manifest support.

The manifest package separates immutable scientific case definitions from
machine-specific execution records. It provides deterministic identifiers,
validated Pydantic models, and atomic JSON serialization.
"""

from scalebridge.integration.energyplus.manifests.identifiers import build_case_id
from scalebridge.integration.energyplus.manifests.models import (
    ArtifactRecord,
    CaseSpec,
    ExecutionMetadata,
    GenerationResult,
    OutputVariableRequest,
    RunManifest,
    RunPeriod,
    RunStatus,
    ScheduleOperation,
    SoftwareMetadata,
    TrackingMetadata,
    ValidationSummary,
)
from scalebridge.integration.energyplus.manifests.serialization import (
    load_case_spec,
    load_run_manifest,
    write_case_spec,
    write_run_manifest,
)

__all__ = [
    "ArtifactRecord",
    "CaseSpec",
    "ExecutionMetadata",
    "GenerationResult",
    "OutputVariableRequest",
    "RunManifest",
    "RunPeriod",
    "RunStatus",
    "ScheduleOperation",
    "SoftwareMetadata",
    "TrackingMetadata",
    "ValidationSummary",
    "build_case_id",
    "load_case_spec",
    "load_run_manifest",
    "write_case_spec",
    "write_run_manifest",
]
