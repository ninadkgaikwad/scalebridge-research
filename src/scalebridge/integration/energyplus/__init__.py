"""ScaleBridge EnergyPlus integration.

This package exposes the stable public contracts used to describe EnergyPlus
simulation cases, identify scientifically equivalent cases across machines,
record execution attempts, and serialize their manifests.

Simulation execution and output parsing are implemented in internal
subpackages. Callers should import shared contracts from this package instead
of depending on internal module locations.
"""

from scalebridge.integration.energyplus.idf import (
    IdfBackend,
    IdfBackendError,
    IdfPreparationError,
    IdfPreparer,
    OpyplusIdfBackend,
    OpyplusNotInstalledError,
    PreparedIdfResult,
    prepare_idf,
)
from scalebridge.integration.energyplus.generation import (
    EnergyPlusGenerationOrchestrator,
    generate_energyplus_case,
)
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
from scalebridge.integration.energyplus.simulation.runner import (
    EnergyPlusExecutionError,
    EnergyPlusInputError,
    EnergyPlusRunResult,
    EnergyPlusRunner,
    EnergyPlusRunnerError,
)
from scalebridge.integration.energyplus.prototypes import (
    COMMERCIAL_TMY3_BY_LOCATION,
    PrototypeInventoryError,
    PrototypeInventoryRecord,
    PrototypeInventoryResult,
    build_and_write_pnnl_inventory,
    resolve_external_data_root,
    resolve_generated_data_root,
    scan_pnnl_commercial_prototypes,
)
from scalebridge.integration.energyplus.outputs import (
    CanonicalExtractionError,
    CanonicalExtractionResult,
    EnergyPlusOutputExtractor,
    extract_canonical_outputs,
)
from scalebridge.integration.energyplus.p1 import (
    P1_BUILDING_TYPES,
    P1_CAMPAIGN_ID,
    P1_CLIMATES,
    P1_REQUIRED_VARIABLE_NAMES,
    build_p1_case_specs,
    p1_output_variables,
    write_p1_campaign_manifest,
)

__all__ = [
    "ArtifactRecord",
    "CaseSpec",
    "CanonicalExtractionError",
    "CanonicalExtractionResult",
    "COMMERCIAL_TMY3_BY_LOCATION",
    "EnergyPlusExecutionError",
    "EnergyPlusGenerationOrchestrator",
    "EnergyPlusInputError",
    "EnergyPlusRunResult",
    "EnergyPlusRunner",
    "EnergyPlusRunnerError",
    "EnergyPlusOutputExtractor",
    "ExecutionMetadata",
    "GenerationResult",
    "IdfBackend",
    "IdfBackendError",
    "IdfPreparationError",
    "IdfPreparer",
    "OutputVariableRequest",
    "OpyplusIdfBackend",
    "OpyplusNotInstalledError",
    "PreparedIdfResult",
    "PrototypeInventoryError",
    "PrototypeInventoryRecord",
    "PrototypeInventoryResult",
    "P1_BUILDING_TYPES",
    "P1_CAMPAIGN_ID",
    "P1_CLIMATES",
    "P1_REQUIRED_VARIABLE_NAMES",
    "RunManifest",
    "RunPeriod",
    "RunStatus",
    "ScheduleOperation",
    "SoftwareMetadata",
    "TrackingMetadata",
    "ValidationSummary",
    "build_case_id",
    "build_and_write_pnnl_inventory",
    "build_p1_case_specs",
    "extract_canonical_outputs",
    "generate_energyplus_case",
    "load_case_spec",
    "load_run_manifest",
    "prepare_idf",
    "p1_output_variables",
    "resolve_external_data_root",
    "resolve_generated_data_root",
    "scan_pnnl_commercial_prototypes",
    "write_case_spec",
    "write_run_manifest",
    "write_p1_campaign_manifest",
]
