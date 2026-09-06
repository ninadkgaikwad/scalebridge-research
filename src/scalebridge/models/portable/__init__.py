"""ScaleBridge Phase E0-7 portable trained-model artifact and forward runtime."""

from .bundle import (
    BUNDLE_MANIFEST_FILENAME,
    PortableModelBundle,
    write_portable_model_bundle,
)
from .contracts import (
    E0_7_BUNDLE_SCHEMA_VERSION,
    E0_7_PAYLOAD_ENVELOPE_VERSION,
    ArtifactStage,
    BundleFileRecord,
    DataLocator,
    MethodPayloadDescriptor,
    ModelFamily,
    NormalizationContract,
    PHVACBundleContract,
    PHVACZoneModelSpec,
    PortableModelError,
    PortableModelManifest,
    RuntimeInputSchema,
    ScalarTransform,
)
from .historical import HistoricalReplayDataset
from .lineage import DataRootRegistry, locator_from_path, sha256_file
from .normalization import denormalize_named_outputs, normalize_named_inputs
from .phvac import PHVACPrediction, PHVACRuntime, prepare_phvac_bundle_contract
from .rc_payload import (
    RC_PHYSICAL_PAYLOAD_SCHEMA_VERSION,
    RCPhysicalPayload,
    default_final_allocation_results,
    load_rc_physical_payload,
    write_rc_physical_payload,
)
from .runtime import ForwardStepResult, RCForwardRuntime

__all__ = [
    "ArtifactStage",
    "BUNDLE_MANIFEST_FILENAME",
    "BundleFileRecord",
    "DataLocator",
    "DataRootRegistry",
    "E0_7_BUNDLE_SCHEMA_VERSION",
    "E0_7_PAYLOAD_ENVELOPE_VERSION",
    "ForwardStepResult",
    "HistoricalReplayDataset",
    "MethodPayloadDescriptor",
    "ModelFamily",
    "NormalizationContract",
    "PHVACBundleContract",
    "PHVACPrediction",
    "PHVACRuntime",
    "PHVACZoneModelSpec",
    "prepare_phvac_bundle_contract",
    "PortableModelBundle",
    "PortableModelError",
    "PortableModelManifest",
    "RCForwardRuntime",
    "RCPhysicalPayload",
    "RC_PHYSICAL_PAYLOAD_SCHEMA_VERSION",
    "RuntimeInputSchema",
    "ScalarTransform",
    "default_final_allocation_results",
    "load_rc_physical_payload",
    "denormalize_named_outputs",
    "locator_from_path",
    "normalize_named_inputs",
    "sha256_file",
    "write_portable_model_bundle",
    "write_rc_physical_payload",
]
