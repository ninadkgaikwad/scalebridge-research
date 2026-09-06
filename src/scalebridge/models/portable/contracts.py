from __future__ import annotations

"""E0-7 portable-model and forward-runtime contracts.

Mathematical authority
----------------------
ScaleBridge_PhaseE0_E0-7_Runtime_PHVAC_Portable_Model_Bundle_Contract_v1_1.tex

This module owns static post-estimation/deployment metadata only.  It contains
no optimizer, training loop, MCMC sampler, controller, Gymnasium environment,
or evaluation-policy implementation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


E0_7_BUNDLE_SCHEMA_VERSION = "phase_e0_e07_bundle_v1"
E0_7_PAYLOAD_ENVELOPE_VERSION = "phase_e0_e07_payload_v1"


class PortableModelError(ValueError):
    """Raised when an E0-7 artifact/runtime contract is invalid."""


class ModelFamily(str, Enum):
    CLASSICAL_ML = "classical_ml"
    SCIML = "sciml"
    OPTIMIZATION = "optimization"
    BAYESIAN = "bayesian"
    GENERIC = "generic"


class ArtifactStage(str, Enum):
    PHASE_B = "phase_b"
    PHASE_C = "phase_c"
    PHASE_D = "phase_d"
    OTHER = "other"


@dataclass(frozen=True)
class DataLocator:
    """Portable logical reference to an upstream artifact.

    ``root_alias`` identifies a machine-local registered data root while
    ``relative_path`` is the portable path below that root.  Absolute paths and
    parent traversal are forbidden as scientific authority.
    """

    stage: ArtifactStage | str
    root_alias: str
    relative_path: str
    artifact_kind: str
    identifiers: Mapping[str, str] = field(default_factory=dict)
    sha256: str | None = None
    required_for_historical_replay: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", ArtifactStage(self.stage))
        alias = str(self.root_alias).strip()
        kind = str(self.artifact_kind).strip()
        if not alias:
            raise PortableModelError("DataLocator root_alias cannot be empty")
        if not kind:
            raise PortableModelError("DataLocator artifact_kind cannot be empty")
        raw = str(self.relative_path).replace("\\", "/").strip()
        path = PurePosixPath(raw)
        if not raw or path.is_absolute() or ".." in path.parts:
            raise PortableModelError(
                "DataLocator relative_path must be a non-empty portable relative path"
            )
        if len(path.parts) and ":" in path.parts[0]:
            raise PortableModelError("DataLocator cannot contain a Windows drive prefix")
        object.__setattr__(self, "root_alias", alias)
        object.__setattr__(self, "artifact_kind", kind)
        object.__setattr__(self, "relative_path", path.as_posix())
        if self.sha256 is not None:
            token = str(self.sha256).strip().lower()
            if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
                raise PortableModelError("DataLocator sha256 must be 64 lowercase hex characters")
            object.__setattr__(self, "sha256", token)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "root_alias": self.root_alias,
            "relative_path": self.relative_path,
            "artifact_kind": self.artifact_kind,
            "identifiers": {str(k): str(v) for k, v in self.identifiers.items()},
            "sha256": self.sha256,
            "required_for_historical_replay": bool(self.required_for_historical_replay),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataLocator":
        return cls(
            stage=str(payload["stage"]),
            root_alias=str(payload["root_alias"]),
            relative_path=str(payload["relative_path"]),
            artifact_kind=str(payload["artifact_kind"]),
            identifiers={str(k): str(v) for k, v in dict(payload.get("identifiers", {})).items()},
            sha256=payload.get("sha256"),
            required_for_historical_replay=bool(
                payload.get("required_for_historical_replay", False)
            ),
        )


@dataclass(frozen=True)
class ScalarTransform:
    """One invertible affine normalization used by a fitted model.

    Model coordinates are ``(physical - offset) / scale`` and inverse
    normalization is ``model * scale + offset``.  Identity is offset=0, scale=1.
    More specialized future method payloads may carry additional metadata, but
    the common E0-7 executable transform remains explicit and invertible.
    """

    offset: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        offset = float(self.offset)
        scale = float(self.scale)
        if offset != offset or abs(offset) == float("inf"):
            raise PortableModelError("Normalization offset must be finite")
        if scale != scale or abs(scale) == float("inf") or scale == 0.0:
            raise PortableModelError("Normalization scale must be finite and non-zero")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)

    def normalize(self, value: Any):
        return (value - self.offset) / self.scale

    def denormalize(self, value: Any):
        return value * self.scale + self.offset

    def to_dict(self) -> dict[str, float]:
        return {"offset": self.offset, "scale": self.scale}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScalarTransform":
        return cls(offset=float(payload.get("offset", 0.0)), scale=float(payload.get("scale", 1.0)))


@dataclass(frozen=True)
class NormalizationContract:
    """Immutable fitted normalization/renormalization carried by the model."""

    inputs: Mapping[str, ScalarTransform] = field(default_factory=dict)
    outputs: Mapping[str, ScalarTransform] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": {str(k): v.to_dict() for k, v in self.inputs.items()},
            "outputs": {str(k): v.to_dict() for k, v in self.outputs.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizationContract":
        return cls(
            inputs={
                str(k): ScalarTransform.from_dict(v)
                for k, v in dict(payload.get("inputs", {})).items()
            },
            outputs={
                str(k): ScalarTransform.from_dict(v)
                for k, v in dict(payload.get("outputs", {})).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class RuntimeInputSchema:
    """Method-declared physical-unit control/disturbance interface."""

    controls: tuple[str, ...] = ()
    disturbances: tuple[str, ...] = ()
    observed_outputs: tuple[str, ...] = ()
    optional_controls: tuple[str, ...] = ()
    optional_disturbances: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        groups: Sequence[tuple[str, ...]] = (
            self.controls,
            self.disturbances,
            self.observed_outputs,
            self.optional_controls,
            self.optional_disturbances,
        )
        for values in groups:
            if len(values) != len(set(values)) or any(not str(v).strip() for v in values):
                raise PortableModelError("RuntimeInputSchema names must be non-empty and unique")
        if set(self.controls) & set(self.optional_controls):
            raise PortableModelError("A control cannot be both required and optional")
        if set(self.disturbances) & set(self.optional_disturbances):
            raise PortableModelError("A disturbance cannot be both required and optional")

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": list(self.controls),
            "disturbances": list(self.disturbances),
            "observed_outputs": list(self.observed_outputs),
            "optional_controls": list(self.optional_controls),
            "optional_disturbances": list(self.optional_disturbances),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeInputSchema":
        return cls(
            controls=tuple(str(v) for v in payload.get("controls", ())),
            disturbances=tuple(str(v) for v in payload.get("disturbances", ())),
            observed_outputs=tuple(str(v) for v in payload.get("observed_outputs", ())),
            optional_controls=tuple(str(v) for v in payload.get("optional_controls", ())),
            optional_disturbances=tuple(str(v) for v in payload.get("optional_disturbances", ())),
        )


@dataclass(frozen=True)
class MethodPayloadDescriptor:
    """Generic E0-7 envelope for a future E.1/E.2/E.3/E.4 payload."""

    family: ModelFamily | str
    method_id: str
    deployment_kind: str
    payload_schema_version: str = E0_7_PAYLOAD_ENVELOPE_VERSION
    embedded_root: str = "payload"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", ModelFamily(self.family))
        if not str(self.method_id).strip() or not str(self.deployment_kind).strip():
            raise PortableModelError("method_id and deployment_kind cannot be empty")
        root = PurePosixPath(str(self.embedded_root).replace("\\", "/"))
        if root.is_absolute() or ".." in root.parts:
            raise PortableModelError("embedded_root must be bundle-relative")
        object.__setattr__(self, "embedded_root", root.as_posix())

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "method_id": self.method_id,
            "deployment_kind": self.deployment_kind,
            "payload_schema_version": self.payload_schema_version,
            "embedded_root": self.embedded_root,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MethodPayloadDescriptor":
        return cls(
            family=str(payload["family"]),
            method_id=str(payload["method_id"]),
            deployment_kind=str(payload["deployment_kind"]),
            payload_schema_version=str(
                payload.get("payload_schema_version", E0_7_PAYLOAD_ENVELOPE_VERSION)
            ),
            embedded_root=str(payload.get("embedded_root", "payload")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PHVACZoneModelSpec:
    zone_id: str
    artifact_relpath: str
    input_transform: str = "absolute_value"
    target_allocation: str = "equal_across_aggregate_zones"
    source_locator: DataLocator | None = None

    def __post_init__(self) -> None:
        if not str(self.zone_id).strip():
            raise PortableModelError("PHVAC zone_id cannot be empty")
        rel = PurePosixPath(str(self.artifact_relpath).replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise PortableModelError("PHVAC artifact_relpath must be bundle-relative")
        object.__setattr__(self, "artifact_relpath", rel.as_posix())
        if self.input_transform not in {"absolute_value", "identity"}:
            raise PortableModelError(
                f"Unsupported common PHVAC input transform {self.input_transform!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "artifact_relpath": self.artifact_relpath,
            "input_transform": self.input_transform,
            "target_allocation": self.target_allocation,
            "source_locator": None if self.source_locator is None else self.source_locator.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PHVACZoneModelSpec":
        source = payload.get("source_locator")
        return cls(
            zone_id=str(payload["zone_id"]),
            artifact_relpath=str(payload["artifact_relpath"]),
            input_transform=str(payload.get("input_transform", "absolute_value")),
            target_allocation=str(
                payload.get("target_allocation", "equal_across_aggregate_zones")
            ),
            source_locator=None if source is None else DataLocator.from_dict(source),
        )


@dataclass(frozen=True)
class PHVACBundleContract:
    """Current equal-allocation Phase-C PHVAC deployment contract.

    ``missing_model_count`` is intentionally derived as N - number of embedded
    PHVAC models.  M=0 means every aggregate zone has a PHVAC model.
    """

    total_aggregate_zones: int
    zone_models: tuple[PHVACZoneModelSpec, ...] = ()
    target_allocation: str = "equal_across_aggregate_zones"
    building_reconstruction: str = "sum_with_equal_allocation_completion"

    def __post_init__(self) -> None:
        n = int(self.total_aggregate_zones)
        if n < 1:
            raise PortableModelError("PHVAC total_aggregate_zones must be >= 1")
        if len(self.zone_models) > n:
            raise PortableModelError("PHVAC model count cannot exceed aggregate-zone count")
        zones = [item.zone_id for item in self.zone_models]
        if len(zones) != len(set(zones)):
            raise PortableModelError("PHVAC zone model IDs must be unique")
        if self.target_allocation != "equal_across_aggregate_zones":
            raise PortableModelError(
                "E0-7 v1 production PHVAC contract requires equal_across_aggregate_zones"
            )
        object.__setattr__(self, "total_aggregate_zones", n)

    @property
    def available_model_count(self) -> int:
        return len(self.zone_models)

    @property
    def missing_model_count(self) -> int:
        return self.total_aggregate_zones - self.available_model_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_aggregate_zones": self.total_aggregate_zones,
            "available_model_count": self.available_model_count,
            "missing_model_count": self.missing_model_count,
            "zone_models": [item.to_dict() for item in self.zone_models],
            "target_allocation": self.target_allocation,
            "building_reconstruction": self.building_reconstruction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PHVACBundleContract":
        obj = cls(
            total_aggregate_zones=int(payload["total_aggregate_zones"]),
            zone_models=tuple(
                PHVACZoneModelSpec.from_dict(item)
                for item in payload.get("zone_models", ())
            ),
            target_allocation=str(
                payload.get("target_allocation", "equal_across_aggregate_zones")
            ),
            building_reconstruction=str(
                payload.get(
                    "building_reconstruction", "sum_with_equal_allocation_completion"
                )
            ),
        )
        # Persisted counts are redundancy checks, never authority.
        if "available_model_count" in payload and int(payload["available_model_count"]) != obj.available_model_count:
            raise PortableModelError("Persisted PHVAC available_model_count is inconsistent")
        if "missing_model_count" in payload and int(payload["missing_model_count"]) != obj.missing_model_count:
            raise PortableModelError("Persisted PHVAC missing_model_count is inconsistent")
        return obj


@dataclass(frozen=True)
class BundleFileRecord:
    relative_path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BundleFileRecord":
        return cls(
            relative_path=str(payload["relative_path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
        )


@dataclass(frozen=True)
class PortableModelManifest:
    """Immutable scientific manifest for one post-estimation model artifact."""

    model_id: str
    payload: MethodPayloadDescriptor
    runtime_inputs: RuntimeInputSchema
    normalization: NormalizationContract = field(default_factory=NormalizationContract)
    lineage: tuple[DataLocator, ...] = ()
    state_contract: Mapping[str, Any] = field(default_factory=dict)
    discretization_contract: Mapping[str, Any] = field(default_factory=dict)
    phvac: PHVACBundleContract | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    files: tuple[BundleFileRecord, ...] = ()
    bundle_schema_version: str = E0_7_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.model_id).strip():
            raise PortableModelError("Portable model_id cannot be empty")
        if self.bundle_schema_version != E0_7_BUNDLE_SCHEMA_VERSION:
            raise PortableModelError(
                f"Unsupported E0-7 bundle schema {self.bundle_schema_version!r}"
            )
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise PortableModelError("Bundle file records must have unique paths")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_schema_version": self.bundle_schema_version,
            "model_id": self.model_id,
            "payload": self.payload.to_dict(),
            "runtime_inputs": self.runtime_inputs.to_dict(),
            "normalization": self.normalization.to_dict(),
            "lineage": [item.to_dict() for item in self.lineage],
            "state_contract": dict(self.state_contract),
            "discretization_contract": dict(self.discretization_contract),
            "phvac": None if self.phvac is None else self.phvac.to_dict(),
            "provenance": dict(self.provenance),
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortableModelManifest":
        return cls(
            bundle_schema_version=str(
                payload.get("bundle_schema_version", E0_7_BUNDLE_SCHEMA_VERSION)
            ),
            model_id=str(payload["model_id"]),
            payload=MethodPayloadDescriptor.from_dict(payload["payload"]),
            runtime_inputs=RuntimeInputSchema.from_dict(payload["runtime_inputs"]),
            normalization=NormalizationContract.from_dict(payload.get("normalization", {})),
            lineage=tuple(DataLocator.from_dict(item) for item in payload.get("lineage", ())),
            state_contract=dict(payload.get("state_contract", {})),
            discretization_contract=dict(payload.get("discretization_contract", {})),
            phvac=(
                None
                if payload.get("phvac") is None
                else PHVACBundleContract.from_dict(payload["phvac"])
            ),
            provenance=dict(payload.get("provenance", {})),
            files=tuple(BundleFileRecord.from_dict(item) for item in payload.get("files", ())),
        )
