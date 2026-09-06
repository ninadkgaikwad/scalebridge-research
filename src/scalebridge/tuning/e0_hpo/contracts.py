from __future__ import annotations

"""Generic E0-8 HPO contracts.

Mathematical authority
----------------------
ScaleBridge_PhaseE0_E0-8_Generic_HPO_Tracking_FrozenConfiguration_Contract_v1.tex

E0-8 is deliberately method-neutral.  Method-specific hyperparameter names,
ranges, conditional semantics, fitting logic, objective science, and optional
multi-objective final-selection rules belong to E.1/E.2/E.3/E.4 providers.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


E0_8_STUDY_SCHEMA_VERSION = "phase_e0_e08_study_v1"
E0_8_FROZEN_SCHEMA_VERSION = "phase_e0_e08_frozen_hyperparameters_v1"
E0_8_SELECTION_SCHEMA_VERSION = "phase_e0_e08_data_selection_v1"


class HPOContractError(ValueError):
    """Raised when an E0-8 scientific/reproducibility contract is invalid."""


class IncompatibleResumeError(HPOContractError):
    """Raised when a persistent Optuna study cannot be resumed scientifically."""


class RecoverableTrialError(RuntimeError):
    """Method-declared trial-level failure that may be recorded and skipped.

    Infrastructure errors should not be wrapped in this exception.  They must
    propagate and stop the study so storage/tracking corruption is never hidden
    as an ordinary poor trial.
    """


class ObjectiveDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class TrialTerminalState(str, Enum):
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAILED = "FAILED"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HPOContractError("Canonical HPO metadata cannot contain NaN/inf")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise HPOContractError(
        "HPO fingerprints require JSON-stable values; unsupported value type "
        f"{type(value).__name__}"
    )


def canonical_json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    normalized = _json_value(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_canonical(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: ObjectiveDirection | str

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise HPOContractError("Objective name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", ObjectiveDirection(self.direction))

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "direction": self.direction.value}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveSpec":
        return cls(name=str(payload["name"]), direction=str(payload["direction"]))


@dataclass(frozen=True)
class HPODataSelection:
    """Exact method-declared HPO data selection, restricted to Phase-D TRAIN."""

    phase_d_contract_id: str
    phase_d_source_manifest_sha256: str
    source_partitions: tuple[str, ...]
    selection_policy: str
    selection_payload: Mapping[str, Any]
    fingerprint: str
    schema_version: str = E0_8_SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != E0_8_SELECTION_SCHEMA_VERSION:
            raise HPOContractError(f"Unsupported E0-8 data-selection schema {self.schema_version!r}")
        if not str(self.phase_d_contract_id).strip():
            raise HPOContractError("phase_d_contract_id cannot be empty")
        sha = str(self.phase_d_source_manifest_sha256).strip().lower()
        if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
            raise HPOContractError("phase_d_source_manifest_sha256 must be 64 lowercase hex characters")
        object.__setattr__(self, "phase_d_source_manifest_sha256", sha)
        parts = tuple(str(value).strip().lower() for value in self.source_partitions)
        if not parts:
            raise HPOContractError("HPO source_partitions cannot be empty")
        if any(not value for value in parts):
            raise HPOContractError("HPO source_partitions cannot contain empty values")
        object.__setattr__(self, "source_partitions", parts)
        if not str(self.selection_policy).strip():
            raise HPOContractError("selection_policy cannot be empty")
        expected = self.compute_fingerprint(
            phase_d_contract_id=self.phase_d_contract_id,
            phase_d_source_manifest_sha256=sha,
            source_partitions=parts,
            selection_policy=self.selection_policy,
            selection_payload=self.selection_payload,
        )
        if str(self.fingerprint).lower() != expected:
            raise HPOContractError("HPODataSelection fingerprint is inconsistent with its content")
        object.__setattr__(self, "fingerprint", expected)

    @staticmethod
    def compute_fingerprint(
        *,
        phase_d_contract_id: str,
        phase_d_source_manifest_sha256: str,
        source_partitions: Sequence[str],
        selection_policy: str,
        selection_payload: Mapping[str, Any],
    ) -> str:
        return sha256_canonical(
            {
                "schema_version": E0_8_SELECTION_SCHEMA_VERSION,
                "phase_d_contract_id": str(phase_d_contract_id),
                "phase_d_source_manifest_sha256": str(phase_d_source_manifest_sha256).lower(),
                "source_partitions": [str(value).lower() for value in source_partitions],
                "selection_policy": str(selection_policy),
                "selection_payload": dict(selection_payload),
            }
        )

    @classmethod
    def create(
        cls,
        *,
        phase_d_contract_id: str,
        phase_d_source_manifest_sha256: str,
        source_partitions: Sequence[str],
        selection_policy: str,
        selection_payload: Mapping[str, Any],
    ) -> "HPODataSelection":
        fingerprint = cls.compute_fingerprint(
            phase_d_contract_id=phase_d_contract_id,
            phase_d_source_manifest_sha256=phase_d_source_manifest_sha256,
            source_partitions=source_partitions,
            selection_policy=selection_policy,
            selection_payload=selection_payload,
        )
        return cls(
            phase_d_contract_id=str(phase_d_contract_id),
            phase_d_source_manifest_sha256=str(phase_d_source_manifest_sha256),
            source_partitions=tuple(str(value) for value in source_partitions),
            selection_policy=str(selection_policy),
            selection_payload=dict(selection_payload),
            fingerprint=fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase_d_contract_id": self.phase_d_contract_id,
            "phase_d_source_manifest_sha256": self.phase_d_source_manifest_sha256,
            "source_partitions": list(self.source_partitions),
            "selection_policy": self.selection_policy,
            "selection_payload": _json_value(self.selection_payload),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class StudySpec:
    study_name: str
    method_id: str
    method_family: str
    provider_version: str
    search_space_snapshot: Mapping[str, Any]
    objectives: tuple[ObjectiveSpec, ...]
    data_selection: HPODataSelection
    study_seed: int
    sampler_name: str
    pruner_name: str
    schema_version: str = E0_8_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("study_name", "method_id", "method_family", "provider_version"):
            if not str(getattr(self, field_name)).strip():
                raise HPOContractError(f"{field_name} cannot be empty")
        if self.schema_version != E0_8_STUDY_SCHEMA_VERSION:
            raise HPOContractError(f"Unsupported E0-8 study schema {self.schema_version!r}")
        if not self.objectives:
            raise HPOContractError("At least one objective is required")
        names = [item.name for item in self.objectives]
        if len(names) != len(set(names)):
            raise HPOContractError("Objective names must be unique")
        _json_value(self.search_space_snapshot)
        object.__setattr__(self, "study_seed", int(self.study_seed))

    @property
    def search_space_fingerprint(self) -> str:
        return sha256_canonical(dict(self.search_space_snapshot))

    @property
    def objective_fingerprint(self) -> str:
        return sha256_canonical([item.to_dict() for item in self.objectives])

    @property
    def fingerprint(self) -> str:
        return sha256_canonical(
            {
                "schema_version": self.schema_version,
                "study_name": self.study_name,
                "method_id": self.method_id,
                "method_family": self.method_family,
                "provider_version": self.provider_version,
                "search_space_snapshot": dict(self.search_space_snapshot),
                "objectives": [item.to_dict() for item in self.objectives],
                "data_selection_fingerprint": self.data_selection.fingerprint,
                "study_seed": self.study_seed,
            }
        )

    @property
    def study_id(self) -> str:
        return f"e08_{self.fingerprint[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_fingerprint": self.fingerprint,
            "study_name": self.study_name,
            "method_id": self.method_id,
            "method_family": self.method_family,
            "provider_version": self.provider_version,
            "study_seed": self.study_seed,
            "sampler_name": self.sampler_name,
            "pruner_name": self.pruner_name,
            "search_space_fingerprint": self.search_space_fingerprint,
            "objective_fingerprint": self.objective_fingerprint,
            "data_selection_fingerprint": self.data_selection.fingerprint,
        }


@dataclass(frozen=True)
class TrialEvaluation:
    objective_values: tuple[float, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.objective_values)
        if not values or any(not math.isfinite(value) for value in values):
            raise RecoverableTrialError("Trial objective values must be finite")
        object.__setattr__(self, "objective_values", values)
        clean_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            number = float(value)
            if not math.isfinite(number):
                raise RecoverableTrialError(f"Trial metric {key!r} is non-finite")
            clean_metrics[str(key)] = number
        object.__setattr__(self, "metrics", clean_metrics)
        _json_value(self.metadata)
        clean_artifacts: dict[str, str] = {}
        for key, value in self.artifact_paths.items():
            artifact_key = str(key).strip()
            artifact_path = str(value).strip()
            if not artifact_key or not artifact_path:
                raise RecoverableTrialError("Trial artifact keys/paths cannot be empty")
            clean_artifacts[artifact_key] = artifact_path
        object.__setattr__(self, "artifact_paths", clean_artifacts)


@dataclass(frozen=True)
class CompletedTrialView:
    trial_number: int
    params: Mapping[str, Any]
    objective_values: tuple[float, ...]
    trial_seed: int


@dataclass(frozen=True)
class FrozenHyperparameters:
    study_id: str
    study_fingerprint: str
    method_id: str
    method_family: str
    provider_version: str
    trial_number: int
    hyperparameters: Mapping[str, Any]
    objective_values: Mapping[str, float]
    data_selection_fingerprint: str
    search_space_fingerprint: str
    objective_fingerprint: str
    selection_policy: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = E0_8_FROZEN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != E0_8_FROZEN_SCHEMA_VERSION:
            raise HPOContractError(f"Unsupported frozen-hyperparameter schema {self.schema_version!r}")
        if not self.hyperparameters:
            raise HPOContractError("Frozen hyperparameter mapping cannot be empty")
        _json_value(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_fingerprint": self.study_fingerprint,
            "method_id": self.method_id,
            "method_family": self.method_family,
            "provider_version": self.provider_version,
            "trial_number": int(self.trial_number),
            "hyperparameters": _json_value(self.hyperparameters),
            "objective_values": {str(key): float(value) for key, value in self.objective_values.items()},
            "data_selection_fingerprint": self.data_selection_fingerprint,
            "search_space_fingerprint": self.search_space_fingerprint,
            "objective_fingerprint": self.objective_fingerprint,
            "selection_policy": self.selection_policy,
            "provenance": _json_value(self.provenance),
        }

    @property
    def content_sha256(self) -> str:
        return sha256_canonical(self.to_dict())
