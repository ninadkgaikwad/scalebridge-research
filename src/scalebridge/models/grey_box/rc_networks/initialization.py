from __future__ import annotations

"""E0-4 runtime initialization policy and state lifting.

Mathematical authority
----------------------
ScaleBridge_PhaseE0_E0-4_Runtime_State_Input_Contract_v1.tex, E0-4B.

This module resolves one starting temperature per modeled zone and lifts that
zone vector into the compiler-owned RC state ordering.  It contains no time
integration and does not alter E0-3 physics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from .compiler import CompiledRCModel
from .specification import RCCompileError


DEFAULT_INITIAL_TEMPERATURE_C = 22.0


class InitializationPolicy(str, Enum):
    AUTO = "auto"
    USER_FIXED = "user_fixed"
    OBSERVED = "observed"
    SETPOINT = "setpoint"
    DEFAULT = "default"

    @classmethod
    def normalize(cls, value: str | "InitializationPolicy") -> "InitializationPolicy":
        if isinstance(value, cls):
            return value
        token = str(value).strip().lower()
        aliases = {
            "auto": cls.AUTO,
            "user": cls.USER_FIXED,
            "fixed": cls.USER_FIXED,
            "user_fixed": cls.USER_FIXED,
            "observed": cls.OBSERVED,
            "measurement": cls.OBSERVED,
            "setpoint": cls.SETPOINT,
            "thermostat": cls.SETPOINT,
            "default": cls.DEFAULT,
        }
        try:
            return aliases[token]
        except KeyError as exc:
            raise RCCompileError(f"Unknown initialization policy: {value!r}") from exc


class InitializationSource(str, Enum):
    USER_FIXED = "user_fixed"
    OBSERVED = "observed"
    SETPOINT = "setpoint"
    DEFAULT = "default"


@dataclass(frozen=True)
class ZoneInitializationEvidence:
    """Initialization evidence available for one modeled zone."""

    observed_air_temperature_c: float | None = None
    scalar_setpoint_c: float | None = None
    heating_setpoint_c: float | None = None
    cooling_setpoint_c: float | None = None
    active_mode: str | None = None


@dataclass(frozen=True)
class InitializationRequest:
    """Runtime initialization policy configuration.

    ``user_temperatures_c`` allows zone-specific overrides.  A global user
    temperature may also be supplied and is used for any zone without a
    zone-specific override.
    """

    policy: InitializationPolicy | str = InitializationPolicy.AUTO
    user_temperatures_c: Mapping[str, float] = field(default_factory=dict)
    global_user_temperature_c: float | None = None
    default_temperature_c: float = DEFAULT_INITIAL_TEMPERATURE_C

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", InitializationPolicy.normalize(self.policy))
        if not np.isfinite(float(self.default_temperature_c)):
            raise RCCompileError("default initialization temperature must be finite")
        if self.global_user_temperature_c is not None and not np.isfinite(
            float(self.global_user_temperature_c)
        ):
            raise RCCompileError("global user initialization temperature must be finite")
        for zone, value in self.user_temperatures_c.items():
            if not zone:
                raise RCCompileError("user initialization zone ID cannot be empty")
            if not np.isfinite(float(value)):
                raise RCCompileError(
                    f"user initialization temperature for {zone!r} must be finite"
                )


@dataclass(frozen=True)
class ZoneInitializationResolution:
    zone_id: str
    value_c: float
    source: InitializationSource
    detail: str


@dataclass(frozen=True)
class InitializationResult:
    policy: InitializationPolicy
    default_temperature_c: float
    resolved_by_zone: Mapping[str, ZoneInitializationResolution]
    zone_vector_c: np.ndarray
    lifting_matrix: np.ndarray
    state: np.ndarray

    @property
    def source_by_zone(self) -> Mapping[str, str]:
        return {z: item.source.value for z, item in self.resolved_by_zone.items()}


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _normalize_active_mode(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"", "none", "unknown", "off", "idle", "deadband"}:
        return None
    if token in {"heat", "heating"}:
        return "heating"
    if token in {"cool", "cooling"}:
        return "cooling"
    raise RCCompileError(f"Unsupported active thermostat mode: {value!r}")


def resolve_setpoint(evidence: ZoneInitializationEvidence) -> tuple[float, str] | None:
    """Resolve one thermostat target according to the locked E0-4 policy."""

    scalar = _finite_or_none(evidence.scalar_setpoint_c)
    if scalar is not None:
        return scalar, "scalar_setpoint"

    heating = _finite_or_none(evidence.heating_setpoint_c)
    cooling = _finite_or_none(evidence.cooling_setpoint_c)
    mode = _normalize_active_mode(evidence.active_mode)

    if mode == "heating":
        if heating is None:
            return None
        return heating, "active_heating_setpoint"
    if mode == "cooling":
        if cooling is None:
            return None
        return cooling, "active_cooling_setpoint"

    if heating is not None and cooling is not None:
        return 0.5 * (heating + cooling), "heating_cooling_midpoint"
    if heating is not None:
        return heating, "single_heating_setpoint"
    if cooling is not None:
        return cooling, "single_cooling_setpoint"
    return None


def build_initialization_lifting(model: CompiledRCModel) -> np.ndarray:
    """Construct S0 so every state copies its modeled zone's resolved T0*."""

    zone_index = {zone: i for i, zone in enumerate(model.spec.zone_ids)}
    lifting = np.zeros((model.state_dimension, len(model.spec.zone_ids)), dtype=float)
    for i, node in enumerate(model.state_nodes):
        try:
            lifting[i, zone_index[node.zone_id]] = 1.0
        except KeyError as exc:  # defensive: compiler should make this impossible
            raise RCCompileError(
                f"Compiled state references unknown modeled zone {node.zone_id!r}"
            ) from exc

    identity = model.observation @ lifting
    expected = np.eye(model.output_dimension, len(model.spec.zone_ids), dtype=float)
    if identity.shape != expected.shape or not np.allclose(identity, expected, atol=0.0, rtol=0.0):
        raise RCCompileError("Initialization lifting violates H S0 = I")
    return lifting


def _user_value_for_zone(request: InitializationRequest, zone_id: str) -> float | None:
    if zone_id in request.user_temperatures_c:
        return float(request.user_temperatures_c[zone_id])
    if request.global_user_temperature_c is not None:
        return float(request.global_user_temperature_c)
    return None


def _resolve_one_zone(
    zone_id: str,
    evidence: ZoneInitializationEvidence,
    request: InitializationRequest,
) -> ZoneInitializationResolution:
    policy = request.policy
    user_value = _user_value_for_zone(request, zone_id)
    observed = _finite_or_none(evidence.observed_air_temperature_c)
    default_value = float(request.default_temperature_c)

    if policy is InitializationPolicy.USER_FIXED:
        if user_value is None:
            raise RCCompileError(
                f"user_fixed initialization requires a finite user value for zone {zone_id!r}"
            )
        return ZoneInitializationResolution(
            zone_id, user_value, InitializationSource.USER_FIXED, "explicit_user_value"
        )

    if policy is InitializationPolicy.OBSERVED:
        if observed is None:
            raise RCCompileError(
                f"observed initialization requires a finite air temperature for zone {zone_id!r}"
            )
        return ZoneInitializationResolution(
            zone_id, observed, InitializationSource.OBSERVED, "observed_air_temperature"
        )

    if policy is InitializationPolicy.SETPOINT:
        setpoint = resolve_setpoint(evidence)
        if setpoint is None:
            raise RCCompileError(
                f"setpoint initialization requires a resolvable thermostat target for zone {zone_id!r}"
            )
        return ZoneInitializationResolution(
            zone_id, setpoint[0], InitializationSource.SETPOINT, setpoint[1]
        )

    if policy is InitializationPolicy.DEFAULT:
        return ZoneInitializationResolution(
            zone_id, default_value, InitializationSource.DEFAULT, "configured_default"
        )

    if policy is not InitializationPolicy.AUTO:  # pragma: no cover - normalize guards this
        raise RCCompileError(f"Unsupported initialization policy {policy!r}")

    if user_value is not None:
        return ZoneInitializationResolution(
            zone_id, user_value, InitializationSource.USER_FIXED, "auto_user_override"
        )
    if observed is not None:
        return ZoneInitializationResolution(
            zone_id, observed, InitializationSource.OBSERVED, "auto_observed_air_temperature"
        )
    setpoint = resolve_setpoint(evidence)
    if setpoint is not None:
        return ZoneInitializationResolution(
            zone_id, setpoint[0], InitializationSource.SETPOINT, f"auto_{setpoint[1]}"
        )
    return ZoneInitializationResolution(
        zone_id, default_value, InitializationSource.DEFAULT, "auto_configured_default"
    )


def initialize_runtime_state(
    model: CompiledRCModel,
    evidence_by_zone: Mapping[str, ZoneInitializationEvidence] | None = None,
    *,
    request: InitializationRequest | None = None,
) -> InitializationResult:
    """Resolve T0* for every modeled zone and construct X0 = S0 T0*."""

    evidence_by_zone = dict(evidence_by_zone or {})
    request = request or InitializationRequest()

    unknown_evidence = set(evidence_by_zone) - set(model.spec.zone_ids)
    if unknown_evidence:
        raise RCCompileError(
            f"Initialization evidence references unknown modeled zones: {sorted(unknown_evidence)}"
        )
    unknown_user = set(request.user_temperatures_c) - set(model.spec.zone_ids)
    if unknown_user:
        raise RCCompileError(
            f"User initialization references unknown modeled zones: {sorted(unknown_user)}"
        )

    resolved: dict[str, ZoneInitializationResolution] = {}
    for zone in model.spec.zone_ids:
        evidence = evidence_by_zone.get(zone, ZoneInitializationEvidence())
        resolved[zone] = _resolve_one_zone(zone, evidence, request)

    zone_vector = np.asarray([resolved[z].value_c for z in model.spec.zone_ids], dtype=float)
    if not np.all(np.isfinite(zone_vector)):
        raise RCCompileError("Resolved initialization vector contains non-finite values")

    lifting = build_initialization_lifting(model)
    state = lifting @ zone_vector
    if not np.all(np.isfinite(state)):
        raise RCCompileError("Initialized RC state contains non-finite values")

    observed = model.observation @ state
    if not np.allclose(observed, zone_vector, atol=0.0, rtol=0.0):
        raise RCCompileError("Initialized state violates H X0 = T0*")

    return InitializationResult(
        policy=request.policy,
        default_temperature_c=float(request.default_temperature_c),
        resolved_by_zone=resolved,
        zone_vector_c=zone_vector,
        lifting_matrix=lifting,
        state=state,
    )
