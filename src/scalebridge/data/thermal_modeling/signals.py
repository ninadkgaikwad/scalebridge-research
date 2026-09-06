# -*- coding: utf-8 -*-
"""Signal registry, grouping rules, and Phase D usability classification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Sequence

from .constants import (
    DEFAULT_INCLUDE_VISIBLE_LIGHTING_IN_ZIR,
    DEFAULT_ZERO_ABSOLUTE_TOLERANCE,
    NullableReason,
    PhaseDSignalStatus,
    SignalRole,
    SourcePhase,
)


@dataclass(frozen=True)
class SignalDefinition:
    """Global definition of one canonical Phase D signal."""

    canonical_name: str
    source_phase: SourcePhase
    source_name: str | None
    units: str | None
    role: SignalRole
    group_name: str | None = None
    auxiliary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "source_phase": self.source_phase.value,
            "source_name": self.source_name,
            "units": self.units,
            "role": self.role.value,
            "group_name": self.group_name,
            "auxiliary": self.auxiliary,
        }


@dataclass(frozen=True)
class SignalClassification:
    """Zone-specific Phase D classification and descriptive statistics."""

    status: PhaseDSignalStatus
    nullable: bool
    nullable_reason: NullableReason
    finite_count: int
    missing_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    constant_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "nullable": self.nullable,
            "nullable_reason": self.nullable_reason.value,
            "finite_count": self.finite_count,
            "missing_count": self.missing_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "constant_value": self.constant_value,
        }


QZIC_COMPONENTS = (
    "qzic_p", "qzic_l", "qzic_ee", "qzic_ge",
    "qzic_oe", "qzic_hwe", "qzic_se",
)
QZIR_COMPONENTS = (
    "qzir_p", "qzir_l", "qzir_ee", "qzir_ge",
    "qzir_oe", "qzir_hwe", "qzir_se",
)
VISIBLE_LIGHTING_COMPONENT = "qzivr_l"


def build_signal_registry(
    *,
    include_visible_lighting_in_zir: bool = DEFAULT_INCLUDE_VISIBLE_LIGHTING_IN_ZIR,
) -> dict[str, SignalDefinition]:
    """Build the canonical D1 signal registry.

    Phase B supplies temperatures. Phase C supplies regression outputs. ZIC and
    ZIR are Phase D derived groups. PHVAC remains auxiliary/provenance.
    """

    registry: dict[str, SignalDefinition] = {
        "timestamp": SignalDefinition("timestamp", SourcePhase.PHASE_B, "timestamp", None, SignalRole.TIME),
        "zone_temperature": SignalDefinition(
            "zone_temperature", SourcePhase.PHASE_B, "Zone_Air_Temperature_", "degC", SignalRole.STATE
        ),
        "outdoor_temperature": SignalDefinition(
            "outdoor_temperature", SourcePhase.PHASE_B,
            "Site_Outdoor_Air_Drybulb_Temperature_", "degC", SignalRole.DISTURBANCE
        ),
        "qsol1": SignalDefinition("qsol1", SourcePhase.PHASE_C, "predicted_QSol1", "W", SignalRole.THERMAL_INPUT),
        "qsol2": SignalDefinition("qsol2", SourcePhase.PHASE_C, "predicted_QSol2", "W", SignalRole.THERMAL_INPUT),
        "qac": SignalDefinition("qac", SourcePhase.PHASE_C, "predicted_QAC", "W", SignalRole.THERMAL_INPUT),
        "phvac": SignalDefinition(
            "phvac", SourcePhase.PHASE_C, "predicted_PHVAC", "W", SignalRole.AUXILIARY, auxiliary=True
        ),
        "zic": SignalDefinition("zic", SourcePhase.PHASE_D_DERIVED, None, "W", SignalRole.DERIVED_GROUP),
        "zir": SignalDefinition("zir", SourcePhase.PHASE_D_DERIVED, None, "W", SignalRole.DERIVED_GROUP),
    }
    for name in QZIC_COMPONENTS:
        registry[name] = SignalDefinition(
            name, SourcePhase.PHASE_C, "predicted_" + _canonical_to_phase_c(name),
            "W", SignalRole.THERMAL_INPUT, group_name="zic"
        )
    for name in QZIR_COMPONENTS:
        registry[name] = SignalDefinition(
            name, SourcePhase.PHASE_C, "predicted_" + _canonical_to_phase_c(name),
            "W", SignalRole.THERMAL_INPUT, group_name="zir"
        )
    registry[VISIBLE_LIGHTING_COMPONENT] = SignalDefinition(
        VISIBLE_LIGHTING_COMPONENT,
        SourcePhase.PHASE_C,
        "predicted_QZivr_L",
        "W",
        SignalRole.THERMAL_INPUT,
        group_name="zir" if include_visible_lighting_in_zir else None,
    )
    return registry


def _canonical_to_phase_c(name: str) -> str:
    prefix, suffix = name.split("_", maxsplit=1)
    phase_c_prefix = {"qzic": "QZic", "qzir": "QZir"}[prefix]
    return f"{phase_c_prefix}_{suffix.upper()}"


def group_components(
    group_name: str,
    *,
    include_visible_lighting_in_zir: bool = DEFAULT_INCLUDE_VISIBLE_LIGHTING_IN_ZIR,
) -> tuple[str, ...]:
    """Return canonical component membership for ZIC or ZIR."""

    normalized = group_name.lower()
    if normalized == "zic":
        return QZIC_COMPONENTS
    if normalized == "zir":
        if include_visible_lighting_in_zir:
            return (*QZIR_COMPONENTS, VISIBLE_LIGHTING_COMPONENT)
        return QZIR_COMPONENTS
    raise ValueError(f"Unsupported heat-input group: {group_name!r}")


def classify_phase_c_signal(
    values: Sequence[float | int | None] | Iterable[float | int | None],
    *,
    phase_c_applicable: bool,
    zero_absolute_tolerance: float = DEFAULT_ZERO_ABSOLUTE_TOLERANCE,
    require_complete: bool = True,
) -> SignalClassification:
    """Classify one Phase C output for Phase D.

    Locked D1 policy:
      * varying outputs are retained;
      * constant nonzero outputs are retained;
      * complete-zero outputs become nullable;
      * non-applicable models become nullable;
      * missing rows in an applicable output are validation failures.
    """

    raw = list(values)
    finite_values = [float(value) for value in raw if value is not None and isfinite(float(value))]
    missing_count = len(raw) - len(finite_values)

    if not phase_c_applicable:
        return SignalClassification(
            PhaseDSignalStatus.NULLABLE_NOT_APPLICABLE,
            True,
            NullableReason.PHASE_C_MODEL_NOT_APPLICABLE,
            len(finite_values),
            missing_count,
            _minimum(finite_values),
            _maximum(finite_values),
            _mean(finite_values),
            _constant_value(finite_values, zero_absolute_tolerance),
        )

    if not finite_values or (require_complete and missing_count > 0):
        return SignalClassification(
            PhaseDSignalStatus.VALIDATION_FAILURE,
            False,
            NullableReason.NONE,
            len(finite_values),
            missing_count,
            _minimum(finite_values),
            _maximum(finite_values),
            _mean(finite_values),
            _constant_value(finite_values, zero_absolute_tolerance),
        )

    minimum = min(finite_values)
    maximum = max(finite_values)
    mean = sum(finite_values) / len(finite_values)
    constant = maximum - minimum <= zero_absolute_tolerance

    if constant and abs(mean) <= zero_absolute_tolerance:
        return SignalClassification(
            PhaseDSignalStatus.NULLABLE_COMPLETE_ZERO,
            True,
            NullableReason.COMPLETE_ZERO_SIGNAL,
            len(finite_values), missing_count, minimum, maximum, mean, 0.0,
        )
    if constant:
        return SignalClassification(
            PhaseDSignalStatus.CONSTANT_NONZERO,
            False,
            NullableReason.NONE,
            len(finite_values), missing_count, minimum, maximum, mean, mean,
        )
    return SignalClassification(
        PhaseDSignalStatus.VARYING,
        False,
        NullableReason.NONE,
        len(finite_values), missing_count, minimum, maximum, mean, None,
    )


def _minimum(values: list[float]) -> float | None:
    return min(values) if values else None


def _maximum(values: list[float]) -> float | None:
    return max(values) if values else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _constant_value(values: list[float], tolerance: float) -> float | None:
    if not values:
        return None
    return sum(values) / len(values) if max(values) - min(values) <= tolerance else None
