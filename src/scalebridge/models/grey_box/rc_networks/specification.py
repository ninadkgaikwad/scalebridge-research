from __future__ import annotations

"""Backend-neutral specification types for the Phase E0 generic RC compiler.

Mathematical authority
----------------------
* ScaleBridge_PhaseE0_E0-3_Generic_Elemental_RC_Contract_v2.tex
* ScaleBridge_PhaseE0_E0-3_Current_RC_Flavours_v2.tex
* ScaleBridge_PhaseE0_E0-3_Generic_Spatial_RC_Compiler_Contract_v1.tex

This module defines model specification only. It contains no training loop,
numerical integrator, Phase-D split logic, or paper-specific building IDs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class RCCompileError(ValueError):
    """Raised when a requested RC model violates the frozen compiler contract."""


class SpatialMode(str, Enum):
    """ScaleBridge spatial physics/information modes."""

    IND = "ind"
    DEP1 = "dep1"
    DEP2 = "dep2"

    @classmethod
    def normalize(cls, value: str | "SpatialMode") -> "SpatialMode":
        if isinstance(value, cls):
            return value
        token = str(value).strip().lower()
        aliases = {
            "ind": cls.IND,
            "independent": cls.IND,
            "dep1": cls.DEP1,
            "dependent1": cls.DEP1,
            "dependent_1": cls.DEP1,
            "dep2": cls.DEP2,
            "dependent2": cls.DEP2,
            "dependent_2": cls.DEP2,
        }
        try:
            return aliases[token]
        except KeyError as exc:
            raise RCCompileError(f"Unknown spatial mode: {value!r}") from exc


class ParameterStatus(str, Enum):
    FIXED = "fixed"
    ESTIMATED = "estimated"


class AllocationMode(str, Enum):
    ESTIMATED = "estimated"
    FIXED = "fixed"
    NEUTRAL_FIXED = "neutral_fixed"


class HeatPortGroup(str, Enum):
    """Within-zone routing family for atomic canonical thermal-power ports."""

    CONVECTIVE = "convective"
    RADIATIVE = "radiative"


@dataclass(frozen=True, order=True)
class StateNode:
    zone_id: str
    state_label: str

    def __post_init__(self) -> None:
        if not self.zone_id or not self.state_label:
            raise RCCompileError("StateNode zone_id/state_label cannot be empty")

    @property
    def key(self) -> str:
        return f"{self.zone_id}::{self.state_label}"


@dataclass(frozen=True, order=True)
class BoundaryNode:
    boundary_label: str

    def __post_init__(self) -> None:
        if not self.boundary_label:
            raise RCCompileError("BoundaryNode boundary_label cannot be empty")

    @property
    def key(self) -> str:
        return f"boundary::{self.boundary_label}"


@dataclass(frozen=True, order=True)
class ThermalPort:
    """Effective zone-local atomic thermal-power input after any DEP2 allocation."""

    zone_id: str
    signal: str

    @property
    def key(self) -> str:
        return f"{self.zone_id}::{self.signal}"


@dataclass(frozen=True)
class ZoneAdjacency:
    zone_a: str
    zone_b: str

    def canonical(self) -> tuple[str, str]:
        if not self.zone_a or not self.zone_b:
            raise RCCompileError("Adjacency zone IDs cannot be empty")
        if self.zone_a == self.zone_b:
            raise RCCompileError(f"Zone self-adjacency is invalid: {self.zone_a!r}")
        return tuple(sorted((self.zone_a, self.zone_b)))  # type: ignore[return-value]


@dataclass(frozen=True)
class ConnectionRule:
    """Unordered state-label pair expanded across every zone adjacency."""

    state_a: str
    state_b: str

    def canonical(self) -> tuple[str, str]:
        if not self.state_a or not self.state_b:
            raise RCCompileError("Connection-rule state labels cannot be empty")
        return tuple(sorted((self.state_a, self.state_b)))  # type: ignore[return-value]


@dataclass(frozen=True)
class ParameterSharingRule:
    """Tie compatible physical parameter instances to one master parameter.

    Select either explicit ``instance_ids`` or a semantic ``family`` with an
    optional modeled-zone subset.  A family selector applies to instances whose
    ``zone_scope`` lies wholly within the supplied zone subset.
    """

    name: str
    instance_ids: tuple[str, ...] = ()
    family: str | None = None
    zone_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise RCCompileError("Sharing rule name cannot be empty")
        if bool(self.instance_ids) == bool(self.family):
            raise RCCompileError(
                "Sharing rule must specify exactly one of instance_ids or family"
            )


@dataclass(frozen=True)
class ParameterConfig:
    """Optional metadata attached to one master parameter."""

    status: ParameterStatus
    initial_value: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    prior: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound > self.upper_bound:
                raise RCCompileError("Parameter lower_bound cannot exceed upper_bound")
        if self.lower_bound is not None and self.initial_value < self.lower_bound:
            raise RCCompileError("Parameter initial_value is below lower_bound")
        if self.upper_bound is not None and self.initial_value > self.upper_bound:
            raise RCCompileError("Parameter initial_value is above upper_bound")


@dataclass(frozen=True)
class AllocationFamilySpec:
    """One DEP2 all-to-one thermal-power allocation family.

    ``weights`` is the authoritative normalized Phase-B row ``w`` over all
    modeled zones.  ``participating_zone_ids`` identifies zones to which this
    family is physically applicable. Non-participating zones are treated as
    fixed lambda=0 in estimated/fixed modes.

    Multiple compatible atomic signals may share one family by listing them in
    ``signals``.  Their participating-zone set must therefore be identical.
    """

    name: str
    signals: tuple[str, ...]
    weights: Mapping[str, float]
    mode: AllocationMode
    participating_zone_ids: tuple[str, ...] = ()
    fixed_lambdas: Mapping[str, float] = field(default_factory=dict)
    lower_bounds: Mapping[str, float] = field(default_factory=dict)
    upper_bounds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RCCompileError("Allocation family name cannot be empty")
        if not self.signals or len(self.signals) != len(set(self.signals)):
            raise RCCompileError(
                "Allocation family signals must be non-empty and unique"
            )
        if not self.weights:
            raise RCCompileError("Allocation family requires authoritative weights")


@dataclass(frozen=True)
class RCCompilerSpec:
    """Complete backend-neutral input to the spatial RC compiler."""

    flavour: str
    zone_ids: tuple[str, ...]
    mode: SpatialMode | str
    adjacency: tuple[ZoneAdjacency, ...] | None = None
    connection_rules: tuple[ConnectionRule, ...] = ()
    zone_port_availability: Mapping[str, Sequence[str]] = field(default_factory=dict)
    port_groups: Mapping[str, HeatPortGroup | str] = field(default_factory=dict)
    parameter_sharing: tuple[ParameterSharingRule, ...] = ()
    dep2_allocations: tuple[AllocationFamilySpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.flavour:
            raise RCCompileError("flavour cannot be empty")
        if not self.zone_ids:
            raise RCCompileError("zone_ids cannot be empty")
        if len(self.zone_ids) != len(set(self.zone_ids)):
            raise RCCompileError("zone_ids must be unique")
        object.__setattr__(self, "mode", SpatialMode.normalize(self.mode))
