from __future__ import annotations

"""E0-6 backend-adapter contracts and tolerance policy."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping

from ..specification import RCCompileError


class BackendAdapterError(RCCompileError):
    """Raised when an E0-6 backend realization violates the frozen contract."""


class ScalarTransformKind(str, Enum):
    FIXED = "fixed"
    POSITIVE_EXP = "positive_exp"
    SHIFTED_EXP = "shifted_exp"
    BOUNDED_SIGMOID = "bounded_sigmoid"


@dataclass(frozen=True)
class RawCoordinate:
    index: int
    name: str
    owner_kind: str
    owner_id: str
    component: str | None = None


@dataclass(frozen=True)
class ScalarParameterPlan:
    master_id: str
    baseline: float
    status: str
    transform: ScalarTransformKind
    raw_index: int | None
    lower_bound: float | None
    upper_bound: float | None
    physical_type: str
    family: str
    units: str


@dataclass(frozen=True)
class SimplexParameterPlan:
    group_id: str
    master_ids: tuple[str, ...]
    labels: tuple[str, ...]
    baseline: tuple[float, ...]
    fixed_mask: tuple[bool, ...]
    raw_indices: tuple[int, ...]
    anchor_position: int | None
    residual: float

    @property
    def trainable_positions(self) -> tuple[int, ...]:
        return tuple(i for i, fixed in enumerate(self.fixed_mask) if not fixed)


@dataclass(frozen=True)
class AllocationParameterPlan:
    family_name: str
    signal_names: tuple[str, ...]
    zone_ids: tuple[str, ...]
    weights: tuple[float, ...]
    fixed_lambdas: tuple[float | None, ...]
    baseline_p: tuple[float, ...]
    estimated_positions: tuple[int, ...]
    raw_indices: tuple[int, ...]
    anchor_position: int | None
    residual: float


@dataclass(frozen=True)
class BackendParameterizationPlan:
    raw_coordinates: tuple[RawCoordinate, ...]
    scalar_parameters: tuple[ScalarParameterPlan, ...]
    simplex_parameters: tuple[SimplexParameterPlan, ...]
    allocation_parameters: tuple[AllocationParameterPlan, ...]
    master_order: tuple[str, ...]
    aggregate_signal_order: tuple[str, ...]

    @property
    def raw_dimension(self) -> int:
        return len(self.raw_coordinates)

    @property
    def raw_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.raw_coordinates)


@dataclass(frozen=True)
class BackendMatrices:
    C: Any
    L_CC: Any
    L_CB: Any
    Gamma: Any
    H: Any
    A: Any
    B_boundary: Any
    B_thermal: Any


@dataclass(frozen=True)
class ParityTolerance:
    atol: float
    rtol: float

    def __post_init__(self) -> None:
        if not (isfinite(self.atol) and isfinite(self.rtol)):
            raise BackendAdapterError("Parity tolerances must be finite")
        if self.atol < 0.0 or self.rtol < 0.0:
            raise BackendAdapterError("Parity tolerances must be nonnegative")


FLOAT64_VALUE_TOLERANCE = ParityTolerance(atol=1e-10, rtol=1e-9)
FLOAT64_DISCRETE_TOLERANCE = ParityTolerance(atol=1e-9, rtol=1e-9)
FLOAT64_DERIVATIVE_TOLERANCE = ParityTolerance(atol=1e-8, rtol=1e-7)
FLOAT32_VALUE_TOLERANCE = ParityTolerance(atol=1e-5, rtol=1e-4)
FLOAT32_DERIVATIVE_TOLERANCE = ParityTolerance(atol=1e-4, rtol=1e-3)


@dataclass(frozen=True)
class ParityComparison:
    label: str
    passed: bool
    normalized_linf: float
    max_abs_error: float
    atol: float
    rtol: float
    metadata: Mapping[str, object] | None = None

@dataclass(frozen=True)
class PhysicalDecisionCoordinate:
    """One direct physical decision variable for CasADi/IPOPT or physical NumPy use."""

    index: int
    name: str
    owner_kind: str
    owner_id: str
    component: str | None
    baseline: float
    lower_bound: float | None
    upper_bound: float | None
    units: str
    physical_type: str


@dataclass(frozen=True)
class PhysicalLinearConstraint:
    """One linear equality/inequality row in direct physical coordinates."""

    constraint_id: str
    indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class PhysicalParameterizationPlan:
    """Direct physical-Theta decision plan used by optimization/Bayesian paths."""

    coordinates: tuple[PhysicalDecisionCoordinate, ...]
    constraints: tuple[PhysicalLinearConstraint, ...]
    master_order: tuple[str, ...]
    aggregate_signal_order: tuple[str, ...]
    fixed_master_values: Mapping[str, float]
    master_decision_index: Mapping[str, int]
    allocation_p_index: Mapping[tuple[str, str], int]

    @property
    def decision_dimension(self) -> int:
        return len(self.coordinates)

    @property
    def decision_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.coordinates)

    @property
    def initial_values(self) -> tuple[float, ...]:
        return tuple(float(item.baseline) for item in self.coordinates)

    @property
    def lower_bounds(self) -> tuple[float, ...]:
        return tuple(
            float('-inf') if item.lower_bound is None else float(item.lower_bound)
            for item in self.coordinates
        )

    @property
    def upper_bounds(self) -> tuple[float, ...]:
        return tuple(
            float('inf') if item.upper_bound is None else float(item.upper_bound)
            for item in self.coordinates
        )
