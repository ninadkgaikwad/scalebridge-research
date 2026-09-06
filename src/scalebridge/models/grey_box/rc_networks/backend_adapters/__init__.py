"""E0-6 v2 backend adapters, physical-coordinate plans, and parity utilities."""

from .casadi_backend import CasadiRCBackend, CasadiTransformedRCBackend
from .casadi_physical_backend import CasadiPhysicalRCBackend
from .contracts import (
    AllocationParameterPlan,
    BackendAdapterError,
    BackendMatrices,
    BackendParameterizationPlan,
    FLOAT32_DERIVATIVE_TOLERANCE,
    FLOAT32_VALUE_TOLERANCE,
    FLOAT64_DERIVATIVE_TOLERANCE,
    FLOAT64_DISCRETE_TOLERANCE,
    FLOAT64_VALUE_TOLERANCE,
    ParityComparison,
    ParityTolerance,
    PhysicalDecisionCoordinate,
    PhysicalLinearConstraint,
    PhysicalParameterizationPlan,
    RawCoordinate,
    ScalarParameterPlan,
    ScalarTransformKind,
    SimplexParameterPlan,
)
from .neuromancer_backend import NeuromancerRCBackend, build_neuromancer_trainable_rc_ode
from .numpy_backend import NumpyRCBackend
from .numpy_physical_backend import NumpyPhysicalRCBackend
from .parity import normalized_linf_error
from .schema import build_parameterization_plan, build_physical_parameterization_plan
from .torch_backend import TorchRCBackend

__all__ = [
    "AllocationParameterPlan",
    "BackendAdapterError",
    "BackendMatrices",
    "BackendParameterizationPlan",
    "CasadiPhysicalRCBackend",
    "CasadiRCBackend",
    "CasadiTransformedRCBackend",
    "FLOAT32_DERIVATIVE_TOLERANCE",
    "FLOAT32_VALUE_TOLERANCE",
    "FLOAT64_DERIVATIVE_TOLERANCE",
    "FLOAT64_DISCRETE_TOLERANCE",
    "FLOAT64_VALUE_TOLERANCE",
    "NeuromancerRCBackend",
    "NumpyPhysicalRCBackend",
    "NumpyRCBackend",
    "ParityComparison",
    "ParityTolerance",
    "PhysicalDecisionCoordinate",
    "PhysicalLinearConstraint",
    "PhysicalParameterizationPlan",
    "RawCoordinate",
    "ScalarParameterPlan",
    "ScalarTransformKind",
    "SimplexParameterPlan",
    "TorchRCBackend",
    "build_neuromancer_trainable_rc_ode",
    "build_parameterization_plan",
    "build_physical_parameterization_plan",
    "normalized_linf_error",
]
