"""ScaleBridge E0-5 fixed-step common discretization engine."""

from .contracts import (
    DiscretizationConfig,
    DiscretizationError,
    DiscretizationProvenance,
    SolverCapability,
    StepDiagnostics,
    TensorStepResult,
    validate_sample_dt,
)
from .diagnostics import modal_rate_max_per_s
from .engine import CommonDiscretizationEngine, RuntimeStepResult
from .linear_oracle import ExactZOHLinearIntegrator, analytical_1r1c_step
from .linear_system import (
    LinearRCStateSpace,
    TorchLinearRCStateSpace,
    compile_linear_state_space,
)
from .neuromancer_ode import build_neuromancer_rc_ode_system
from .solver_registry import (
    available_solver_names,
    build_neuromancer_integrator,
    normalize_solver_name,
    solver_capabilities,
    solver_capability,
)

__all__ = [
    "CommonDiscretizationEngine",
    "DiscretizationConfig",
    "DiscretizationError",
    "DiscretizationProvenance",
    "ExactZOHLinearIntegrator",
    "LinearRCStateSpace",
    "RuntimeStepResult",
    "SolverCapability",
    "StepDiagnostics",
    "TensorStepResult",
    "TorchLinearRCStateSpace",
    "analytical_1r1c_step",
    "available_solver_names",
    "build_neuromancer_integrator",
    "build_neuromancer_rc_ode_system",
    "compile_linear_state_space",
    "modal_rate_max_per_s",
    "normalize_solver_name",
    "solver_capabilities",
    "solver_capability",
    "validate_sample_dt",
]
