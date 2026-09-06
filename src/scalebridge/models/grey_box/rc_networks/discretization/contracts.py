from __future__ import annotations

"""Frozen E0-5 common-discretization contracts."""

from dataclasses import dataclass
from typing import Any

from ..specification import RCCompileError


class DiscretizationError(RCCompileError):
    """Raised when E0-5 numerical configuration violates the frozen contract."""


@dataclass(frozen=True)
class DiscretizationConfig:
    """Configuration for one fixed-step E0-5 numerical method.

    ``sample_dt_s`` is supplied per step because the E0-5 mathematics permits
    irregular canonical intervals. ``substeps`` is the explicit number of
    equal fixed solver steps within one canonical interval.
    """

    solver: str = "rk4"
    substeps: int = 1
    diagnostics_per_step: bool = False
    stability_safety_factor: float = 0.9

    def __post_init__(self) -> None:
        token = str(self.solver).strip()
        if not token:
            raise DiscretizationError("E0-5 solver name cannot be empty")
        if isinstance(self.substeps, bool) or int(self.substeps) != self.substeps:
            raise DiscretizationError("E0-5 substeps must be an integer >= 1")
        if int(self.substeps) < 1:
            raise DiscretizationError("E0-5 substeps must be >= 1")
        eta = float(self.stability_safety_factor)
        if not (0.0 < eta <= 1.0):
            raise DiscretizationError(
                "E0-5 stability_safety_factor must lie in (0, 1]"
            )

    @property
    def normalized_solver(self) -> str:
        return str(self.solver).strip().lower().replace("-", "_").replace(" ", "_")

    def step_size_s(self, sample_dt_s: float) -> float:
        dt = validate_sample_dt(sample_dt_s)
        return dt / int(self.substeps)


def validate_sample_dt(sample_dt_s: float) -> float:
    try:
        dt = float(sample_dt_s)
    except Exception as exc:  # pragma: no cover - defensive
        raise DiscretizationError("E0-5 sample_dt_s must be finite and positive") from exc
    if not (dt > 0.0) or dt == float("inf") or dt != dt:
        raise DiscretizationError("E0-5 sample_dt_s must be finite and positive")
    return dt


@dataclass(frozen=True)
class SolverCapability:
    key: str
    backend: str
    class_name: str | None
    fixed_step: bool
    first_order_state: bool
    history_required: bool
    available: bool
    default: bool = False
    exact_linear: bool = False
    local_error_available: bool = False
    notes: str = ""


@dataclass(frozen=True)
class DiscretizationProvenance:
    solver: str
    backend: str
    sample_dt_s: float
    substeps: int
    integration_h_s: float
    input_hold: str
    dtype: str
    device: str
    diagnostics_per_step: bool


@dataclass(frozen=True)
class StepDiagnostics:
    enabled: bool
    finite_input: bool | None = None
    finite_output: bool | None = None
    exact_oracle_available: bool = False
    exact_oracle_linf_abs: float | None = None
    exact_oracle_l2: float | None = None
    stability_check_available: bool = False
    modal_rate_max_per_s: float | None = None
    method_stability_radius: float | None = None
    stability_metric: float | None = None
    stability_limit_with_safety: float | None = None
    stability_passed: bool | None = None
    recommended_minimum_substeps: int | None = None
    local_error_linf_abs: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TensorStepResult:
    state: Any
    provenance: DiscretizationProvenance
    diagnostics: StepDiagnostics
