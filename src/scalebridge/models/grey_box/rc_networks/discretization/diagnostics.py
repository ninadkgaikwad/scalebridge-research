from __future__ import annotations

"""Optional E0-5 per-step diagnostics. The normal execution path bypasses this."""

from math import ceil

import numpy as np
import torch

from .contracts import DiscretizationConfig, StepDiagnostics
from .linear_oracle import ExactZOHLinearIntegrator
from .linear_system import LinearRCStateSpace, TorchLinearRCStateSpace
from .solver_registry import normalize_solver_name


_STABILITY_RADIUS = {
    "euler": 2.0,
    "rk4": 2.785293563405282,
}


def modal_rate_max_per_s(system: LinearRCStateSpace) -> float:
    """Largest eigenvalue of C^-1/2 L_CC C^-1/2 for the linear RC graph."""

    c_root_inv = 1.0 / np.sqrt(system.C)
    symmetric = (c_root_inv[:, None] * system.L_CC) * c_root_inv[None, :]
    symmetric = 0.5 * (symmetric + symmetric.T)
    eig = np.linalg.eigvalsh(symmetric)
    return float(max(0.0, np.max(eig))) if eig.size else 0.0


def _finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.all(torch.isfinite(item)).item()) for item in tensors)


def build_step_diagnostics(
    *,
    config: DiscretizationConfig,
    sample_dt_s: float,
    linear_system_numpy: LinearRCStateSpace,
    linear_system_torch: TorchLinearRCStateSpace,
    initial_state: torch.Tensor,
    boundary: torch.Tensor,
    thermal: torch.Tensor,
    numerical_state: torch.Tensor,
    local_error_tensors: tuple[torch.Tensor, ...] = (),
) -> StepDiagnostics:
    if not config.diagnostics_per_step:
        return StepDiagnostics(enabled=False)

    finite_input = _finite(initial_state, boundary, thermal)
    finite_output = _finite(numerical_state)

    exact = ExactZOHLinearIntegrator(linear_system_torch).step(
        initial_state,
        boundary,
        thermal,
        sample_dt_s=sample_dt_s,
    )
    diff = numerical_state - exact
    oracle_linf = float(torch.max(torch.abs(diff)).detach().cpu().item())
    oracle_l2 = float(torch.linalg.vector_norm(diff).detach().cpu().item())

    solver = normalize_solver_name(config.solver)
    rate = modal_rate_max_per_s(linear_system_numpy)
    radius = _STABILITY_RADIUS.get(solver)
    stability_available = radius is not None
    metric = None
    limit = None
    passed = None
    recommended = None
    notes: list[str] = []

    if radius is not None:
        h = float(sample_dt_s) / int(config.substeps)
        metric = h * rate
        limit = float(config.stability_safety_factor) * radius
        passed = bool(metric <= limit)
        denom = max(limit, np.finfo(float).tiny)
        recommended = max(1, int(ceil(float(sample_dt_s) * rate / denom)))
    else:
        notes.append(
            f"No frozen analytical negative-real-axis stability radius for solver {solver!r}"
        )

    local_error = None
    if local_error_tensors:
        local_error = max(
            float(torch.max(torch.abs(item)).detach().cpu().item())
            for item in local_error_tensors
        )

    return StepDiagnostics(
        enabled=True,
        finite_input=finite_input,
        finite_output=finite_output,
        exact_oracle_available=True,
        exact_oracle_linf_abs=oracle_linf,
        exact_oracle_l2=oracle_l2,
        stability_check_available=stability_available,
        modal_rate_max_per_s=rate,
        method_stability_radius=radius,
        stability_metric=metric,
        stability_limit_with_safety=limit,
        stability_passed=passed,
        recommended_minimum_substeps=recommended,
        local_error_linf_abs=local_error,
        notes=tuple(notes),
    )
