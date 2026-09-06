from __future__ import annotations

"""Compatibility-filtered Neuromancer 1.5.6 fixed-step solver registry."""

from dataclasses import replace
import inspect
from typing import Mapping

from .contracts import DiscretizationError, SolverCapability


# Audited directly against the user's installed Neuromancer 1.5.6 API.
# Only one-step, first-order, fixed-h methods compatible with Xdot=f(X,*args)
# and held *args are included. SDE, multistep-history, second-order-state, and
# the generic DiffEqIntegrator wrapper are deliberately excluded.
_NATIVE_SOLVERS: Mapping[str, SolverCapability] = {
    "euler": SolverCapability(
        key="euler", backend="neuromancer", class_name="Euler",
        fixed_step=True, first_order_state=True, history_required=False,
        available=False, notes="Forward Euler",
    ),
    "euler_trap": SolverCapability(
        key="euler_trap", backend="neuromancer", class_name="Euler_Trap",
        fixed_step=True, first_order_state=True, history_required=False,
        available=False, notes="Euler predictor with trapezoidal corrector",
    ),
    "rk2": SolverCapability(
        key="rk2", backend="neuromancer", class_name="RK2",
        fixed_step=True, first_order_state=True, history_required=False,
        available=False, notes="Explicit midpoint RK2",
    ),
    "rk4": SolverCapability(
        key="rk4", backend="neuromancer", class_name="RK4",
        fixed_step=True, first_order_state=True, history_required=False,
        available=False, default=True, notes="Classical RK4",
    ),
    "rk4_trap": SolverCapability(
        key="rk4_trap", backend="neuromancer", class_name="RK4_Trap",
        fixed_step=True, first_order_state=True, history_required=False,
        available=False, notes="RK4 predictor with trapezoidal corrector",
    ),
    "luther": SolverCapability(
        key="luther", backend="neuromancer", class_name="Luther",
        fixed_step=True, first_order_state=True, history_required=False,
        available=False, notes="Neuromancer Luther explicit Runge-Kutta method",
    ),
    "runge_kutta_fehlberg": SolverCapability(
        key="runge_kutta_fehlberg", backend="neuromancer",
        class_name="Runge_Kutta_Fehlberg", fixed_step=True,
        first_order_state=True, history_required=False, available=False,
        local_error_available=True,
        notes="Fixed-h embedded RKF high-order state plus local-error estimate",
    ),
}

_EXACT = SolverCapability(
    key="exact_zoh_linear",
    backend="scalebridge",
    class_name="ExactZOHLinearIntegrator",
    fixed_step=True,
    first_order_state=True,
    history_required=False,
    available=True,
    exact_linear=True,
    notes="Graph-general augmented matrix-exponential ZOH propagator",
)

_ALIASES = {
    "eulertrap": "euler_trap",
    "euler_trapezoid": "euler_trap",
    "runge_kutta_fehlberg": "runge_kutta_fehlberg",
    "runge_kutta_fehlberg_45": "runge_kutta_fehlberg",
    "rkf": "runge_kutta_fehlberg",
    "rkf45": "runge_kutta_fehlberg",
    "exact": "exact_zoh_linear",
    "exact_zoh": "exact_zoh_linear",
    "linear_exact": "exact_zoh_linear",
}


def normalize_solver_name(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    token = _ALIASES.get(token, token)
    return token


def _load_nm_integrators():
    try:
        from neuromancer.dynamics import integrators
    except Exception:
        return None
    return integrators


def solver_capabilities() -> tuple[SolverCapability, ...]:
    nm = _load_nm_integrators()
    results: list[SolverCapability] = []
    for key, capability in _NATIVE_SOLVERS.items():
        available = False
        notes = capability.notes
        if nm is not None and capability.class_name is not None:
            cls = getattr(nm, capability.class_name, None)
            if inspect.isclass(cls):
                try:
                    sig = inspect.signature(cls)
                    available = "h" in sig.parameters
                except Exception:
                    available = False
            if not available:
                notes += "; installed class missing fixed-h constructor"
        results.append(replace(capability, available=available, notes=notes))
    results.append(_EXACT)
    return tuple(results)


def solver_capability(name: str) -> SolverCapability:
    key = normalize_solver_name(name)
    lookup = {item.key: item for item in solver_capabilities()}
    try:
        return lookup[key]
    except KeyError as exc:
        raise DiscretizationError(
            f"Unsupported E0-5 solver {name!r}; supported={sorted(lookup)}"
        ) from exc


def available_solver_names() -> tuple[str, ...]:
    return tuple(item.key for item in solver_capabilities() if item.available)


def build_neuromancer_integrator(name: str, block, *, h: float):
    capability = solver_capability(name)
    if capability.exact_linear:
        raise DiscretizationError("exact_zoh_linear is not a Neuromancer integrator")
    if not capability.available:
        raise DiscretizationError(
            f"Neuromancer solver {capability.key!r} is unavailable in this environment"
        )
    nm = _load_nm_integrators()
    assert nm is not None
    cls = getattr(nm, capability.class_name)
    return cls(block=block, h=float(h))
