from __future__ import annotations

"""Neuromancer ODESystem adapter for the frozen E0-3 linear RC RHS."""

from .contracts import DiscretizationError
from .linear_system import TorchLinearRCStateSpace


def _require_odesystem():
    try:
        from neuromancer.dynamics.ode import ODESystem
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DiscretizationError(
            "Neuromancer is required for E0-5 native fixed-step solvers"
        ) from exc
    return ODESystem


def build_neuromancer_rc_ode_system(system: TorchLinearRCStateSpace):
    """Return an actual installed-Neuromancer ``ODESystem`` instance.

    The class is created lazily so importing ScaleBridge does not require
    Neuromancer when only ``exact_zoh_linear`` is used.
    """

    ODESystem = _require_odesystem()

    class NeuromancerRCODESystem(ODESystem):
        def __init__(self, linear_system: TorchLinearRCStateSpace):
            super().__init__(
                insize=linear_system.state_dimension + linear_system.input_dimension,
                outsize=linear_system.state_dimension,
            )
            self.linear_system = linear_system

        def ode_equations(self, x, boundary, thermal):
            return self.linear_system.rhs(x, boundary, thermal)

    return NeuromancerRCODESystem(system)
