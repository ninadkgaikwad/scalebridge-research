from __future__ import annotations

"""Neuromancer facade over the E0-6 PyTorch physical parameter owner."""

from dataclasses import dataclass, field

import torch

from ..discretization.solver_registry import build_neuromancer_integrator
from .contracts import BackendAdapterError
from .torch_backend import TorchRCBackend


def _require_odesystem():
    try:
        from neuromancer.dynamics.ode import ODESystem
    except Exception as exc:  # pragma: no cover - environment dependent
        raise BackendAdapterError("Neuromancer is required for the E0-6 Neuromancer backend") from exc
    return ODESystem


def build_neuromancer_trainable_rc_ode(backend: TorchRCBackend):
    ODESystem = _require_odesystem()

    class TrainableRCODESystem(ODESystem):
        def __init__(self):
            insize = (
                backend.model.state_dimension
                + len(backend.model.boundary_nodes)
                + len(backend.model.thermal_ports)
                + len(backend.plan.aggregate_signal_order)
            )
            super().__init__(insize=insize, outsize=backend.model.state_dimension)
            # Registering the backend as a child module ensures the exact same
            # nn.Parameter owner is visible through the Neuromancer facade.
            self.backend = backend

        def ode_equations(self, x, boundary, local_thermal, aggregate_thermal):
            return self.backend.rhs(x, boundary, local_thermal, aggregate_thermal)

    return TrainableRCODESystem()


@dataclass
class NeuromancerRCBackend:
    torch_backend: TorchRCBackend
    _integrator_cache: dict[tuple[str, float], object] = field(default_factory=dict, init=False)

    @property
    def raw(self):
        return self.torch_backend.raw

    def ode_system(self):
        return build_neuromancer_trainable_rc_ode(self.torch_backend)

    def rhs(self, state, boundary, local_thermal, aggregate_thermal=None):
        return self.torch_backend.rhs(state, boundary, local_thermal, aggregate_thermal)

    def step(self, solver: str, state, boundary, local_thermal, aggregate_thermal=None, *, sample_dt_s: float, substeps: int = 1):
        key = str(solver).strip().lower().replace("-", "_")
        if key in {"exact", "exact_zoh", "exact_zoh_linear"}:
            return self.torch_backend.step(
                "exact_zoh_linear",
                state,
                boundary,
                local_thermal,
                aggregate_thermal,
                sample_dt_s=sample_dt_s,
                substeps=substeps,
            )
        if key not in {"euler", "rk2", "rk4"}:
            raise BackendAdapterError(
                "E0-6 cross-backend Neuromancer parity is required for euler/rk2/rk4/exact_zoh_linear"
            )
        if substeps < 1:
            raise BackendAdapterError("substeps must be >= 1")
        h = float(sample_dt_s) / int(substeps)
        cache_key = (key, h)
        integrator = self._integrator_cache.get(cache_key)
        if integrator is None:
            integrator = build_neuromancer_integrator(key, self.ode_system(), h=h)
            self._integrator_cache[cache_key] = integrator

        dtype = self.torch_backend.dtype
        device = self.torch_backend.device
        x = state if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=dtype, device=device)
        tb = boundary if isinstance(boundary, torch.Tensor) else torch.as_tensor(boundary, dtype=dtype, device=device)
        local = local_thermal if isinstance(local_thermal, torch.Tensor) else torch.as_tensor(local_thermal, dtype=dtype, device=device)
        if aggregate_thermal is None:
            aggregate = torch.empty(
                (0,), dtype=dtype, device=device
            ) if not len(self.torch_backend.plan.aggregate_signal_order) else None
            if aggregate is None:
                raise BackendAdapterError("DEP2 Neuromancer realization requires aggregate_thermal")
        else:
            aggregate = aggregate_thermal if isinstance(aggregate_thermal, torch.Tensor) else torch.as_tensor(aggregate_thermal, dtype=dtype, device=device)

        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        if tb.ndim == 1:
            tb = tb.unsqueeze(0)
        if local.ndim == 1:
            local = local.unsqueeze(0)
        if aggregate.ndim == 1:
            aggregate = aggregate.unsqueeze(0)

        out = x
        for _ in range(int(substeps)):
            out = integrator(out, tb, local, aggregate)
        return out.squeeze(0) if squeeze else out
