from __future__ import annotations

"""Independent exact linear-ZOH oracle for any compiled linear RC graph."""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch

from .contracts import DiscretizationError, validate_sample_dt
from .linear_system import TorchLinearRCStateSpace


def _as_batch(value: torch.Tensor, width: int, label: str) -> tuple[torch.Tensor, bool]:
    if value.ndim == 1:
        if value.shape[0] != width:
            raise DiscretizationError(
                f"{label} width must be {width}, got {tuple(value.shape)}"
            )
        return value.unsqueeze(0), True
    if value.ndim == 2 and value.shape[1] == width:
        return value, False
    raise DiscretizationError(
        f"{label} must have shape ({width},) or (batch,{width}); got {tuple(value.shape)}"
    )


@dataclass
class ExactZOHLinearIntegrator:
    """Exact ZOH propagator based on the Van-Loan augmented exponential.

    Transition matrices are cached by ``dt`` for a fixed materialized linear
    system. The kernel is independent of Neuromancer's RK formulas.
    """

    system: TorchLinearRCStateSpace
    _transition_cache: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    def transition(self, sample_dt_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        dt = validate_sample_dt(sample_dt_s)
        cached = self._transition_cache.get(dt)
        if cached is not None:
            return cached

        n = self.system.state_dimension
        b = self.system.B
        m = int(b.shape[1])
        augmented = torch.zeros(
            (n + m, n + m), dtype=self.system.A.dtype, device=self.system.A.device
        )
        augmented[:n, :n] = self.system.A
        augmented[:n, n:] = b
        expm = torch.linalg.matrix_exp(augmented * dt)
        ad = expm[:n, :n]
        bd = expm[:n, n:]
        self._transition_cache[dt] = (ad, bd)
        return ad, bd

    def step(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
        thermal: torch.Tensor,
        *,
        sample_dt_s: float,
    ) -> torch.Tensor:
        x, squeeze = _as_batch(state, self.system.state_dimension, "state")
        tb, _ = _as_batch(boundary, self.system.boundary_dimension, "boundary")
        q, _ = _as_batch(thermal, self.system.thermal_dimension, "thermal")
        if not (x.shape[0] == tb.shape[0] == q.shape[0]):
            raise DiscretizationError("State/boundary/thermal batch sizes must match")

        ad, bd = self.transition(sample_dt_s)
        u = torch.cat((tb, q), dim=1)
        out = x @ ad.transpose(0, 1) + u @ bd.transpose(0, 1)
        return out.squeeze(0) if squeeze else out


def analytical_1r1c_step(
    temperature_c: torch.Tensor | float,
    outdoor_temperature_c: torch.Tensor | float,
    thermal_power_w: torch.Tensor | float,
    *,
    resistance_k_per_w: float,
    capacitance_j_per_k: float,
    sample_dt_s: float,
) -> torch.Tensor:
    """Scalar analytical oracle used only for focused 1R1C verification."""

    r = float(resistance_k_per_w)
    c = float(capacitance_j_per_k)
    dt = validate_sample_dt(sample_dt_s)
    if r <= 0.0 or c <= 0.0:
        raise DiscretizationError("1R1C analytical oracle requires positive R and C")
    t = torch.as_tensor(temperature_c)
    to = torch.as_tensor(outdoor_temperature_c, dtype=t.dtype, device=t.device)
    q = torch.as_tensor(thermal_power_w, dtype=t.dtype, device=t.device)
    tau = r * c
    t_inf = to + r * q
    return t_inf + (t - t_inf) * torch.exp(
        torch.as_tensor(-dt / tau, dtype=t.dtype, device=t.device)
    )
