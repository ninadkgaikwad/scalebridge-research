from __future__ import annotations

"""Differentiable PyTorch physical realization for E0-6."""

from math import log

import torch
from torch import nn

from ..compiler import CompiledRCModel
from ..discretization.linear_oracle import ExactZOHLinearIntegrator
from ..discretization.linear_system import TorchLinearRCStateSpace
from ..specification import HeatPortGroup, SpatialMode, StateNode
from .contracts import BackendAdapterError, BackendMatrices, ScalarTransformKind
from .schema import build_parameterization_plan, build_physical_parameterization_plan


def _as_batch(value, width: int, label: str, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, bool]:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=dtype, device=device)
    if tensor.ndim == 1:
        if tensor.shape[0] != width:
            raise BackendAdapterError(
                f"{label} width must be {width}, got {tuple(tensor.shape)}"
            )
        return tensor.unsqueeze(0), True
    if tensor.ndim == 2 and tensor.shape[1] == width:
        return tensor, False
    raise BackendAdapterError(
        f"{label} must have shape ({width},) or (batch,{width}); got {tuple(tensor.shape)}"
    )


class TorchRCBackend(nn.Module):
    """Own one canonical raw vector and realize live RC physics in PyTorch."""

    def __init__(
        self,
        model: CompiledRCModel,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model = model
        self.plan = build_parameterization_plan(model)
        self.physical_plan = build_physical_parameterization_plan(model)
        self.raw = nn.Parameter(
            torch.zeros(self.plan.raw_dimension, dtype=dtype, device=torch.device(device))
        )
        self._aggregate_index = {
            signal: i for i, signal in enumerate(self.plan.aggregate_signal_order)
        }
        self._allocation_by_family = {
            item.family_name: item for item in self.plan.allocation_parameters
        }

        self.register_buffer(
            "_incidence",
            torch.as_tensor(model.incidence, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_observation",
            torch.as_tensor(model.observation, dtype=dtype, device=device),
            persistent=False,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.raw.dtype

    @property
    def device(self) -> torch.device:
        return self.raw.device

    def zero_raw(self) -> torch.Tensor:
        return torch.zeros_like(self.raw)

    def _rho(self, raw: torch.Tensor | None) -> torch.Tensor:
        rho = self.raw if raw is None else raw
        if not isinstance(rho, torch.Tensor):
            rho = torch.as_tensor(rho, dtype=self.dtype, device=self.device)
        rho = rho.to(dtype=self.dtype, device=self.device).reshape(-1)
        if tuple(rho.shape) != (self.plan.raw_dimension,):
            raise BackendAdapterError(
                f"Raw coordinate shape must be {(self.plan.raw_dimension,)}, got {tuple(rho.shape)}"
            )
        return rho

    def master_values(self, raw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        rho = self._rho(raw)
        values: dict[str, torch.Tensor] = {}

        for spec in self.plan.scalar_parameters:
            baseline = torch.as_tensor(spec.baseline, dtype=self.dtype, device=self.device)
            if spec.transform is ScalarTransformKind.FIXED:
                value = baseline
            elif spec.transform is ScalarTransformKind.POSITIVE_EXP:
                assert spec.raw_index is not None
                value = baseline * torch.exp(rho[spec.raw_index])
            elif spec.transform is ScalarTransformKind.SHIFTED_EXP:
                assert spec.raw_index is not None and spec.lower_bound is not None
                lower = torch.as_tensor(spec.lower_bound, dtype=self.dtype, device=self.device)
                value = lower + (baseline - lower) * torch.exp(rho[spec.raw_index])
            elif spec.transform is ScalarTransformKind.BOUNDED_SIGMOID:
                assert spec.raw_index is not None
                assert spec.lower_bound is not None and spec.upper_bound is not None
                lower = torch.as_tensor(spec.lower_bound, dtype=self.dtype, device=self.device)
                upper = torch.as_tensor(spec.upper_bound, dtype=self.dtype, device=self.device)
                z0 = (baseline - lower) / (upper - lower)
                a0 = torch.log(z0 / (1.0 - z0))
                value = lower + (upper - lower) * torch.sigmoid(a0 + rho[spec.raw_index])
            else:  # pragma: no cover
                raise BackendAdapterError(f"Unhandled transform {spec.transform!r}")
            values[spec.master_id] = value

        for group in self.plan.simplex_parameters:
            out = [torch.as_tensor(v, dtype=self.dtype, device=self.device) for v in group.baseline]
            positions = group.trainable_positions
            if positions:
                if len(positions) == 1:
                    out[positions[0]] = torch.as_tensor(group.residual, dtype=self.dtype, device=self.device)
                else:
                    anchor = group.anchor_position
                    logits = []
                    raw_iter = iter(group.raw_indices)
                    for pos in positions:
                        base_share = group.baseline[pos] / group.residual
                        base_logit = torch.as_tensor(log(base_share), dtype=self.dtype, device=self.device)
                        offset = torch.zeros((), dtype=self.dtype, device=self.device) if pos == anchor else rho[next(raw_iter)]
                        logits.append(base_logit + offset)
                    shares = torch.softmax(torch.stack(logits), dim=0)
                    for pos, share in zip(positions, shares):
                        out[pos] = torch.as_tensor(group.residual, dtype=self.dtype, device=self.device) * share
            for master_id, value in zip(group.master_ids, out):
                values[master_id] = value

        missing = set(self.plan.master_order) - set(values)
        if missing:
            raise BackendAdapterError(f"Backend failed to realize masters: {sorted(missing)}")
        return values

    def master_vector(self, raw: torch.Tensor | None = None) -> torch.Tensor:
        values = self.master_values(raw)
        return torch.stack([values[mid] for mid in self.plan.master_order])


    def physical_decision_vector(self, raw: torch.Tensor | None = None) -> torch.Tensor:
        """Map transformed SciML coordinates to the v2 physical-Theta decision vector."""
        values = self.master_values(raw)
        lambdas = self.allocation_lambdas(raw)
        inverse_master = {idx: mid for mid, idx in self.physical_plan.master_decision_index.items()}
        inverse_alloc = {idx: key for key, idx in self.physical_plan.allocation_p_index.items()}
        entries = []
        zone_index = {z: i for i, z in enumerate(self.model.spec.zone_ids)}
        for coord in self.physical_plan.coordinates:
            if coord.index in inverse_master:
                entries.append(values[inverse_master[coord.index]])
            elif coord.index in inverse_alloc:
                family, zone = inverse_alloc[coord.index]
                spec = self.model.allocation_families[family]
                lam = lambdas[family][zone_index[zone]]
                entries.append(torch.as_tensor(float(spec.weights[zone]), dtype=self.dtype, device=self.device) * lam)
            else:  # pragma: no cover
                raise BackendAdapterError(f"Unmapped physical decision coordinate {coord.name!r}")
        return torch.stack(entries) if entries else torch.zeros(0, dtype=self.dtype, device=self.device)

    def allocation_lambdas(self, raw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        rho = self._rho(raw)
        results: dict[str, torch.Tensor] = {}
        for group in self.plan.allocation_parameters:
            entries = [
                torch.as_tensor(0.0 if fixed is None else fixed, dtype=self.dtype, device=self.device)
                for fixed in group.fixed_lambdas
            ]
            positions = group.estimated_positions
            if positions:
                if len(positions) == 1:
                    pvals = [torch.as_tensor(group.residual, dtype=self.dtype, device=self.device)]
                else:
                    anchor = group.anchor_position
                    logits = []
                    raw_iter = iter(group.raw_indices)
                    for pos in positions:
                        base_share = group.baseline_p[pos] / group.residual
                        base_logit = torch.as_tensor(log(base_share), dtype=self.dtype, device=self.device)
                        offset = torch.zeros((), dtype=self.dtype, device=self.device) if pos == anchor else rho[next(raw_iter)]
                        logits.append(base_logit + offset)
                    pvals = list(torch.as_tensor(group.residual, dtype=self.dtype, device=self.device) * torch.softmax(torch.stack(logits), dim=0))
                for pos, pval in zip(positions, pvals):
                    entries[pos] = pval / torch.as_tensor(group.weights[pos], dtype=self.dtype, device=self.device)
            results[group.family_name] = torch.stack(entries)
        return results

    def matrices(self, raw: torch.Tensor | None = None) -> BackendMatrices:
        values = self.master_values(raw)
        i2m = self.model.parameter_registry.instance_to_master
        c = torch.stack(
            [values[i2m[self.model.state_capacitance_parameter[node.key]]] for node in self.model.state_nodes]
        )
        g = torch.stack(
            [1.0 / values[i2m[edge.parameter_instance_id]] for edge in self.model.resistance_edges]
        )
        D = self._incidence
        L = (D * g.unsqueeze(0)) @ D.transpose(0, 1)
        n = self.model.state_dimension
        lcc = L[:n, :n]
        lcb = L[:n, n:]
        gamma = self._gamma(values)
        h = self._observation
        a = -lcc / c.unsqueeze(1)
        bt = -lcb / c.unsqueeze(1)
        bq = gamma / c.unsqueeze(1)
        return BackendMatrices(c, lcc, lcb, gamma, h, a, bt, bq)

    def _gamma(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        gamma = torch.zeros(
            (self.model.state_dimension, len(self.model.thermal_ports)),
            dtype=self.dtype,
            device=self.device,
        )
        sidx = self.model.state_index
        i2m = self.model.parameter_registry.instance_to_master
        for j, port in enumerate(self.model.thermal_ports):
            group = self.model.port_groups[port.signal]
            zone = port.zone_id
            if self.model.flavour.routing_kind == "all_to_air" or group is HeatPortGroup.CONVECTIVE:
                gamma[sidx[StateNode(zone, "a").key], j] = 1.0
            elif self.model.flavour.routing_kind == "eta_r":
                eta = values[i2m[self.model.routing_parameter_ids[(zone, "eta_r")]]]
                gamma[sidx[StateNode(zone, "a").key], j] = 1.0 - eta
                gamma[sidx[StateNode(zone, "m").key], j] = eta
            elif self.model.flavour.routing_kind == "gamma_r_3way":
                for label, state in zip(("gamma_a_r", "gamma_e_r", "gamma_m_r"), ("a", "e", "m")):
                    value = values[i2m[self.model.routing_parameter_ids[(zone, label)]]]
                    gamma[sidx[StateNode(zone, state).key], j] = value
            else:  # pragma: no cover
                raise BackendAdapterError(f"Unhandled routing kind {self.model.flavour.routing_kind!r}")
        return gamma

    def effective_thermal(self, local_thermal, aggregate_thermal=None, *, raw: torch.Tensor | None = None) -> torch.Tensor:
        local, squeeze = _as_batch(
            local_thermal,
            len(self.model.thermal_ports),
            "local_thermal",
            dtype=self.dtype,
            device=self.device,
        )
        if self.model.spec.mode is not SpatialMode.DEP2:
            if aggregate_thermal is not None:
                aggregate_tensor = torch.as_tensor(aggregate_thermal)
                if aggregate_tensor.numel():
                    raise BackendAdapterError("Aggregate thermal inputs are only valid in DEP2")
            return local.squeeze(0) if squeeze else local

        if aggregate_thermal is None:
            raise BackendAdapterError("DEP2 backend realization requires aggregate_thermal")
        aggregate, _ = _as_batch(
            aggregate_thermal,
            len(self.plan.aggregate_signal_order),
            "aggregate_thermal",
            dtype=self.dtype,
            device=self.device,
        )
        if aggregate.shape[0] != local.shape[0]:
            raise BackendAdapterError("Local and aggregate thermal batch sizes must match")
        lambdas = self.allocation_lambdas(raw)
        zone_index = {z: i for i, z in enumerate(self.model.spec.zone_ids)}
        columns = []
        for j, port in enumerate(self.model.thermal_ports):
            if port.signal == "qac":
                columns.append(local[:, j])
            else:
                family = self.model.signal_to_allocation_family[port.signal]
                lam = lambdas[family][zone_index[port.zone_id]]
                columns.append(lam * aggregate[:, self._aggregate_index[port.signal]])
        out = torch.stack(columns, dim=1)
        return out.squeeze(0) if squeeze else out

    def linear_system(self, raw: torch.Tensor | None = None) -> TorchLinearRCStateSpace:
        m = self.matrices(raw)
        return TorchLinearRCStateSpace(
            C=m.C,
            L_CC=m.L_CC,
            L_CB=m.L_CB,
            Gamma=m.Gamma,
            H=m.H,
            A=m.A,
            B_boundary=m.B_boundary,
            B_thermal=m.B_thermal,
            state_keys=tuple(node.key for node in self.model.state_nodes),
            boundary_labels=tuple(node.boundary_label for node in self.model.boundary_nodes),
            thermal_port_keys=tuple(port.key for port in self.model.thermal_ports),
        )

    def rhs_effective(self, state, boundary, effective_thermal, *, raw: torch.Tensor | None = None) -> torch.Tensor:
        system = self.linear_system(raw)
        x, squeeze = _as_batch(state, system.state_dimension, "state", dtype=self.dtype, device=self.device)
        tb, _ = _as_batch(boundary, system.boundary_dimension, "boundary", dtype=self.dtype, device=self.device)
        q, _ = _as_batch(effective_thermal, system.thermal_dimension, "effective_thermal", dtype=self.dtype, device=self.device)
        if not (x.shape[0] == tb.shape[0] == q.shape[0]):
            raise BackendAdapterError("State/boundary/thermal batch sizes must match")
        out = system.rhs(x, tb, q)
        return out.squeeze(0) if squeeze else out

    def rhs(self, state, boundary, local_thermal, aggregate_thermal=None, *, raw: torch.Tensor | None = None) -> torch.Tensor:
        q = self.effective_thermal(local_thermal, aggregate_thermal, raw=raw)
        return self.rhs_effective(state, boundary, q, raw=raw)

    def step(self, solver: str, state, boundary, local_thermal, aggregate_thermal=None, *, sample_dt_s: float, substeps: int = 1, raw: torch.Tensor | None = None) -> torch.Tensor:
        key = str(solver).strip().lower().replace("-", "_")
        if substeps < 1:
            raise BackendAdapterError("substeps must be >= 1")
        h = float(sample_dt_s) / int(substeps)
        x = state if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=self.dtype, device=self.device)
        x = x.to(dtype=self.dtype, device=self.device)
        q = self.effective_thermal(local_thermal, aggregate_thermal, raw=raw)
        for _ in range(int(substeps)):
            if key == "euler":
                x = x + h * self.rhs_effective(x, boundary, q, raw=raw)
            elif key == "rk2":
                k1 = self.rhs_effective(x, boundary, q, raw=raw)
                k2 = self.rhs_effective(x + 0.5 * h * k1, boundary, q, raw=raw)
                x = x + h * k2
            elif key == "rk4":
                k1 = self.rhs_effective(x, boundary, q, raw=raw)
                k2 = self.rhs_effective(x + 0.5 * h * k1, boundary, q, raw=raw)
                k3 = self.rhs_effective(x + 0.5 * h * k2, boundary, q, raw=raw)
                k4 = self.rhs_effective(x + h * k3, boundary, q, raw=raw)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif key in {"exact_zoh_linear", "exact_zoh", "exact"}:
                # A fresh E0-5 exact integrator is intentional: live trainable
                # matrices must never reuse a transition cached at older rho.
                system = self.linear_system(raw)
                integrator = ExactZOHLinearIntegrator(system)
                x = integrator.step(x, torch.as_tensor(boundary, dtype=self.dtype, device=self.device), q, sample_dt_s=h)
            else:
                raise BackendAdapterError(
                    f"E0-6 common solver must be one of euler/rk2/rk4/exact_zoh_linear; got {solver!r}"
                )
        return x

    def forward(self, state, boundary, local_thermal, aggregate_thermal=None) -> torch.Tensor:
        return self.rhs(state, boundary, local_thermal, aggregate_thermal)
