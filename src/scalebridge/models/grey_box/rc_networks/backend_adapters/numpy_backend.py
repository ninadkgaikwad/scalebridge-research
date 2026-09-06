from __future__ import annotations

"""Independent NumPy/SciPy numerical reference for E0-6."""

from dataclasses import dataclass
from math import exp, log
from typing import Mapping

import numpy as np

from ..compiler import CompiledRCModel
from ..specification import HeatPortGroup, SpatialMode, StateNode
from .contracts import BackendAdapterError, BackendMatrices, ScalarTransformKind
from .schema import build_parameterization_plan, build_physical_parameterization_plan


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def _as_batch(value, width: int, label: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != width:
            raise BackendAdapterError(f"{label} width must be {width}, got {arr.shape}")
        return arr[None, :], True
    if arr.ndim == 2 and arr.shape[1] == width:
        return arr, False
    raise BackendAdapterError(
        f"{label} must have shape ({width},) or (batch,{width}); got {arr.shape}"
    )


@dataclass
class NumpyRCBackend:
    model: CompiledRCModel

    def __post_init__(self) -> None:
        self.plan = build_parameterization_plan(self.model)
        self.physical_plan = build_physical_parameterization_plan(self.model)
        self._master_lookup = {m.master_id: m for m in self.model.parameter_registry.masters}
        self._aggregate_index = {
            signal: i for i, signal in enumerate(self.plan.aggregate_signal_order)
        }
        self._allocation_by_family = {
            item.family_name: item for item in self.plan.allocation_parameters
        }

    def zero_raw(self) -> np.ndarray:
        return np.zeros(self.plan.raw_dimension, dtype=float)

    def _validate_raw(self, raw) -> np.ndarray:
        rho = np.asarray(raw, dtype=float).reshape(-1)
        if rho.shape != (self.plan.raw_dimension,):
            raise BackendAdapterError(
                f"Raw coordinate shape must be {(self.plan.raw_dimension,)}, got {rho.shape}"
            )
        if not np.all(np.isfinite(rho)):
            raise BackendAdapterError("Raw coordinates contain non-finite values")
        return rho

    def master_values(self, raw) -> dict[str, float]:
        rho = self._validate_raw(raw)
        values: dict[str, float] = {}

        for spec in self.plan.scalar_parameters:
            if spec.transform is ScalarTransformKind.FIXED:
                value = spec.baseline
            elif spec.transform is ScalarTransformKind.POSITIVE_EXP:
                assert spec.raw_index is not None
                value = spec.baseline * exp(float(rho[spec.raw_index]))
            elif spec.transform is ScalarTransformKind.SHIFTED_EXP:
                assert spec.raw_index is not None and spec.lower_bound is not None
                value = spec.lower_bound + (spec.baseline - spec.lower_bound) * exp(
                    float(rho[spec.raw_index])
                )
            elif spec.transform is ScalarTransformKind.BOUNDED_SIGMOID:
                assert spec.raw_index is not None
                assert spec.lower_bound is not None and spec.upper_bound is not None
                z0 = (spec.baseline - spec.lower_bound) / (
                    spec.upper_bound - spec.lower_bound
                )
                a0 = log(z0 / (1.0 - z0))
                value = spec.lower_bound + (spec.upper_bound - spec.lower_bound) * _sigmoid(
                    a0 + float(rho[spec.raw_index])
                )
            else:  # pragma: no cover - defensive
                raise BackendAdapterError(f"Unhandled transform {spec.transform!r}")
            values[spec.master_id] = float(value)

        for group in self.plan.simplex_parameters:
            positions = group.trainable_positions
            out = list(group.baseline)
            if positions:
                if len(positions) == 1:
                    out[positions[0]] = group.residual
                else:
                    anchor = group.anchor_position
                    logits: list[float] = []
                    raw_iter = iter(group.raw_indices)
                    for pos in positions:
                        base_share = group.baseline[pos] / group.residual
                        offset = 0.0 if pos == anchor else float(rho[next(raw_iter)])
                        logits.append(log(base_share) + offset)
                    shift = max(logits)
                    exps = [exp(x - shift) for x in logits]
                    denom = sum(exps)
                    for pos, numer in zip(positions, exps):
                        out[pos] = group.residual * numer / denom
            for master_id, value in zip(group.master_ids, out):
                values[master_id] = float(value)

        missing = set(self.plan.master_order) - set(values)
        if missing:
            raise BackendAdapterError(f"Backend failed to realize masters: {sorted(missing)}")
        return values


    def physical_decision_vector(self, raw) -> np.ndarray:
        """Map transformed reference coordinates to the v2 physical-Theta decision vector."""
        values = self.master_values(raw)
        lambdas = self.allocation_lambdas(raw)
        inverse_master = {idx: mid for mid, idx in self.physical_plan.master_decision_index.items()}
        inverse_alloc = {idx: key for key, idx in self.physical_plan.allocation_p_index.items()}
        zone_index = {z: i for i, z in enumerate(self.model.spec.zone_ids)}
        out = np.empty(self.physical_plan.decision_dimension, dtype=float)
        for coord in self.physical_plan.coordinates:
            if coord.index in inverse_master:
                out[coord.index] = float(values[inverse_master[coord.index]])
            elif coord.index in inverse_alloc:
                family, zone = inverse_alloc[coord.index]
                spec = self.model.allocation_families[family]
                out[coord.index] = float(spec.weights[zone]) * float(lambdas[family][zone_index[zone]])
            else:  # pragma: no cover
                raise BackendAdapterError(f"Unmapped physical decision coordinate {coord.name!r}")
        return out

    def allocation_lambdas(self, raw) -> dict[str, np.ndarray]:
        rho = self._validate_raw(raw)
        results: dict[str, np.ndarray] = {}
        for group in self.plan.allocation_parameters:
            lam = np.zeros(len(group.zone_ids), dtype=float)
            for i, fixed in enumerate(group.fixed_lambdas):
                if fixed is not None:
                    lam[i] = fixed
            positions = group.estimated_positions
            if positions:
                if len(positions) == 1:
                    p_values = {positions[0]: group.residual}
                else:
                    anchor = group.anchor_position
                    logits: list[float] = []
                    raw_iter = iter(group.raw_indices)
                    for pos in positions:
                        base_share = group.baseline_p[pos] / group.residual
                        offset = 0.0 if pos == anchor else float(rho[next(raw_iter)])
                        logits.append(log(base_share) + offset)
                    shift = max(logits)
                    exps = [exp(x - shift) for x in logits]
                    denom = sum(exps)
                    p_values = {
                        pos: group.residual * numer / denom
                        for pos, numer in zip(positions, exps)
                    }
                for pos, pval in p_values.items():
                    lam[pos] = pval / group.weights[pos]
            weighted_mass = float(np.dot(np.asarray(group.weights), lam))
            if abs(weighted_mass - 1.0) > 1e-9:
                raise BackendAdapterError(
                    f"Allocation family {group.family_name!r} violates A_g B_g = 1: {weighted_mass}"
                )
            results[group.family_name] = lam
        return results

    def matrices(self, raw) -> BackendMatrices:
        master_values = self.master_values(raw)
        instance_to_master = self.model.parameter_registry.instance_to_master

        c = np.asarray(
            [
                master_values[instance_to_master[self.model.state_capacitance_parameter[node.key]]]
                for node in self.model.state_nodes
            ],
            dtype=float,
        )
        g = np.asarray(
            [
                1.0 / master_values[instance_to_master[edge.parameter_instance_id]]
                for edge in self.model.resistance_edges
            ],
            dtype=float,
        )
        D = np.asarray(self.model.incidence, dtype=float)
        L = (D * g[None, :]) @ D.T
        n = self.model.state_dimension
        lcc = L[:n, :n]
        lcb = L[:n, n:]
        gamma = self._gamma(master_values)
        h = np.asarray(self.model.observation, dtype=float).copy()
        a = -lcc / c[:, None]
        bt = -lcb / c[:, None]
        bq = gamma / c[:, None]
        return BackendMatrices(c, lcc, lcb, gamma, h, a, bt, bq)

    def _gamma(self, master_values: Mapping[str, float]) -> np.ndarray:
        gamma = np.zeros((self.model.state_dimension, len(self.model.thermal_ports)), dtype=float)
        sidx = self.model.state_index
        instance_to_master = self.model.parameter_registry.instance_to_master

        for j, port in enumerate(self.model.thermal_ports):
            group = self.model.port_groups[port.signal]
            zone = port.zone_id
            if self.model.flavour.routing_kind == "all_to_air" or group is HeatPortGroup.CONVECTIVE:
                gamma[sidx[StateNode(zone, "a").key], j] = 1.0
            elif self.model.flavour.routing_kind == "eta_r":
                instance_id = self.model.routing_parameter_ids[(zone, "eta_r")]
                eta = master_values[instance_to_master[instance_id]]
                gamma[sidx[StateNode(zone, "a").key], j] = 1.0 - eta
                gamma[sidx[StateNode(zone, "m").key], j] = eta
            elif self.model.flavour.routing_kind == "gamma_r_3way":
                for label, state in zip(("gamma_a_r", "gamma_e_r", "gamma_m_r"), ("a", "e", "m")):
                    instance_id = self.model.routing_parameter_ids[(zone, label)]
                    value = master_values[instance_to_master[instance_id]]
                    gamma[sidx[StateNode(zone, state).key], j] = value
            else:  # pragma: no cover - compiler would already reject
                raise BackendAdapterError(
                    f"Unhandled routing kind {self.model.flavour.routing_kind!r}"
                )
        return gamma

    def effective_thermal(self, raw, local_thermal, aggregate_thermal=None) -> np.ndarray:
        local, squeeze = _as_batch(local_thermal, len(self.model.thermal_ports), "local_thermal")
        if self.model.spec.mode is not SpatialMode.DEP2:
            if aggregate_thermal is not None and np.asarray(aggregate_thermal).size:
                raise BackendAdapterError("Aggregate thermal inputs are only valid in DEP2")
            return local[0] if squeeze else local

        agg_width = len(self.plan.aggregate_signal_order)
        if aggregate_thermal is None:
            raise BackendAdapterError("DEP2 backend realization requires aggregate_thermal")
        aggregate, _ = _as_batch(aggregate_thermal, agg_width, "aggregate_thermal")
        if aggregate.shape[0] != local.shape[0]:
            raise BackendAdapterError("Local and aggregate thermal batch sizes must match")

        lambdas = self.allocation_lambdas(raw)
        zone_index = {z: i for i, z in enumerate(self.model.spec.zone_ids)}
        out = np.zeros_like(local)
        for j, port in enumerate(self.model.thermal_ports):
            if port.signal == "qac":
                out[:, j] = local[:, j]
            else:
                family = self.model.signal_to_allocation_family[port.signal]
                lam = lambdas[family][zone_index[port.zone_id]]
                out[:, j] = lam * aggregate[:, self._aggregate_index[port.signal]]
        return out[0] if squeeze else out

    def rhs_effective(self, raw, state, boundary, effective_thermal) -> np.ndarray:
        matrices = self.matrices(raw)
        x, squeeze = _as_batch(state, self.model.state_dimension, "state")
        tb, _ = _as_batch(boundary, matrices.B_boundary.shape[1], "boundary")
        q, _ = _as_batch(effective_thermal, len(self.model.thermal_ports), "effective_thermal")
        if not (x.shape[0] == tb.shape[0] == q.shape[0]):
            raise BackendAdapterError("State/boundary/thermal batch sizes must match")
        out = x @ matrices.A.T + tb @ matrices.B_boundary.T + q @ matrices.B_thermal.T
        return out[0] if squeeze else out

    def rhs(self, raw, state, boundary, local_thermal, aggregate_thermal=None) -> np.ndarray:
        q = self.effective_thermal(raw, local_thermal, aggregate_thermal)
        return self.rhs_effective(raw, state, boundary, q)

    def step(self, solver: str, raw, state, boundary, local_thermal, aggregate_thermal=None, *, sample_dt_s: float, substeps: int = 1) -> np.ndarray:
        key = str(solver).strip().lower().replace("-", "_")
        if substeps < 1:
            raise BackendAdapterError("substeps must be >= 1")
        h = float(sample_dt_s) / int(substeps)
        x = np.asarray(state, dtype=float)
        q = self.effective_thermal(raw, local_thermal, aggregate_thermal)
        for _ in range(int(substeps)):
            if key == "euler":
                x = x + h * self.rhs_effective(raw, x, boundary, q)
            elif key == "rk2":
                k1 = self.rhs_effective(raw, x, boundary, q)
                k2 = self.rhs_effective(raw, x + 0.5 * h * k1, boundary, q)
                x = x + h * k2
            elif key == "rk4":
                k1 = self.rhs_effective(raw, x, boundary, q)
                k2 = self.rhs_effective(raw, x + 0.5 * h * k1, boundary, q)
                k3 = self.rhs_effective(raw, x + 0.5 * h * k2, boundary, q)
                k4 = self.rhs_effective(raw, x + h * k3, boundary, q)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif key in {"exact_zoh_linear", "exact_zoh", "exact"}:
                x = self._exact_step(raw, x, boundary, q, h)
            else:
                raise BackendAdapterError(
                    f"E0-6 common solver must be one of euler/rk2/rk4/exact_zoh_linear; got {solver!r}"
                )
        return x

    def _exact_step(self, raw, state, boundary, effective_thermal, dt: float) -> np.ndarray:
        try:
            from scipy.linalg import expm
        except Exception as exc:  # pragma: no cover - environment dependent
            raise BackendAdapterError("SciPy is required for independent NumPy exact-ZOH parity") from exc
        matrices = self.matrices(raw)
        x, squeeze = _as_batch(state, self.model.state_dimension, "state")
        tb, _ = _as_batch(boundary, matrices.B_boundary.shape[1], "boundary")
        q, _ = _as_batch(effective_thermal, len(self.model.thermal_ports), "effective_thermal")
        B = np.concatenate((matrices.B_boundary, matrices.B_thermal), axis=1)
        n, m = matrices.A.shape[0], B.shape[1]
        aug = np.zeros((n + m, n + m), dtype=float)
        aug[:n, :n] = matrices.A
        aug[:n, n:] = B
        transition = expm(aug * float(dt))
        ad = transition[:n, :n]
        bd = transition[:n, n:]
        u = np.concatenate((tb, q), axis=1)
        out = x @ ad.T + u @ bd.T
        return out[0] if squeeze else out
