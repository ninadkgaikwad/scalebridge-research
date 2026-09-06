from __future__ import annotations

"""Transformed-coordinate CasADi reference realization for E0-6 v2 parity diagnostics."""

from dataclasses import dataclass
from math import log

import numpy as np

from ..compiler import CompiledRCModel
from ..specification import HeatPortGroup, SpatialMode, StateNode
from .contracts import BackendAdapterError, ScalarTransformKind
from .schema import build_parameterization_plan


def _require_casadi():
    try:
        import casadi as ca
    except Exception as exc:  # pragma: no cover - environment dependent
        raise BackendAdapterError("CasADi is required for the E0-6 CasADi backend") from exc
    return ca


@dataclass
class CasadiTransformedRCBackend:
    model: CompiledRCModel
    symbol_type: str = "SX"

    def __post_init__(self) -> None:
        self.ca = _require_casadi()
        token = str(self.symbol_type).strip().upper()
        if token not in {"SX", "MX"}:
            raise BackendAdapterError("CasADi symbol_type must be 'SX' or 'MX'")
        self.symbol_type = token
        self.sym = self.ca.SX if token == "SX" else self.ca.MX
        self.plan = build_parameterization_plan(self.model)
        self._aggregate_index = {
            signal: i for i, signal in enumerate(self.plan.aggregate_signal_order)
        }
        self._build_functions()

    def _scalar(self, value: float):
        return self.ca.DM(float(value))

    def _master_expressions(self, rho):
        ca = self.ca
        values = {}
        for spec in self.plan.scalar_parameters:
            baseline = self._scalar(spec.baseline)
            if spec.transform is ScalarTransformKind.FIXED:
                value = baseline
            elif spec.transform is ScalarTransformKind.POSITIVE_EXP:
                value = baseline * ca.exp(rho[spec.raw_index])
            elif spec.transform is ScalarTransformKind.SHIFTED_EXP:
                lower = self._scalar(spec.lower_bound)
                value = lower + (baseline - lower) * ca.exp(rho[spec.raw_index])
            elif spec.transform is ScalarTransformKind.BOUNDED_SIGMOID:
                lower = self._scalar(spec.lower_bound)
                upper = self._scalar(spec.upper_bound)
                z0 = (spec.baseline - spec.lower_bound) / (spec.upper_bound - spec.lower_bound)
                a0 = self._scalar(log(z0 / (1.0 - z0)))
                sigmoid = 1.0 / (1.0 + ca.exp(-(a0 + rho[spec.raw_index])))
                value = lower + (upper - lower) * sigmoid
            else:  # pragma: no cover
                raise BackendAdapterError(f"Unhandled transform {spec.transform!r}")
            values[spec.master_id] = value

        for group in self.plan.simplex_parameters:
            out = [self._scalar(v) for v in group.baseline]
            positions = group.trainable_positions
            if positions:
                if len(positions) == 1:
                    out[positions[0]] = self._scalar(group.residual)
                else:
                    anchor = group.anchor_position
                    terms = []
                    raw_iter = iter(group.raw_indices)
                    for pos in positions:
                        base_share = group.baseline[pos] / group.residual
                        offset = self._scalar(0.0) if pos == anchor else rho[next(raw_iter)]
                        terms.append(ca.exp(self._scalar(log(base_share)) + offset))
                    denom = sum(terms[1:], terms[0])
                    for pos, term in zip(positions, terms):
                        out[pos] = self._scalar(group.residual) * term / denom
            for master_id, value in zip(group.master_ids, out):
                values[master_id] = value
        return values

    def _allocation_expressions(self, rho):
        ca = self.ca
        results = {}
        for group in self.plan.allocation_parameters:
            entries = [self._scalar(0.0 if fixed is None else fixed) for fixed in group.fixed_lambdas]
            positions = group.estimated_positions
            if positions:
                if len(positions) == 1:
                    pvals = [self._scalar(group.residual)]
                else:
                    anchor = group.anchor_position
                    terms = []
                    raw_iter = iter(group.raw_indices)
                    for pos in positions:
                        base_share = group.baseline_p[pos] / group.residual
                        offset = self._scalar(0.0) if pos == anchor else rho[next(raw_iter)]
                        terms.append(ca.exp(self._scalar(log(base_share)) + offset))
                    denom = sum(terms[1:], terms[0])
                    pvals = [self._scalar(group.residual) * term / denom for term in terms]
                for pos, pval in zip(positions, pvals):
                    entries[pos] = pval / self._scalar(group.weights[pos])
            results[group.family_name] = ca.vertcat(*entries)
        return results

    def _gamma_expression(self, values):
        ca = self.ca
        gamma = self.sym.zeros(self.model.state_dimension, len(self.model.thermal_ports))
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

    def _matrix_expressions(self, rho):
        ca = self.ca
        values = self._master_expressions(rho)
        i2m = self.model.parameter_registry.instance_to_master
        c = ca.vertcat(*[
            values[i2m[self.model.state_capacitance_parameter[node.key]]]
            for node in self.model.state_nodes
        ])
        g = ca.vertcat(*[
            1.0 / values[i2m[edge.parameter_instance_id]]
            for edge in self.model.resistance_edges
        ])
        D = ca.DM(np.asarray(self.model.incidence, dtype=float))
        L = ca.mtimes(D * ca.repmat(g.T, D.size1(), 1), D.T)
        n = self.model.state_dimension
        lcc = L[:n, :n]
        lcb = L[:n, n:]
        gamma = self._gamma_expression(values)
        h = ca.DM(np.asarray(self.model.observation, dtype=float))
        inv_c = 1.0 / c
        a = -ca.repmat(inv_c, 1, n) * lcc
        bt = -ca.repmat(inv_c, 1, lcb.size2()) * lcb
        bq = ca.repmat(inv_c, 1, gamma.size2()) * gamma
        master_vec = ca.vertcat(*[values[mid] for mid in self.plan.master_order])
        return master_vec, c, lcc, lcb, gamma, h, a, bt, bq

    def _effective_expression(self, rho, local, aggregate):
        ca = self.ca
        if self.model.spec.mode is not SpatialMode.DEP2:
            return local
        lambdas = self._allocation_expressions(rho)
        zone_index = {z: i for i, z in enumerate(self.model.spec.zone_ids)}
        entries = []
        for j, port in enumerate(self.model.thermal_ports):
            if port.signal == "qac":
                entries.append(local[j])
            else:
                family = self.model.signal_to_allocation_family[port.signal]
                lam = lambdas[family][zone_index[port.zone_id]]
                entries.append(lam * aggregate[self._aggregate_index[port.signal]])
        return ca.vertcat(*entries)

    def _build_functions(self) -> None:
        ca = self.ca
        rho = self.sym.sym("rho", self.plan.raw_dimension)
        x = self.sym.sym("x", self.model.state_dimension)
        tb = self.sym.sym("tb", len(self.model.boundary_nodes))
        local = self.sym.sym("local", len(self.model.thermal_ports))
        aggregate = self.sym.sym("aggregate", len(self.plan.aggregate_signal_order))

        master_vec, c, lcc, lcb, gamma, h, a, bt, bq = self._matrix_expressions(rho)
        effective = self._effective_expression(rho, local, aggregate)
        rhs = ca.mtimes(a, x) + ca.mtimes(bt, tb) + ca.mtimes(bq, effective)

        prefix = f"e06_{self.symbol_type.lower()}"
        self.master_function = ca.Function(prefix + "_masters", [rho], [master_vec])
        self.matrix_function = ca.Function(
            prefix + "_matrices", [rho], [c, lcc, lcb, gamma, h, a, bt, bq]
        )
        self.effective_function = ca.Function(
            prefix + "_effective", [rho, local, aggregate], [effective]
        )
        self.rhs_function = ca.Function(
            prefix + "_rhs", [rho, x, tb, local, aggregate], [rhs]
        )
        self.state_jacobian_function = ca.Function(
            prefix + "_jx", [rho, x, tb, local, aggregate], [ca.jacobian(rhs, x)]
        )
        self.boundary_jacobian_function = ca.Function(
            prefix + "_jtb", [rho, x, tb, local, aggregate], [ca.jacobian(rhs, tb)]
        )
        self.local_input_jacobian_function = ca.Function(
            prefix + "_jlocal", [rho, x, tb, local, aggregate], [ca.jacobian(rhs, local)]
        )
        self.raw_jacobian_function = ca.Function(
            prefix + "_jrho", [rho, x, tb, local, aggregate], [ca.jacobian(rhs, rho)]
        )
        self._symbols = (rho, x, tb, local, aggregate)
        self._rhs_expr = rhs

    def zero_raw(self) -> np.ndarray:
        return np.zeros(self.plan.raw_dimension, dtype=float)

    def _vec(self, value, width: int, label: str):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape != (width,):
            raise BackendAdapterError(f"{label} shape must be {(width,)}, got {arr.shape}")
        return self.ca.DM(arr)

    def _args(self, raw, state=None, boundary=None, local_thermal=None, aggregate_thermal=None):
        rho = self._vec(raw, self.plan.raw_dimension, "raw")
        if state is None:
            return (rho,)
        x = self._vec(state, self.model.state_dimension, "state")
        tb = self._vec(boundary, len(self.model.boundary_nodes), "boundary")
        local = self._vec(local_thermal, len(self.model.thermal_ports), "local_thermal")
        if len(self.plan.aggregate_signal_order):
            if aggregate_thermal is None:
                raise BackendAdapterError("DEP2 CasADi realization requires aggregate_thermal")
            aggregate = self._vec(
                aggregate_thermal, len(self.plan.aggregate_signal_order), "aggregate_thermal"
            )
        else:
            aggregate = self.ca.DM.zeros(0, 1)
        return rho, x, tb, local, aggregate

    @staticmethod
    def _numpy(value) -> np.ndarray:
        return np.asarray(value, dtype=float)

    def master_values(self, raw) -> np.ndarray:
        return self._numpy(self.master_function(*self._args(raw))).reshape(-1)

    def matrices(self, raw):
        outputs = self.matrix_function(*self._args(raw))
        return tuple(self._numpy(item) for item in outputs)

    def effective_thermal(self, raw, local_thermal, aggregate_thermal=None) -> np.ndarray:
        rho = self._vec(raw, self.plan.raw_dimension, "raw")
        local = self._vec(local_thermal, len(self.model.thermal_ports), "local_thermal")
        aggregate = (
            self._vec(aggregate_thermal, len(self.plan.aggregate_signal_order), "aggregate_thermal")
            if len(self.plan.aggregate_signal_order)
            else self.ca.DM.zeros(0, 1)
        )
        return self._numpy(self.effective_function(rho, local, aggregate)).reshape(-1)

    def rhs(self, raw, state, boundary, local_thermal, aggregate_thermal=None) -> np.ndarray:
        return self._numpy(
            self.rhs_function(*self._args(raw, state, boundary, local_thermal, aggregate_thermal))
        ).reshape(-1)

    def raw_jacobian(self, raw, state, boundary, local_thermal, aggregate_thermal=None) -> np.ndarray:
        return self._numpy(
            self.raw_jacobian_function(
                *self._args(raw, state, boundary, local_thermal, aggregate_thermal)
            )
        )

    def step(self, solver: str, raw, state, boundary, local_thermal, aggregate_thermal=None, *, sample_dt_s: float, substeps: int = 1) -> np.ndarray:
        key = str(solver).strip().lower().replace("-", "_")
        if key in {"exact", "exact_zoh", "exact_zoh_linear"}:
            return self._exact_numeric(raw, state, boundary, local_thermal, aggregate_thermal, sample_dt_s=sample_dt_s, substeps=substeps)
        if key not in {"euler", "rk2", "rk4"}:
            raise BackendAdapterError(f"Unsupported common CasADi solver {solver!r}")
        if substeps < 1:
            raise BackendAdapterError("substeps must be >= 1")

        ca = self.ca
        rho = self.sym.sym("rho_step", self.plan.raw_dimension)
        x = self.sym.sym("x_step", self.model.state_dimension)
        tb = self.sym.sym("tb_step", len(self.model.boundary_nodes))
        local = self.sym.sym("local_step", len(self.model.thermal_ports))
        aggregate = self.sym.sym("aggregate_step", len(self.plan.aggregate_signal_order))
        h = float(sample_dt_s) / int(substeps)

        def f(xx):
            master_vec, c, lcc, lcb, gamma, hmat, a, bt, bq = self._matrix_expressions(rho)
            qeff = self._effective_expression(rho, local, aggregate)
            return ca.mtimes(a, xx) + ca.mtimes(bt, tb) + ca.mtimes(bq, qeff)

        out = x
        for _ in range(int(substeps)):
            if key == "euler":
                out = out + h * f(out)
            elif key == "rk2":
                k1 = f(out)
                k2 = f(out + 0.5 * h * k1)
                out = out + h * k2
            else:
                k1 = f(out)
                k2 = f(out + 0.5 * h * k1)
                k3 = f(out + 0.5 * h * k2)
                k4 = f(out + h * k3)
                out = out + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        fn = ca.Function(
            f"e06_{self.symbol_type.lower()}_{key}_step",
            [rho, x, tb, local, aggregate],
            [out],
        )
        args = self._args(raw, state, boundary, local_thermal, aggregate_thermal)
        return self._numpy(fn(*args)).reshape(-1)

    def _exact_numeric(self, raw, state, boundary, local_thermal, aggregate_thermal, *, sample_dt_s: float, substeps: int) -> np.ndarray:
        """P5 exact value parity using CasADi-generated live A/B matrices.

        CasADi distributions do not always ship the optional SLICOT ``expm``
        plugin (and SX matrix exponential support varies). E0-6 therefore uses
        CasADi to realize the live matrices, then applies the independent SciPy
        augmented exponential for value parity. Symbolic exact-ZOH AD is only
        claimed when a compatible CasADi expm path is available.
        """
        try:
            from scipy.linalg import expm
        except Exception as exc:  # pragma: no cover
            raise BackendAdapterError("SciPy is required for CasADi exact-ZOH value parity") from exc
        if substeps < 1:
            raise BackendAdapterError("substeps must be >= 1")
        _, _, _, _, _, a, bt, bq = self.matrices(raw)
        qeff = self.effective_thermal(raw, local_thermal, aggregate_thermal)
        x = np.asarray(state, dtype=float).reshape(-1)
        tbv = np.asarray(boundary, dtype=float).reshape(-1)
        h = float(sample_dt_s) / int(substeps)
        B = np.concatenate((bt, bq), axis=1)
        n, m = a.shape[0], B.shape[1]
        for _ in range(int(substeps)):
            aug = np.zeros((n + m, n + m), dtype=float)
            aug[:n, :n] = a
            aug[:n, n:] = B
            transition = expm(aug * h)
            u = np.concatenate((tbv, qeff))
            x = transition[:n, :n] @ x + transition[:n, n:] @ u
        return x

    def parameter_probe_gradient(self, raw, state, boundary, local_thermal, aggregate_thermal, probe) -> np.ndarray:
        ca = self.ca
        rho, x, tb, local, aggregate = self._symbols
        v = self.sym.sym("probe", self.model.state_dimension)
        loss = ca.dot(v, self._rhs_expr)
        grad = ca.gradient(loss, rho)
        fn = ca.Function(
            f"e06_{self.symbol_type.lower()}_probe_grad",
            [rho, x, tb, local, aggregate, v],
            [grad],
        )
        args = self._args(raw, state, boundary, local_thermal, aggregate_thermal)
        return self._numpy(fn(*args, self._vec(probe, self.model.state_dimension, "probe"))).reshape(-1)


# Backward-compatible alias. Production IPOPT estimation should use
# CasadiPhysicalRCBackend from casadi_physical_backend.py.
CasadiRCBackend = CasadiTransformedRCBackend
