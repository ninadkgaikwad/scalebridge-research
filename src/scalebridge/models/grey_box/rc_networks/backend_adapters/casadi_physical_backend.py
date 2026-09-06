from __future__ import annotations

"""Direct physical-Theta CasADi realization for IPOPT-ready E0-6 v2 optimization."""

from dataclasses import dataclass

import numpy as np

from ..compiler import CompiledRCModel
from ..specification import AllocationMode, HeatPortGroup, SpatialMode, StateNode
from .contracts import BackendAdapterError
from .schema import build_physical_parameterization_plan


def _require_casadi():
    try:
        import casadi as ca
    except Exception as exc:  # pragma: no cover - environment dependent
        raise BackendAdapterError("CasADi is required for the E0-6 physical backend") from exc
    return ca


@dataclass
class CasadiPhysicalRCBackend:
    """CasADi backend whose NLP variables are physical parameters directly.

    The class deliberately does not apply exp/sigmoid/softmax transforms.
    Physical bounds are exposed as ``lbx``/``ubx`` and conservation/simplex
    equations are exposed as explicit ``g`` constraints for IPOPT.
    """

    model: CompiledRCModel
    symbol_type: str = "MX"

    def __post_init__(self) -> None:
        self.ca = _require_casadi()
        token = str(self.symbol_type).strip().upper()
        if token not in {"SX", "MX"}:
            raise BackendAdapterError("CasADi symbol_type must be 'SX' or 'MX'")
        self.symbol_type = token
        self.sym = self.ca.SX if token == "SX" else self.ca.MX
        self.plan = build_physical_parameterization_plan(self.model)
        self._aggregate_index = {
            signal: i for i, signal in enumerate(self.plan.aggregate_signal_order)
        }
        self._build_functions()

    def initial_physical(self) -> np.ndarray:
        return np.asarray(self.plan.initial_values, dtype=float)

    def lower_bounds(self) -> np.ndarray:
        return np.asarray(self.plan.lower_bounds, dtype=float)

    def upper_bounds(self) -> np.ndarray:
        return np.asarray(self.plan.upper_bounds, dtype=float)

    def constraint_lower_bounds(self) -> np.ndarray:
        return np.asarray([c.lower_bound for c in self.plan.constraints], dtype=float)

    def constraint_upper_bounds(self) -> np.ndarray:
        return np.asarray([c.upper_bound for c in self.plan.constraints], dtype=float)

    def _master_expressions(self, theta):
        values = {mid: self.ca.DM(float(v)) for mid, v in self.plan.fixed_master_values.items()}
        for master_id, idx in self.plan.master_decision_index.items():
            values[master_id] = theta[idx]
        missing = set(self.plan.master_order) - set(values)
        if missing:  # pragma: no cover - plan construction protects this
            raise BackendAdapterError(f"Missing physical master expressions: {sorted(missing)}")
        return values

    def _allocation_expressions(self, theta):
        ca = self.ca
        zones = tuple(self.model.spec.zone_ids)
        results = {}
        for family_name, spec in self.model.allocation_families.items():
            entries = []
            participants = set(spec.participating_zone_ids or zones)
            for zone in zones:
                weight = float(spec.weights[zone])
                if spec.mode is AllocationMode.NEUTRAL_FIXED:
                    value = ca.DM(1.0)
                elif spec.mode is AllocationMode.FIXED:
                    value = ca.DM(float(spec.fixed_lambdas[zone]))
                elif spec.mode is AllocationMode.ESTIMATED:
                    if zone not in participants:
                        value = ca.DM(0.0)
                    elif zone in spec.fixed_lambdas:
                        value = ca.DM(float(spec.fixed_lambdas[zone]))
                    else:
                        pidx = self.plan.allocation_p_index[(family_name, zone)]
                        value = theta[pidx] / weight
                else:  # pragma: no cover
                    raise BackendAdapterError(f"Unsupported allocation mode {spec.mode!r}")
                entries.append(value)
            results[family_name] = ca.vertcat(*entries)
        return results

    def _gamma_expression(self, values):
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

    def _matrix_expressions(self, theta):
        ca = self.ca
        values = self._master_expressions(theta)
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

    def _effective_expression(self, theta, local, aggregate):
        ca = self.ca
        if self.model.spec.mode is not SpatialMode.DEP2:
            return local
        lambdas = self._allocation_expressions(theta)
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

    def constraint_expression(self, theta):
        rows = []
        for constraint in self.plan.constraints:
            value = 0
            for idx, coeff in zip(constraint.indices, constraint.coefficients):
                value = value + float(coeff) * theta[idx]
            rows.append(value)
        return self.ca.vertcat(*rows) if rows else self.sym.zeros(0, 1)

    def rhs_expression(self, theta, x, tb, local, aggregate):
        ca = self.ca
        _, _, _, _, _, _, a, bt, bq = self._matrix_expressions(theta)
        qeff = self._effective_expression(theta, local, aggregate)
        return ca.mtimes(a, x) + ca.mtimes(bt, tb) + ca.mtimes(bq, qeff)

    def rk_expression(self, solver: str, theta, x, tb, local, aggregate, *, sample_dt_s: float, substeps: int = 1):
        key = str(solver).strip().lower().replace("-", "_")
        if key not in {"euler", "rk2", "rk4"}:
            raise BackendAdapterError("Physical CasADi symbolic propagation supports euler/rk2/rk4")
        if substeps < 1:
            raise BackendAdapterError("substeps must be >= 1")
        h = float(sample_dt_s) / int(substeps)
        out = x
        for _ in range(int(substeps)):
            if key == "euler":
                out = out + h * self.rhs_expression(theta, out, tb, local, aggregate)
            elif key == "rk2":
                k1 = self.rhs_expression(theta, out, tb, local, aggregate)
                k2 = self.rhs_expression(theta, out + 0.5 * h * k1, tb, local, aggregate)
                out = out + h * k2
            else:
                k1 = self.rhs_expression(theta, out, tb, local, aggregate)
                k2 = self.rhs_expression(theta, out + 0.5 * h * k1, tb, local, aggregate)
                k3 = self.rhs_expression(theta, out + 0.5 * h * k2, tb, local, aggregate)
                k4 = self.rhs_expression(theta, out + h * k3, tb, local, aggregate)
                out = out + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return out

    def _build_functions(self) -> None:
        ca = self.ca
        theta = self.sym.sym("theta", self.plan.decision_dimension)
        x = self.sym.sym("x", self.model.state_dimension)
        tb = self.sym.sym("tb", len(self.model.boundary_nodes))
        local = self.sym.sym("local", len(self.model.thermal_ports))
        aggregate = self.sym.sym("aggregate", len(self.plan.aggregate_signal_order))

        master_vec, c, lcc, lcb, gamma, h, a, bt, bq = self._matrix_expressions(theta)
        effective = self._effective_expression(theta, local, aggregate)
        rhs = ca.mtimes(a, x) + ca.mtimes(bt, tb) + ca.mtimes(bq, effective)
        constraints = self.constraint_expression(theta)
        prefix = f"e06_physical_{self.symbol_type.lower()}"
        self.master_function = ca.Function(prefix + "_masters", [theta], [master_vec])
        self.matrix_function = ca.Function(prefix + "_matrices", [theta], [c, lcc, lcb, gamma, h, a, bt, bq])
        self.effective_function = ca.Function(prefix + "_effective", [theta, local, aggregate], [effective])
        self.rhs_function = ca.Function(prefix + "_rhs", [theta, x, tb, local, aggregate], [rhs])
        self.constraint_function = ca.Function(prefix + "_constraints", [theta], [constraints])
        self.state_jacobian_function = ca.Function(prefix + "_jx", [theta, x, tb, local, aggregate], [ca.jacobian(rhs, x)])
        self.physical_jacobian_function = ca.Function(prefix + "_jtheta", [theta, x, tb, local, aggregate], [ca.jacobian(rhs, theta)])
        self._symbols = (theta, x, tb, local, aggregate)
        self._rhs_expr = rhs

    def _vec(self, value, width: int, label: str):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape != (width,):
            raise BackendAdapterError(f"{label} shape must be {(width,)}, got {arr.shape}")
        return self.ca.DM(arr)

    def _args(self, theta, state=None, boundary=None, local_thermal=None, aggregate_thermal=None):
        t = self._vec(theta, self.plan.decision_dimension, "theta")
        if state is None:
            return (t,)
        x = self._vec(state, self.model.state_dimension, "state")
        tb = self._vec(boundary, len(self.model.boundary_nodes), "boundary")
        local = self._vec(local_thermal, len(self.model.thermal_ports), "local_thermal")
        if len(self.plan.aggregate_signal_order):
            if aggregate_thermal is None:
                raise BackendAdapterError("DEP2 physical CasADi backend requires aggregate_thermal")
            aggregate = self._vec(aggregate_thermal, len(self.plan.aggregate_signal_order), "aggregate_thermal")
        else:
            aggregate = self.ca.DM.zeros(0, 1)
        return t, x, tb, local, aggregate

    @staticmethod
    def _numpy(value) -> np.ndarray:
        return np.asarray(value, dtype=float)

    def master_values(self, theta) -> np.ndarray:
        return self._numpy(self.master_function(*self._args(theta))).reshape(-1)

    def matrices(self, theta):
        return tuple(self._numpy(item) for item in self.matrix_function(*self._args(theta)))

    def constraints(self, theta) -> np.ndarray:
        return self._numpy(self.constraint_function(*self._args(theta))).reshape(-1)

    def rhs(self, theta, state, boundary, local_thermal, aggregate_thermal=None) -> np.ndarray:
        return self._numpy(self.rhs_function(*self._args(theta, state, boundary, local_thermal, aggregate_thermal))).reshape(-1)

    def parameter_probe_gradient(self, theta, state, boundary, local_thermal, aggregate_thermal, probe) -> np.ndarray:
        ca = self.ca
        th, x, tb, local, aggregate = self._symbols
        v = self.sym.sym("probe", self.model.state_dimension)
        loss = ca.dot(v, self._rhs_expr)
        grad = ca.gradient(loss, th)
        fn = ca.Function(
            f"e06_physical_{self.symbol_type.lower()}_probe_grad",
            [th, x, tb, local, aggregate, v],
            [grad],
        )
        args = self._args(theta, state, boundary, local_thermal, aggregate_thermal)
        return self._numpy(fn(*args, self._vec(probe, self.model.state_dimension, "probe"))).reshape(-1)

    def step(self, solver: str, theta, state, boundary, local_thermal, aggregate_thermal=None, *, sample_dt_s: float, substeps: int = 1) -> np.ndarray:
        key = str(solver).strip().lower().replace("-", "_")
        if key in {"exact", "exact_zoh", "exact_zoh_linear"}:
            # Value oracle: live physical CasADi matrices + independent augmented expm.
            from scipy.linalg import expm
            _, _, _, _, _, a, bt, bq = self.matrices(theta)
            local = np.asarray(local_thermal, dtype=float).reshape(-1)
            if self.model.spec.mode is SpatialMode.DEP2:
                aggregate = np.asarray(aggregate_thermal, dtype=float).reshape(-1)
                qeff = self._numpy(self.effective_function(self._vec(theta, self.plan.decision_dimension, "theta"), self._vec(local, len(self.model.thermal_ports), "local"), self._vec(aggregate, len(self.plan.aggregate_signal_order), "aggregate"))).reshape(-1)
            else:
                qeff = local
            xnum = np.asarray(state, dtype=float).reshape(-1)
            tbnum = np.asarray(boundary, dtype=float).reshape(-1)
            h = float(sample_dt_s) / int(substeps)
            B = np.concatenate((bt, bq), axis=1)
            u = np.concatenate((tbnum, qeff))
            n, m = a.shape[0], B.shape[1]
            for _ in range(int(substeps)):
                aug = np.zeros((n + m, n + m), dtype=float)
                aug[:n, :n] = a
                aug[:n, n:] = B
                transition = expm(aug * h)
                xnum = transition[:n, :n] @ xnum + transition[:n, n:] @ u
            return xnum

        th = self.sym.sym("theta_step", self.plan.decision_dimension)
        x = self.sym.sym("x_step", self.model.state_dimension)
        tb = self.sym.sym("tb_step", len(self.model.boundary_nodes))
        local = self.sym.sym("local_step", len(self.model.thermal_ports))
        aggregate = self.sym.sym("aggregate_step", len(self.plan.aggregate_signal_order))
        out = self.rk_expression(key, th, x, tb, local, aggregate, sample_dt_s=sample_dt_s, substeps=substeps)
        fn = self.ca.Function(f"e06_physical_{self.symbol_type.lower()}_{key}_step", [th, x, tb, local, aggregate], [out])
        return self._numpy(fn(*self._args(theta, state, boundary, local_thermal, aggregate_thermal))).reshape(-1)

    def build_ipopt_nlp(self, objective):
        """Return an IPOPT-ready physical-parameter NLP shell.

        ``objective`` is caller-owned.  The returned ``g/lbg/ubg`` contain only
        E0-6 physical-domain constraints (routing/simplex and DEP2 conservation).
        A parameter-estimation method that introduces state-trajectory dynamics
        should concatenate its transcription constraints and matching bounds at
        the method layer rather than hiding them inside this backend adapter.
        """
        theta, _, _, _, _ = self._symbols
        g = self.constraint_expression(theta)
        return {
            "nlp": {"x": theta, "f": objective, "g": g},
            "x0": self.initial_physical(),
            "lbx": self.lower_bounds(),
            "ubx": self.upper_bounds(),
            "lbg": self.constraint_lower_bounds(),
            "ubg": self.constraint_upper_bounds(),
        }

    @property
    def theta_symbol(self):
        return self._symbols[0]
