from __future__ import annotations

"""Direct physical-Theta NumPy reference backend for E0-6 v2.

This backend is intentionally transform-free.  It is appropriate for Bayesian
likelihood/reference evaluation and for checking the physical-coordinate
CasADi/IPOPT realization.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..compiler import CompiledRCModel
from ..discretization.linear_oracle import ExactZOHLinearIntegrator
from ..discretization.linear_system import TorchLinearRCStateSpace
from ..specification import AllocationMode, HeatPortGroup, SpatialMode, StateNode
from .contracts import BackendAdapterError, BackendMatrices
from .schema import build_physical_parameterization_plan


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
class NumpyPhysicalRCBackend:
    model: CompiledRCModel

    def __post_init__(self) -> None:
        self.plan = build_physical_parameterization_plan(self.model)
        self._aggregate_index = {
            signal: i for i, signal in enumerate(self.plan.aggregate_signal_order)
        }
        self._inverse_master_index = {
            idx: master_id for master_id, idx in self.plan.master_decision_index.items()
        }
        self._inverse_allocation_index = {
            idx: key for key, idx in self.plan.allocation_p_index.items()
        }

    def initial_physical(self) -> np.ndarray:
        return np.asarray(self.plan.initial_values, dtype=float)

    def lower_bounds(self) -> np.ndarray:
        return np.asarray(self.plan.lower_bounds, dtype=float)

    def upper_bounds(self) -> np.ndarray:
        return np.asarray(self.plan.upper_bounds, dtype=float)

    def _theta(self, theta) -> np.ndarray:
        arr = np.asarray(theta, dtype=float).reshape(-1)
        if arr.shape != (self.plan.decision_dimension,):
            raise BackendAdapterError(
                f"Physical decision shape must be {(self.plan.decision_dimension,)}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise BackendAdapterError("Physical decisions contain non-finite values")
        lb = self.lower_bounds()
        ub = self.upper_bounds()
        if np.any(arr < lb - 1e-12) or np.any(arr > ub + 1e-12):
            raise BackendAdapterError("Physical decisions violate declared bounds")
        for row in self.plan.constraints:
            value = float(sum(c * arr[i] for i, c in zip(row.indices, row.coefficients)))
            if value < row.lower_bound - 1e-10 or value > row.upper_bound + 1e-10:
                raise BackendAdapterError(
                    f"Physical decisions violate constraint {row.constraint_id!r}: {value}"
                )
        return arr

    def master_values(self, theta) -> dict[str, float]:
        arr = self._theta(theta)
        values = {k: float(v) for k, v in self.plan.fixed_master_values.items()}
        for master_id, idx in self.plan.master_decision_index.items():
            values[master_id] = float(arr[idx])
        missing = set(self.plan.master_order) - set(values)
        if missing:
            raise BackendAdapterError(f"Physical backend failed to realize masters: {sorted(missing)}")
        # Reuse E0-3 physical-domain validation without changing the compiler.
        self.model.parameter_registry.resolve_instance_values(values)
        return values

    def allocation_lambdas(self, theta) -> dict[str, np.ndarray]:
        arr = self._theta(theta)
        out: dict[str, np.ndarray] = {}
        zones = tuple(self.model.spec.zone_ids)
        for family_name, spec in self.model.allocation_families.items():
            weights = np.asarray([float(spec.weights[z]) for z in zones], dtype=float)
            lam = np.zeros(len(zones), dtype=float)
            participants = set(spec.participating_zone_ids or zones)
            if spec.mode is AllocationMode.NEUTRAL_FIXED:
                lam[:] = 1.0
            elif spec.mode is AllocationMode.FIXED:
                lam[:] = [float(spec.fixed_lambdas[z]) for z in zones]
            elif spec.mode is AllocationMode.ESTIMATED:
                for i, zone in enumerate(zones):
                    if zone not in participants:
                        lam[i] = 0.0
                    elif zone in spec.fixed_lambdas:
                        lam[i] = float(spec.fixed_lambdas[zone])
                    else:
                        pidx = self.plan.allocation_p_index[(family_name, zone)]
                        lam[i] = float(arr[pidx]) / weights[i]
            else:  # pragma: no cover
                raise BackendAdapterError(f"Unsupported allocation mode {spec.mode!r}")
            mass = float(np.dot(weights, lam))
            if abs(mass - 1.0) > 1e-9:
                raise BackendAdapterError(
                    f"Allocation family {family_name!r} violates A_g B_g = 1: {mass}"
                )
            out[family_name] = lam
        return out

    def matrices(self, theta) -> BackendMatrices:
        masters = self.master_values(theta)
        rc = self.model.matrices(masters)
        c = np.asarray(rc.C, dtype=float)
        lcc = np.asarray(rc.L_CC, dtype=float)
        lcb = np.asarray(rc.L_CB, dtype=float)
        gamma = np.asarray(rc.Gamma, dtype=float)
        h = np.asarray(rc.H, dtype=float)
        a = -lcc / c[:, None]
        bt = -lcb / c[:, None]
        bq = gamma / c[:, None]
        return BackendMatrices(c, lcc, lcb, gamma, h, a, bt, bq)

    def effective_thermal(self, theta, local_thermal, aggregate_thermal=None) -> np.ndarray:
        local, squeeze = _as_batch(local_thermal, len(self.model.thermal_ports), "local_thermal")
        if self.model.spec.mode is not SpatialMode.DEP2:
            return local[0] if squeeze else local
        if aggregate_thermal is None:
            raise BackendAdapterError("DEP2 physical backend requires aggregate_thermal")
        aggregate, _ = _as_batch(
            aggregate_thermal, len(self.plan.aggregate_signal_order), "aggregate_thermal"
        )
        if aggregate.shape[0] != local.shape[0]:
            raise BackendAdapterError("Local and aggregate thermal batch sizes must match")
        lambdas = self.allocation_lambdas(theta)
        zone_index = {z: i for i, z in enumerate(self.model.spec.zone_ids)}
        columns = []
        for j, port in enumerate(self.model.thermal_ports):
            if port.signal == "qac":
                columns.append(local[:, j])
            else:
                family = self.model.signal_to_allocation_family[port.signal]
                lam = lambdas[family][zone_index[port.zone_id]]
                columns.append(lam * aggregate[:, self._aggregate_index[port.signal]])
        result = np.stack(columns, axis=1)
        return result[0] if squeeze else result

    def rhs(self, theta, state, boundary, local_thermal, aggregate_thermal=None) -> np.ndarray:
        mats = self.matrices(theta)
        x, squeeze = _as_batch(state, self.model.state_dimension, "state")
        tb, _ = _as_batch(boundary, len(self.model.boundary_nodes), "boundary")
        q = self.effective_thermal(theta, local_thermal, aggregate_thermal)
        q, _ = _as_batch(q, len(self.model.thermal_ports), "effective_thermal")
        out = x @ mats.A.T + tb @ mats.B_boundary.T + q @ mats.B_thermal.T
        return out[0] if squeeze else out

    def step(self, solver: str, theta, state, boundary, local_thermal, aggregate_thermal=None, *, sample_dt_s: float, substeps: int = 1) -> np.ndarray:
        key = str(solver).strip().lower().replace("-", "_")
        if substeps < 1:
            raise BackendAdapterError("substeps must be >= 1")
        x = np.asarray(state, dtype=float).reshape(-1)
        tb = np.asarray(boundary, dtype=float).reshape(-1)
        q = self.effective_thermal(theta, local_thermal, aggregate_thermal).reshape(-1)
        h = float(sample_dt_s) / int(substeps)
        mats = self.matrices(theta)

        def f(xx):
            return mats.A @ xx + mats.B_boundary @ tb + mats.B_thermal @ q

        if key in {"exact", "exact_zoh", "exact_zoh_linear"}:
            from scipy.linalg import expm
            B = np.concatenate((mats.B_boundary, mats.B_thermal), axis=1)
            u = np.concatenate((tb, q))
            n, m = mats.A.shape[0], B.shape[1]
            for _ in range(int(substeps)):
                aug = np.zeros((n + m, n + m), dtype=float)
                aug[:n, :n] = mats.A
                aug[:n, n:] = B
                transition = expm(aug * h)
                x = transition[:n, :n] @ x + transition[:n, n:] @ u
            return x

        if key not in {"euler", "rk2", "rk4"}:
            raise BackendAdapterError(f"Unsupported physical NumPy solver {solver!r}")
        for _ in range(int(substeps)):
            if key == "euler":
                x = x + h * f(x)
            elif key == "rk2":
                k1 = f(x)
                k2 = f(x + 0.5 * h * k1)
                x = x + h * k2
            else:
                k1 = f(x)
                k2 = f(x + 0.5 * h * k1)
                k3 = f(x + 0.5 * h * k2)
                k4 = f(x + h * k3)
                x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x
