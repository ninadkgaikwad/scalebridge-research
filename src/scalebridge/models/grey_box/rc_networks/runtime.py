from __future__ import annotations

"""Reference continuous-time evaluator for a compiled RC model.

This is deliberately a NumPy reference implementation, not a discretizer.  It
evaluates f(X,U;Theta) at one instant so later Torch/CasADi/Neuromancer adapters
can be checked against the same compiled graph.
"""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .allocation import AllocationResult
from .compiler import CompiledRCModel
from .specification import RCCompileError, SpatialMode, StateNode


@dataclass(frozen=True)
class RCInputSnapshot:
    """Canonical instantaneous inputs before/after DEP2 spatial allocation."""

    boundary_temperatures: Mapping[str, float]
    local_thermal_powers: Mapping[tuple[str, str], float] = field(default_factory=dict)
    aggregate_thermal_powers: Mapping[str, float] = field(default_factory=dict)


def default_initial_state(
    model: CompiledRCModel,
    observed_air_temperatures: Mapping[str, float],
) -> np.ndarray:
    """Initialize every latent state to its zone's observed air temperature."""

    if set(observed_air_temperatures) != set(model.spec.zone_ids):
        raise RCCompileError(
            "Default initialization requires one observed air temperature for every "
            "modeled zone"
        )
    values = []
    for node in model.state_nodes:
        value = float(observed_air_temperatures[node.zone_id])
        if not np.isfinite(value):
            raise RCCompileError(
                f"Non-finite observed air temperature for {node.zone_id!r}"
            )
        values.append(value)
    return np.asarray(values, dtype=float)


def _allocation_by_signal(
    model: CompiledRCModel,
    allocation_results: Mapping[str, AllocationResult],
) -> dict[str, AllocationResult]:
    out: dict[str, AllocationResult] = {}
    for signal, family_name in model.signal_to_allocation_family.items():
        try:
            result = allocation_results[family_name]
        except KeyError as exc:
            raise RCCompileError(
                f"Missing DEP2 allocation result for family {family_name!r}"
            ) from exc
        if result.family_name != family_name:
            raise RCCompileError(
                f"Allocation result family mismatch: expected {family_name!r}, "
                f"got {result.family_name!r}"
            )
        if result.max_consistency_error > 1e-9:
            raise RCCompileError(
                f"Allocation family {family_name!r} violates A_g B_g = 1"
            )
        out[signal] = result
    return out


def assemble_effective_thermal_vector(
    model: CompiledRCModel,
    snapshot: RCInputSnapshot,
    *,
    allocation_results: Mapping[str, AllocationResult] | None = None,
) -> np.ndarray:
    """Build the zone-local atomic thermal-power vector used by Gamma.

    IND/DEP1 read local canonical thermal powers directly.
    DEP2 keeps QAC local and obtains every applicable non-HVAC thermal signal
    exclusively from its all-to-one scalar plus the explicit B_g result.
    """

    result = np.zeros(len(model.thermal_ports), dtype=float)

    if model.spec.mode is SpatialMode.DEP2:
        by_signal = _allocation_by_signal(model, allocation_results or {})
    else:
        if allocation_results:
            raise RCCompileError("Allocation results are only valid in DEP2 mode")
        by_signal = {}

    for j, port in enumerate(model.thermal_ports):
        key = (port.zone_id, port.signal)
        if model.spec.mode is not SpatialMode.DEP2 or port.signal == "qac":
            if key not in snapshot.local_thermal_powers:
                raise RCCompileError(
                    f"Missing required local thermal input {key!r}; structurally "
                    "unavailable signals must be omitted at compile time, not zero-filled"
                )
            value = float(snapshot.local_thermal_powers[key])
        else:
            if port.signal not in snapshot.aggregate_thermal_powers:
                raise RCCompileError(
                    f"Missing DEP2 all-to-one thermal signal {port.signal!r}"
                )
            qbar = float(snapshot.aggregate_thermal_powers[port.signal])
            allocation = by_signal[port.signal]
            try:
                lam = float(allocation.lambda_by_zone[port.zone_id])
            except KeyError as exc:
                raise RCCompileError(
                    f"Allocation result lacks modeled zone {port.zone_id!r}"
                ) from exc
            value = lam * qbar

        if not np.isfinite(value):
            raise RCCompileError(f"Non-finite thermal input for {port.key}")
        result[j] = value

    return result


def assemble_boundary_vector(
    model: CompiledRCModel,
    snapshot: RCInputSnapshot,
) -> np.ndarray:
    values = []
    for boundary in model.boundary_nodes:
        if boundary.boundary_label not in snapshot.boundary_temperatures:
            raise RCCompileError(
                f"Missing required boundary temperature {boundary.boundary_label!r}"
            )
        value = float(snapshot.boundary_temperatures[boundary.boundary_label])
        if not np.isfinite(value):
            raise RCCompileError(
                f"Non-finite boundary temperature {boundary.boundary_label!r}"
            )
        values.append(value)
    return np.asarray(values, dtype=float)


def rhs(
    model: CompiledRCModel,
    state: np.ndarray,
    snapshot: RCInputSnapshot,
    parameter_values: Mapping[str, float],
    *,
    allocation_results: Mapping[str, AllocationResult] | None = None,
) -> np.ndarray:
    """Evaluate the authoritative continuous-time equation.

    C Xdot = -L_CC X - L_CB T_B + Gamma Q
    """

    x = np.asarray(state, dtype=float)
    if x.shape != (model.state_dimension,):
        raise RCCompileError(
            f"State shape must be {(model.state_dimension,)}, got {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise RCCompileError("State contains non-finite values")

    matrices = model.matrices(parameter_values)
    tb = assemble_boundary_vector(model, snapshot)
    q = assemble_effective_thermal_vector(
        model,
        snapshot,
        allocation_results=allocation_results,
    )
    net = -matrices.L_CC @ x - matrices.L_CB @ tb + matrices.Gamma @ q
    return net / matrices.C


def observe(model: CompiledRCModel, state: np.ndarray) -> np.ndarray:
    x = np.asarray(state, dtype=float)
    if x.shape != (model.state_dimension,):
        raise RCCompileError("State dimension mismatch")
    return model.observation @ x
