from __future__ import annotations

"""Compiler-invariant diagnostics for the generic RC implementation."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .compiler import CompiledRCModel, RCMatrices
from .specification import RCCompileError, SpatialMode


@dataclass(frozen=True)
class InvariantReport:
    incidence_column_balance_max_abs: float
    laplacian_row_sum_max_abs: float
    laplacian_symmetry_max_abs: float
    gamma_column_sum_max_abs_error: float
    gamma_min_entry: float
    observation_binary_error: float
    dep1_dep2_graph_signature: str

    @property
    def passed(self) -> bool:
        tol = 1e-9
        return (
            self.incidence_column_balance_max_abs <= tol
            and self.laplacian_row_sum_max_abs <= tol
            and self.laplacian_symmetry_max_abs <= tol
            and self.gamma_column_sum_max_abs_error <= tol
            and self.gamma_min_entry >= -tol
            and self.observation_binary_error <= tol
        )


def graph_signature(model: CompiledRCModel) -> str:
    """Deterministic textual signature of physical topology, independent of values."""

    states = ",".join(node.key for node in model.state_nodes)
    boundaries = ",".join(node.key for node in model.boundary_nodes)
    edges = ";".join(
        f"{'--'.join(edge.endpoint_keys())}|{edge.family}|{edge.kind}"
        for edge in model.resistance_edges
    )
    return f"states={states}||boundaries={boundaries}||edges={edges}"


def validate_compiler_invariants(
    model: CompiledRCModel,
    parameter_values: Mapping[str, float],
) -> InvariantReport:
    matrices = model.matrices(parameter_values)
    D = matrices.D
    L = matrices.L
    Gamma = matrices.Gamma
    H = matrices.H

    incidence_balance = (
        float(np.max(np.abs(D.sum(axis=0)))) if D.shape[1] else 0.0
    )
    row_sum = float(np.max(np.abs(L.sum(axis=1)))) if L.size else 0.0
    symmetry = float(np.max(np.abs(L - L.T))) if L.size else 0.0
    if Gamma.shape[1]:
        gamma_sum_error = float(np.max(np.abs(Gamma.sum(axis=0) - 1.0)))
        gamma_min = float(np.min(Gamma))
    else:
        gamma_sum_error = 0.0
        gamma_min = 0.0

    h_binary_error = (
        float(np.max(np.minimum(np.abs(H), np.abs(H - 1.0)))) if H.size else 0.0
    )

    report = InvariantReport(
        incidence_column_balance_max_abs=incidence_balance,
        laplacian_row_sum_max_abs=row_sum,
        laplacian_symmetry_max_abs=symmetry,
        gamma_column_sum_max_abs_error=gamma_sum_error,
        gamma_min_entry=gamma_min,
        observation_binary_error=h_binary_error,
        dep1_dep2_graph_signature=graph_signature(model),
    )
    if not report.passed:
        raise RCCompileError(f"Compiled RC invariant validation failed: {report}")
    return report


def assert_dep1_dep2_physics_equivalent(
    dep1: CompiledRCModel,
    dep2: CompiledRCModel,
) -> None:
    """Enforce the frozen architecture invariant L/topology_DEP1 == L/topology_DEP2."""

    if dep1.spec.mode is not SpatialMode.DEP1 or dep2.spec.mode is not SpatialMode.DEP2:
        raise RCCompileError("Expected DEP1 and DEP2 models")
    if dep1.flavour.name != dep2.flavour.name:
        raise RCCompileError("DEP1/DEP2 flavour mismatch")
    if dep1.spec.zone_ids != dep2.spec.zone_ids:
        raise RCCompileError("DEP1/DEP2 modeled-zone mismatch")
    if graph_signature(dep1) != graph_signature(dep2):
        raise RCCompileError("DEP1 and DEP2 physical graphs are not identical")
