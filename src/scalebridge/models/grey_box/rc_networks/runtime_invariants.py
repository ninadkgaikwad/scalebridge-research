from __future__ import annotations

"""E0-4 runtime invariant diagnostics."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .compiler import CompiledRCModel
from .initialization import InitializationResult
from .invariants import graph_signature
from .runtime_binding import CanonicalRuntimeFrame, RuntimeBinding
from .specification import RCCompileError, SpatialMode


@dataclass(frozen=True)
class RuntimeInvariantReport:
    initialization_projection_max_abs: float
    lifting_identity_max_abs: float
    dep2_coordinate_error_max_abs: float
    structural_absence_respected: bool
    physics_signature: str
    phvac_excluded: bool
    recursive_reset_requires_explicit_api: bool

    @property
    def passed(self) -> bool:
        tol = 1e-9
        return (
            self.initialization_projection_max_abs <= tol
            and self.lifting_identity_max_abs <= tol
            and self.dep2_coordinate_error_max_abs <= tol
            and self.structural_absence_respected
            and self.phvac_excluded
            and self.recursive_reset_requires_explicit_api
        )


def validate_runtime_invariants(
    model: CompiledRCModel,
    initialization: InitializationResult,
    frame: CanonicalRuntimeFrame,
    binding: RuntimeBinding,
) -> RuntimeInvariantReport:
    """Validate directly testable E0-4 runtime invariants for one realization."""

    observed = model.observation @ initialization.state
    projection_error = float(
        np.max(np.abs(observed - initialization.zone_vector_c))
    ) if observed.size else 0.0

    hs0 = model.observation @ initialization.lifting_matrix
    expected = np.eye(model.output_dimension, len(model.spec.zone_ids), dtype=float)
    lifting_error = float(np.max(np.abs(hs0 - expected))) if hs0.size else 0.0

    active = {(p.zone_id, p.signal) for p in model.thermal_ports}
    # Binding snapshot must contain only model-applicable local signals (plus DEP2
    # local QAC, which is itself an active port). Rich extra frame signals remain
    # outside the low-level E0-3 snapshot.
    structural_ok = set(binding.snapshot.local_thermal_powers).issubset(active)

    phvac_excluded = all(
        signal.lower() != "phvac"
        for _, signal in binding.snapshot.local_thermal_powers
    ) and all(signal.lower() != "phvac" for signal in binding.snapshot.aggregate_thermal_powers)

    report = RuntimeInvariantReport(
        initialization_projection_max_abs=projection_error,
        lifting_identity_max_abs=lifting_error,
        dep2_coordinate_error_max_abs=float(binding.dep2_coordinate_error_max_abs),
        structural_absence_respected=bool(structural_ok),
        physics_signature=graph_signature(model),
        phvac_excluded=bool(phvac_excluded),
        recursive_reset_requires_explicit_api=True,
    )
    if not report.passed:
        raise RCCompileError(f"E0-4 runtime invariant validation failed: {report}")
    return report


def assert_runtime_binding_does_not_change_physics(
    model: CompiledRCModel,
    signature_before: str,
) -> None:
    """Invariant 4: runtime binding cannot mutate the compiled physical graph."""

    after = graph_signature(model)
    if signature_before != after:
        raise RCCompileError("Runtime binding altered the compiled RC physics graph")


def assert_dep1_dep2_runtime_physics_equivalent(
    dep1: CompiledRCModel,
    dep2: CompiledRCModel,
) -> None:
    """E0-4 form of the frozen DEP1/DEP2 physical-equivalence invariant."""

    if dep1.spec.mode is not SpatialMode.DEP1 or dep2.spec.mode is not SpatialMode.DEP2:
        raise RCCompileError("Expected DEP1 and DEP2 models")
    if graph_signature(dep1) != graph_signature(dep2):
        raise RCCompileError("DEP1/DEP2 runtime models do not share one physical graph")
    if not np.array_equal(dep1.observation, dep2.observation):
        raise RCCompileError("DEP1/DEP2 observation matrices differ")
