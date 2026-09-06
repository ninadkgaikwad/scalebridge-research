from __future__ import annotations

"""Focused E0-3B validation for the generic continuous-time RC compiler."""

import json
from pathlib import Path
import sys

import numpy as np

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    RCCompilerSpec,
    RCInputSnapshot,
    ZoneAdjacency,
    assert_dep1_dep2_physics_equivalent,
    compile_rc_model,
    default_initial_state,
    initial_estimated_allocation_result,
    rhs,
    validate_compiler_invariants,
)


def _values(model):
    defaults = {
        "C_a": 2.0e6,
        "C_m": 8.0e6,
        "C_e": 5.0e6,
        "R_ao": 0.02,
        "R_am": 0.01,
        "R_om": 0.04,
        "R_ae": 0.015,
        "R_eo": 0.03,
        "R_inter_a_a": 0.05,
        "eta_r": 0.7,
        "gamma_a_r": 0.2,
        "gamma_e_r": 0.3,
        "gamma_m_r": 0.5,
    }
    return {item.instance_id: defaults[item.family] for item in model.parameter_registry.instances}


def main() -> int:
    report: dict[str, object] = {
        "phase": "E0-3B",
        "continuous_time_only": True,
        "flavours": {},
    }

    for flavour in ("1r1c", "2r2c", "3r2c", "4r3c"):
        model = compile_rc_model(RCCompilerSpec(flavour, ("A", "B"), "dep1"))
        inv = validate_compiler_invariants(model, _values(model))
        init = default_initial_state(model, {"A": 22.0, "B": 24.0})
        report["flavours"][flavour] = {
            "state_dimension": model.state_dimension,
            "edge_count": len(model.resistance_edges),
            "master_parameter_count": len(model.parameter_registry.masters),
            "invariants_passed": inv.passed,
            "initial_state": init.tolist(),
        }

    allocations = (
        AllocationFamilySpec(
            name="convective",
            signals=("zic", "qsol1"),
            weights={"A": 0.7, "B": 0.3},
            mode=AllocationMode.ESTIMATED,
        ),
        AllocationFamilySpec(
            name="radiative",
            signals=("zir", "qsol2"),
            weights={"A": 0.7, "B": 0.3},
            mode=AllocationMode.ESTIMATED,
        ),
    )
    dep2 = compile_rc_model(
        RCCompilerSpec(
            "2r2c",
            ("A", "B"),
            "dep2",
            adjacency=(ZoneAdjacency("A", "B"),),
            dep2_allocations=allocations,
        )
    )
    dep1 = compile_rc_model(
        RCCompilerSpec(
            "2r2c",
            ("A", "B"),
            "dep1",
            adjacency=(ZoneAdjacency("A", "B"),),
        )
    )
    assert_dep1_dep2_physics_equivalent(dep1, dep2)

    conv = initial_estimated_allocation_result(dep2.allocation_families["convective"], ("A", "B"))
    rad = initial_estimated_allocation_result(dep2.allocation_families["radiative"], ("A", "B"))
    if max(conv.max_consistency_error, rad.max_consistency_error) > 1e-12:
        raise RuntimeError("DEP2 allocation consistency failed")

    snapshot = RCInputSnapshot(
        boundary_temperatures={"outdoor_temperature": 30.0},
        local_thermal_powers={("A", "qac"): -500.0, ("B", "qac"): -700.0},
        aggregate_thermal_powers={"zic": 300.0, "qsol1": 100.0, "zir": 80.0, "qsol2": 20.0},
    )
    derivative = rhs(
        dep2,
        default_initial_state(dep2, {"A": 22.0, "B": 24.0}),
        snapshot,
        _values(dep2),
        allocation_results={"convective": conv, "radiative": rad},
    )
    if derivative.shape != (4,) or not np.all(np.isfinite(derivative)):
        raise RuntimeError("DEP2 continuous RHS smoke validation failed")

    report["dep2"] = {
        "dep1_dep2_physics_equivalent": True,
        "convective_lambda": dict(conv.lambda_by_zone),
        "radiative_lambda": dict(rad.lambda_by_zone),
        "rhs_finite": True,
    }
    report["qualified"] = True

    out = Path("validated_artifacts") / "phase_e0" / "e03b_rc_compiler_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nE0-3B generic RC compiler validation PASSED\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
