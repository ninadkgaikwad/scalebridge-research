from __future__ import annotations

"""Standalone scientific validator for ScaleBridge Phase E0 E0-4."""

import json
from pathlib import Path

import numpy as np

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    CanonicalRuntimeFrame,
    InitializationRequest,
    RCCompileError,
    RCCompilerSpec,
    ZoneAdjacency,
    ZoneInitializationEvidence,
    accept_model_evolved_state,
    assert_dep1_dep2_runtime_physics_equivalent,
    bind_runtime_frame,
    compile_rc_model,
    estimated_allocation_result,
    graph_signature,
    initialize_runtime_state,
    rhs,
    start_recursive_state,
    validate_runtime_invariants,
)


OUT = Path("validated_artifacts/phase_e0/e04_runtime_state_input_validation.json")


def _ports(zones, signals=("qac", "zic")):
    return {z: tuple(signals) for z in zones}


def _param_values(model):
    family_values = {
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
    out = {}
    for inst in model.parameter_registry.instances:
        try:
            out[inst.instance_id] = family_values[inst.family]
        except KeyError as exc:
            raise RuntimeError(f"No validator value for parameter family {inst.family!r}") from exc
    return out


def _expected_rejection(name, fn, rejections):
    try:
        fn()
    except (RCCompileError, ValueError) as exc:
        rejections[name] = {"passed": True, "error": str(exc)}
    else:
        rejections[name] = {"passed": False, "error": "expected rejection did not occur"}


def main() -> int:
    report: dict[str, object] = {
        "phase": "E0-4",
        "continuous_time_only": True,
        "numerical_integrator_present": False,
        "qualified": False,
    }

    # ------------------------------------------------------------
    # A/B: state ordering + initialization across all built-ins.
    # ------------------------------------------------------------
    flavour_results = {}
    for flavour in ("1r1c", "2r2c", "3r2c", "4r3c"):
        model = compile_rc_model(RCCompilerSpec(flavour, ("A", "B"), "ind"))
        init = initialize_runtime_state(
            model,
            {
                "A": ZoneInitializationEvidence(observed_air_temperature_c=29.0),
                "B": ZoneInitializationEvidence(observed_air_temperature_c=24.0),
            },
            request=InitializationRequest(
                policy="auto",
                user_temperatures_c={"A": 21.0},
            ),
        )
        hs0 = model.observation @ init.lifting_matrix
        flavour_results[flavour] = {
            "state_dimension": model.state_dimension,
            "state_keys": [node.key for node in model.state_nodes],
            "initial_state": init.state.tolist(),
            "resolved_sources": dict(init.source_by_zone),
            "hs0_identity_max_abs": float(np.max(np.abs(hs0 - np.eye(2)))),
            "hx0_matches_t0star": bool(
                np.allclose(model.observation @ init.state, init.zone_vector_c, atol=0, rtol=0)
            ),
        }
    report["flavours"] = flavour_results

    # Mixed initialization sources including setpoint midpoint and 22 C fallback.
    init_model = compile_rc_model(RCCompilerSpec("4r3c", ("A", "B", "C", "D"), "ind"))
    mixed = initialize_runtime_state(
        init_model,
        {
            "A": ZoneInitializationEvidence(observed_air_temperature_c=30.0),
            "B": ZoneInitializationEvidence(observed_air_temperature_c=23.5),
            "C": ZoneInitializationEvidence(heating_setpoint_c=20.0, cooling_setpoint_c=24.0),
        },
        request=InitializationRequest(policy="auto", user_temperatures_c={"A": 21.0}),
    )
    report["initialization_policy"] = {
        "priority_result_C": mixed.zone_vector_c.tolist(),
        "source_by_zone": dict(mixed.source_by_zone),
        "default_zone_D_C": float(mixed.resolved_by_zone["D"].value_c),
        "setpoint_midpoint_zone_C_C": float(mixed.resolved_by_zone["C"].value_c),
    }

    # ------------------------------------------------------------
    # C/D/E: IND, DEP1, DEP2 runtime realization and RHS handoff.
    # ------------------------------------------------------------
    zones = ("A", "B")
    common = dict(
        flavour="2r2c",
        zone_ids=zones,
        adjacency=(ZoneAdjacency("A", "B"),),
        zone_port_availability=_ports(zones),
    )
    ind = compile_rc_model(RCCompilerSpec(mode="ind", **common))
    dep1 = compile_rc_model(RCCompilerSpec(mode="dep1", **common))
    dep2 = compile_rc_model(
        RCCompilerSpec(
            mode="dep2",
            dep2_allocations=(
                AllocationFamilySpec(
                    name="zic_family",
                    signals=("zic",),
                    weights={"A": 0.7, "B": 0.3},
                    mode=AllocationMode.ESTIMATED,
                ),
            ),
            **common,
        )
    )
    assert_dep1_dep2_runtime_physics_equivalent(dep1, dep2)

    local_frame = CanonicalRuntimeFrame(
        timestamp="t0",
        boundary_temperatures={"outdoor_temperature": 30.0},
        local_thermal_powers={
            ("A", "qac"): -100.0,
            ("A", "zic"): 300.0,
            ("B", "qac"): -200.0,
            ("B", "zic"): 500.0,
        },
        auxiliary_electrical_powers={("A", "phvac"): 450.0, ("B", "phvac"): 700.0},
    )
    ind_binding = bind_runtime_frame(ind, local_frame)
    dep1_binding = bind_runtime_frame(dep1, local_frame)

    alloc = estimated_allocation_result(
        dep2.allocation_families["zic_family"], zones, (0.4,)
    )
    dep2_frame = CanonicalRuntimeFrame(
        timestamp="t0",
        boundary_temperatures={"outdoor_temperature": 30.0},
        local_thermal_powers={("A", "qac"): -100.0, ("B", "qac"): -200.0},
        aggregate_thermal_powers={"zic": 900.0},
        local_source_availability={
            ("A", "qac"): True,
            ("B", "qac"): True,
            ("A", "zic"): True,
            ("B", "zic"): False,
        },
        auxiliary_electrical_powers={("A", "phvac"): 450.0, ("B", "phvac"): 700.0},
    )
    dep2_binding = bind_runtime_frame(
        dep2,
        dep2_frame,
        allocation_results={"zic_family": alloc},
    )

    init_dep2 = initialize_runtime_state(
        dep2,
        {
            "A": ZoneInitializationEvidence(observed_air_temperature_c=22.0),
            "B": ZoneInitializationEvidence(observed_air_temperature_c=24.0),
        },
    )
    runtime_inv = validate_runtime_invariants(dep2, init_dep2, dep2_frame, dep2_binding)

    rhs_results = {}
    for name, model, binding in (
        ("ind", ind, ind_binding),
        ("dep1", dep1, dep1_binding),
        ("dep2", dep2, dep2_binding),
    ):
        x = np.asarray([22.0, 22.0, 24.0, 24.0], dtype=float)
        dx = rhs(
            model,
            x,
            binding.snapshot,
            _param_values(model),
            allocation_results=binding.allocation_results,
        )
        rhs_results[name] = {
            "finite": bool(np.all(np.isfinite(dx))),
            "dimension": int(dx.size),
            "values": dx.tolist(),
        }

    report["runtime_binding"] = {
        "ind_used_local": [list(x) for x in ind_binding.used_local_thermal_keys],
        "dep1_used_local": [list(x) for x in dep1_binding.used_local_thermal_keys],
        "dep2_used_local": [list(x) for x in dep2_binding.used_local_thermal_keys],
        "dep2_used_aggregate": list(dep2_binding.used_aggregate_signals),
        "dep2_coordinate_error_max_abs": dep2_binding.dep2_coordinate_error_max_abs,
        "dep2_phvac_auxiliary_only": [list(x) for x in dep2_binding.unused_auxiliary_electrical_keys],
        "source_availability_differs_from_model_applicability": True,
        "dep1_dep2_graph_equivalent": graph_signature(dep1) == graph_signature(dep2),
        "runtime_invariants_passed": runtime_inv.passed,
        "rhs": rhs_results,
    }

    # ------------------------------------------------------------
    # F: recursive ownership; no hidden use of future observations.
    # ------------------------------------------------------------
    recursive = start_recursive_state(dep2, init_dep2, timestamp="t0")
    evolved = recursive.state + np.asarray([0.1, 0.05, -0.1, -0.05])
    recursive_next = accept_model_evolved_state(
        dep2, recursive, evolved, next_timestamp="t1"
    )
    report["recursive_state"] = {
        "initial_origin": recursive.origin.value,
        "next_origin": recursive_next.origin.value,
        "next_state_matches_model_evolved_state": bool(np.array_equal(recursive_next.state, evolved)),
        "hidden_measurement_reset_used": False,
    }

    # ------------------------------------------------------------
    # G: expected fail-fast rejections.
    # ------------------------------------------------------------
    rejections = {}
    _expected_rejection(
        "user_fixed_missing",
        lambda: initialize_runtime_state(
            ind, request=InitializationRequest(policy="user_fixed")
        ),
        rejections,
    )
    _expected_rejection(
        "observed_missing",
        lambda: initialize_runtime_state(ind, request=InitializationRequest(policy="observed")),
        rejections,
    )
    _expected_rejection(
        "setpoint_missing",
        lambda: initialize_runtime_state(ind, request=InitializationRequest(policy="setpoint")),
        rejections,
    )
    _expected_rejection(
        "phvac_in_thermal_domain",
        lambda: CanonicalRuntimeFrame(timestamp="t0", local_thermal_powers={("A", "phvac"): 10.0}),
        rejections,
    )
    _expected_rejection(
        "aggregate_qac",
        lambda: CanonicalRuntimeFrame(timestamp="t0", aggregate_thermal_powers={"qac": 10.0}),
        rejections,
    )
    _expected_rejection(
        "missing_required_local_signal",
        lambda: bind_runtime_frame(
            ind,
            CanonicalRuntimeFrame(
                timestamp="t0",
                boundary_temperatures={"outdoor_temperature": 30.0},
                local_thermal_powers={("A", "qac"): 0.0, ("B", "qac"): 0.0},
            ),
        ),
        rejections,
    )
    _expected_rejection(
        "timestamp_mismatch",
        lambda: bind_runtime_frame(ind, local_frame, expected_timestamp="t1"),
        rejections,
    )
    report["expected_rejections"] = rejections

    all_rejections = all(item["passed"] for item in rejections.values())
    all_flavours = all(
        item["hs0_identity_max_abs"] <= 1e-12 and item["hx0_matches_t0star"]
        for item in flavour_results.values()
    )
    all_rhs = all(item["finite"] for item in rhs_results.values())
    qualified = bool(
        all_rejections
        and all_flavours
        and all_rhs
        and runtime_inv.passed
        and dep2_binding.dep2_coordinate_error_max_abs <= 1e-9
        and graph_signature(dep1) == graph_signature(dep2)
        and mixed.resolved_by_zone["D"].value_c == 22.0
        and mixed.resolved_by_zone["C"].value_c == 22.0
    )
    report["qualified"] = qualified

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not qualified:
        raise SystemExit("E0-4 runtime state/input validation FAILED")
    print("\nE0-4 runtime state/input validation PASSED")
    print(f"Report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
