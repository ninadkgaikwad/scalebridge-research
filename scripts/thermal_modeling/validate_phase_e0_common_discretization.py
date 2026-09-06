from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    CanonicalRuntimeFrame,
    CommonDiscretizationEngine,
    DiscretizationConfig,
    RCCompilerSpec,
    ZoneAdjacency,
    analytical_1r1c_step,
    available_solver_names,
    bind_runtime_frame,
    compile_rc_model,
    estimated_allocation_result,
    solver_capabilities,
)


def _values(model, *, c_scale=1.0):
    values = {}
    for inst in model.parameter_registry.instances:
        if inst.physical_type == "capacitance":
            values[inst.instance_id] = (1.0e6 if inst.family == "C_a" else 5.0e6) * c_scale
        elif inst.physical_type == "resistance":
            values[inst.instance_id] = 0.01
        elif inst.family == "eta_r":
            values[inst.instance_id] = 0.7
        elif inst.family == "gamma_a_r":
            values[inst.instance_id] = 0.2
        elif inst.family == "gamma_e_r":
            values[inst.instance_id] = 0.3
        elif inst.family == "gamma_m_r":
            values[inst.instance_id] = 0.5
        else:
            raise RuntimeError(f"No validator value for {inst}")
    return values


def _ports(zones, signals=("qac", "zic")):
    return {z: tuple(signals) for z in zones}


def _model(flavour, zones=("A", "B"), mode="dep1"):
    adjacency = None
    if len(zones) > 1:
        adjacency = tuple(ZoneAdjacency(zones[i], zones[i + 1]) for i in range(len(zones) - 1))
    return compile_rc_model(
        RCCompilerSpec(
            flavour,
            tuple(zones),
            mode,
            adjacency=adjacency,
            zone_port_availability=_ports(zones),
        )
    )


def main() -> int:
    report: dict[str, object] = {
        "phase": "E0-5",
        "fixed_step_only": True,
        "input_hold": "zoh_left",
        "default_solver": "rk4",
        "diagnostics_default": False,
        "qualified": False,
    }

    capabilities = solver_capabilities()
    report["solver_registry"] = [
        {
            "key": item.key,
            "backend": item.backend,
            "class_name": item.class_name,
            "available": item.available,
            "fixed_step": item.fixed_step,
            "first_order_state": item.first_order_state,
            "history_required": item.history_required,
            "default": item.default,
            "exact_linear": item.exact_linear,
            "local_error_available": item.local_error_available,
        }
        for item in capabilities
    ]
    available = set(available_solver_names())
    expected_native = {
        "euler",
        "euler_trap",
        "rk2",
        "rk4",
        "rk4_trap",
        "luther",
        "runge_kutta_fehlberg",
    }
    missing_native = sorted(expected_native - available)
    report["solver_registry_summary"] = {
        "available": sorted(available),
        "missing_expected_neuromancer_1_5_6": missing_native,
        "exact_zoh_linear_available": "exact_zoh_linear" in available,
    }
    if missing_native:
        raise RuntimeError(f"Missing audited Neuromancer 1.5.6 solvers: {missing_native}")

    # Scalar exact oracle.
    scalar = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            ("A",),
            "ind",
            zone_port_availability={"A": ("qac",)},
        )
    )
    scalar_values = _values(scalar)
    exact_engine = CommonDiscretizationEngine(
        scalar,
        scalar_values,
        config=DiscretizationConfig(solver="exact_zoh_linear"),
    )
    exact_state = exact_engine.step_tensor(
        torch.tensor([22.0], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([1000.0], dtype=torch.float64),
        sample_dt_s=600.0,
    ).state[0, 0]
    analytical = analytical_1r1c_step(
        torch.tensor(22.0, dtype=torch.float64),
        10.0,
        1000.0,
        resistance_k_per_w=0.01,
        capacitance_j_per_k=1.0e6,
        sample_dt_s=600.0,
    )
    scalar_error = abs(float(exact_state - analytical))
    report["scalar_1r1c_oracle"] = {
        "exact_zoh_C": float(exact_state),
        "analytical_C": float(analytical),
        "abs_error_C": scalar_error,
        "passed": scalar_error <= 1e-11,
    }
    if scalar_error > 1e-11:
        raise RuntimeError("Exact ZOH failed scalar analytical 1R1C oracle")

    # Graph-general exact oracle across all frozen flavours.
    flavour_results = {}
    for flavour in ("1r1c", "2r2c", "3r2c", "4r3c"):
        model = _model(flavour)
        values = _values(model)
        engine = CommonDiscretizationEngine(
            model,
            values,
            config=DiscretizationConfig(solver="exact_zoh_linear", substeps=4),
        )
        state = torch.linspace(21.0, 24.0, model.state_dimension, dtype=torch.float64)
        boundary = torch.tensor([30.0], dtype=torch.float64)
        thermal = torch.zeros(len(model.thermal_ports), dtype=torch.float64)
        result = engine.step_tensor(state, boundary, thermal, sample_dt_s=600.0)
        finite = bool(torch.all(torch.isfinite(result.state)).item())
        flavour_results[flavour] = {
            "state_dimension": model.state_dimension,
            "boundary_dimension": len(model.boundary_nodes),
            "thermal_dimension": len(model.thermal_ports),
            "finite": finite,
            "integration_h_s": result.provenance.integration_h_s,
        }
        if not finite:
            raise RuntimeError(f"Non-finite exact oracle result for {flavour}")
    report["graph_general_exact_oracle"] = flavour_results

    # Every audited native Neuromancer solver executes the same compiled RC ODE.
    native_results = {}
    model = _model("2r2c", zones=("A",), mode="ind")
    values = _values(model)
    state = torch.tensor([22.0, 22.0], dtype=torch.float64)
    boundary = torch.tensor([30.0], dtype=torch.float64)
    thermal = torch.tensor([-300.0, 200.0], dtype=torch.float64)
    exact_ref = CommonDiscretizationEngine(
        model,
        values,
        config=DiscretizationConfig(solver="exact_zoh_linear"),
    ).step_tensor(state, boundary, thermal, sample_dt_s=60.0).state

    for solver in sorted(expected_native):
        result = CommonDiscretizationEngine(
            model,
            values,
            config=DiscretizationConfig(solver=solver, substeps=4),
        ).step_tensor(state, boundary, thermal, sample_dt_s=60.0)
        err = float(torch.max(torch.abs(result.state - exact_ref)).item())
        native_results[solver] = {
            "finite": bool(torch.all(torch.isfinite(result.state)).item()),
            "linf_vs_exact": err,
            "integration_h_s": result.provenance.integration_h_s,
        }
    report["neuromancer_native_solvers"] = native_results

    # Diagnostics neutrality + stability recommendation.
    stiff_values = _values(scalar, c_scale=0.01)
    normal = CommonDiscretizationEngine(
        scalar,
        stiff_values,
        config=DiscretizationConfig(solver="euler", diagnostics_per_step=False),
    ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=600.0)
    diagnostic = CommonDiscretizationEngine(
        scalar,
        stiff_values,
        config=DiscretizationConfig(solver="euler", diagnostics_per_step=True),
    ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=600.0)
    same_state = bool(torch.equal(normal.state, diagnostic.state))
    report["diagnostics"] = {
        "normal_enabled": normal.diagnostics.enabled,
        "diagnostic_enabled": diagnostic.diagnostics.enabled,
        "state_identical": same_state,
        "recommended_minimum_substeps": diagnostic.diagnostics.recommended_minimum_substeps,
        "stability_passed": diagnostic.diagnostics.stability_passed,
        "exact_oracle_linf_abs": diagnostic.diagnostics.exact_oracle_linf_abs,
    }
    if not same_state:
        raise RuntimeError("Diagnostics changed the integrated E0-5 state")

    # DEP2 is upstream forcing semantics; the E0-5 engine consumes only the bound vectors.
    family = AllocationFamilySpec(
        name="non_hvac",
        signals=("zic",),
        weights={"A": 0.4, "B": 0.6},
        mode=AllocationMode.ESTIMATED,
    )
    dep2 = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            ("A", "B"),
            "dep2",
            adjacency=(ZoneAdjacency("A", "B"),),
            zone_port_availability=_ports(("A", "B")),
            dep2_allocations=(family,),
        )
    )
    allocation = {
        "non_hvac": estimated_allocation_result(family, dep2.spec.zone_ids, logits=(0.0,))
    }
    binding = bind_runtime_frame(
        dep2,
        CanonicalRuntimeFrame(
            timestamp="t0",
            boundary_temperatures={"outdoor_temperature": 30.0},
            local_thermal_powers={("A", "qac"): -500.0, ("B", "qac"): -800.0},
            aggregate_thermal_powers={"zic": 1000.0},
        ),
        allocation_results=allocation,
    )
    dep2_result = CommonDiscretizationEngine(
        dep2,
        _values(dep2),
        config=DiscretizationConfig(solver="rk4", substeps=2),
    ).step_binding_tensor(
        torch.tensor([22.0, 24.0], dtype=torch.float64),
        binding,
        sample_dt_s=300.0,
    )
    report["spatial_mode_neutrality"] = {
        "dep2_finite": bool(torch.all(torch.isfinite(dep2_result.state)).item()),
        "e0_5_consumed_boundary_dimension": int(binding.boundary_vector.size),
        "e0_5_consumed_effective_thermal_dimension": int(binding.effective_thermal_vector.size),
    }

    # Differentiability of common Neuromancer path in state and forcing.
    grad_engine = CommonDiscretizationEngine(
        scalar,
        scalar_values,
        config=DiscretizationConfig(solver="rk4"),
    )
    x = torch.tensor([22.0], dtype=torch.float64, requires_grad=True)
    tb = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([1000.0], dtype=torch.float64, requires_grad=True)
    loss = grad_engine.step_tensor(x, tb, q, sample_dt_s=60.0).state.sum()
    loss.backward()
    gradients_finite = all(
        item.grad is not None and bool(torch.all(torch.isfinite(item.grad)).item())
        for item in (x, tb, q)
    )
    report["differentiability"] = {
        "state_and_forcing_gradients_finite": gradients_finite,
    }
    if not gradients_finite:
        raise RuntimeError("Neuromancer E0-5 tensor path lost differentiation")

    report["qualified"] = True
    path = Path("validated_artifacts/phase_e0/e05_common_discretization_validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nE0-5 common discretization validation PASSED")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
