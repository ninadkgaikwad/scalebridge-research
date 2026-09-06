from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    CanonicalRuntimeFrame,
    CommonDiscretizationEngine,
    ConnectionRule,
    DiscretizationConfig,
    DiscretizationError,
    RCCompilerSpec,
    RuntimeStateOrigin,
    ZoneAdjacency,
    ZoneInitializationEvidence,
    analytical_1r1c_step,
    available_solver_names,
    bind_runtime_frame,
    compile_linear_state_space,
    compile_rc_model,
    estimated_allocation_result,
    initialize_runtime_state,
    rhs,
    solver_capabilities,
    start_recursive_state,
)


def _parameter_values(model, *, c_scale=1.0, r_scale=1.0):
    values = {}
    for inst in model.parameter_registry.instances:
        if inst.physical_type == "capacitance":
            base = 1.0e6 if inst.family == "C_a" else 5.0e6
            values[inst.instance_id] = base * c_scale
        elif inst.physical_type == "resistance":
            values[inst.instance_id] = 0.01 * r_scale
        elif inst.family == "eta_r":
            values[inst.instance_id] = 0.7
        elif inst.family == "gamma_a_r":
            values[inst.instance_id] = 0.2
        elif inst.family == "gamma_e_r":
            values[inst.instance_id] = 0.3
        elif inst.family == "gamma_m_r":
            values[inst.instance_id] = 0.5
        else:
            raise AssertionError(inst)
    return values


def _ports(zones, signals=("qac", "zic")):
    return {zone: tuple(signals) for zone in zones}


def _model(flavour="2r2c", zones=("A", "B"), mode="dep1", *, ports=None, rules=()):
    adjacency = None
    if len(zones) > 1:
        adjacency = tuple(ZoneAdjacency(zones[i], zones[i + 1]) for i in range(len(zones) - 1))
    return compile_rc_model(
        RCCompilerSpec(
            flavour=flavour,
            zone_ids=tuple(zones),
            mode=mode,
            adjacency=adjacency,
            connection_rules=tuple(rules),
            zone_port_availability=ports or _ports(zones),
        )
    )


def _binding(model, *, timestamp="t0"):
    local = {}
    for port in model.thermal_ports:
        local[(port.zone_id, port.signal)] = 250.0 if port.signal == "zic" else -400.0
    frame = CanonicalRuntimeFrame(
        timestamp=timestamp,
        boundary_temperatures={"outdoor_temperature": 30.0},
        local_thermal_powers=local,
    )
    return bind_runtime_frame(model, frame)


def test_solver_registry_exposes_all_audited_native_fixed_step_methods_plus_exact():
    names = set(available_solver_names())
    assert names == {
        "euler",
        "euler_trap",
        "rk2",
        "rk4",
        "rk4_trap",
        "luther",
        "runge_kutta_fehlberg",
        "exact_zoh_linear",
    }
    default = [item.key for item in solver_capabilities() if item.default]
    assert default == ["rk4"]


def test_registry_does_not_expose_incompatible_neuromancer_families():
    names = set(available_solver_names())
    for excluded in (
        "DiffEqIntegrator",
        "MultiStep_PredictorCorrector",
        "LeapFrog",
        "Yoshida4",
        "BasicSDEIntegrator",
        "LatentSDEIntegrator",
    ):
        assert excluded.lower() not in names


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_sample_dt_rejected(bad):
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    engine = CommonDiscretizationEngine(model, _parameter_values(model))
    with pytest.raises(DiscretizationError):
        engine.step_tensor([22.0], [10.0], [1000.0], sample_dt_s=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_substeps_rejected(bad):
    with pytest.raises(DiscretizationError):
        DiscretizationConfig(substeps=bad)


def test_compiled_linear_state_space_matches_authoritative_e0_3_rhs():
    model = _model("4r3c", ("A", "B"), "dep1")
    params = _parameter_values(model)
    binding = _binding(model)
    linear = compile_linear_state_space(model, params).to_torch(dtype=torch.float64, device="cpu")
    x_np = np.linspace(21.0, 25.0, model.state_dimension)
    got = linear.rhs(
        torch.tensor(x_np, dtype=torch.float64).unsqueeze(0),
        torch.tensor(binding.boundary_vector, dtype=torch.float64).unsqueeze(0),
        torch.tensor(binding.effective_thermal_vector, dtype=torch.float64).unsqueeze(0),
    )[0].detach().numpy()
    expected = rhs(model, x_np, binding.snapshot, params)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-14)


def test_exact_oracle_matches_scalar_1r1c_analytical_solution():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    params = _parameter_values(model)
    engine = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear")
    )
    got = engine.step_tensor(
        torch.tensor([22.0], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor([1000.0], dtype=torch.float64),
        sample_dt_s=600.0,
    ).state[0, 0]
    expected = analytical_1r1c_step(
        torch.tensor(22.0, dtype=torch.float64),
        10.0,
        1000.0,
        resistance_k_per_w=0.01,
        capacitance_j_per_k=1.0e6,
        sample_dt_s=600.0,
    )
    assert float(got) == pytest.approx(float(expected), abs=1e-12)
    assert float(got) == pytest.approx(21.883529067168497, abs=1e-12)


@pytest.mark.parametrize("flavour", ["1r1c", "2r2c", "3r2c", "4r3c"])
@pytest.mark.parametrize("zone_count", [1, 2, 4])
def test_exact_oracle_is_topology_agnostic_for_all_compiled_flavours(flavour, zone_count):
    zones = tuple(chr(ord("A") + i) for i in range(zone_count))
    mode = "ind" if zone_count == 1 else "dep1"
    rules = ()
    if flavour == "4r3c" and zone_count > 1:
        rules = (ConnectionRule("a", "e"), ConnectionRule("m", "m"))
    model = _model(flavour, zones, mode, rules=rules)
    params = _parameter_values(model)
    engine = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear")
    )
    result = engine.step_tensor(
        torch.linspace(21.0, 24.0, model.state_dimension, dtype=torch.float64),
        torch.tensor([30.0], dtype=torch.float64),
        torch.zeros(len(model.thermal_ports), dtype=torch.float64),
        sample_dt_s=300.0,
    )
    assert result.state.shape == (1, model.state_dimension)
    assert torch.all(torch.isfinite(result.state))


def test_exact_zoh_semigroup_full_step_equals_repeated_substeps():
    model = _model("3r2c", ("A", "B", "C"), "dep1")
    params = _parameter_values(model)
    engine = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear")
    )
    x0 = torch.linspace(20.0, 25.0, model.state_dimension, dtype=torch.float64)
    tb = torch.tensor([31.0], dtype=torch.float64)
    q = torch.linspace(-300.0, 500.0, len(model.thermal_ports), dtype=torch.float64)
    full = engine.step_tensor(x0, tb, q, sample_dt_s=600.0).state
    repeated = x0
    for _ in range(4):
        repeated = engine.step_tensor(repeated, tb, q, sample_dt_s=150.0).state
    torch.testing.assert_close(full, repeated, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "solver",
    ["euler", "euler_trap", "rk2", "rk4", "rk4_trap", "luther", "runge_kutta_fehlberg"],
)
def test_all_audited_neuromancer_native_solvers_execute(solver):
    model = _model("2r2c", ("A",), "ind")
    engine = CommonDiscretizationEngine(
        model, _parameter_values(model), config=DiscretizationConfig(solver=solver, substeps=2)
    )
    result = engine.step_tensor(
        torch.tensor([22.0, 22.0], dtype=torch.float64),
        torch.tensor([30.0], dtype=torch.float64),
        torch.tensor([-300.0, 200.0], dtype=torch.float64),
        sample_dt_s=10.0,
    )
    assert result.state.shape == (1, 2)
    assert torch.all(torch.isfinite(result.state))
    assert result.provenance.backend == "neuromancer"
    assert result.provenance.input_hold == "zoh_left"
    assert result.provenance.integration_h_s == pytest.approx(5.0)


def test_rk4_converges_quickly_to_exact_linear_oracle():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    params = _parameter_values(model, c_scale=0.1)
    exact_engine = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear")
    )
    exact = exact_engine.step_tensor([22.0], [10.0], [1000.0], sample_dt_s=600.0).state
    errors = []
    for substeps in (1, 2, 4):
        rk = CommonDiscretizationEngine(
            model, params, config=DiscretizationConfig(solver="rk4", substeps=substeps)
        )
        got = rk.step_tensor([22.0], [10.0], [1000.0], sample_dt_s=600.0).state
        errors.append(float(torch.max(torch.abs(got - exact))))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]


def test_euler_error_decreases_with_substeps():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    params = _parameter_values(model, c_scale=0.1)
    exact = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear")
    ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=100.0).state
    errors = []
    for substeps in (1, 2, 4, 8):
        got = CommonDiscretizationEngine(
            model, params, config=DiscretizationConfig(solver="euler", substeps=substeps)
        ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=100.0).state
        errors.append(float(torch.max(torch.abs(got - exact))))
    assert all(right < left for left, right in zip(errors, errors[1:]))


def test_diagnostics_off_does_not_call_exact_oracle_for_neuromancer(monkeypatch):
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    engine = CommonDiscretizationEngine(
        model, _parameter_values(model), config=DiscretizationConfig(solver="rk4")
    )
    import scalebridge.models.grey_box.rc_networks.discretization.diagnostics as diag_module

    class Bomb:
        def __init__(self, *args, **kwargs):
            raise AssertionError("oracle should not be constructed in fast path")

    monkeypatch.setattr(diag_module, "ExactZOHLinearIntegrator", Bomb)
    result = engine.step_tensor([22.0], [10.0], [1000.0], sample_dt_s=60.0)
    assert result.diagnostics.enabled is False


def test_diagnostics_do_not_change_integrated_state():
    model = _model("2r2c", ("A", "B"), "dep1")
    params = _parameter_values(model)
    args = dict(
        state=torch.tensor([22.0, 22.0, 24.0, 24.0], dtype=torch.float64),
        boundary=torch.tensor([31.0], dtype=torch.float64),
        thermal=torch.tensor([-300.0, 200.0, -400.0, 250.0], dtype=torch.float64),
        sample_dt_s=300.0,
    )
    normal = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="rk4", substeps=3, diagnostics_per_step=False)
    ).step_tensor(**args)
    diag = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="rk4", substeps=3, diagnostics_per_step=True)
    ).step_tensor(**args)
    torch.testing.assert_close(normal.state, diag.state, rtol=0.0, atol=0.0)
    assert diag.diagnostics.enabled is True
    assert diag.diagnostics.exact_oracle_available is True


def test_euler_stability_diagnostic_recommends_substeps_for_stiff_case():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    # C=1e4, R=0.01 => mu_max = 0.01 s^-1; dt*mu=6.
    params = _parameter_values(model, c_scale=0.01)
    result = CommonDiscretizationEngine(
        model,
        params,
        config=DiscretizationConfig(solver="euler", substeps=1, diagnostics_per_step=True),
    ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=600.0)
    assert result.diagnostics.stability_check_available is True
    assert result.diagnostics.recommended_minimum_substeps == 4
    assert result.diagnostics.stability_passed is False


def test_non_frozen_stability_theory_reports_unavailable_not_failure():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    result = CommonDiscretizationEngine(
        model,
        _parameter_values(model),
        config=DiscretizationConfig(solver="luther", diagnostics_per_step=True),
    ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=60.0)
    assert result.diagnostics.stability_check_available is False
    assert result.diagnostics.stability_passed is None
    assert result.diagnostics.notes


def test_runge_kutta_fehlberg_reports_local_error_only_when_diagnostics_enabled():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    params = _parameter_values(model)
    diag = CommonDiscretizationEngine(
        model,
        params,
        config=DiscretizationConfig(
            solver="runge_kutta_fehlberg", substeps=2, diagnostics_per_step=True
        ),
    ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=60.0)
    assert diag.diagnostics.local_error_linf_abs is not None
    normal = CommonDiscretizationEngine(
        model,
        params,
        config=DiscretizationConfig(
            solver="runge_kutta_fehlberg", substeps=2, diagnostics_per_step=False
        ),
    ).step_tensor([22.0], [10.0], [1000.0], sample_dt_s=60.0)
    assert normal.diagnostics.local_error_linf_abs is None


def test_rk4_tensor_path_is_differentiable_in_state_and_forcing():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    engine = CommonDiscretizationEngine(
        model, _parameter_values(model), config=DiscretizationConfig(solver="rk4")
    )
    x = torch.tensor([22.0], dtype=torch.float64, requires_grad=True)
    tb = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([1000.0], dtype=torch.float64, requires_grad=True)
    out = engine.step_tensor(x, tb, q, sample_dt_s=60.0).state.sum()
    out.backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert tb.grad is not None and torch.all(torch.isfinite(tb.grad))
    assert q.grad is not None and torch.all(torch.isfinite(q.grad))


def test_exact_tensor_path_is_differentiable_in_state_and_forcing():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    engine = CommonDiscretizationEngine(
        model, _parameter_values(model), config=DiscretizationConfig(solver="exact_zoh_linear")
    )
    x = torch.tensor([22.0], dtype=torch.float64, requires_grad=True)
    tb = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([1000.0], dtype=torch.float64, requires_grad=True)
    out = engine.step_tensor(x, tb, q, sample_dt_s=60.0).state.sum()
    out.backward()
    assert x.grad is not None
    assert tb.grad is not None
    assert q.grad is not None


def test_variable_canonical_intervals_are_supported_with_fixed_h_per_call():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    engine = CommonDiscretizationEngine(
        model, _parameter_values(model), config=DiscretizationConfig(solver="rk4", substeps=3)
    )
    a = engine.step_tensor([22.0], [10.0], [1000.0], sample_dt_s=300.0)
    b = engine.step_tensor([22.0], [10.0], [1000.0], sample_dt_s=600.0)
    assert a.provenance.integration_h_s == pytest.approx(100.0)
    assert b.provenance.integration_h_s == pytest.approx(200.0)


def test_runtime_step_consumes_e0_4_binding_and_preserves_model_owned_state_semantics():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    params = _parameter_values(model)
    init = initialize_runtime_state(
        model, {"A": ZoneInitializationEvidence(observed_air_temperature_c=22.0)}
    )
    current = start_recursive_state(model, init, timestamp="t0")
    frame = CanonicalRuntimeFrame(
        timestamp="t0",
        boundary_temperatures={"outdoor_temperature": 10.0},
        local_thermal_powers={("A", "qac"): 1000.0},
    )
    binding = bind_runtime_frame(model, frame)
    result = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear")
    ).step_runtime(
        current,
        binding,
        next_timestamp="t1",
        sample_dt_s=600.0,
    )
    assert result.runtime_state.timestamp == "t1"
    assert result.runtime_state.origin is RuntimeStateOrigin.MODEL_EVOLUTION
    assert result.runtime_state.state[0] == pytest.approx(21.883529067168497, abs=1e-12)


def test_dep2_effective_forcing_uses_same_e0_5_engine_without_mode_specific_logic():
    zones = ("A", "B")
    family = AllocationFamilySpec(
        name="non_hvac",
        signals=("zic",),
        weights={"A": 0.4, "B": 0.6},
        mode=AllocationMode.ESTIMATED,
    )
    model = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            zones,
            "dep2",
            adjacency=(ZoneAdjacency("A", "B"),),
            zone_port_availability=_ports(zones),
            dep2_allocations=(family,),
        )
    )
    allocation = {
        "non_hvac": estimated_allocation_result(family, zones, logits=(0.0,))
    }
    frame = CanonicalRuntimeFrame(
        timestamp="t0",
        boundary_temperatures={"outdoor_temperature": 30.0},
        local_thermal_powers={("A", "qac"): -500.0, ("B", "qac"): -800.0},
        aggregate_thermal_powers={"zic": 1000.0},
    )
    binding = bind_runtime_frame(model, frame, allocation_results=allocation)
    result = CommonDiscretizationEngine(
        model, _parameter_values(model), config=DiscretizationConfig(solver="exact_zoh_linear")
    ).step_binding_tensor(
        torch.tensor([22.0, 24.0], dtype=torch.float64),
        binding,
        sample_dt_s=300.0,
    )
    assert result.state.shape == (1, 2)
    assert torch.all(torch.isfinite(result.state))


def test_equilibrium_is_preserved_by_core_methods():
    model = _model("1r1c", ("A",), "ind", ports=_ports(("A",), ("qac",)))
    params = _parameter_values(model)
    # For To=10, R=.01, Q=1000, equilibrium is 20 C.
    for solver in ("euler", "rk4", "exact_zoh_linear"):
        result = CommonDiscretizationEngine(
            model, params, config=DiscretizationConfig(solver=solver, substeps=3)
        ).step_tensor([20.0], [10.0], [1000.0], sample_dt_s=600.0)
        assert result.state[0, 0].item() == pytest.approx(20.0, abs=1e-12)

def test_exact_selected_substeps_are_executed_and_provenance_is_truthful():
    model = _model("2r2c", ("A", "B"), "dep1")
    params = _parameter_values(model)
    x = torch.tensor([22.0, 22.0, 24.0, 24.0], dtype=torch.float64)
    tb = torch.tensor([30.0], dtype=torch.float64)
    q = torch.tensor([-400.0, 250.0, -400.0, 250.0], dtype=torch.float64)
    one = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear", substeps=1)
    ).step_tensor(x, tb, q, sample_dt_s=600.0)
    four = CommonDiscretizationEngine(
        model, params, config=DiscretizationConfig(solver="exact_zoh_linear", substeps=4)
    ).step_tensor(x, tb, q, sample_dt_s=600.0)
    torch.testing.assert_close(one.state, four.state, rtol=1e-12, atol=1e-12)
    assert four.provenance.integration_h_s == pytest.approx(150.0)
    assert four.provenance.substeps == 4
