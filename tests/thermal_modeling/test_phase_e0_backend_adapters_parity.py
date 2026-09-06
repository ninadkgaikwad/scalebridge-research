from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    ParameterConfig,
    ParameterSharingRule,
    ParameterStatus,
    RCCompilerSpec,
    SpatialMode,
    ZoneAdjacency,
    compile_rc_model,
)
from scalebridge.models.grey_box.rc_networks.backend_adapters import (
    BackendAdapterError,
    CasadiRCBackend,
    FLOAT64_DERIVATIVE_TOLERANCE,
    FLOAT64_DISCRETE_TOLERANCE,
    FLOAT64_VALUE_TOLERANCE,
    NeuromancerRCBackend,
    NumpyRCBackend,
    TorchRCBackend,
    build_parameterization_plan,
    normalized_linf_error,
)


def _ports(zones):
    return {z: ("qac", "zic", "zir", "qsol1", "qsol2") for z in zones}


def _default_family_value(family: str, zone: str | None = None) -> float:
    values = {
        "C_a": 2.0e6,
        "C_m": 8.0e6,
        "C_e": 5.0e6,
        "R_ao": 0.02,
        "R_am": 0.01,
        "R_om": 0.04,
        "R_ae": 0.015,
        "R_eo": 0.03,
        "R_inter_a_a": 0.025,
        "eta_r": 0.7,
        "gamma_a_r": 0.2,
        "gamma_e_r": 0.3,
        "gamma_m_r": 0.5,
    }
    return values[family]


def _configured_model(spec, *, estimated_families=(), value_overrides=None, bound_overrides=None):
    provisional = compile_rc_model(spec)
    value_overrides = value_overrides or {}
    bound_overrides = bound_overrides or {}
    configs = {}
    for master in provisional.parameter_registry.masters:
        zone = provisional.parameter_registry.instance(master.member_instance_ids[0]).zone_scope[0]
        key = (master.family, zone)
        value = value_overrides.get(key, value_overrides.get(master.family, _default_family_value(master.family, zone)))
        bounds = bound_overrides.get(key, bound_overrides.get(master.family, (None, None)))
        configs[master.master_id] = ParameterConfig(
            status=(ParameterStatus.ESTIMATED if master.family in set(estimated_families) else ParameterStatus.FIXED),
            initial_value=float(value),
            lower_bound=bounds[0],
            upper_bound=bounds[1],
        )
    return compile_rc_model(spec, parameter_configs=configs)


@pytest.mark.parametrize("flavour", ["1r1c", "2r2c", "3r2c", "4r3c"])
def test_zero_raw_reconstructs_e03_matrices_for_every_flavour(flavour):
    estimated = {
        "1r1c": ("C_a", "R_ao"),
        "2r2c": ("C_a", "R_ao", "eta_r"),
        "3r2c": ("C_m", "R_om", "eta_r"),
        "4r3c": ("C_e", "R_ae", "gamma_a_r", "gamma_e_r", "gamma_m_r"),
    }[flavour]
    model = _configured_model(
        RCCompilerSpec(flavour, ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated_families=estimated,
    )
    reference = model.matrices(model.parameter_registry.initial_master_values())
    backend = NumpyRCBackend(model)
    got = backend.matrices(backend.zero_raw())
    np.testing.assert_allclose(got.C, reference.C, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.L_CC, reference.L_CC, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.L_CB, reference.L_CB, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.Gamma, reference.Gamma, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.H, reference.H, rtol=0, atol=0)


def test_current_five_channel_semantics_are_inherited_from_flavour_metadata():
    model = _configured_model(
        RCCompilerSpec("2r2c", ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated_families=("eta_r",),
        value_overrides={"eta_r": 0.7},
    )
    backend = NumpyRCBackend(model)
    gamma = backend.matrices(backend.zero_raw()).Gamma
    pidx = model.port_index
    np.testing.assert_array_equal(gamma[:, pidx["A::qac"]], [1.0, 0.0])
    np.testing.assert_array_equal(gamma[:, pidx["A::zic"]], [1.0, 0.0])
    np.testing.assert_array_equal(gamma[:, pidx["A::qsol1"]], [1.0, 0.0])
    np.testing.assert_allclose(gamma[:, pidx["A::zir"]], [0.3, 0.7], atol=1e-14)
    np.testing.assert_allclose(gamma[:, pidx["A::qsol2"]], [0.3, 0.7], atol=1e-14)


def test_4r3c_radiative_simplex_uses_two_raw_dof_and_stays_normalized():
    model = _configured_model(
        RCCompilerSpec("4r3c", ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated_families=("gamma_a_r", "gamma_e_r", "gamma_m_r"),
    )
    plan = build_parameterization_plan(model)
    group = plan.simplex_parameters[0]
    assert len(group.raw_indices) == 2
    backend = NumpyRCBackend(model)
    rho = backend.zero_raw()
    rho[list(group.raw_indices)] = [0.4, -0.2]
    values = backend.master_values(rho)
    gamma_values = np.asarray([values[mid] for mid in group.master_ids])
    assert np.all(gamma_values > 0.0)
    assert gamma_values.sum() == pytest.approx(1.0)


def test_shared_scalar_master_is_one_raw_coordinate_and_exactly_shared():
    spec = RCCompilerSpec(
        "2r2c",
        ("A", "B"),
        "dep1",
        adjacency=(ZoneAdjacency("A", "B"),),
        zone_port_availability=_ports(("A", "B")),
        parameter_sharing=(ParameterSharingRule(name="shared_ram", family="R_am"),),
    )
    model = _configured_model(spec, estimated_families=("R_am",))
    backend = NumpyRCBackend(model)
    matching = [c for c in backend.plan.raw_coordinates if c.owner_id == "shared|shared_ram"]
    assert len(matching) == 1
    rho = backend.zero_raw()
    rho[matching[0].index] = math.log(1.5)
    values = backend.master_values(rho)
    assert values["shared|shared_ram"] == pytest.approx(0.015)
    mats = backend.matrices(rho)
    assert np.all(np.isfinite(mats.A))


def test_bounded_eta_and_positive_rc_transforms_are_centered_and_feasible():
    model = _configured_model(
        RCCompilerSpec("2r2c", ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated_families=("C_a", "R_ao", "eta_r"),
        bound_overrides={"R_ao": (0.005, 0.04)},
    )
    backend = NumpyRCBackend(model)
    baseline = backend.master_values(backend.zero_raw())
    rho = backend.zero_raw()
    for coord in backend.plan.raw_coordinates:
        rho[coord.index] = 3.0
    moved = backend.master_values(rho)
    assert baseline[next(mid for mid in baseline if "C_a" in mid)] == pytest.approx(2e6)
    rmid = next(mid for mid in moved if "R_ao" in mid)
    emid = next(mid for mid in moved if "eta_r" in mid)
    assert 0.005 < moved[rmid] < 0.04
    assert 0.0 < moved[emid] < 1.0


def _dep2_model():
    zones = ("A", "B")
    spec = RCCompilerSpec(
        "2r2c",
        zones,
        SpatialMode.DEP2,
        adjacency=(ZoneAdjacency("A", "B"),),
        zone_port_availability=_ports(zones),
        dep2_allocations=(
            AllocationFamilySpec(
                "zic_family", ("zic",), {"A": 0.4, "B": 0.6}, AllocationMode.NEUTRAL_FIXED
            ),
            AllocationFamilySpec(
                "zir_family", ("zir",), {"A": 0.4, "B": 0.6}, AllocationMode.ESTIMATED
            ),
            AllocationFamilySpec(
                "sol1_family", ("qsol1",), {"A": 0.4, "B": 0.6}, AllocationMode.NEUTRAL_FIXED
            ),
            AllocationFamilySpec(
                "sol2_family", ("qsol2",), {"A": 0.4, "B": 0.6}, AllocationMode.NEUTRAL_FIXED
            ),
        ),
    )
    return _configured_model(
        spec,
        estimated_families=("C_a", "R_ao", "R_am", "eta_r"),
        value_overrides={
            ("C_a", "A"): 1.0e6,
            ("C_a", "B"): 1.2e6,
            ("C_m", "A"): 5.0e6,
            ("C_m", "B"): 6.0e6,
            ("R_ao", "A"): 0.01,
            ("R_ao", "B"): 0.012,
            ("R_am", "A"): 0.02,
            ("R_am", "B"): 0.02,
            ("eta_r", "A"): 0.8,
            ("eta_r", "B"): 0.7,
            "R_inter_a_a": 0.03,
        },
    )


def _running_inputs(model):
    local = np.zeros(len(model.thermal_ports), dtype=float)
    local[model.port_index["A::qac"]] = -1200.0
    local[model.port_index["B::qac"]] = -800.0
    # DEP2 non-HVAC local entries are deliberately garbage: they must be ignored.
    for key, idx in model.port_index.items():
        if not key.endswith("::qac"):
            local[idx] = 999999.0
    aggregate_by_signal = {"zic": 1000.0, "zir": 600.0, "qsol1": 800.0, "qsol2": 300.0}
    return local, aggregate_by_signal


def _aggregate_vector(backend, values):
    return np.asarray([values[s] for s in backend.plan.aggregate_signal_order], dtype=float)


def test_dep2_contribution_simplex_is_centered_and_preserves_coordinate():
    model = _dep2_model()
    backend = NumpyRCBackend(model)
    alloc = next(a for a in backend.plan.allocation_parameters if a.family_name == "zir_family")
    assert len(alloc.raw_indices) == 1
    zero_lam = backend.allocation_lambdas(backend.zero_raw())["zir_family"]
    np.testing.assert_allclose(zero_lam, [1.0, 1.0], atol=1e-14)
    rho = backend.zero_raw()
    rho[alloc.raw_indices[0]] = math.log(2.0)  # Zone A anchor, Zone B tilt -> p=[.25,.75]
    lam = backend.allocation_lambdas(rho)["zir_family"]
    np.testing.assert_allclose(lam, [0.625, 1.25], atol=1e-12)
    assert 0.4 * lam[0] + 0.6 * lam[1] == pytest.approx(1.0)


def test_dep2_keeps_qac_local_and_realizes_all_non_hvac_channels():
    model = _dep2_model()
    backend = NumpyRCBackend(model)
    local, aggregate_by_signal = _running_inputs(model)
    qeff = backend.effective_thermal(
        backend.zero_raw(), local, _aggregate_vector(backend, aggregate_by_signal)
    )
    assert qeff[model.port_index["A::qac"]] == -1200.0
    assert qeff[model.port_index["B::qac"]] == -800.0
    assert qeff[model.port_index["A::zic"]] == pytest.approx(1000.0)
    assert qeff[model.port_index["B::qsol2"]] == pytest.approx(300.0)


def test_numpy_torch_casadi_p0_p1_p2_parity_on_running_dep2_example():
    model = _dep2_model()
    nb = NumpyRCBackend(model)
    tb = TorchRCBackend(model, dtype=torch.float64)
    cb = CasadiRCBackend(model, symbol_type="SX")
    rho = nb.zero_raw()
    # perturb every raw coordinate diversely but moderately
    rho[:] = np.linspace(-0.35, 0.45, len(rho))
    local, aggregate_by_signal = _running_inputs(model)
    agg = _aggregate_vector(nb, aggregate_by_signal)
    x = np.array([22.0, 22.0, 24.0, 24.0])
    boundary = np.array([10.0])

    np_masters = np.asarray([nb.master_values(rho)[mid] for mid in nb.plan.master_order])
    torch_masters = tb.master_vector(torch.tensor(rho, dtype=torch.float64)).detach().numpy()
    casadi_masters = cb.master_values(rho)
    assert normalized_linf_error(np_masters, torch_masters, FLOAT64_VALUE_TOLERANCE).passed
    assert normalized_linf_error(np_masters, casadi_masters, FLOAT64_VALUE_TOLERANCE).passed

    nm = nb.matrices(rho)
    tm = tb.matrices(torch.tensor(rho, dtype=torch.float64))
    cm = cb.matrices(rho)
    for a, b in [
        (nm.C, tm.C.detach().numpy()),
        (nm.L_CC, tm.L_CC.detach().numpy()),
        (nm.L_CB, tm.L_CB.detach().numpy()),
        (nm.Gamma, tm.Gamma.detach().numpy()),
        (nm.A, tm.A.detach().numpy()),
        (nm.B_boundary, tm.B_boundary.detach().numpy()),
        (nm.B_thermal, tm.B_thermal.detach().numpy()),
        (nm.C, cm[0].reshape(-1)),
        (nm.L_CC, cm[1]),
        (nm.L_CB, cm[2]),
        (nm.Gamma, cm[3]),
        (nm.A, cm[5]),
        (nm.B_boundary, cm[6]),
        (nm.B_thermal, cm[7]),
    ]:
        assert normalized_linf_error(a, b, FLOAT64_VALUE_TOLERANCE).passed

    nrhs = nb.rhs(rho, x, boundary, local, agg)
    trhs = tb.rhs(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(boundary, dtype=torch.float64),
        torch.tensor(local, dtype=torch.float64),
        torch.tensor(agg, dtype=torch.float64),
        raw=torch.tensor(rho, dtype=torch.float64),
    ).detach().numpy()
    crhs = cb.rhs(rho, x, boundary, local, agg)
    assert normalized_linf_error(nrhs, trhs, FLOAT64_VALUE_TOLERANCE).passed
    assert normalized_linf_error(nrhs, crhs, FLOAT64_VALUE_TOLERANCE).passed


def test_torch_and_casadi_parameter_gradient_parity():
    model = _dep2_model()
    nb = NumpyRCBackend(model)
    tb = TorchRCBackend(model, dtype=torch.float64)
    cb = CasadiRCBackend(model, symbol_type="MX")
    rho_np = np.linspace(-0.2, 0.3, nb.plan.raw_dimension)
    rho = torch.tensor(rho_np, dtype=torch.float64, requires_grad=True)
    local, aggregate_by_signal = _running_inputs(model)
    agg = _aggregate_vector(nb, aggregate_by_signal)
    x = np.array([22.0, 22.0, 24.0, 24.0])
    boundary = np.array([10.0])
    probe = np.array([1.0, -0.5, 0.75, 0.2])

    out = tb.rhs(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(boundary, dtype=torch.float64),
        torch.tensor(local, dtype=torch.float64),
        torch.tensor(agg, dtype=torch.float64),
        raw=rho,
    )
    loss = torch.dot(torch.tensor(probe, dtype=torch.float64), out)
    torch_grad = torch.autograd.grad(loss, rho)[0].detach().numpy()
    casadi_grad = cb.parameter_probe_gradient(rho_np, x, boundary, local, agg, probe)
    result = normalized_linf_error(torch_grad, casadi_grad, FLOAT64_DERIVATIVE_TOLERANCE)
    assert result.passed, result


@pytest.mark.parametrize("solver", ["euler", "rk2", "rk4", "exact_zoh_linear"])
def test_p5_numpy_torch_casadi_common_solver_parity(solver):
    model = _dep2_model()
    nb = NumpyRCBackend(model)
    tb = TorchRCBackend(model, dtype=torch.float64)
    cb = CasadiRCBackend(model, symbol_type="SX")
    rho = np.linspace(-0.15, 0.25, nb.plan.raw_dimension)
    local, aggregate_by_signal = _running_inputs(model)
    agg = _aggregate_vector(nb, aggregate_by_signal)
    x = np.array([22.0, 22.0, 24.0, 24.0])
    boundary = np.array([10.0])
    n = nb.step(solver, rho, x, boundary, local, agg, sample_dt_s=600.0, substeps=4)
    t = tb.step(
        solver,
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(boundary, dtype=torch.float64),
        torch.tensor(local, dtype=torch.float64),
        torch.tensor(agg, dtype=torch.float64),
        sample_dt_s=600.0,
        substeps=4,
        raw=torch.tensor(rho, dtype=torch.float64),
    ).detach().numpy()
    c = cb.step(solver, rho, x, boundary, local, agg, sample_dt_s=600.0, substeps=4)
    assert normalized_linf_error(n, t, FLOAT64_DISCRETE_TOLERANCE).passed
    assert normalized_linf_error(n, c, FLOAT64_DISCRETE_TOLERANCE).passed


def test_torch_exact_zoh_remains_differentiable_in_live_parameters():
    model = _dep2_model()
    backend = TorchRCBackend(model, dtype=torch.float64)
    local, aggregate_by_signal = _running_inputs(model)
    agg = _aggregate_vector(NumpyRCBackend(model), aggregate_by_signal)
    out = backend.step(
        "exact_zoh_linear",
        torch.tensor([22.0, 22.0, 24.0, 24.0], dtype=torch.float64),
        torch.tensor([10.0], dtype=torch.float64),
        torch.tensor(local, dtype=torch.float64),
        torch.tensor(agg, dtype=torch.float64),
        sample_dt_s=600.0,
        substeps=1,
    )
    grad = torch.autograd.grad(out.sum(), backend.raw)[0]
    assert torch.isfinite(grad).all()
    assert torch.linalg.vector_norm(grad) > 0.0


def test_neuromancer_facade_reuses_exact_torch_parameter_owner_when_available():
    pytest.importorskip("neuromancer")
    model = _dep2_model()
    torch_backend = TorchRCBackend(model, dtype=torch.float64)
    nm = NeuromancerRCBackend(torch_backend)
    ode = nm.ode_system()
    assert ode.backend is torch_backend
    assert nm.raw is torch_backend.raw


def test_missing_parameter_config_fails_before_backend_realization():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    with pytest.raises(BackendAdapterError, match="requires ParameterConfig"):
        build_parameterization_plan(model)


def test_fully_fixed_model_has_zero_raw_dimension_across_backends():
    model = _configured_model(
        RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated_families=(),
    )
    nb = NumpyRCBackend(model)
    tb = TorchRCBackend(model, dtype=torch.float64)
    cb = CasadiRCBackend(model, symbol_type="SX")
    assert nb.plan.raw_dimension == 0
    assert tb.plan.raw_dimension == 0
    assert cb.plan.raw_dimension == 0
    np.testing.assert_allclose(list(nb.master_values([]).values()), cb.master_values([]))


def test_partial_cross_zone_4r3c_simplex_sharing_fails_fast():
    zones = ("A", "B")
    spec = RCCompilerSpec(
        "4r3c",
        zones,
        "dep1",
        adjacency=(ZoneAdjacency("A", "B"),),
        zone_port_availability=_ports(zones),
        parameter_sharing=(ParameterSharingRule(name="only_gamma_a", family="gamma_a_r"),),
    )
    model = _configured_model(
        spec,
        estimated_families=("gamma_a_r", "gamma_e_r", "gamma_m_r"),
    )
    with pytest.raises(BackendAdapterError, match="Partial cross-zone sharing"):
        build_parameterization_plan(model)
