from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    ParameterConfig,
    ParameterStatus,
    RCCompilerSpec,
    SpatialMode,
    ZoneAdjacency,
    compile_rc_model,
)
from scalebridge.models.grey_box.rc_networks.backend_adapters import (
    CasadiPhysicalRCBackend,
    CasadiTransformedRCBackend,
    NumpyPhysicalRCBackend,
    NumpyRCBackend,
    TorchRCBackend,
    build_physical_parameterization_plan,
)


def _ports(zones):
    return {z: ("qac", "zic", "zir", "qsol1", "qsol2") for z in zones}


def _default_value(family):
    return {
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
    }[family]


def _configured(spec, estimated=(), values=None, bounds=None):
    provisional = compile_rc_model(spec)
    values = values or {}
    bounds = bounds or {}
    configs = {}
    for master in provisional.parameter_registry.masters:
        instance = provisional.parameter_registry.instance(master.member_instance_ids[0])
        zone = instance.zone_scope[0] if instance.zone_scope else None
        val = values.get((master.family, zone), values.get(master.family, _default_value(master.family)))
        lo, hi = bounds.get((master.family, zone), bounds.get(master.family, (None, None)))
        configs[master.master_id] = ParameterConfig(
            status=ParameterStatus.ESTIMATED if master.family in set(estimated) else ParameterStatus.FIXED,
            initial_value=float(val),
            lower_bound=lo,
            upper_bound=hi,
        )
    return compile_rc_model(spec, parameter_configs=configs)


def _dep2_model():
    zones = ("A", "B")
    spec = RCCompilerSpec(
        "2r2c",
        zones,
        SpatialMode.DEP2,
        adjacency=(ZoneAdjacency("A", "B"),),
        zone_port_availability=_ports(zones),
        dep2_allocations=(
            AllocationFamilySpec("zic_family", ("zic",), {"A": 0.4, "B": 0.6}, AllocationMode.NEUTRAL_FIXED),
            AllocationFamilySpec("zir_family", ("zir",), {"A": 0.4, "B": 0.6}, AllocationMode.ESTIMATED),
            AllocationFamilySpec("sol1_family", ("qsol1",), {"A": 0.4, "B": 0.6}, AllocationMode.NEUTRAL_FIXED),
            AllocationFamilySpec("sol2_family", ("qsol2",), {"A": 0.4, "B": 0.6}, AllocationMode.NEUTRAL_FIXED),
        ),
    )
    return _configured(
        spec,
        estimated=("C_a", "R_ao", "R_am", "eta_r"),
        values={
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
        bounds={
            "C_a": (1.0e4, 1.0e8),
            "R_ao": (1.0e-4, 1.0),
            "R_am": (1.0e-4, 1.0),
            "eta_r": (0.0, 1.0),
        },
    )


def _inputs(model, aggregate_order):
    local = np.zeros(len(model.thermal_ports), dtype=float)
    local[model.port_index["A::qac"]] = -1200.0
    local[model.port_index["B::qac"]] = -800.0
    amap = {"zic": 1000.0, "zir": 600.0, "qsol1": 800.0, "qsol2": 300.0}
    aggregate = np.asarray([amap[s] for s in aggregate_order], dtype=float)
    return local, aggregate


@pytest.mark.parametrize("flavour", ["1r1c", "2r2c", "3r2c", "4r3c"])
def test_direct_physical_plan_covers_all_frozen_flavours(flavour):
    estimated = {
        "1r1c": ("C_a", "R_ao"),
        "2r2c": ("C_a", "R_ao", "eta_r"),
        "3r2c": ("C_m", "R_om", "eta_r"),
        "4r3c": ("C_e", "R_ae", "gamma_a_r", "gamma_e_r", "gamma_m_r"),
    }[flavour]
    model = _configured(
        RCCompilerSpec(flavour, ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated=estimated,
    )
    plan = build_physical_parameterization_plan(model)
    pb = NumpyPhysicalRCBackend(model)
    theta0 = pb.initial_physical()
    assert theta0.shape == (plan.decision_dimension,)
    assert np.all(np.isfinite(theta0))
    mats = pb.matrices(theta0)
    assert np.all(np.isfinite(mats.A))
    if flavour == "4r3c":
        assert any(c.constraint_id.startswith("sum|routing_simplex") for c in plan.constraints)


def test_4r3c_physical_simplex_is_explicit_not_softmax_hidden():
    model = _configured(
        RCCompilerSpec("4r3c", ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated=("gamma_a_r", "gamma_e_r", "gamma_m_r"),
    )
    plan = build_physical_parameterization_plan(model)
    gamma_coords = [c for c in plan.coordinates if c.owner_kind == "routing_simplex"]
    assert len(gamma_coords) == 3
    row = next(c for c in plan.constraints if c.constraint_id.startswith("sum|routing_simplex"))
    assert len(row.indices) == 3
    assert row.lower_bound == pytest.approx(1.0)
    assert row.upper_bound == pytest.approx(1.0)


def test_dep2_physical_plan_optimizes_p_directly_and_enforces_mass():
    model = _dep2_model()
    plan = build_physical_parameterization_plan(model)
    pcoords = [c for c in plan.coordinates if c.owner_kind == "allocation_p"]
    assert [c.component for c in pcoords] == ["A", "B"]
    row = next(c for c in plan.constraints if c.constraint_id == "sum|allocation|zir_family")
    assert len(row.indices) == 2
    assert row.lower_bound == pytest.approx(1.0)
    pb = NumpyPhysicalRCBackend(model)
    theta = pb.initial_physical()
    lam = pb.allocation_lambdas(theta)["zir_family"]
    assert 0.4 * lam[0] + 0.6 * lam[1] == pytest.approx(1.0)


def test_transformed_and_physical_numpy_are_same_at_mapped_theta():
    model = _dep2_model()
    transformed = NumpyRCBackend(model)
    physical = NumpyPhysicalRCBackend(model)
    rho = np.linspace(-0.25, 0.3, transformed.plan.raw_dimension)
    theta = transformed.physical_decision_vector(rho)
    tm = transformed.matrices(rho)
    pm = physical.matrices(theta)
    np.testing.assert_allclose(tm.C, pm.C, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tm.L_CC, pm.L_CC, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tm.Gamma, pm.Gamma, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tm.A, pm.A, rtol=1e-12, atol=1e-12)


def test_casadi_physical_matches_numpy_physical_and_exposes_ipopt_bounds():
    model = _dep2_model()
    nb = NumpyPhysicalRCBackend(model)
    cb = CasadiPhysicalRCBackend(model, symbol_type="MX")
    theta = nb.initial_physical()
    local, aggregate = _inputs(model, nb.plan.aggregate_signal_order)
    x = np.asarray([22.0, 22.0, 24.0, 24.0])
    boundary = np.asarray([10.0])
    nm = nb.matrices(theta)
    cm = cb.matrices(theta)
    np.testing.assert_allclose(nm.C, cm[0].reshape(-1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(nm.L_CC, cm[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(nm.Gamma, cm[3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(nb.rhs(theta, x, boundary, local, aggregate), cb.rhs(theta, x, boundary, local, aggregate), rtol=1e-11, atol=1e-11)
    assert cb.lower_bounds().shape == theta.shape
    assert cb.upper_bounds().shape == theta.shape
    assert len(cb.plan.constraints) >= 1


def test_physical_casadi_rk4_matches_numpy_and_builds_nlp_schema():
    model = _dep2_model()
    nb = NumpyPhysicalRCBackend(model)
    cb = CasadiPhysicalRCBackend(model, symbol_type="MX")
    theta = nb.initial_physical()
    local, aggregate = _inputs(model, nb.plan.aggregate_signal_order)
    x = np.asarray([22.0, 22.0, 24.0, 24.0])
    boundary = np.asarray([10.0])
    nstep = nb.step("rk4", theta, x, boundary, local, aggregate, sample_dt_s=600.0, substeps=4)
    cstep = cb.step("rk4", theta, x, boundary, local, aggregate, sample_dt_s=600.0, substeps=4)
    np.testing.assert_allclose(nstep, cstep, rtol=1e-10, atol=1e-10)
    objective = cb.ca.sumsqr(cb.theta_symbol - cb.ca.DM(theta))
    spec = cb.build_ipopt_nlp(objective)
    assert set(spec["nlp"]) == {"x", "f", "g"}
    assert spec["x0"].shape == theta.shape
    assert spec["lbx"].shape == theta.shape
    assert spec["ubx"].shape == theta.shape
    assert spec["lbg"].shape == spec["ubg"].shape


def test_p4_chain_rule_torch_raw_equals_physical_casadi_gradient():
    model = _dep2_model()
    tb = TorchRCBackend(model, dtype=torch.float64)
    cb = CasadiPhysicalRCBackend(model, symbol_type="SX")
    rho = torch.linspace(-0.2, 0.25, tb.plan.raw_dimension, dtype=torch.float64, requires_grad=True)
    theta_t = tb.physical_decision_vector(rho)
    theta = theta_t.detach().cpu().numpy()
    local_np, aggregate_np = _inputs(model, tb.physical_plan.aggregate_signal_order)
    x_np = np.asarray([22.0, 22.0, 24.0, 24.0])
    boundary_np = np.asarray([10.0])
    probe_np = np.asarray([1.0, -0.5, 0.75, 0.2])

    out = tb.rhs(
        torch.tensor(x_np, dtype=torch.float64),
        torch.tensor(boundary_np, dtype=torch.float64),
        torch.tensor(local_np, dtype=torch.float64),
        torch.tensor(aggregate_np, dtype=torch.float64),
        raw=rho,
    )
    loss = torch.dot(torch.tensor(probe_np, dtype=torch.float64), out)
    grad_rho = torch.autograd.grad(loss, rho, retain_graph=True)[0]

    jac = torch.autograd.functional.jacobian(
        lambda rr: tb.physical_decision_vector(rr), rho
    )
    grad_theta = torch.tensor(
        cb.parameter_probe_gradient(theta, x_np, boundary_np, local_np, aggregate_np, probe_np),
        dtype=torch.float64,
    )
    mapped = jac.transpose(0, 1) @ grad_theta
    torch.testing.assert_close(grad_rho, mapped, rtol=1e-7, atol=1e-8)


def test_transformed_casadi_remains_reference_mode_and_backward_alias():
    model = _configured(
        RCCompilerSpec("2r2c", ("A",), "ind", zone_port_availability=_ports(("A",))),
        estimated=("C_a", "R_ao", "eta_r"),
    )
    cb = CasadiTransformedRCBackend(model, symbol_type="SX")
    assert cb.zero_raw().shape == (cb.plan.raw_dimension,)


def test_casadi_physical_ipopt_smoke_solves_bounded_constrained_theta_problem():
    model = _dep2_model()
    cb = CasadiPhysicalRCBackend(model, symbol_type="MX")
    theta0 = cb.initial_physical()
    objective = cb.ca.sumsqr(cb.theta_symbol - cb.ca.DM(theta0))
    spec = cb.build_ipopt_nlp(objective)
    solver = cb.ca.nlpsol(
        "e06_v2_ipopt_smoke",
        "ipopt",
        spec["nlp"],
        {"ipopt.print_level": 0, "print_time": False},
    )
    result = solver(
        x0=spec["x0"],
        lbx=spec["lbx"],
        ubx=spec["ubx"],
        lbg=spec["lbg"],
        ubg=spec["ubg"],
    )
    solved = np.asarray(result["x"], dtype=float).reshape(-1)
    np.testing.assert_allclose(solved, theta0, rtol=2e-5, atol=2e-7)
