from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    ConnectionRule,
    ParameterSharingRule,
    RCCompileError,
    RCCompilerSpec,
    RCInputSnapshot,
    SpatialMode,
    ZoneAdjacency,
    allocation_degrees_of_freedom,
    assert_dep1_dep2_physics_equivalent,
    compile_rc_model,
    default_initial_state,
    estimated_allocation_result,
    initial_estimated_allocation_result,
    initial_reference_logits,
    rhs,
    validate_compiler_invariants,
)


def _oracle_rc():
    path = Path("Paper_PINODE_EPSR/src/pinode_epsr/physics/rc.py")
    spec = importlib.util.spec_from_file_location("scale_e03b_oracle_rc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load controlled RC oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _values(model, *, by_family=None, by_instance=None):
    by_family = dict(by_family or {})
    by_instance = dict(by_instance or {})
    values = {}
    for inst in model.parameter_registry.instances:
        if inst.instance_id in by_instance:
            values[inst.instance_id] = float(by_instance[inst.instance_id])
        elif inst.family in by_family:
            values[inst.instance_id] = float(by_family[inst.family])
        elif inst.physical_type == "routing":
            values[inst.instance_id] = 1.0
        else:
            raise AssertionError(f"No test value for {inst}")
    return values


def _all_ports(zones):
    return {z: ("qac", "zic", "zir", "qsol1", "qsol2") for z in zones}


def _dep2_allocations(zones=("A", "B"), weights=None):
    weights = weights or {"A": 0.5, "B": 0.5}
    return (
        AllocationFamilySpec(
            name="convective_non_hvac",
            signals=("zic", "qsol1"),
            weights=weights,
            mode=AllocationMode.ESTIMATED,
        ),
        AllocationFamilySpec(
            name="radiative_non_hvac",
            signals=("zir", "qsol2"),
            weights=weights,
            mode=AllocationMode.ESTIMATED,
        ),
    )


def test_1r1c_single_matches_controlled_oracle():
    oracle = _oracle_rc()
    model = compile_rc_model(
        RCCompilerSpec(
            flavour="1r1c",
            zone_ids=("Dining",),
            mode=SpatialMode.IND,
            zone_port_availability=_all_ports(("Dining",)),
        )
    )
    params = _values(model, by_family={"C_a": 2.0e6, "R_ao": 0.02})
    snapshot = RCInputSnapshot(
        boundary_temperatures={"outdoor_temperature": 30.0},
        local_thermal_powers={
            ("Dining", "qac"): -1200.0,
            ("Dining", "zic"): 300.0,
            ("Dining", "zir"): 100.0,
            ("Dining", "qsol1"): 200.0,
            ("Dining", "qsol2"): 50.0,
        },
    )
    got = rhs(model, np.array([24.0]), snapshot, params)[0]
    heat = oracle.HeatInputs(qac=-1200.0, zic=300.0, zir=100.0, qsol1=200.0, qsol2=50.0)
    p = oracle.RC1ZoneParams(c_air=2.0e6, r_out=0.02, eta_rad=1.0, eta_rad_mode="full")
    expected = oracle.rhs_1r1c_single(24.0, 30.0, heat, p)
    assert got == pytest.approx(expected)


def test_2r2c_single_matches_controlled_oracle():
    oracle = _oracle_rc()
    model = compile_rc_model(
        RCCompilerSpec(
            flavour="2r2c",
            zone_ids=("Dining",),
            mode="independent",
            zone_port_availability=_all_ports(("Dining",)),
        )
    )
    params = _values(
        model,
        by_family={
            "C_a": 2.0e6,
            "C_m": 8.0e6,
            "R_ao": 0.02,
            "R_am": 0.01,
            "eta_r": 0.7,
        },
    )
    snapshot = RCInputSnapshot(
        boundary_temperatures={"outdoor_temperature": 31.0},
        local_thermal_powers={
            ("Dining", "qac"): -1000.0,
            ("Dining", "zic"): 300.0,
            ("Dining", "zir"): 120.0,
            ("Dining", "qsol1"): 80.0,
            ("Dining", "qsol2"): 40.0,
        },
    )
    state = np.array([24.0, 25.0])
    got = rhs(model, state, snapshot, params)
    heat = oracle.HeatInputs(qac=-1000.0, zic=300.0, zir=120.0, qsol1=80.0, qsol2=40.0)
    p = oracle.RC2ZoneParams(
        c_air=2.0e6,
        c_mass=8.0e6,
        r_out=0.02,
        r_mass=0.01,
        eta_rad=0.7,
        eta_rad_mode="fixed",
    )
    expected = oracle.rhs_2r2c_single(state, 31.0, heat, p)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-14)


def test_1r1c_two_zone_dep1_matches_controlled_oracle():
    oracle = _oracle_rc()
    model = compile_rc_model(
        RCCompilerSpec(
            flavour="1r1c",
            zone_ids=("Dining", "Kitchen"),
            mode="dep1",
            adjacency=(ZoneAdjacency("Dining", "Kitchen"),),
            zone_port_availability=_all_ports(("Dining", "Kitchen")),
        )
    )
    instance_values = {}
    for inst in model.parameter_registry.instances:
        if inst.family == "C_a":
            instance_values[inst.instance_id] = 2.0e6 if "Dining" in inst.zone_scope else 3.0e6
        elif inst.family == "R_ao":
            instance_values[inst.instance_id] = 0.02 if "Dining" in inst.zone_scope else 0.03
        elif inst.family == "R_inter_a_a":
            instance_values[inst.instance_id] = 0.015
        else:
            raise AssertionError(inst)
    snapshot = RCInputSnapshot(
        boundary_temperatures={"outdoor_temperature": 32.0},
        local_thermal_powers={
            ("Dining", "qac"): -900.0,
            ("Dining", "zic"): 200.0,
            ("Dining", "zir"): 80.0,
            ("Dining", "qsol1"): 50.0,
            ("Dining", "qsol2"): 20.0,
            ("Kitchen", "qac"): -1500.0,
            ("Kitchen", "zic"): 600.0,
            ("Kitchen", "zir"): 200.0,
            ("Kitchen", "qsol1"): 100.0,
            ("Kitchen", "qsol2"): 50.0,
        },
    )
    state = np.array([24.0, 26.0])
    got = rhs(model, state, snapshot, instance_values)
    heats = (
        oracle.HeatInputs(-900.0, 200.0, 80.0, 50.0, 20.0),
        oracle.HeatInputs(-1500.0, 600.0, 200.0, 100.0, 50.0),
    )
    params = (
        oracle.RC1ZoneParams(2.0e6, 0.02),
        oracle.RC1ZoneParams(3.0e6, 0.03),
    )
    expected = oracle.rhs_1r1c_coupled(state, 32.0, heats, params, 0.015)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-14)


def test_2r2c_two_zone_dep1_matches_controlled_oracle():
    oracle = _oracle_rc()
    model = compile_rc_model(
        RCCompilerSpec(
            flavour="2r2c",
            zone_ids=("Dining", "Kitchen"),
            mode="dependent1",
            adjacency=(ZoneAdjacency("Dining", "Kitchen"),),
            zone_port_availability=_all_ports(("Dining", "Kitchen")),
        )
    )
    values = {}
    family_zone = {
        ("C_a", "Dining"): 2.0e6,
        ("C_m", "Dining"): 8.0e6,
        ("R_ao", "Dining"): 0.02,
        ("R_am", "Dining"): 0.01,
        ("eta_r", "Dining"): 0.6,
        ("C_a", "Kitchen"): 3.0e6,
        ("C_m", "Kitchen"): 9.0e6,
        ("R_ao", "Kitchen"): 0.03,
        ("R_am", "Kitchen"): 0.012,
        ("eta_r", "Kitchen"): 0.8,
    }
    for inst in model.parameter_registry.instances:
        if inst.family == "R_inter_a_a":
            values[inst.instance_id] = 0.015
        else:
            values[inst.instance_id] = family_zone[(inst.family, inst.zone_scope[0])]
    snapshot = RCInputSnapshot(
        boundary_temperatures={"outdoor_temperature": 32.0},
        local_thermal_powers={
            ("Dining", "qac"): -900.0,
            ("Dining", "zic"): 200.0,
            ("Dining", "zir"): 80.0,
            ("Dining", "qsol1"): 50.0,
            ("Dining", "qsol2"): 20.0,
            ("Kitchen", "qac"): -1500.0,
            ("Kitchen", "zic"): 600.0,
            ("Kitchen", "zir"): 200.0,
            ("Kitchen", "qsol1"): 100.0,
            ("Kitchen", "qsol2"): 50.0,
        },
    )
    state = np.array([24.0, 25.0, 26.0, 27.0])
    got = rhs(model, state, snapshot, values)
    heats = (
        oracle.HeatInputs(-900.0, 200.0, 80.0, 50.0, 20.0),
        oracle.HeatInputs(-1500.0, 600.0, 200.0, 100.0, 50.0),
    )
    params = (
        oracle.RC2ZoneParams(2.0e6, 8.0e6, 0.02, 0.01, 0.6, "fixed"),
        oracle.RC2ZoneParams(3.0e6, 9.0e6, 0.03, 0.012, 0.8, "fixed"),
    )
    expected = oracle.rhs_2r2c_coupled(state, 32.0, heats, params, 0.015)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-14)


def test_3r2c_matches_hand_equations():
    model = compile_rc_model(
        RCCompilerSpec(
            flavour="3r2c",
            zone_ids=("A",),
            mode="ind",
            zone_port_availability=_all_ports(("A",)),
        )
    )
    values = _values(
        model,
        by_family={
            "C_a": 2e6, "C_m": 8e6, "R_ao": 0.02, "R_am": 0.01,
            "R_om": 0.04, "eta_r": 0.25,
        },
    )
    x = np.array([24.0, 25.0])
    snap = RCInputSnapshot(
        {"outdoor_temperature": 30.0},
        {
            ("A", "qac"): -500.0, ("A", "zic"): 200.0, ("A", "zir"): 100.0,
            ("A", "qsol1"): 50.0, ("A", "qsol2"): 20.0,
        },
    )
    got = rhs(model, x, snap, values)
    qc, qr = -500.0 + 200.0 + 50.0, 100.0 + 20.0
    expected = np.array([
        ((30-24)/0.02 + (25-24)/0.01 + qc + 0.75*qr)/2e6,
        ((30-25)/0.04 + (24-25)/0.01 + 0.25*qr)/8e6,
    ])
    np.testing.assert_allclose(got, expected, atol=1e-14, rtol=0)


def test_4r3c_matches_hand_equations():
    model = compile_rc_model(
        RCCompilerSpec(
            flavour="4r3c",
            zone_ids=("A",),
            mode="ind",
            zone_port_availability=_all_ports(("A",)),
        )
    )
    values = _values(
        model,
        by_family={
            "C_a": 2e6, "C_e": 5e6, "C_m": 8e6, "R_ao": 0.02,
            "R_ae": 0.015, "R_eo": 0.03, "R_am": 0.01,
            "gamma_a_r": 0.2, "gamma_e_r": 0.3, "gamma_m_r": 0.5,
        },
    )
    x = np.array([24.0, 26.0, 25.0])
    snap = RCInputSnapshot(
        {"outdoor_temperature": 30.0},
        {
            ("A", "qac"): -500.0, ("A", "zic"): 200.0, ("A", "zir"): 100.0,
            ("A", "qsol1"): 50.0, ("A", "qsol2"): 20.0,
        },
    )
    got = rhs(model, x, snap, values)
    qc, qr = -500 + 200 + 50, 100 + 20
    expected = np.array([
        ((30-24)/0.02 + (26-24)/0.015 + (25-24)/0.01 + qc + 0.2*qr)/2e6,
        ((30-26)/0.03 + (24-26)/0.015 + 0.3*qr)/5e6,
        ((24-25)/0.01 + 0.5*qr)/8e6,
    ])
    np.testing.assert_allclose(got, expected, atol=1e-14, rtol=0)


def test_default_chain_is_only_used_when_adjacency_absent():
    chain = compile_rc_model(RCCompilerSpec("1r1c", ("A", "B", "C"), "dep1"))
    assert chain.resolved_adjacency == (("A", "B"), ("B", "C"))

    explicit_empty = compile_rc_model(
        RCCompilerSpec("1r1c", ("A", "B", "C"), "dep1", adjacency=())
    )
    assert explicit_empty.resolved_adjacency == ()

    partial = compile_rc_model(
        RCCompilerSpec(
            "1r1c", ("A", "B", "C"), "dep1",
            adjacency=(ZoneAdjacency("A", "C"),),
        )
    )
    assert partial.resolved_adjacency == (("A", "C"),)


def test_cross_type_rule_expands_symmetrically():
    model = compile_rc_model(
        RCCompilerSpec(
            "2r2c",
            ("A", "B"),
            "dep1",
            adjacency=(ZoneAdjacency("A", "B"),),
            connection_rules=(ConnectionRule("a", "m"),),
        )
    )
    inter = [e for e in model.resistance_edges if e.kind == "inter_zone"]
    assert len(inter) == 2
    endpoints = {e.endpoint_keys() for e in inter}
    assert ("A::a", "B::m") in endpoints
    assert ("A::m", "B::a") in endpoints


def test_duplicate_reverse_adjacency_canonicalizes_once():
    model = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            ("A", "B"),
            "dep1",
            adjacency=(ZoneAdjacency("A", "B"), ZoneAdjacency("B", "A")),
        )
    )
    assert model.resolved_adjacency == (("A", "B"),)
    assert len([e for e in model.resistance_edges if e.kind == "inter_zone"]) == 1


def test_invalid_connection_state_fails():
    with pytest.raises(RCCompileError):
        compile_rc_model(
            RCCompilerSpec(
                "1r1c", ("A", "B"), "dep1",
                connection_rules=(ConnectionRule("a", "m"),),
            )
        )


def test_parameter_sharing_reduces_master_dimension_without_merging_states():
    independent = compile_rc_model(RCCompilerSpec("1r1c", ("A", "B"), "ind"))
    shared = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            ("A", "B"),
            "ind",
            parameter_sharing=(ParameterSharingRule(name="air_caps", family="C_a"),),
        )
    )
    assert independent.state_dimension == shared.state_dimension == 2
    assert len(shared.parameter_registry.masters) == len(independent.parameter_registry.masters) - 1
    c_masters = [
        m for m in shared.parameter_registry.masters if m.family == "C_a"
    ]
    assert len(c_masters) == 1
    assert len(c_masters[0].member_instance_ids) == 2


def test_equal_weight_dep2_has_n_minus_one_dof_and_neutral_lambda_one():
    family = _dep2_allocations()[0]
    assert allocation_degrees_of_freedom(family, ("A", "B")) == 1
    assert initial_reference_logits(family, ("A", "B")) == pytest.approx((0.0,))
    result = initial_estimated_allocation_result(family, ("A", "B"))
    assert result.lambda_by_zone == pytest.approx({"A": 1.0, "B": 1.0})
    assert result.p_by_zone == pytest.approx({"A": 0.5, "B": 0.5})


def test_unequal_weight_neutral_initialization_is_lambda_one():
    family = AllocationFamilySpec(
        name="g",
        signals=("zic",),
        weights={"A": 0.7, "B": 0.3},
        mode=AllocationMode.ESTIMATED,
    )
    logits = initial_reference_logits(family, ("A", "B"))
    assert logits[0] == pytest.approx(np.log(0.7/0.3))
    result = estimated_allocation_result(family, ("A", "B"), logits)
    assert result.lambda_by_zone == pytest.approx({"A": 1.0, "B": 1.0})


def test_unequal_weight_estimated_allocation_always_preserves_ab():
    family = AllocationFamilySpec(
        name="g",
        signals=("zic",),
        weights={"A": 0.7, "B": 0.3},
        mode=AllocationMode.ESTIMATED,
    )
    result = estimated_allocation_result(family, ("A", "B"), (1.2,))
    mass = 0.7*result.lambda_by_zone["A"] + 0.3*result.lambda_by_zone["B"]
    assert mass == pytest.approx(1.0)
    assert result.ab_error <= 1e-12


def test_partial_fixed_allocation_uses_residual_simplex():
    family = AllocationFamilySpec(
        name="g",
        signals=("zic",),
        weights={"A": 0.5, "B": 0.3, "C": 0.2},
        mode=AllocationMode.ESTIMATED,
        fixed_lambdas={"A": 1.2},
    )
    assert allocation_degrees_of_freedom(family, ("A", "B", "C")) == 1
    result = estimated_allocation_result(
        family,
        ("A", "B", "C"),
        (np.log(0.625/0.375),),
    )
    assert result.residual == pytest.approx(0.4)
    assert result.lambda_by_zone["A"] == pytest.approx(1.2)
    assert result.lambda_by_zone["B"] == pytest.approx(0.25/0.3)
    assert result.lambda_by_zone["C"] == pytest.approx(0.15/0.2)


def test_partial_fixed_negative_residual_fails():
    family = AllocationFamilySpec(
        name="g",
        signals=("zic",),
        weights={"A": 0.5, "B": 0.3, "C": 0.2},
        mode=AllocationMode.ESTIMATED,
        fixed_lambdas={"A": 2.4},
    )
    with pytest.raises(RCCompileError):
        initial_reference_logits(family, ("A", "B", "C"))


def test_dep2_requires_explicit_non_hvac_allocation_coverage():
    with pytest.raises(RCCompileError, match="missing"):
        compile_rc_model(
            RCCompilerSpec(
                "1r1c",
                ("A", "B"),
                "dep2",
                zone_port_availability=_all_ports(("A", "B")),
                dep2_allocations=(),
            )
        )


def test_dep2_keeps_qac_local_and_allocates_non_hvac():
    model = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            ("A", "B"),
            "dep2",
            adjacency=(ZoneAdjacency("A", "B"),),
            zone_port_availability=_all_ports(("A", "B")),
            dep2_allocations=_dep2_allocations(),
        )
    )
    conv = estimated_allocation_result(
        model.allocation_families["convective_non_hvac"],
        ("A", "B"),
        (np.log(0.7/0.3),),
    )
    rad = initial_estimated_allocation_result(
        model.allocation_families["radiative_non_hvac"],
        ("A", "B"),
    )
    params = _values(
        model,
        by_family={"C_a": 2e6, "R_ao": 0.02, "R_inter_a_a": 0.015},
    )
    snap = RCInputSnapshot(
        {"outdoor_temperature": 30.0},
        local_thermal_powers={("A","qac"):-100.0, ("B","qac"):-200.0},
        aggregate_thermal_powers={"zic":1000.0, "qsol1":100.0, "zir":200.0, "qsol2":20.0},
    )
    # Successful RHS proves local non-HVAC inputs were not required.
    got = rhs(
        model,
        np.array([24.0, 25.0]),
        snap,
        params,
        allocation_results={
            "convective_non_hvac": conv,
            "radiative_non_hvac": rad,
        },
    )
    assert got.shape == (2,)


def test_dep1_dep2_compile_identical_physics_graph():
    common = dict(
        flavour="2r2c",
        zone_ids=("A", "B"),
        adjacency=(ZoneAdjacency("A", "B"),),
        zone_port_availability=_all_ports(("A", "B")),
    )
    dep1 = compile_rc_model(RCCompilerSpec(mode="dep1", **common))
    dep2 = compile_rc_model(
        RCCompilerSpec(mode="dep2", dep2_allocations=_dep2_allocations(), **common)
    )
    assert_dep1_dep2_physics_equivalent(dep1, dep2)


@pytest.mark.parametrize(
    ("flavour", "expected"),
    [
        ("1r1c", [22.0]),
        ("2r2c", [22.0, 22.0]),
        ("3r2c", [22.0, 22.0]),
        ("4r3c", [22.0, 22.0, 22.0]),
    ],
)
def test_default_initialization_sets_all_latent_states_equal_air(flavour, expected):
    model = compile_rc_model(RCCompilerSpec(flavour, ("A",), "ind"))
    got = default_initial_state(model, {"A": 22.0})
    np.testing.assert_allclose(got, expected)


def test_multi_zone_initialization_uses_each_zones_observed_air():
    model = compile_rc_model(RCCompilerSpec("2r2c", ("A", "B"), "ind"))
    got = default_initial_state(model, {"A": 22.0, "B": 24.0})
    np.testing.assert_allclose(got, [22.0, 22.0, 24.0, 24.0])


def test_phvac_cannot_be_declared_as_thermal_port():
    with pytest.raises(RCCompileError, match="PHVAC"):
        compile_rc_model(
            RCCompilerSpec(
                "1r1c", ("A",), "ind",
                zone_port_availability={"A": ("qac", "phvac")},
                port_groups={"phvac": "convective"},
            )
        )


def test_structurally_available_input_must_not_be_silently_zero_filled():
    model = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            ("A",),
            "ind",
            zone_port_availability={"A": ("qac", "zic")},
        )
    )
    values = _values(model, by_family={"C_a": 2e6, "R_ao": 0.02})
    snap = RCInputSnapshot(
        {"outdoor_temperature": 30.0},
        local_thermal_powers={("A", "qac"): -100.0},
    )
    with pytest.raises(RCCompileError, match="Missing required local thermal input"):
        rhs(model, np.array([24.0]), snap, values)


def test_structurally_unavailable_input_is_omitted_not_zero_inserted():
    model = compile_rc_model(
        RCCompilerSpec(
            "1r1c",
            ("A",),
            "ind",
            zone_port_availability={"A": ("qac",)},
        )
    )
    assert [p.key for p in model.thermal_ports] == ["A::qac"]
    values = _values(model, by_family={"C_a": 2e6, "R_ao": 0.02})
    snap = RCInputSnapshot(
        {"outdoor_temperature": 30.0},
        local_thermal_powers={("A", "qac"): -100.0},
    )
    assert rhs(model, np.array([24.0]), snap, values).shape == (1,)


def test_one_zone_dep1_has_no_inter_zone_edges():
    model = compile_rc_model(RCCompilerSpec("2r2c", ("A",), "dep1"))
    assert model.resolved_adjacency == ()
    assert not [e for e in model.resistance_edges if e.kind == "inter_zone"]


@pytest.mark.parametrize("flavour", ["1r1c", "2r2c", "3r2c", "4r3c"])
def test_compiler_invariants_hold(flavour):
    model = compile_rc_model(RCCompilerSpec(flavour, ("A", "B"), "dep1"))
    family_defaults = {
        "C_a": 2e6, "C_m": 8e6, "C_e": 5e6,
        "R_ao": 0.02, "R_am": 0.01, "R_om": 0.04,
        "R_ae": 0.015, "R_eo": 0.03, "R_inter_a_a": 0.05,
        "eta_r": 0.7,
        "gamma_a_r": 0.2, "gamma_e_r": 0.3, "gamma_m_r": 0.5,
    }
    values = _values(model, by_family=family_defaults)
    report = validate_compiler_invariants(model, values)
    assert report.passed


def test_zero_resistance_is_rejected():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    values = _values(model, by_family={"C_a": 2e6, "R_ao": 0.02})
    rid = next(i.instance_id for i in model.parameter_registry.instances if i.family == "R_ao")
    values[rid] = 0.0
    with pytest.raises(RCCompileError, match="strictly positive"):
        model.matrices(values)


def test_routing_coefficients_must_be_conservative():
    model = compile_rc_model(RCCompilerSpec("4r3c", ("A",), "ind"))
    values = _values(
        model,
        by_family={
            "C_a":2e6, "C_e":5e6, "C_m":8e6, "R_ao":0.02,
            "R_ae":0.015, "R_eo":0.03, "R_am":0.01,
            "gamma_a_r":0.2, "gamma_e_r":0.3, "gamma_m_r":0.4,
        },
    )
    with pytest.raises(RCCompileError, match="must sum to 1"):
        model.matrices(values)
