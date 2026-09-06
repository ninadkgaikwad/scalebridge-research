import numpy as np

from Paper_PINODE_EPSR.aggregation import aggregation_residual, equal_weight_temperature
from Paper_PINODE_EPSR.config import canonical_case_specs
from Paper_PINODE_EPSR.rc import (
    HeatInputs,
    RC1ZoneParams,
    RC2ZoneParams,
    allocate_heat_2c,
    effective_heat_1c,
    rhs_1r1c_single,
    rhs_2r2c_single,
)


def test_case_matrix_collapses_all_to_one_aliases():
    specs = canonical_case_specs()
    assert set(specs) == {"all_to_one", "identity_ind", "identity_dep1", "identity_dep2"}
    assert specs["all_to_one"].all_to_one_aliases == ("ind", "dep1", "dep2")
    assert specs["identity_dep1"].dependency_mode == "dependent1"
    assert specs["identity_dep2"].dependency_mode == "dependent2"


def test_equal_weight_temperature():
    got = equal_weight_temperature(np.array([20.0, 22.0]), np.array([24.0, 26.0]))
    assert np.allclose(got, [22.0, 24.0])


def test_heat_classification_is_locked():
    heat = HeatInputs(qac=10.0, zic=20.0, zir=30.0, qsol1=40.0, qsol2=50.0)
    assert np.isclose(heat.convective, 70.0)   # QAC + QZIC + QSol1
    assert np.isclose(heat.radiative, 80.0)    # QZIR + QSol2
    assert np.isclose(heat.total, 150.0)


def test_1r1c_default_uses_every_heat_channel_with_eta_rad_full():
    p = RC1ZoneParams(c_air=10.0, r_out=2.0)
    base = rhs_1r1c_single(20.0, 20.0, HeatInputs(qac=0.0, zic=0.0), p)
    channels = ("qac", "zic", "zir", "qsol1", "qsol2")
    for channel in channels:
        values = dict(qac=0.0, zic=0.0, zir=0.0, qsol1=0.0, qsol2=0.0)
        values[channel] = 10.0
        got = rhs_1r1c_single(20.0, 20.0, HeatInputs(**values), p)
        assert np.isclose(got - base, 1.0), channel


def test_1r1c_eta_rad_fixed_scales_only_radiative_channels():
    p = RC1ZoneParams(c_air=10.0, r_out=2.0, eta_rad=0.25, eta_rad_mode="fixed")
    heat = HeatInputs(qac=10.0, zic=20.0, zir=30.0, qsol1=40.0, qsol2=50.0)
    assert np.isclose(effective_heat_1c(heat, p), 70.0 + 0.25 * 80.0)


def test_1r1c_eta_rad_modes_and_bounds():
    heat = HeatInputs(qac=1.0, zic=2.0, zir=3.0, qsol1=4.0, qsol2=5.0)
    assert np.isclose(effective_heat_1c(heat, RC1ZoneParams(10.0, 2.0, eta_rad_mode="full")), 15.0)
    assert np.isclose(effective_heat_1c(heat, RC1ZoneParams(10.0, 2.0, eta_rad_mode="zero")), 7.0)
    assert np.isclose(RC1ZoneParams(10.0, 2.0, eta_rad=0.6, eta_rad_mode="learnable").resolved_eta_rad(), 0.6)
    try:
        RC1ZoneParams(10.0, 2.0, eta_rad=-0.1, eta_rad_mode="fixed").resolved_eta_rad()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid 1R1C eta_rad was accepted")


def test_2r2c_default_routes_convective_to_air_radiative_to_mass():
    p = RC2ZoneParams(c_air=10.0, c_mass=20.0, r_out=2.0, r_mass=3.0)
    heat = HeatInputs(qac=10.0, zic=20.0, zir=30.0, qsol1=40.0, qsol2=50.0)
    air, mass = allocate_heat_2c(heat, p)
    assert np.isclose(air, 70.0)
    assert np.isclose(mass, 80.0)
    assert np.isclose(air + mass, heat.total)


def test_eta_rad_fixed_split_conserves_radiative_heat():
    p = RC2ZoneParams(
        c_air=10.0,
        c_mass=20.0,
        r_out=2.0,
        r_mass=3.0,
        eta_rad=0.25,
        eta_rad_mode="fixed",
    )
    heat = HeatInputs(qac=10.0, zic=20.0, zir=30.0, qsol1=40.0, qsol2=50.0)
    air, mass = allocate_heat_2c(heat, p)
    assert np.isclose(air, 70.0 + 0.75 * 80.0)
    assert np.isclose(mass, 0.25 * 80.0)
    assert np.isclose(air + mass, heat.total)


def test_eta_rad_modes_mass_and_air_only():
    heat = HeatInputs(qac=1.0, zic=2.0, zir=3.0, qsol1=4.0, qsol2=5.0)
    p_mass = RC2ZoneParams(10.0, 20.0, 2.0, 3.0, eta_rad_mode="mass_only")
    p_air = RC2ZoneParams(10.0, 20.0, 2.0, 3.0, eta_rad_mode="air_only")
    assert allocate_heat_2c(heat, p_mass) == (7.0, 8.0)
    assert allocate_heat_2c(heat, p_air) == (15.0, 0.0)


def test_eta_rad_learnable_mode_accepts_bounded_initial_value():
    p = RC2ZoneParams(
        10.0,
        20.0,
        2.0,
        3.0,
        eta_rad=0.6,
        eta_rad_mode="learnable",
    )
    assert np.isclose(p.resolved_eta_rad(), 0.6)


def test_eta_rad_rejects_out_of_range_fixed_or_learnable_value():
    p = RC2ZoneParams(
        10.0,
        20.0,
        2.0,
        3.0,
        eta_rad=1.1,
        eta_rad_mode="fixed",
    )
    try:
        p.resolved_eta_rad()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid eta_rad was accepted")


def test_2r2c_rhs_uses_complete_heat_input():
    p = RC2ZoneParams(c_air=10.0, c_mass=20.0, r_out=2.0, r_mass=3.0)
    rhs = rhs_2r2c_single(
        np.array([20.0, 20.0]),
        20.0,
        HeatInputs(qac=10.0, zic=20.0, zir=30.0, qsol1=40.0, qsol2=50.0),
        p,
    )
    assert np.isclose(rhs[0], 70.0 / 10.0)
    assert np.isclose(rhs[1], 80.0 / 20.0)


def test_aggregation_residual_zero_when_consistent():
    identity = np.array([[1.0, 3.0], [2.0, 4.0]])
    aggregate = np.array([2.0, 3.0])
    assert np.allclose(aggregation_residual(aggregate, identity), 0.0)
