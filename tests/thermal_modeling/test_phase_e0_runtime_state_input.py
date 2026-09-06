from __future__ import annotations

import numpy as np
import pytest

from scalebridge.models.grey_box.rc_networks import (
    AllocationFamilySpec,
    AllocationMode,
    CanonicalRuntimeFrame,
    InitializationPolicy,
    InitializationRequest,
    InitializationSource,
    RCCompileError,
    RCCompilerSpec,
    RuntimeStateOrigin,
    ZoneAdjacency,
    ZoneInitializationEvidence,
    accept_model_evolved_state,
    assert_dep1_dep2_runtime_physics_equivalent,
    assert_runtime_binding_does_not_change_physics,
    assert_state_binding_timestamp,
    bind_runtime_frame,
    build_initialization_lifting,
    compile_rc_model,
    estimated_allocation_result,
    explicit_state_reset,
    graph_signature,
    initialize_runtime_state,
    model_forcing_applicability,
    resolve_setpoint,
    runtime_source_availability,
    start_recursive_state,
    validate_runtime_invariants,
)


def _ports(zones, signals=("qac", "zic", "zir", "qsol1", "qsol2")):
    return {z: tuple(signals) for z in zones}


def _dep2_family(zones=("A", "B"), weights=None, signals=("zic",)):
    weights = weights or {z: 1.0 / len(zones) for z in zones}
    return AllocationFamilySpec(
        name="non_hvac",
        signals=tuple(signals),
        weights=weights,
        mode=AllocationMode.ESTIMATED,
    )


def _dep2_model(*, weights=None, ports=None):
    zones = ("A", "B")
    port_map = ports or _ports(zones, ("qac", "zic"))
    non_hvac = sorted({s for values in port_map.values() for s in values if s != "qac"})
    return compile_rc_model(
        RCCompilerSpec(
            "2r2c",
            zones,
            "dep2",
            adjacency=(ZoneAdjacency("A", "B"),),
            zone_port_availability=port_map,
            dep2_allocations=(
                _dep2_family(zones, weights=weights, signals=tuple(non_hvac)),
            ) if non_hvac else (),
        )
    )


def _dep2_alloc(model, logits=(0.0,)):
    family = model.allocation_families["non_hvac"]
    return {"non_hvac": estimated_allocation_result(family, model.spec.zone_ids, logits)}


def _frame(
    timestamp="2026-08-27T12:00:00",
    *,
    local=None,
    aggregate=None,
    boundary=None,
    observed=None,
    aux=None,
    local_avail=None,
    aggregate_avail=None,
):
    return CanonicalRuntimeFrame(
        timestamp=timestamp,
        boundary_temperatures=boundary or {"outdoor_temperature": 30.0},
        local_thermal_powers=local or {},
        aggregate_thermal_powers=aggregate or {},
        observed_air_temperatures=observed or {},
        auxiliary_electrical_powers=aux or {},
        local_source_availability=local_avail or {},
        aggregate_source_availability=aggregate_avail or {},
    )


# ---------------------------------------------------------------------------
# E0-4A/B: deterministic state ordering and initialization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("flavour", "expected_state"),
    [
        ("1r1c", [21.0, 24.0]),
        ("2r2c", [21.0, 21.0, 24.0, 24.0]),
        ("3r2c", [21.0, 21.0, 24.0, 24.0]),
        ("4r3c", [21.0, 21.0, 21.0, 24.0, 24.0, 24.0]),
    ],
)
def test_initialization_lifts_resolved_zone_values_in_compiler_order(flavour, expected_state):
    model = compile_rc_model(RCCompilerSpec(flavour, ("A", "B"), "ind"))
    result = initialize_runtime_state(
        model,
        {
            "A": ZoneInitializationEvidence(observed_air_temperature_c=21.0),
            "B": ZoneInitializationEvidence(observed_air_temperature_c=24.0),
        },
    )
    np.testing.assert_allclose(result.state, expected_state)
    np.testing.assert_allclose(model.observation @ result.state, [21.0, 24.0])


@pytest.mark.parametrize("flavour", ["1r1c", "2r2c", "3r2c", "4r3c"])
def test_lifting_matrix_satisfies_hs0_identity(flavour):
    model = compile_rc_model(RCCompilerSpec(flavour, ("A", "B", "C"), "ind"))
    s0 = build_initialization_lifting(model)
    np.testing.assert_array_equal(model.observation @ s0, np.eye(3))


def test_auto_priority_user_over_observed_setpoint_and_default():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        {
            "A": ZoneInitializationEvidence(
                observed_air_temperature_c=23.0,
                scalar_setpoint_c=22.0,
            )
        },
        request=InitializationRequest(
            policy="auto",
            user_temperatures_c={"A": 21.0},
            default_temperature_c=19.0,
        ),
    )
    assert result.state[0] == pytest.approx(21.0)
    assert result.resolved_by_zone["A"].source is InitializationSource.USER_FIXED


def test_auto_user_override_does_not_parse_invalid_lower_priority_setpoint():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        {"A": ZoneInitializationEvidence(heating_setpoint_c=20.0, active_mode="invalid_mode")},
        request=InitializationRequest(policy="auto", user_temperatures_c={"A": 21.0}),
    )
    assert result.state[0] == pytest.approx(21.0)


def test_auto_observation_does_not_parse_invalid_lower_priority_setpoint():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        {
            "A": ZoneInitializationEvidence(
                observed_air_temperature_c=23.0,
                heating_setpoint_c=20.0,
                active_mode="invalid_mode",
            )
        },
    )
    assert result.state[0] == pytest.approx(23.0)


def test_auto_priority_observed_over_setpoint_and_default():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        {"A": ZoneInitializationEvidence(observed_air_temperature_c=23.0, scalar_setpoint_c=22.0)},
    )
    assert result.state[0] == pytest.approx(23.0)
    assert result.resolved_by_zone["A"].source is InitializationSource.OBSERVED


def test_auto_uses_setpoint_when_observation_missing():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        {"A": ZoneInitializationEvidence(scalar_setpoint_c=22.5)},
    )
    assert result.state[0] == pytest.approx(22.5)
    assert result.resolved_by_zone["A"].source is InitializationSource.SETPOINT


def test_auto_falls_back_to_configured_default():
    model = compile_rc_model(RCCompilerSpec("2r2c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        request=InitializationRequest(default_temperature_c=20.5),
    )
    np.testing.assert_allclose(result.state, [20.5, 20.5])
    assert result.resolved_by_zone["A"].source is InitializationSource.DEFAULT


def test_auto_default_is_22c():
    model = compile_rc_model(RCCompilerSpec("4r3c", ("A",), "ind"))
    result = initialize_runtime_state(model)
    np.testing.assert_allclose(result.state, [22.0, 22.0, 22.0])


def test_global_user_temperature_applies_to_all_zones():
    model = compile_rc_model(RCCompilerSpec("2r2c", ("A", "B"), "ind"))
    result = initialize_runtime_state(
        model,
        request=InitializationRequest(policy="user_fixed", global_user_temperature_c=19.5),
    )
    np.testing.assert_allclose(result.state, [19.5, 19.5, 19.5, 19.5])


def test_zone_user_override_precedes_global_user_temperature():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A", "B"), "ind"))
    result = initialize_runtime_state(
        model,
        request=InitializationRequest(
            policy="user_fixed",
            global_user_temperature_c=20.0,
            user_temperatures_c={"B": 24.0},
        ),
    )
    np.testing.assert_allclose(result.state, [20.0, 24.0])


@pytest.mark.parametrize(
    ("policy", "evidence", "user", "expected"),
    [
        ("observed", ZoneInitializationEvidence(observed_air_temperature_c=23.0), {}, 23.0),
        ("setpoint", ZoneInitializationEvidence(scalar_setpoint_c=21.5), {}, 21.5),
        ("default", ZoneInitializationEvidence(observed_air_temperature_c=29.0), {}, 22.0),
        ("user_fixed", ZoneInitializationEvidence(observed_air_temperature_c=29.0), {"A": 20.0}, 20.0),
    ],
)
def test_explicit_initialization_modes(policy, evidence, user, expected):
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        {"A": evidence},
        request=InitializationRequest(policy=policy, user_temperatures_c=user),
    )
    assert result.state[0] == pytest.approx(expected)
    assert result.policy is InitializationPolicy.normalize(policy)


def test_setpoint_scalar_has_highest_setpoint_resolution_priority():
    got = resolve_setpoint(
        ZoneInitializationEvidence(
            scalar_setpoint_c=22.25,
            heating_setpoint_c=20.0,
            cooling_setpoint_c=26.0,
            active_mode="cooling",
        )
    )
    assert got == (22.25, "scalar_setpoint")


def test_setpoint_active_heating_mode_uses_heating_setpoint():
    assert resolve_setpoint(
        ZoneInitializationEvidence(heating_setpoint_c=20.0, cooling_setpoint_c=24.0, active_mode="heat")
    ) == (20.0, "active_heating_setpoint")


def test_setpoint_active_cooling_mode_uses_cooling_setpoint():
    assert resolve_setpoint(
        ZoneInitializationEvidence(heating_setpoint_c=20.0, cooling_setpoint_c=24.0, active_mode="cooling")
    ) == (24.0, "active_cooling_setpoint")


def test_setpoint_midpoint_when_both_exist_without_active_mode():
    assert resolve_setpoint(
        ZoneInitializationEvidence(heating_setpoint_c=20.0, cooling_setpoint_c=24.0)
    ) == (22.0, "heating_cooling_midpoint")


def test_setpoint_single_mode_value_can_resolve_without_active_mode():
    assert resolve_setpoint(ZoneInitializationEvidence(heating_setpoint_c=20.5)) == (
        20.5,
        "single_heating_setpoint",
    )


def test_auto_can_skip_nan_observation_and_use_setpoint():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    result = initialize_runtime_state(
        model,
        {"A": ZoneInitializationEvidence(observed_air_temperature_c=np.nan, scalar_setpoint_c=22.0)},
    )
    assert result.state[0] == pytest.approx(22.0)


@pytest.mark.parametrize("policy", ["user_fixed", "observed", "setpoint"])
def test_explicit_policy_fails_when_required_source_cannot_resolve(policy):
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    request = InitializationRequest(policy=policy)
    with pytest.raises(RCCompileError):
        initialize_runtime_state(model, request=request)


def test_invalid_active_mode_fails_in_setpoint_resolution():
    with pytest.raises(RCCompileError, match="Unsupported active thermostat mode"):
        resolve_setpoint(ZoneInitializationEvidence(heating_setpoint_c=20.0, active_mode="economizer"))


def test_unknown_initialization_zone_fails_exact_identity_contract():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    with pytest.raises(RCCompileError, match="unknown modeled zones"):
        initialize_runtime_state(model, {"a": ZoneInitializationEvidence(observed_air_temperature_c=22.0)})


# ---------------------------------------------------------------------------
# E0-4C/D/E: runtime frame, applicability, and spatial forcing realization
# ---------------------------------------------------------------------------

def test_ind_binding_uses_compiler_port_order_and_local_sources():
    model = compile_rc_model(
        RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac", "zic")})
    )
    frame = _frame(local={("A", "qac"): -100.0, ("A", "zic"): 300.0})
    binding = bind_runtime_frame(model, frame)
    assert binding.used_local_thermal_keys == (("A", "qac"), ("A", "zic"))
    np.testing.assert_allclose(binding.effective_thermal_vector, [-100.0, 300.0])


def test_dep1_binding_uses_local_non_hvac_sources():
    model = compile_rc_model(
        RCCompilerSpec(
            "1r1c", ("A", "B"), "dep1",
            zone_port_availability=_ports(("A", "B"), ("qac", "zic")),
        )
    )
    frame = _frame(local={
        ("A", "qac"): -100.0, ("A", "zic"): 300.0,
        ("B", "qac"): -200.0, ("B", "zic"): 500.0,
    })
    binding = bind_runtime_frame(model, frame)
    np.testing.assert_allclose(binding.effective_thermal_vector, [-100.0, 300.0, -200.0, 500.0])


def test_dep2_keeps_qac_local_and_allocates_non_hvac():
    model = _dep2_model(weights={"A": 0.7, "B": 0.3})
    alloc = _dep2_alloc(model, logits=(np.log(0.7 / 0.3),))
    frame = _frame(
        local={("A", "qac"): -100.0, ("B", "qac"): -200.0},
        aggregate={"zic": 1000.0},
    )
    binding = bind_runtime_frame(model, frame, allocation_results=alloc)
    # Neutral initialization for unequal weights gives lambda_A=lambda_B=1.
    np.testing.assert_allclose(binding.effective_thermal_vector, [-100.0, 1000.0, -200.0, 1000.0])
    assert binding.dep2_coordinate_error_max_abs <= 1e-12


def test_dep2_runtime_coordinate_consistency_for_non_neutral_allocation():
    model = _dep2_model(weights={"A": 0.7, "B": 0.3})
    alloc = _dep2_alloc(model, logits=(1.2,))
    frame = _frame(
        local={("A", "qac"): 0.0, ("B", "qac"): 0.0},
        aggregate={"zic": 1234.5},
    )
    binding = bind_runtime_frame(model, frame, allocation_results=alloc)
    by_port = dict(zip(binding.model_applicable_ports, binding.effective_thermal_vector))
    recovered = 0.7 * by_port[("A", "zic")] + 0.3 * by_port[("B", "zic")]
    assert recovered == pytest.approx(1234.5)
    assert binding.dep2_coordinate_error_max_abs <= 1e-12


def test_source_availability_and_model_applicability_are_independent_in_dep2():
    model = _dep2_model()
    frame = _frame(
        local={("A", "qac"): -100.0, ("B", "qac"): -200.0},
        aggregate={"zic": 800.0},
        local_avail={("A", "qac"): True, ("B", "qac"): True, ("A", "zic"): True, ("B", "zic"): False},
    )
    applicability = model_forcing_applicability(model)
    availability = runtime_source_availability(frame)
    assert applicability[("B", "zic")] is True
    assert availability[("B", "zic")] is False
    binding = bind_runtime_frame(model, frame, allocation_results=_dep2_alloc(model))
    assert ("B", "zic") not in binding.used_local_thermal_keys
    assert "zic" in binding.used_aggregate_signals


def test_same_missing_local_non_hvac_source_fails_in_dep1():
    model = compile_rc_model(
        RCCompilerSpec(
            "1r1c", ("A", "B"), "dep1",
            zone_port_availability=_ports(("A", "B"), ("qac", "zic")),
        )
    )
    frame = _frame(
        local={("A", "qac"): -100.0, ("A", "zic"): 300.0, ("B", "qac"): -200.0},
        local_avail={("B", "zic"): False},
    )
    with pytest.raises(RCCompileError, match="unavailable"):
        bind_runtime_frame(model, frame)


def test_structurally_absent_extra_signal_is_unused_not_zero_inserted():
    model = compile_rc_model(
        RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)})
    )
    frame = _frame(local={("A", "qac"): -100.0, ("A", "zic"): 999.0})
    binding = bind_runtime_frame(model, frame)
    np.testing.assert_allclose(binding.effective_thermal_vector, [-100.0])
    assert binding.unused_local_thermal_keys == (("A", "zic"),)


def test_structurally_required_missing_signal_is_not_zero_filled():
    model = compile_rc_model(
        RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac", "zic")})
    )
    with pytest.raises(RCCompileError, match="unavailable"):
        bind_runtime_frame(model, _frame(local={("A", "qac"): -100.0}))


def test_rich_frame_extra_aggregate_signal_is_retained_as_unused():
    model = compile_rc_model(
        RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)})
    )
    binding = bind_runtime_frame(
        model,
        _frame(local={("A", "qac"): -100.0}, aggregate={"zic": 500.0}),
    )
    assert binding.unused_aggregate_signals == ("zic",)


def test_extra_boundary_is_allowed_but_not_consumed():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    binding = bind_runtime_frame(
        model,
        _frame(
            local={("A", "qac"): 0.0},
            boundary={"outdoor_temperature": 30.0, "ground_temperature": 18.0},
        ),
    )
    assert binding.used_boundary_labels == ("outdoor_temperature",)
    assert binding.unused_boundary_labels == ("ground_temperature",)


def test_phvac_is_allowed_as_auxiliary_electrical_signal_and_excluded_from_q():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    binding = bind_runtime_frame(
        model,
        _frame(local={("A", "qac"): -100.0}, aux={("A", "phvac"): 450.0}),
    )
    assert binding.unused_auxiliary_electrical_keys == (("A", "phvac"),)
    assert ("A", "phvac") not in binding.snapshot.local_thermal_powers


def test_phvac_in_local_thermal_mapping_is_rejected_at_frame_boundary():
    with pytest.raises(RCCompileError, match="PHVAC"):
        _frame(local={("A", "phvac"): 450.0})


def test_phvac_in_aggregate_thermal_mapping_is_rejected():
    with pytest.raises(RCCompileError, match="PHVAC"):
        _frame(aggregate={"phvac": 450.0})


def test_aggregate_qac_is_rejected():
    with pytest.raises(RCCompileError, match="QAC must remain local"):
        _frame(aggregate={"qac": -500.0})


def test_local_source_marked_unavailable_cannot_supply_value():
    with pytest.raises(RCCompileError, match="marked unavailable"):
        _frame(local={("A", "qac"): 0.0}, local_avail={("A", "qac"): False})


def test_aggregate_source_marked_unavailable_cannot_supply_value():
    with pytest.raises(RCCompileError, match="marked unavailable"):
        _frame(aggregate={"zic": 1.0}, aggregate_avail={"zic": False})


def test_unknown_runtime_zone_identity_fails():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    with pytest.raises(RCCompileError, match="outside the compiled model"):
        bind_runtime_frame(model, _frame(local={("B", "qac"): 0.0}))


def test_expected_timestamp_mismatch_fails():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    with pytest.raises(RCCompileError, match="timestamp mismatch"):
        bind_runtime_frame(
            model,
            _frame(timestamp="t0", local={("A", "qac"): 0.0}),
            expected_timestamp="t1",
        )


def test_matching_expected_timestamp_passes():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    binding = bind_runtime_frame(
        model,
        _frame(timestamp="t0", local={("A", "qac"): 0.0}),
        expected_timestamp="t0",
    )
    assert binding.timestamp == "t0"


def test_none_runtime_timestamp_is_rejected():
    with pytest.raises(RCCompileError, match="requires a timestamp"):
        _frame(timestamp=None)


@pytest.mark.parametrize(
    ("kind", "kwargs", "pattern"),
    [
        ("boundary", {"boundary": {"outdoor_temperature": np.nan}, "local": {("A", "qac"): 0.0}}, "Non-finite boundary"),
        ("local", {"local": {("A", "qac"): np.inf}}, "Non-finite local"),
    ],
)
def test_nonfinite_required_runtime_values_fail(kind, kwargs, pattern):
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    with pytest.raises(RCCompileError, match=pattern):
        bind_runtime_frame(model, _frame(**kwargs))


def test_nonfinite_dep2_aggregate_value_fails():
    model = _dep2_model()
    frame = _frame(local={("A", "qac"): 0.0, ("B", "qac"): 0.0}, aggregate={"zic": np.nan})
    with pytest.raises(RCCompileError, match="Non-finite aggregate"):
        bind_runtime_frame(model, frame, allocation_results=_dep2_alloc(model))


def test_missing_dep2_allocation_result_fails():
    model = _dep2_model()
    frame = _frame(local={("A", "qac"): 0.0, ("B", "qac"): 0.0}, aggregate={"zic": 1.0})
    with pytest.raises(RCCompileError, match="Missing DEP2 allocation results"):
        bind_runtime_frame(model, frame)


def test_allocation_result_is_invalid_for_non_dep2_mode():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    with pytest.raises(RCCompileError, match="only valid in DEP2"):
        bind_runtime_frame(
            model,
            _frame(local={("A", "qac"): 0.0}),
            allocation_results={"x": object()},
        )


# ---------------------------------------------------------------------------
# E0-4F/H: recursive state ownership and runtime invariants
# ---------------------------------------------------------------------------

def test_recursive_state_starts_from_e04_initialization():
    model = compile_rc_model(RCCompilerSpec("2r2c", ("A",), "ind"))
    init = initialize_runtime_state(model, {"A": ZoneInitializationEvidence(observed_air_temperature_c=23.0)})
    state = start_recursive_state(model, init, timestamp="t0")
    assert state.origin is RuntimeStateOrigin.INITIALIZATION
    np.testing.assert_allclose(state.state, [23.0, 23.0])


def test_accept_model_evolved_state_does_not_take_observations():
    model = compile_rc_model(RCCompilerSpec("2r2c", ("A",), "ind"))
    init = initialize_runtime_state(model)
    current = start_recursive_state(model, init, timestamp="t0")
    evolved = np.array([22.7, 22.2])
    nxt = accept_model_evolved_state(model, current, evolved, next_timestamp="t1")
    assert nxt.origin is RuntimeStateOrigin.MODEL_EVOLUTION
    np.testing.assert_allclose(nxt.state, evolved)


def test_rich_frame_observation_does_not_mutate_recursive_state():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    init = initialize_runtime_state(model, {"A": ZoneInitializationEvidence(observed_air_temperature_c=22.0)})
    state = start_recursive_state(model, init, timestamp="t0")
    before = state.state.copy()
    bind_runtime_frame(
        model,
        _frame(timestamp="t0", local={("A", "qac"): 0.0}, observed={"A": 29.0}),
    )
    np.testing.assert_array_equal(state.state, before)


def test_explicit_reset_requires_named_reason():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    with pytest.raises(RCCompileError, match="non-empty reason"):
        explicit_state_reset(model, np.array([25.0]), timestamp="t1", reason="")


def test_explicit_reset_is_visible_in_state_origin_and_reason():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    reset = explicit_state_reset(model, np.array([25.0]), timestamp="t1", reason="observer assimilation")
    assert reset.origin is RuntimeStateOrigin.EXPLICIT_RESET
    assert reset.reset_reason == "observer assimilation"


def test_runtime_state_dimension_is_enforced():
    model = compile_rc_model(RCCompilerSpec("2r2c", ("A",), "ind"))
    with pytest.raises(RCCompileError, match="shape"):
        explicit_state_reset(model, np.array([22.0]), timestamp="t1", reason="test")


def test_runtime_state_finiteness_is_enforced():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind"))
    with pytest.raises(RCCompileError, match="non-finite"):
        explicit_state_reset(model, np.array([np.nan]), timestamp="t1", reason="test")


def test_state_and_input_timestamps_must_match():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    init = initialize_runtime_state(model)
    state = start_recursive_state(model, init, timestamp="t0")
    binding = bind_runtime_frame(model, _frame(timestamp="t1", local={("A", "qac"): 0.0}))
    with pytest.raises(RCCompileError, match="State/input timestamp mismatch"):
        assert_state_binding_timestamp(state, binding)


def test_state_and_input_matching_timestamps_pass():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    init = initialize_runtime_state(model)
    state = start_recursive_state(model, init, timestamp="t0")
    binding = bind_runtime_frame(model, _frame(timestamp="t0", local={("A", "qac"): 0.0}))
    assert_state_binding_timestamp(state, binding)


def test_runtime_invariant_report_passes_for_valid_realization():
    model = _dep2_model(weights={"A": 0.7, "B": 0.3})
    init = initialize_runtime_state(
        model,
        {
            "A": ZoneInitializationEvidence(observed_air_temperature_c=22.0),
            "B": ZoneInitializationEvidence(observed_air_temperature_c=24.0),
        },
    )
    frame = _frame(
        local={("A", "qac"): -100.0, ("B", "qac"): -200.0},
        aggregate={"zic": 900.0},
        aux={("A", "phvac"): 500.0, ("B", "phvac"): 700.0},
    )
    binding = bind_runtime_frame(model, frame, allocation_results=_dep2_alloc(model, (0.4,)))
    report = validate_runtime_invariants(model, init, frame, binding)
    assert report.passed
    assert report.phvac_excluded
    assert report.structural_absence_respected


def test_runtime_binding_cannot_change_physics_graph():
    model = compile_rc_model(RCCompilerSpec("1r1c", ("A",), "ind", zone_port_availability={"A": ("qac",)}))
    before = graph_signature(model)
    bind_runtime_frame(model, _frame(local={("A", "qac"): 0.0}))
    assert_runtime_binding_does_not_change_physics(model, before)


def test_dep1_dep2_runtime_physics_are_equivalent():
    common = dict(
        flavour="2r2c",
        zone_ids=("A", "B"),
        adjacency=(ZoneAdjacency("A", "B"),),
        zone_port_availability=_ports(("A", "B"), ("qac", "zic")),
    )
    dep1 = compile_rc_model(RCCompilerSpec(mode="dep1", **common))
    dep2 = compile_rc_model(
        RCCompilerSpec(mode="dep2", dep2_allocations=(_dep2_family(),), **common)
    )
    assert_dep1_dep2_runtime_physics_equivalent(dep1, dep2)


def test_initialization_is_deterministic_for_identical_inputs():
    model = compile_rc_model(RCCompilerSpec("4r3c", ("A", "B"), "ind"))
    evidence = {
        "A": ZoneInitializationEvidence(observed_air_temperature_c=22.1),
        "B": ZoneInitializationEvidence(heating_setpoint_c=20.0, cooling_setpoint_c=24.0),
    }
    first = initialize_runtime_state(model, evidence)
    second = initialize_runtime_state(model, evidence)
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(first.lifting_matrix, second.lifting_matrix)
    assert first.source_by_zone == second.source_by_zone


# ---------------------------------------------------------------------------
# Complete worked example from the locked E0-4 contract
# ---------------------------------------------------------------------------

def test_locked_complete_two_zone_example():
    model = compile_rc_model(
        RCCompilerSpec(
            "2r2c",
            ("Dining", "Kitchen"),
            "dep2",
            adjacency=(ZoneAdjacency("Dining", "Kitchen"),),
            zone_port_availability=_ports(("Dining", "Kitchen"), ("qac", "zic")),
            dep2_allocations=(
                AllocationFamilySpec(
                    name="zic_family",
                    signals=("zic",),
                    weights={"Dining": 0.5, "Kitchen": 0.5},
                    mode=AllocationMode.FIXED,
                    fixed_lambdas={"Dining": 1.4, "Kitchen": 0.6},
                ),
            ),
        )
    )
    init = initialize_runtime_state(
        model,
        {"Kitchen": ZoneInitializationEvidence(observed_air_temperature_c=24.0)},
        request=InitializationRequest(policy="auto", user_temperatures_c={"Dining": 21.5}),
    )
    np.testing.assert_allclose(init.state, [21.5, 21.5, 24.0, 24.0])

    from scalebridge.models.grey_box.rc_networks import fixed_allocation_result

    allocation = {
        "zic_family": fixed_allocation_result(
            model.allocation_families["zic_family"], model.spec.zone_ids
        )
    }
    frame = _frame(
        local={("Dining", "qac"): -1500.0, ("Kitchen", "qac"): -900.0},
        aggregate={"zic": 1000.0},
        boundary={"outdoor_temperature": 5.0},
        aux={("Dining", "phvac"): 1800.0, ("Kitchen", "phvac"): 1200.0},
    )
    binding = bind_runtime_frame(model, frame, allocation_results=allocation)
    by_port = dict(zip(binding.model_applicable_ports, binding.effective_thermal_vector))
    assert by_port[("Dining", "qac")] == pytest.approx(-1500.0)
    assert by_port[("Kitchen", "qac")] == pytest.approx(-900.0)
    assert by_port[("Dining", "zic")] == pytest.approx(1400.0)
    assert by_port[("Kitchen", "zic")] == pytest.approx(600.0)
    recovered = 0.5 * by_port[("Dining", "zic")] + 0.5 * by_port[("Kitchen", "zic")]
    assert recovered == pytest.approx(1000.0)
    assert binding.unused_auxiliary_electrical_keys == (
        ("Dining", "phvac"),
        ("Kitchen", "phvac"),
    )
