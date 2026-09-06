from scalebridge.data.thermal_modeling.constants import (
    NullableReason,
    PhaseDSignalStatus,
    SourcePhase,
)
from scalebridge.data.thermal_modeling.signals import (
    build_signal_registry,
    classify_phase_c_signal,
    group_components,
)


def test_registry_uses_phase_b_temperatures_and_phase_c_heat_inputs():
    registry = build_signal_registry()
    assert registry["zone_temperature"].source_phase is SourcePhase.PHASE_B
    assert registry["outdoor_temperature"].source_phase is SourcePhase.PHASE_B
    assert registry["qsol1"].source_name == "predicted_QSol1"
    assert registry["qac"].source_name == "predicted_QAC"
    assert registry["phvac"].auxiliary is True


def test_visible_lighting_is_in_zir_by_default_and_optional():
    assert "qzivr_l" in group_components("zir")
    assert "qzivr_l" not in group_components("zir", include_visible_lighting_in_zir=False)


def test_varying_signal_is_retained():
    result = classify_phase_c_signal([1.0, 2.0, 3.0], phase_c_applicable=True)
    assert result.status is PhaseDSignalStatus.VARYING
    assert result.nullable is False


def test_constant_nonzero_signal_is_retained():
    result = classify_phase_c_signal([250.0, 250.0, 250.0], phase_c_applicable=True)
    assert result.status is PhaseDSignalStatus.CONSTANT_NONZERO
    assert result.nullable is False
    assert result.constant_value == 250.0


def test_complete_zero_signal_becomes_nullable():
    result = classify_phase_c_signal([0.0, 0.0, 0.0], phase_c_applicable=True)
    assert result.status is PhaseDSignalStatus.NULLABLE_COMPLETE_ZERO
    assert result.nullable is True
    assert result.nullable_reason is NullableReason.COMPLETE_ZERO_SIGNAL


def test_non_applicable_signal_becomes_nullable():
    result = classify_phase_c_signal([], phase_c_applicable=False)
    assert result.status is PhaseDSignalStatus.NULLABLE_NOT_APPLICABLE
    assert result.nullable is True
    assert result.nullable_reason is NullableReason.PHASE_C_MODEL_NOT_APPLICABLE


def test_missing_rows_in_applicable_signal_are_validation_failure():
    result = classify_phase_c_signal([1.0, None, 2.0], phase_c_applicable=True)
    assert result.status is PhaseDSignalStatus.VALIDATION_FAILURE
    assert result.nullable is False
