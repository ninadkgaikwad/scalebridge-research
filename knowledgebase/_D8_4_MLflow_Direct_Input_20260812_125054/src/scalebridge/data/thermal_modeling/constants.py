# -*- coding: utf-8 -*-
"""Constants and enums for Phase D thermal-model data contracts."""

from __future__ import annotations

from enum import Enum

PHASE_D_SCHEMA_VERSION = "phase_d_d1_v1"
DEFAULT_INCLUDE_VISIBLE_LIGHTING_IN_ZIR = True
DEFAULT_ZERO_ABSOLUTE_TOLERANCE = 1.0e-9


class PhaseDMode(str, Enum):
    """Supported Phase D spatial organization modes."""

    INDEPENDENT = "independent"
    DEPENDENT1 = "dependent1"
    DEPENDENT2 = "dependent2"


class ModelingSilo(str, Enum):
    """Downstream modeling-data silos supported by Phase D."""

    ML_SCIML = "ml_sciml"
    OPT_BAYES = "opt_bayes"


class SourcePhase(str, Enum):
    """Authoritative upstream source phase for a Phase D signal."""

    PHASE_B = "phase_b"
    PHASE_C = "phase_c"
    PHASE_D_DERIVED = "phase_d_derived"


class SignalRole(str, Enum):
    """Semantic role of a signal in Phase D."""

    TIME = "time"
    STATE = "state"
    DISTURBANCE = "disturbance"
    THERMAL_INPUT = "thermal_input"
    AUXILIARY = "auxiliary"
    DERIVED_GROUP = "derived_group"


class PhaseDSignalStatus(str, Enum):
    """Phase D usability classification for one zone-specific signal."""

    VARYING = "varying"
    CONSTANT_NONZERO = "constant_nonzero"
    NULLABLE_COMPLETE_ZERO = "nullable_complete_zero"
    NULLABLE_NOT_APPLICABLE = "nullable_not_applicable"
    VALIDATION_FAILURE = "validation_failure"


class NullableReason(str, Enum):
    """Why a canonical Phase D signal column is nullable."""

    COMPLETE_ZERO_SIGNAL = "complete_zero_signal"
    PHASE_C_MODEL_NOT_APPLICABLE = "phase_c_model_not_applicable"
    NONE = "none"
