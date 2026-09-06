"""PINODE/EPSR reproducible research package.

Canonical implementation lives under this ``src/pinode_epsr`` package.
The parent ``Paper_PINODE_EPSR`` modules are compatibility shims for the
historical ScaleBridge paper-development API.
"""

from .core.config import PaperConfig, canonical_case_specs
from .data.phase_d import PhaseDTrajectory, load_case, load_manifest_only
from .methods import (
    BaiCuiResidentialRCReference, InversePINNConfig, InversePINNRC,
    NeuralODEConfig, NeuralODEModel, BasePINODEConfig, BasePINODEModel,
    EBPPINODEConfig, EBPPINODEModel,
)
from .data.phase_c import PhaseCModelBundle, discover_and_load_phase_c_bundle
from .data.thermostat_data import (
    TrainingAlignmentDiagnostics, calibrate_controlled_thermostats,
)
from .evaluation.thermostat import (ThermostatCalibration, ThermostatActuationProfile, ThermostatModeActuation, LegacyHeatingCoolingThermostat, resolve_actuation_profile)
from .evaluation.runtime import EvaluationResult, PaperModelRuntime, sim1, sim2, sim3

__all__ = [
    "PaperConfig", "canonical_case_specs", "PhaseDTrajectory", "load_case",
    "load_manifest_only", "BaiCuiResidentialRCReference", "InversePINNConfig",
    "InversePINNRC", "NeuralODEConfig", "NeuralODEModel", "BasePINODEConfig",
    "BasePINODEModel", "EBPPINODEConfig", "EBPPINODEModel",
    "PhaseCModelBundle", "discover_and_load_phase_c_bundle",
    "TrainingAlignmentDiagnostics", "calibrate_controlled_thermostats",
    "ThermostatCalibration", "ThermostatActuationProfile", "ThermostatModeActuation", "LegacyHeatingCoolingThermostat", "resolve_actuation_profile",
    "EvaluationResult", "PaperModelRuntime", "sim1", "sim2", "sim3",
]
