from __future__ import annotations

"""E0-4 canonical runtime frame and model-input realization.

Mathematical authority
----------------------
ScaleBridge_PhaseE0_E0-4_Runtime_State_Input_Contract_v1.tex, E0-4C--E0-4E
and E0-4G.

This layer binds named runtime evidence into the already-compiled E0-3 input
ordering.  It does not integrate the model and cannot alter RC topology.
"""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .allocation import AllocationResult
from .compiler import CompiledRCModel
from .runtime import (
    RCInputSnapshot,
    assemble_boundary_vector,
    assemble_effective_thermal_vector,
)
from .specification import RCCompileError, SpatialMode


SignalKey = tuple[str, str]


@dataclass(frozen=True)
class CanonicalRuntimeFrame:
    """One timestamped canonical E0-4 runtime frame.

    Rich frames are allowed: values not consumed by a particular compiled model
    remain explicit unused inputs.  PHVAC belongs in ``auxiliary_electrical_powers``
    rather than a thermal-power mapping.
    """

    timestamp: object
    boundary_temperatures: Mapping[str, float] = field(default_factory=dict)
    local_thermal_powers: Mapping[SignalKey, float] = field(default_factory=dict)
    aggregate_thermal_powers: Mapping[str, float] = field(default_factory=dict)
    observed_air_temperatures: Mapping[str, float] = field(default_factory=dict)
    auxiliary_electrical_powers: Mapping[SignalKey, float] = field(default_factory=dict)
    local_source_availability: Mapping[SignalKey, bool] = field(default_factory=dict)
    aggregate_source_availability: Mapping[str, bool] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            raise RCCompileError("Canonical runtime frame requires a timestamp")

        for key in self.local_thermal_powers:
            _validate_signal_key(key, "local thermal power")
            if key[1].lower() == "phvac":
                raise RCCompileError(
                    "PHVAC is electrical power and cannot appear in local_thermal_powers"
                )
        for signal in self.aggregate_thermal_powers:
            token = str(signal).strip().lower()
            if not token:
                raise RCCompileError("Aggregate thermal signal name cannot be empty")
            if token == "phvac":
                raise RCCompileError(
                    "PHVAC is electrical power and cannot appear in aggregate_thermal_powers"
                )
            if token == "qac":
                raise RCCompileError("QAC must remain local and cannot be an aggregate DEP2 source")

        for key in self.auxiliary_electrical_powers:
            _validate_signal_key(key, "auxiliary electrical power")
        for key in self.local_source_availability:
            _validate_signal_key(key, "local source availability")

        for key, available in self.local_source_availability.items():
            if not bool(available) and key in self.local_thermal_powers:
                raise RCCompileError(
                    f"Local source {key!r} is marked unavailable but a thermal value was supplied"
                )
        for signal, available in self.aggregate_source_availability.items():
            if not bool(available) and signal in self.aggregate_thermal_powers:
                raise RCCompileError(
                    f"Aggregate source {signal!r} is marked unavailable but a value was supplied"
                )


@dataclass(frozen=True)
class RuntimeBinding:
    timestamp: object
    snapshot: RCInputSnapshot
    boundary_vector: np.ndarray
    effective_thermal_vector: np.ndarray
    allocation_results: Mapping[str, AllocationResult]
    model_applicable_ports: tuple[SignalKey, ...]
    used_boundary_labels: tuple[str, ...]
    used_local_thermal_keys: tuple[SignalKey, ...]
    used_aggregate_signals: tuple[str, ...]
    unused_boundary_labels: tuple[str, ...]
    unused_local_thermal_keys: tuple[SignalKey, ...]
    unused_aggregate_signals: tuple[str, ...]
    unused_auxiliary_electrical_keys: tuple[SignalKey, ...]
    dep2_coordinate_error_max_abs: float


def _validate_signal_key(key: object, label: str) -> None:
    if not isinstance(key, tuple) or len(key) != 2:
        raise RCCompileError(f"{label} key must be a (zone_id, signal) tuple: {key!r}")
    zone, signal = key
    if not str(zone).strip() or not str(signal).strip():
        raise RCCompileError(f"{label} key cannot contain empty zone/signal: {key!r}")


def _timestamps_equal(left: object, right: object) -> bool:
    try:
        value = left == right
        if isinstance(value, np.ndarray):
            return bool(np.all(value))
        return bool(value)
    except Exception as exc:  # pragma: no cover - defensive for exotic timestamp objects
        raise RCCompileError("Unable to compare canonical runtime timestamps") from exc


def _validate_frame_zone_identity(model: CompiledRCModel, frame: CanonicalRuntimeFrame) -> None:
    valid = set(model.spec.zone_ids)
    referenced: set[str] = set(frame.observed_air_temperatures)
    referenced.update(zone for zone, _ in frame.local_thermal_powers)
    referenced.update(zone for zone, _ in frame.auxiliary_electrical_powers)
    referenced.update(zone for zone, _ in frame.local_source_availability)
    unknown = referenced - valid
    if unknown:
        raise RCCompileError(
            "Runtime frame references zone identities outside the compiled model: "
            f"{sorted(unknown)}"
        )


def _local_source_is_available(frame: CanonicalRuntimeFrame, key: SignalKey) -> bool:
    if key in frame.local_source_availability:
        return bool(frame.local_source_availability[key])
    return key in frame.local_thermal_powers


def _aggregate_source_is_available(frame: CanonicalRuntimeFrame, signal: str) -> bool:
    if signal in frame.aggregate_source_availability:
        return bool(frame.aggregate_source_availability[signal])
    return signal in frame.aggregate_thermal_powers


def model_forcing_applicability(model: CompiledRCModel) -> Mapping[SignalKey, bool]:
    """Return the E0-3 model-forcing applicability relation a_model."""

    return {(port.zone_id, port.signal): True for port in model.thermal_ports}


def runtime_source_availability(frame: CanonicalRuntimeFrame) -> Mapping[SignalKey, bool]:
    """Return local source availability a_src independently of model applicability."""

    keys = set(frame.local_source_availability) | set(frame.local_thermal_powers)
    return {key: _local_source_is_available(frame, key) for key in sorted(keys)}


def _required_boundary_values(
    model: CompiledRCModel,
    frame: CanonicalRuntimeFrame,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for node in model.boundary_nodes:
        label = node.boundary_label
        if label not in frame.boundary_temperatures:
            raise RCCompileError(f"Missing required boundary temperature {label!r}")
        value = float(frame.boundary_temperatures[label])
        if not np.isfinite(value):
            raise RCCompileError(f"Non-finite boundary temperature {label!r}")
        out[label] = value
    return out


def _required_forcing_values(
    model: CompiledRCModel,
    frame: CanonicalRuntimeFrame,
) -> tuple[dict[SignalKey, float], dict[str, float], set[SignalKey], set[str]]:
    local: dict[SignalKey, float] = {}
    aggregate: dict[str, float] = {}
    used_local: set[SignalKey] = set()
    used_aggregate: set[str] = set()

    for port in model.thermal_ports:
        key = (port.zone_id, port.signal)
        if model.spec.mode is SpatialMode.DEP2 and port.signal != "qac":
            signal = port.signal
            if not _aggregate_source_is_available(frame, signal):
                raise RCCompileError(
                    f"DEP2 requires authoritative all-to-one source for {signal!r}"
                )
            if signal not in frame.aggregate_thermal_powers:
                raise RCCompileError(f"Missing DEP2 all-to-one thermal signal {signal!r}")
            value = float(frame.aggregate_thermal_powers[signal])
            if not np.isfinite(value):
                raise RCCompileError(f"Non-finite aggregate thermal input {signal!r}")
            aggregate[signal] = value
            used_aggregate.add(signal)
            continue

        if not _local_source_is_available(frame, key):
            raise RCCompileError(
                f"Required local thermal source {key!r} is unavailable for {model.spec.mode.value}"
            )
        if key not in frame.local_thermal_powers:
            raise RCCompileError(f"Missing required local thermal input {key!r}")
        value = float(frame.local_thermal_powers[key])
        if not np.isfinite(value):
            raise RCCompileError(f"Non-finite local thermal input {key!r}")
        local[key] = value
        used_local.add(key)

    return local, aggregate, used_local, used_aggregate


def _dep2_coordinate_error(
    model: CompiledRCModel,
    aggregate_values: Mapping[str, float],
    allocation_results: Mapping[str, AllocationResult],
) -> float:
    if model.spec.mode is not SpatialMode.DEP2:
        return 0.0

    max_error = 0.0
    for signal, family_name in model.signal_to_allocation_family.items():
        family = model.allocation_families[family_name]
        allocation = allocation_results[family_name]
        qbar = float(aggregate_values[signal])
        reconstructed = {
            zone: float(allocation.lambda_by_zone[zone]) * qbar
            for zone in model.spec.zone_ids
        }
        recovered = sum(float(family.weights[z]) * reconstructed[z] for z in model.spec.zone_ids)
        max_error = max(max_error, abs(recovered - qbar))
    return max_error


def bind_runtime_frame(
    model: CompiledRCModel,
    frame: CanonicalRuntimeFrame,
    *,
    allocation_results: Mapping[str, AllocationResult] | None = None,
    expected_timestamp: object | None = None,
) -> RuntimeBinding:
    """Bind one canonical runtime frame into the exact E0-3 input ordering."""

    _validate_frame_zone_identity(model, frame)
    if expected_timestamp is not None and not _timestamps_equal(frame.timestamp, expected_timestamp):
        raise RCCompileError(
            f"Runtime timestamp mismatch: frame={frame.timestamp!r}, expected={expected_timestamp!r}"
        )

    if model.spec.mode is SpatialMode.DEP2:
        allocation_results = dict(allocation_results or {})
        required_families = set(model.allocation_families)
        missing = required_families - set(allocation_results)
        extra = set(allocation_results) - required_families
        if missing:
            raise RCCompileError(f"Missing DEP2 allocation results: {sorted(missing)}")
        if extra:
            raise RCCompileError(f"Unexpected DEP2 allocation results: {sorted(extra)}")
    else:
        if allocation_results:
            raise RCCompileError("Allocation results are only valid in DEP2 mode")
        allocation_results = {}

    boundary = _required_boundary_values(model, frame)
    local, aggregate, used_local, used_aggregate = _required_forcing_values(model, frame)

    snapshot = RCInputSnapshot(
        boundary_temperatures=boundary,
        local_thermal_powers=local,
        aggregate_thermal_powers=aggregate,
    )
    boundary_vector = assemble_boundary_vector(model, snapshot)
    effective = assemble_effective_thermal_vector(
        model,
        snapshot,
        allocation_results=allocation_results,
    )

    dep2_error = _dep2_coordinate_error(model, aggregate, allocation_results)
    if dep2_error > 1e-9:
        raise RCCompileError(
            f"Runtime DEP2 aggregate-coordinate consistency failed: error={dep2_error}"
        )

    used_boundary = set(boundary)
    return RuntimeBinding(
        timestamp=frame.timestamp,
        snapshot=snapshot,
        boundary_vector=boundary_vector,
        effective_thermal_vector=effective,
        allocation_results=allocation_results,
        model_applicable_ports=tuple((p.zone_id, p.signal) for p in model.thermal_ports),
        used_boundary_labels=tuple(sorted(used_boundary)),
        used_local_thermal_keys=tuple(sorted(used_local)),
        used_aggregate_signals=tuple(sorted(used_aggregate)),
        unused_boundary_labels=tuple(sorted(set(frame.boundary_temperatures) - used_boundary)),
        unused_local_thermal_keys=tuple(sorted(set(frame.local_thermal_powers) - used_local)),
        unused_aggregate_signals=tuple(sorted(set(frame.aggregate_thermal_powers) - used_aggregate)),
        unused_auxiliary_electrical_keys=tuple(sorted(frame.auxiliary_electrical_powers)),
        dep2_coordinate_error_max_abs=float(dep2_error),
    )
