from __future__ import annotations

"""Forward-only E0-7 runtime instance for physical RC payloads."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from scalebridge.models.grey_box.rc_networks import (
    CanonicalRuntimeFrame,
    CommonDiscretizationEngine,
    InitializationRequest,
    RuntimeStateSnapshot,
    ZoneInitializationEvidence,
    bind_runtime_frame,
    initialize_runtime_state,
    observe,
    start_recursive_state,
)

from .contracts import PortableModelError
from .phvac import PHVACPrediction, PHVACRuntime
from .rc_payload import RCPhysicalPayload, load_rc_physical_payload


@dataclass(frozen=True)
class ForwardStepResult:
    timestamp: object
    next_timestamp: object
    observed_output: np.ndarray
    state: RuntimeStateSnapshot
    phvac: PHVACPrediction | None
    diagnostics: Mapping[str, object] | None = None


class RCForwardRuntime:
    """Post-estimation forward runtime for a final physical RC model.

    This is suitable for ODE/RC, Inverse-PINN deployment, physical-theta
    optimization results, and deterministic Bayesian point summaries.  It does
    not own estimation/training and never resets hidden state implicitly.
    """

    def __init__(
        self,
        payload: RCPhysicalPayload,
        *,
        phvac_runtime: PHVACRuntime | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.payload = payload
        self.model = payload.compiled_model()
        self.engine = CommonDiscretizationEngine(
            self.model,
            payload.theta,
            config=payload.discretization,
        )
        self.phvac_runtime = phvac_runtime
        self.dtype = dtype
        self.device = device
        self.state: RuntimeStateSnapshot | None = None

    @classmethod
    def from_bundle(
        cls,
        bundle,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "RCForwardRuntime":
        rel = str(bundle.manifest.payload.metadata.get("rc_payload_relpath", "")).strip()
        if not rel:
            raise PortableModelError(
                "Portable model payload metadata does not declare rc_payload_relpath"
            )
        payload = load_rc_physical_payload(bundle.path(rel))
        phvac_runtime = PHVACRuntime.from_bundle(bundle)
        return cls(payload, phvac_runtime=phvac_runtime, dtype=dtype, device=device)

    def initialize(
        self,
        *,
        timestamp: object,
        observed_air_temperatures_c: Mapping[str, float],
        request: InitializationRequest | None = None,
    ) -> RuntimeStateSnapshot:
        evidence = {
            zone: ZoneInitializationEvidence(
                observed_air_temperature_c=float(observed_air_temperatures_c[zone]),
            )
            for zone in self.model.spec.zone_ids
        }
        init = initialize_runtime_state(
            self.model,
            evidence,
            request=request or InitializationRequest(),
        )
        self.state = start_recursive_state(self.model, init, timestamp=timestamp)
        return self.state

    def reset_to_observation(
        self,
        *,
        timestamp: object,
        observed_air_temperatures_c: Mapping[str, float],
    ) -> RuntimeStateSnapshot:
        """Explicit Sim1-style reset; never called automatically during recursion."""
        return self.initialize(
            timestamp=timestamp,
            observed_air_temperatures_c=observed_air_temperatures_c,
        )

    def step(
        self,
        frame: CanonicalRuntimeFrame,
        *,
        next_timestamp: object,
        sample_dt_s: float,
        include_diagnostics: bool = False,
    ) -> ForwardStepResult:
        if self.state is None:
            raise PortableModelError("RCForwardRuntime must be initialized before step()")
        binding = bind_runtime_frame(
            self.model,
            frame,
            allocation_results=self.payload.dep2_allocation_results,
            expected_timestamp=self.state.timestamp,
        )
        stepped = self.engine.step_runtime(
            self.state,
            binding,
            next_timestamp=next_timestamp,
            sample_dt_s=sample_dt_s,
            dtype=self.dtype,
            device=self.device,
        )
        self.state = stepped.runtime_state
        y = np.asarray(observe(self.model, self.state.state), dtype=float)

        phvac = None
        if self.phvac_runtime is not None:
            qac = {
                zone: float(value)
                for (zone, signal), value in frame.local_thermal_powers.items()
                if str(signal).lower() == "qac"
            }
            phvac = self.phvac_runtime.predict(qac)

        diagnostics = None
        if include_diagnostics:
            diagnostics = {
                "controls_used": {
                    f"{zone}::{signal}": float(value)
                    for (zone, signal), value in frame.local_thermal_powers.items()
                    if str(signal).lower() == "qac"
                },
                "disturbances_used": {
                    "boundary_temperatures": dict(frame.boundary_temperatures),
                    "local_thermal_powers": {
                        f"{zone}::{signal}": float(value)
                        for (zone, signal), value in frame.local_thermal_powers.items()
                        if str(signal).lower() != "qac"
                    },
                    "aggregate_thermal_powers": dict(frame.aggregate_thermal_powers),
                },
                "binding": {
                    "used_boundary_labels": list(binding.used_boundary_labels),
                    "used_local_thermal_keys": [list(v) for v in binding.used_local_thermal_keys],
                    "used_aggregate_signals": list(binding.used_aggregate_signals),
                },
                "discretization": {
                    "solver": stepped.provenance.solver,
                    "sample_dt_s": stepped.provenance.sample_dt_s,
                    "substeps": stepped.provenance.substeps,
                },
            }

        return ForwardStepResult(
            timestamp=frame.timestamp,
            next_timestamp=next_timestamp,
            observed_output=y,
            state=self.state,
            phvac=phvac,
            diagnostics=diagnostics,
        )
