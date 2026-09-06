from __future__ import annotations

"""E0-5 fixed-step ZOH common discretization engine."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from ..compiler import CompiledRCModel
from ..runtime_binding import RuntimeBinding
from ..runtime_state import (
    RuntimeStateSnapshot,
    accept_model_evolved_state,
    assert_state_binding_timestamp,
)
from .contracts import (
    DiscretizationConfig,
    DiscretizationError,
    DiscretizationProvenance,
    StepDiagnostics,
    TensorStepResult,
    validate_sample_dt,
)
from .diagnostics import build_step_diagnostics
from .linear_oracle import ExactZOHLinearIntegrator
from .linear_system import LinearRCStateSpace, compile_linear_state_space
from .neuromancer_ode import build_neuromancer_rc_ode_system
from .solver_registry import (
    build_neuromancer_integrator,
    normalize_solver_name,
    solver_capability,
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _as_2d_tensor(
    value,
    *,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=dtype, device=device)
    if tensor.ndim == 1:
        if tensor.shape[0] != width:
            raise DiscretizationError(
                f"{label} width must be {width}, got {tuple(tensor.shape)}"
            )
        return tensor.unsqueeze(0)
    if tensor.ndim == 2 and tensor.shape[1] == width:
        return tensor
    raise DiscretizationError(
        f"{label} must have shape ({width},) or (batch,{width}); got {tuple(tensor.shape)}"
    )


@dataclass(frozen=True)
class RuntimeStepResult:
    runtime_state: RuntimeStateSnapshot
    provenance: DiscretizationProvenance
    diagnostics: StepDiagnostics


class CommonDiscretizationEngine:
    """Numerically evolve one materialized compiled linear RC graph.

    Normal execution intentionally performs no eigendecomposition or oracle
    comparison. Those scientific diagnostics are entered only when
    ``diagnostics_per_step=True``.
    """

    def __init__(
        self,
        model: CompiledRCModel,
        parameter_values: Mapping[str, float],
        *,
        config: DiscretizationConfig | None = None,
    ) -> None:
        self.model = model
        self.parameter_values = dict(parameter_values)
        self.config = config or DiscretizationConfig()
        self.linear_system: LinearRCStateSpace = compile_linear_state_space(
            model, self.parameter_values
        )
        self._torch_system_cache = {}
        self._exact_integrator_cache = {}
        self._neuromancer_integrator_cache = {}
        capability = solver_capability(self.config.normalized_solver)
        if not capability.available:
            raise DiscretizationError(
                f"Configured E0-5 solver {capability.key!r} is unavailable"
            )

    def _provenance(
        self,
        *,
        sample_dt_s: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> DiscretizationProvenance:
        dt = validate_sample_dt(sample_dt_s)
        h = self.config.step_size_s(dt)
        capability = solver_capability(self.config.normalized_solver)
        return DiscretizationProvenance(
            solver=capability.key,
            backend=capability.backend,
            sample_dt_s=dt,
            substeps=int(self.config.substeps),
            integration_h_s=h,
            input_hold="zoh_left",
            dtype=_dtype_name(dtype),
            device=str(device),
            diagnostics_per_step=bool(self.config.diagnostics_per_step),
        )

    def step_tensor(
        self,
        state,
        boundary,
        thermal,
        *,
        sample_dt_s: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> TensorStepResult:
        dt = validate_sample_dt(sample_dt_s)

        if isinstance(state, torch.Tensor):
            inferred_dtype = state.dtype if state.is_floating_point() else torch.get_default_dtype()
            inferred_device = state.device
        else:
            inferred_dtype = torch.get_default_dtype()
            inferred_device = torch.device("cpu")
        dtype = dtype or inferred_dtype
        device_obj = torch.device(device) if device is not None else inferred_device

        system_key = (dtype, str(device_obj))
        system = self._torch_system_cache.get(system_key)
        if system is None:
            system = self.linear_system.to_torch(dtype=dtype, device=device_obj)
            self._torch_system_cache[system_key] = system
        x = _as_2d_tensor(
            state, width=system.state_dimension, dtype=dtype, device=device_obj, label="state"
        )
        tb = _as_2d_tensor(
            boundary,
            width=system.boundary_dimension,
            dtype=dtype,
            device=device_obj,
            label="boundary",
        )
        q = _as_2d_tensor(
            thermal,
            width=system.thermal_dimension,
            dtype=dtype,
            device=device_obj,
            label="thermal",
        )
        if not (x.shape[0] == tb.shape[0] == q.shape[0]):
            raise DiscretizationError("State/boundary/thermal batch sizes must match")

        capability = solver_capability(self.config.normalized_solver)
        initial = x
        local_error_tensors: tuple[torch.Tensor, ...] = ()

        if capability.exact_linear:
            exact = self._exact_integrator_cache.get(system_key)
            if exact is None:
                exact = ExactZOHLinearIntegrator(system)
                self._exact_integrator_cache[system_key] = exact
            h = self.config.step_size_s(dt)
            out = x
            for _ in range(int(self.config.substeps)):
                out = exact.step(out, tb, q, sample_dt_s=h)
                if out.ndim == 1:
                    out = out.unsqueeze(0)
        else:
            h = self.config.step_size_s(dt)
            integrator_key = (capability.key, float(h), dtype, str(device_obj))
            integrator = self._neuromancer_integrator_cache.get(integrator_key)
            if integrator is None:
                ode_system = build_neuromancer_rc_ode_system(system)
                integrator = build_neuromancer_integrator(capability.key, ode_system, h=h)
                self._neuromancer_integrator_cache[integrator_key] = integrator
            error_start = len(getattr(integrator, "local_error", ()))
            out = x
            for _ in range(int(self.config.substeps)):
                out = integrator(out, tb, q)
            if hasattr(integrator, "local_error"):
                new_errors = tuple(getattr(integrator, "local_error")[error_start:])
                if self.config.diagnostics_per_step:
                    local_error_tensors = new_errors
                # Neuromancer 1.5.6 RKF accumulates this list indefinitely.
                # E0-5 owns the wrapper lifecycle and prevents normal-run memory growth.
                getattr(integrator, "local_error").clear()

        diagnostics = build_step_diagnostics(
            config=self.config,
            sample_dt_s=dt,
            linear_system_numpy=self.linear_system,
            linear_system_torch=system,
            initial_state=initial,
            boundary=tb,
            thermal=q,
            numerical_state=out,
            local_error_tensors=local_error_tensors,
        )
        provenance = self._provenance(sample_dt_s=dt, dtype=dtype, device=device_obj)
        return TensorStepResult(state=out, provenance=provenance, diagnostics=diagnostics)

    def step_binding_tensor(
        self,
        state,
        binding: RuntimeBinding,
        *,
        sample_dt_s: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> TensorStepResult:
        return self.step_tensor(
            state,
            binding.boundary_vector,
            binding.effective_thermal_vector,
            sample_dt_s=sample_dt_s,
            dtype=dtype,
            device=device,
        )

    def step_runtime(
        self,
        current: RuntimeStateSnapshot,
        binding: RuntimeBinding,
        *,
        next_timestamp: object,
        sample_dt_s: float,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> RuntimeStepResult:
        """Advance E0-4 recursive state and return a MODEL_EVOLUTION snapshot."""

        assert_state_binding_timestamp(current, binding)
        tensor_result = self.step_binding_tensor(
            current.state,
            binding,
            sample_dt_s=sample_dt_s,
            dtype=dtype,
            device=device,
        )
        evolved = tensor_result.state.detach().cpu().numpy()
        if evolved.shape[0] != 1:
            raise DiscretizationError("Runtime state API expects exactly one trajectory")
        next_state = accept_model_evolved_state(
            self.model,
            current,
            np.asarray(evolved[0], dtype=float),
            next_timestamp=next_timestamp,
        )
        return RuntimeStepResult(
            runtime_state=next_state,
            provenance=tensor_result.provenance,
            diagnostics=tensor_result.diagnostics,
        )
