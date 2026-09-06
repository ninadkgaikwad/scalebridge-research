from __future__ import annotations

"""E0-4 recursive runtime state ownership without numerical integration."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .compiler import CompiledRCModel
from .initialization import InitializationResult
from .runtime_binding import RuntimeBinding
from .specification import RCCompileError


class RuntimeStateOrigin(str, Enum):
    INITIALIZATION = "initialization"
    MODEL_EVOLUTION = "model_evolution"
    EXPLICIT_RESET = "explicit_reset"


@dataclass(frozen=True)
class RuntimeStateSnapshot:
    timestamp: object
    state: np.ndarray
    origin: RuntimeStateOrigin
    reset_reason: str | None = None


def _validated_state(model: CompiledRCModel, value: np.ndarray) -> np.ndarray:
    state = np.asarray(value, dtype=float).reshape(-1)
    if state.shape != (model.state_dimension,):
        raise RCCompileError(
            f"Runtime state shape must be {(model.state_dimension,)}, got {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise RCCompileError("Runtime state contains non-finite values")
    return state.copy()


def start_recursive_state(
    model: CompiledRCModel,
    initialization: InitializationResult,
    *,
    timestamp: object,
) -> RuntimeStateSnapshot:
    """Create the model-owned recursive state at t0 from E0-4 initialization."""

    if timestamp is None:
        raise RCCompileError("Runtime state requires a timestamp")
    state = _validated_state(model, initialization.state)
    return RuntimeStateSnapshot(
        timestamp=timestamp,
        state=state,
        origin=RuntimeStateOrigin.INITIALIZATION,
    )


def accept_model_evolved_state(
    model: CompiledRCModel,
    current: RuntimeStateSnapshot,
    evolved_state: np.ndarray,
    *,
    next_timestamp: object,
) -> RuntimeStateSnapshot:
    """Accept a state produced externally by the future E0-5 integrator.

    This function deliberately accepts only the evolved model state; it has no
    measured-temperature argument, preventing hidden teacher forcing.
    """

    if next_timestamp is None:
        raise RCCompileError("Evolved runtime state requires a timestamp")
    _validated_state(model, current.state)
    state = _validated_state(model, evolved_state)
    return RuntimeStateSnapshot(
        timestamp=next_timestamp,
        state=state,
        origin=RuntimeStateOrigin.MODEL_EVOLUTION,
    )


def explicit_state_reset(
    model: CompiledRCModel,
    state: np.ndarray,
    *,
    timestamp: object,
    reason: str,
) -> RuntimeStateSnapshot:
    """Perform a named higher-level reset; silent resets are not supported."""

    if timestamp is None:
        raise RCCompileError("Explicit reset requires a timestamp")
    if not str(reason).strip():
        raise RCCompileError("Explicit state reset requires a non-empty reason")
    return RuntimeStateSnapshot(
        timestamp=timestamp,
        state=_validated_state(model, state),
        origin=RuntimeStateOrigin.EXPLICIT_RESET,
        reset_reason=str(reason),
    )


def assert_state_binding_timestamp(
    state: RuntimeStateSnapshot,
    binding: RuntimeBinding,
) -> None:
    """Require X_k and U_k to belong to the same canonical runtime timestamp."""

    try:
        equal = bool(state.timestamp == binding.timestamp)
    except Exception as exc:  # pragma: no cover
        raise RCCompileError("Unable to compare state/binding timestamps") from exc
    if not equal:
        raise RCCompileError(
            f"State/input timestamp mismatch: state={state.timestamp!r}, "
            f"binding={binding.timestamp!r}"
        )
