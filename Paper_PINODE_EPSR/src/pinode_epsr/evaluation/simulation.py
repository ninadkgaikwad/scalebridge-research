from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..data.phase_d import PhaseDTrajectory


class OneStepModel(Protocol):
    def __call__(self, state: np.ndarray, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray: ...


@dataclass
class SimulationResult:
    case_name: str
    simulation: str
    prediction: np.ndarray
    target: np.ndarray
    timestamps: np.ndarray


def sim1_one_step(model: OneStepModel, trajectory: PhaseDTrajectory, partition: str = "test") -> SimulationResult:
    """Sim1: state, disturbance, and QAC/control all come from the test dataset."""
    data = trajectory.split(partition)
    prediction = np.vstack(
        [model(x, u, d) for x, u, d in zip(data.state, data.control, data.disturbance)]
    )
    return SimulationResult(
        case_name=trajectory.case_name,
        simulation="sim1",
        prediction=prediction,
        target=data.target,
        timestamps=data.timestamp.to_numpy(),
    )


def sim2_recorded_control_rollout(
    model: OneStepModel,
    trajectory: PhaseDTrajectory,
    partition: str = "test",
) -> SimulationResult:
    """Sim2: predicted state is recursively fed back; QAC and disturbances are recorded test values.

    The state dimension must match the prediction target dimension.  Hidden-state
    models (e.g. 2R2C) should expose a paper-model wrapper whose observable input
    and output are zone-air temperatures while it retains hidden state internally.
    """
    data = trajectory.split(partition)
    if len(data.state) == 0:
        raise ValueError(f"No included rows in partition {partition}")
    if data.state.shape[1] != data.target.shape[1]:
        raise ValueError(
            "Sim2 observable rollout expects state dimension == target dimension. "
            "Use a model wrapper for hidden RC states."
        )

    current = data.state[0].copy()
    preds = []
    for u, d in zip(data.control, data.disturbance):
        nxt = np.asarray(model(current, u, d), dtype=float).reshape(-1)
        preds.append(nxt)
        current = nxt
    return SimulationResult(
        case_name=trajectory.case_name,
        simulation="sim2",
        prediction=np.vstack(preds),
        target=data.target,
        timestamps=data.timestamp.to_numpy(),
    )
