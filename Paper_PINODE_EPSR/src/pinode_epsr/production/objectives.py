from __future__ import annotations

import numpy as np

from ..evaluation.runtime import PaperModelRuntime
from ..core.common import RolloutWindow


def recursive_temperature_predictions(runtime: PaperModelRuntime, windows: list[RolloutWindow]) -> tuple[np.ndarray, np.ndarray]:
    truth: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    for window in windows:
        state = runtime.initialize(int(window.start))
        for k in range(int(window.start), int(window.stop)):
            state = runtime.step(state, k)
            pred.append(runtime.observe(state))
            truth.append(np.asarray(runtime.arrays.y[k + 1], dtype=float))
    if not truth:
        raise ValueError("No legal HPO holdout rollout predictions")
    return np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)


def score_temperature_objective(runtime: PaperModelRuntime, windows: list[RolloutWindow], objective: str) -> float:
    truth, pred = recursive_temperature_predictions(runtime, windows)
    err = pred - truth
    if objective == "recursive_temperature_rmse_C":
        return float(np.sqrt(np.mean(err ** 2)))
    if objective == "recursive_temperature_mae_C":
        return float(np.mean(np.abs(err)))
    if objective == "recursive_temperature_cvrmse":
        denom = float(np.mean(np.abs(truth)))
        if denom <= 1e-12:
            raise FloatingPointError("CVRMSE denominator is zero")
        return float(np.sqrt(np.mean(err ** 2)) / denom)
    if objective == "recursive_temperature_normalized":
        scale = np.asarray(runtime.model.S_y.detach().cpu(), dtype=float).reshape(1, -1)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return float(np.mean((err / scale) ** 2))
    raise ValueError(f"Unsupported HPO objective {objective!r}")
