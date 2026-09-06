from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import time
from typing import Any, Sequence

import numpy as np
import torch

from ..backends.neuromancer import scalar_objective_problem
from ..core.common import RolloutWindow, set_deterministic
from ..data.method_data import inverse_pinn_forcing, node_method_arrays
from ..evaluation.runtime import PaperModelRuntime
from ..methods.inverse_pinn import InversePINNRC
from ..training.trainer import OptimizationConfig, build_optimizer
from .objectives import score_temperature_objective


@dataclass(frozen=True)
class FitOutcome:
    train_history: tuple[float, ...]
    validation_history: tuple[float, ...]
    best_validation_score: float
    best_epoch: int
    epochs_completed: int
    wall_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _node_window_loss(model, arrays, window: RolloutWindow) -> torch.Tensor:
    N_r = int(window.stop - window.start)
    k = int(window.start)
    rc_order = int(model.config.rc_order)
    if rc_order == 1:
        cy = cv = None
    else:
        cs = int(window.context_start)
        cy = torch.as_tensor(arrays.y[cs:k + 1], dtype=torch.float64)
        if isinstance(arrays.v, dict):
            cv = {key: torch.as_tensor(value[cs:k + 1], dtype=torch.float64) for key, value in arrays.v.items()}
        else:
            cv = torch.as_tensor(arrays.v[cs:k + 1], dtype=torch.float64)
    yt = torch.as_tensor(arrays.y[k:k + N_r + 1], dtype=torch.float64)
    if isinstance(arrays.v, dict):
        vs = {key: torch.as_tensor(value[k:k + N_r], dtype=torch.float64) for key, value in arrays.v.items()}
    else:
        vs = torch.as_tensor(arrays.v[k:k + N_r], dtype=torch.float64)
    return model.rollout_loss(y_true=yt, v_sequence=vs, context_y=cy, context_v=cv)["total"]


def fit_model(
    model,
    trajectory,
    *,
    fit_indices: np.ndarray,
    fit_windows: Sequence[RolloutWindow],
    validation_windows: Sequence[RolloutWindow],
    objective: str,
    optimization: OptimizationConfig,
    batch_size: int = 32,
) -> FitOutcome:
    """Train one method while selecting the best epoch by predictive rollout quality."""
    set_deterministic(optimization.seed)
    start_time = time.perf_counter()
    runtime = PaperModelRuntime(model, trajectory)
    rng = np.random.default_rng(optimization.seed)
    train_history: list[float] = []
    validation_history: list[float] = []
    best = float("inf"); best_epoch = -1; stale = 0
    best_state = copy.deepcopy(model.state_dict())

    if isinstance(model, InversePINNRC):
        fit_indices = np.asarray(fit_indices, dtype=int)
        arrays = node_method_arrays(trajectory, row_indices=fit_indices)
        t0 = trajectory.timestamp.iloc[int(fit_indices[0])]
        t_seconds = np.asarray([(trajectory.timestamp.iloc[int(i)] - t0).total_seconds() for i in fit_indices], dtype=float)
        forcing = inverse_pinn_forcing(trajectory, row_indices=fit_indices)
        def closure():
            return model.loss(
                t_seconds=torch.as_tensor(t_seconds, dtype=torch.float64),
                y_measured=torch.as_tensor(arrays.y, dtype=torch.float64),
                forcing=forcing,
            )["total"]
        current_windows = None
    else:
        arrays = node_method_arrays(trajectory)
        current_windows: list[RolloutWindow] = []
        def closure():
            if not current_windows:
                raise RuntimeError("No current training windows")
            return torch.stack([_node_window_loss(model, arrays, w) for w in current_windows]).mean()

    problem, loader = scalar_objective_problem(model, closure, dataset_name="train")
    problem_input = next(iter(loader))
    optimizer = build_optimizer(problem.parameters(), optimization)

    n_epochs = int(optimization.max_epochs)
    for epoch in range(n_epochs):
        if current_windows is not None:
            if not fit_windows:
                raise ValueError("No legal training rollout windows")
            n_batch = min(max(1, int(batch_size)), len(fit_windows))
            chosen = rng.choice(len(fit_windows), size=n_batch, replace=False)
            current_windows[:] = [fit_windows[int(i)] for i in chosen]

        optimizer.zero_grad(set_to_none=True)
        out = problem(problem_input)
        loss = out["train_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite production training loss")
        loss.backward()
        if optimization.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimization.gradient_clip_norm)
        optimizer.step()
        train_value = float(loss.detach().cpu())
        train_history.append(train_value)

        score = score_temperature_objective(runtime, list(validation_windows), objective)
        validation_history.append(score)
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite predictive validation objective")
        if score < best - optimization.min_delta:
            best = float(score); best_epoch = epoch; stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= optimization.patience:
            break

    model.load_state_dict(best_state)
    return FitOutcome(
        train_history=tuple(train_history), validation_history=tuple(validation_history),
        best_validation_score=float(best), best_epoch=int(best_epoch),
        epochs_completed=len(train_history), wall_seconds=float(time.perf_counter() - start_time),
    )
