from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import DT_SECONDS


@dataclass(frozen=True)
class TensorStandardizer:
    """Training-only affine normalization x_tilde=(x-mu)/scale."""

    mean: torch.Tensor
    scale: torch.Tensor
    names: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        values: np.ndarray | torch.Tensor,
        *,
        names: Sequence[str] = (),
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "TensorStandardizer":
        x = torch.as_tensor(values, dtype=dtype, device=device)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] < 1:
            raise ValueError("values must have shape [samples, features]")
        mean = x.mean(dim=0)
        scale = x.std(dim=0, unbiased=False)
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        return cls(mean=mean.detach(), scale=scale.detach(), names=tuple(names))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.scale.to(x)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return self.mean.to(z) + self.scale.to(z) * z

    def to_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "mean": self.mean.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class RolloutWindow:
    segment_id: int
    context_start: int
    start: int
    stop: int
    partition: str

    @property
    def prediction_steps(self) -> int:
        return self.stop - self.start


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def contiguous_segments(
    timestamp: Sequence[object] | pd.Series,
    partition: Sequence[str] | np.ndarray,
    included: Sequence[bool] | np.ndarray,
    *,
    partition_name: str,
    dt_seconds: float = DT_SECONDS,
) -> list[np.ndarray]:
    """Return maximal row-index segments that obey the Phase-D temporal contract."""

    ts = pd.to_datetime(pd.Series(timestamp), errors="raise")
    part = np.asarray(partition, dtype=str)
    inc = np.asarray(included, dtype=bool)
    if not (len(ts) == len(part) == len(inc)):
        raise ValueError("timestamp/partition/included lengths differ")
    eligible = (part == partition_name) & inc
    segments: list[list[int]] = []
    current: list[int] = []
    for i in range(len(ts)):
        if not eligible[i]:
            if current:
                segments.append(current)
                current = []
            continue
        if current:
            prev = current[-1]
            delta = (ts.iloc[i] - ts.iloc[prev]).total_seconds()
            if not math.isclose(delta, dt_seconds, rel_tol=0.0, abs_tol=1e-6):
                segments.append(current)
                current = []
        current.append(i)
    if current:
        segments.append(current)
    return [np.asarray(seg, dtype=int) for seg in segments]


def build_rollout_windows(
    segments: Sequence[np.ndarray],
    *,
    partition: str,
    N_r: int,
    L_e: int = 1,
    is_2c: bool = False,
) -> list[RolloutWindow]:
    """Construct Part-3 windows without crossing any segment/split boundary."""

    if N_r < 1:
        raise ValueError("N_r must be >= 1")
    if L_e < 1:
        raise ValueError("L_e must be >= 1")
    context = L_e if is_2c else 1
    windows: list[RolloutWindow] = []
    for segment_id, seg in enumerate(segments):
        n = len(seg)
        if n < context + N_r:
            continue
        # Tex: 2C k in [s+L_e-1, e-N_r]; 1C is the L_e=1 special case.
        for local_start in range(context - 1, n - N_r):
            start = int(seg[local_start])
            context_start = int(seg[local_start - context + 1])
            stop = int(seg[local_start + N_r])
            if stop - start != N_r:
                # Row indices themselves can only remain consecutive inside a segment;
                # keep the guard explicit for future data representations.
                continue
            windows.append(
                RolloutWindow(
                    segment_id=segment_id,
                    context_start=context_start,
                    start=start,
                    stop=stop,
                    partition=partition,
                )
            )
    return windows


def representative_window_subset(
    windows: Sequence[RolloutWindow],
    *,
    max_windows: int,
    seed: int = 42,
) -> list[RolloutWindow]:
    """Deterministic training-only subset spread across the available windows."""

    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    if len(windows) <= max_windows:
        return list(windows)
    rng = np.random.default_rng(seed)
    # Stratify by segment first, then fill from remaining windows.
    by_segment: dict[int, list[int]] = {}
    for i, w in enumerate(windows):
        by_segment.setdefault(w.segment_id, []).append(i)
    chosen: list[int] = []
    for segment_id in sorted(by_segment):
        if len(chosen) >= max_windows:
            break
        inds = by_segment[segment_id]
        chosen.append(inds[len(inds) // 2])
    remaining = np.setdiff1d(np.arange(len(windows)), np.asarray(chosen, dtype=int))
    if len(chosen) < max_windows:
        extra = rng.choice(remaining, size=max_windows - len(chosen), replace=False)
        chosen.extend(int(i) for i in extra)
    return [windows[i] for i in sorted(chosen)]


def save_checkpoint(path: Path, *, model: nn.Module, provenance: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "provenance": provenance}, path)


def load_checkpoint(path: Path, *, model: nn.Module, map_location: str | torch.device = "cpu") -> dict[str, object]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return dict(payload.get("provenance", {}))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
