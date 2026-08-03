# -*- coding: utf-8 -*-
"""Abstract model contract for Stage C heat-input regression."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import numpy as np

MODEL_ARTIFACT_SCHEMA_VERSION = "1.0"


def as_1d_float_array(values: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {arr.shape}.")
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


@dataclass(frozen=True)
class ModelFitSummary:
    estimator_type: str
    sample_count: int
    coefficient: float
    intercept: float
    fit_intercept: bool
    training_loss: float
    converged: bool
    epochs_completed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeatInputRegressionModel(ABC):
    """Common API implemented by all Stage C scalar regression estimators."""

    estimator_type: str = "abstract"

    def __init__(self, *, fit_intercept: bool = True, model_id: str = "", metadata: Mapping[str, Any] | None = None) -> None:
        self.fit_intercept = bool(fit_intercept)
        self.model_id = str(model_id)
        self.metadata = dict(metadata or {})
        self._is_fitted = False
        self._fit_summary: ModelFitSummary | None = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def fit_summary(self) -> ModelFitSummary:
        if self._fit_summary is None:
            raise RuntimeError("Model has not been fitted.")
        return self._fit_summary

    @property
    @abstractmethod
    def coefficient(self) -> float: ...

    @property
    @abstractmethod
    def intercept(self) -> float: ...

    @abstractmethod
    def fit(self, x: Any, y: Any) -> "HeatInputRegressionModel": ...

    @abstractmethod
    def predict(self, x: Any) -> np.ndarray: ...

    def predict_one(self, x: float) -> float:
        return float(self.predict([x])[0])

    def predict_batch(self, x: Any) -> np.ndarray:
        return self.predict(x)

    @abstractmethod
    def save(self, output_dir: str | Path) -> Path: ...

    def _validate_xy(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        x_arr = as_1d_float_array(x, name="x")
        y_arr = as_1d_float_array(y, name="y")
        if x_arr.size != y_arr.size:
            raise ValueError(f"x and y lengths differ: {x_arr.size} != {y_arr.size}.")
        if x_arr.size < 2:
            raise ValueError("At least two samples are required.")
        if float(np.ptp(x_arr)) == 0.0:
            raise ValueError("x is constant; a regression slope cannot be identified.")
        return x_arr, y_arr

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction or serialization.")
