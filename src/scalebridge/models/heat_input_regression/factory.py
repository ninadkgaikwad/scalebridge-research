# -*- coding: utf-8 -*-
"""Factory for reusable Stage C regression estimators."""
from __future__ import annotations
from typing import Any
from .base import HeatInputRegressionModel
from .linear_closed_form import ClosedFormLinearRegression
from .linear_pytorch import PyTorchLinearRegression

ESTIMATOR_ALIASES={
    "closed_form_linear":"closed_form_linear","ols":"closed_form_linear","linear_closed_form":"closed_form_linear",
    "pytorch_linear":"pytorch_linear","torch_linear":"pytorch_linear",
}

def create_heat_input_regression_model(estimator_type: str, **kwargs: Any) -> HeatInputRegressionModel:
    key=ESTIMATOR_ALIASES.get(str(estimator_type).strip().lower())
    if key is None: raise ValueError(f"Unsupported estimator_type {estimator_type!r}. Supported: {sorted(set(ESTIMATOR_ALIASES.values()))}")
    return ClosedFormLinearRegression(**kwargs) if key=="closed_form_linear" else PyTorchLinearRegression(**kwargs)
