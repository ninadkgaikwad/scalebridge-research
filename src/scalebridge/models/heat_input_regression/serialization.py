# -*- coding: utf-8 -*-
"""Load saved Stage C regression artifacts without caller-side type branching."""
from __future__ import annotations
from pathlib import Path
import json
from .base import HeatInputRegressionModel
from .linear_closed_form import ClosedFormLinearRegression
from .linear_pytorch import PyTorchLinearRegression

_LOADERS={"closed_form_linear":ClosedFormLinearRegression.load,"pytorch_linear":PyTorchLinearRegression.load}

def load_heat_input_regression_model(artifact_dir: str | Path) -> HeatInputRegressionModel:
    root=Path(artifact_dir); payload=json.loads((root/"model_manifest.json").read_text(encoding="utf-8")); kind=payload.get("estimator_type")
    if kind not in _LOADERS: raise ValueError(f"Unsupported saved estimator_type {kind!r}.")
    return _LOADERS[kind](root)
