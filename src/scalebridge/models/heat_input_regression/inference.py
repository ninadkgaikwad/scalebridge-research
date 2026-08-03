# -*- coding: utf-8 -*-
"""Inference helpers for scalar heat-input regression components."""
from __future__ import annotations
from typing import Any
import numpy as np
from .base import HeatInputRegressionModel

def predict_component(model: HeatInputRegressionModel, predictor_values: Any) -> np.ndarray:
    return model.predict_batch(predictor_values)

def predict_component_one(model: HeatInputRegressionModel, predictor_value: float) -> float:
    return model.predict_one(predictor_value)
