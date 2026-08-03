# -*- coding: utf-8 -*-
"""Closed-form scalar ordinary/ridge least-squares estimator."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import json
import numpy as np

from .base import HeatInputRegressionModel, ModelFitSummary, MODEL_ARTIFACT_SCHEMA_VERSION, as_1d_float_array


class ClosedFormLinearRegression(HeatInputRegressionModel):
    estimator_type = "closed_form_linear"

    def __init__(self, *, fit_intercept: bool = True, ridge_alpha: float = 0.0, model_id: str = "", metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(fit_intercept=fit_intercept, model_id=model_id, metadata=metadata)
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be nonnegative.")
        self.ridge_alpha = float(ridge_alpha)
        self._coefficient = 0.0
        self._intercept = 0.0

    @property
    def coefficient(self) -> float:
        self._require_fitted(); return float(self._coefficient)

    @property
    def intercept(self) -> float:
        self._require_fitted(); return float(self._intercept)

    def fit(self, x: Any, y: Any) -> "ClosedFormLinearRegression":
        x_arr, y_arr = self._validate_xy(x, y)
        if self.fit_intercept:
            x_mean, y_mean = float(np.mean(x_arr)), float(np.mean(y_arr))
            xc, yc = x_arr - x_mean, y_arr - y_mean
            denom = float(np.dot(xc, xc) + self.ridge_alpha)
            if denom <= 0:
                raise ValueError("Regression denominator is zero.")
            coef = float(np.dot(xc, yc) / denom)
            intercept = float(y_mean - coef * x_mean)
        else:
            denom = float(np.dot(x_arr, x_arr) + self.ridge_alpha)
            if denom <= 0:
                raise ValueError("Regression denominator is zero.")
            coef = float(np.dot(x_arr, y_arr) / denom)
            intercept = 0.0
        self._coefficient, self._intercept = coef, intercept
        self._is_fitted = True
        residual = y_arr - self.predict(x_arr)
        mse = float(np.mean(residual ** 2))
        self._fit_summary = ModelFitSummary(self.estimator_type, int(x_arr.size), coef, intercept, self.fit_intercept, mse, True, 1)
        return self

    def predict(self, x: Any) -> np.ndarray:
        self._require_fitted()
        x_arr = as_1d_float_array(x, name="x", allow_empty=True)
        return self._intercept + self._coefficient * x_arr

    def save(self, output_dir: str | Path) -> Path:
        self._require_fitted()
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "estimator_type": self.estimator_type,
            "model_id": self.model_id,
            "fit_intercept": self.fit_intercept,
            "ridge_alpha": self.ridge_alpha,
            "coefficient": self.coefficient,
            "intercept": self.intercept,
            "fit_summary": self.fit_summary.to_dict(),
            "metadata": self.metadata,
        }
        path = output / "model_manifest.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "ClosedFormLinearRegression":
        path = Path(artifact_dir) / "model_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("estimator_type") != cls.estimator_type:
            raise ValueError(f"Artifact estimator_type is {payload.get('estimator_type')!r}, expected {cls.estimator_type!r}.")
        model = cls(fit_intercept=payload["fit_intercept"], ridge_alpha=payload.get("ridge_alpha", 0.0), model_id=payload.get("model_id", ""), metadata=payload.get("metadata", {}))
        model._coefficient = float(payload["coefficient"]); model._intercept = float(payload["intercept"]); model._is_fitted = True
        fs = payload["fit_summary"]; model._fit_summary = ModelFitSummary(**fs)
        return model
