# -*- coding: utf-8 -*-
"""Deterministic PyTorch scalar linear regression estimator with CPU/GPU support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import (
    HeatInputRegressionModel,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    ModelFitSummary,
    as_1d_float_array,
)


class PyTorchLinearRegression(HeatInputRegressionModel):
    """Scalar linear regression optimized by PyTorch on an explicit device."""

    estimator_type = "pytorch_linear"
    SUPPORTED_DEVICES = {"cpu", "cuda", "auto"}

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        learning_rate: float = 1e-2,
        max_epochs: int = 2000,
        tolerance: float = 1e-10,
        patience: int = 100,
        seed: int = 42,
        device: str = "cpu",
        model_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(fit_intercept=fit_intercept, model_id=model_id, metadata=metadata)
        if learning_rate <= 0 or max_epochs <= 0 or tolerance < 0 or patience <= 0:
            raise ValueError("Invalid PyTorch optimization configuration.")
        requested = str(device).strip().lower()
        if requested not in self.SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported PyTorch device {device!r}; expected one of "
                f"{sorted(self.SUPPORTED_DEVICES)}."
            )
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.seed = int(seed)
        self.requested_device = requested
        self.resolved_device = ""
        self.torch_version = ""
        self.cuda_version = ""
        self.cuda_device_name = ""
        self._coefficient = 0.0
        self._intercept = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorchLinearRegression requires torch.") from exc
        return torch

    def _resolve_device(self, torch: Any) -> Any:
        if self.requested_device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = self.requested_device
        if resolved == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA training was requested, but torch.cuda.is_available() is False."
            )
        device = torch.device(resolved)
        self.resolved_device = str(device)
        self.torch_version = str(torch.__version__)
        self.cuda_version = str(torch.version.cuda or "")
        self.cuda_device_name = (
            str(torch.cuda.get_device_name(device)) if device.type == "cuda" else ""
        )
        return device

    @property
    def coefficient(self) -> float:
        self._require_fitted()
        return float(self._coefficient)

    @property
    def intercept(self) -> float:
        self._require_fitted()
        return float(self._intercept)

    def fit(self, x: Any, y: Any) -> "PyTorchLinearRegression":
        torch = self._torch()
        x_arr, y_arr = self._validate_xy(x, y)
        device = self._resolve_device(torch)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        xm = float(np.mean(x_arr))
        xs = float(np.std(x_arr))
        ym = float(np.mean(y_arr))
        ys = float(np.std(y_arr))
        if xs == 0.0:
            raise ValueError("x is constant.")
        if ys == 0.0:
            ys = 1.0

        if self.fit_intercept:
            x_t = torch.tensor(
                ((x_arr - xm) / xs).reshape(-1, 1),
                dtype=torch.float64,
                device=device,
            )
            y_t = torch.tensor(
                ((y_arr - ym) / ys).reshape(-1, 1),
                dtype=torch.float64,
                device=device,
            )
            weight = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64, device=device))
            bias = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float64, device=device))
            parameters = [weight, bias]
        else:
            x_scale = float(np.sqrt(np.mean(x_arr**2)))
            y_scale = float(np.sqrt(np.mean(y_arr**2)))
            if x_scale == 0.0:
                raise ValueError("x is constant at zero.")
            if y_scale == 0.0:
                y_scale = 1.0
            x_t = torch.tensor(
                (x_arr / x_scale).reshape(-1, 1),
                dtype=torch.float64,
                device=device,
            )
            y_t = torch.tensor(
                (y_arr / y_scale).reshape(-1, 1),
                dtype=torch.float64,
                device=device,
            )
            weight = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64, device=device))
            bias = None
            parameters = [weight]

        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        best = float("inf")
        stale = 0
        converged = False
        self.loss_history = []
        epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = x_t @ weight
            if bias is not None:
                prediction = prediction + bias
            loss = torch.mean((prediction - y_t) ** 2)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu().item())
            self.loss_history.append(value)
            if best - value > self.tolerance:
                best = value
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                converged = True
                break

        weight_value = float(weight.detach().cpu().item())
        if self.fit_intercept:
            bias_value = float(bias.detach().cpu().item()) if bias is not None else 0.0
            coefficient = (ys / xs) * weight_value
            intercept = ym + ys * bias_value - coefficient * xm
        else:
            coefficient = (y_scale / x_scale) * weight_value
            intercept = 0.0

        self._coefficient = float(coefficient)
        self._intercept = float(intercept)
        self._is_fitted = True
        mse = float(np.mean((y_arr - self.predict(x_arr)) ** 2))
        self._fit_summary = ModelFitSummary(
            self.estimator_type,
            int(x_arr.size),
            self._coefficient,
            self._intercept,
            self.fit_intercept,
            mse,
            converged,
            epoch,
        )
        return self

    def predict(self, x: Any) -> np.ndarray:
        self._require_fitted()
        x_arr = as_1d_float_array(x, name="x", allow_empty=True)
        return self._intercept + self._coefficient * x_arr

    def save(self, output_dir: str | Path) -> Path:
        self._require_fitted()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "estimator_type": self.estimator_type,
            "model_id": self.model_id,
            "fit_intercept": self.fit_intercept,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "patience": self.patience,
            "seed": self.seed,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cuda_device_name": self.cuda_device_name,
            "coefficient": self.coefficient,
            "intercept": self.intercept,
            "fit_summary": self.fit_summary.to_dict(),
            "loss_history": self.loss_history,
            "metadata": self.metadata,
        }
        path = output / "model_manifest.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "PyTorchLinearRegression":
        payload = json.loads(
            (Path(artifact_dir) / "model_manifest.json").read_text(encoding="utf-8")
        )
        if payload.get("estimator_type") != cls.estimator_type:
            raise ValueError("Artifact estimator type mismatch.")
        model = cls(
            fit_intercept=payload["fit_intercept"],
            learning_rate=payload.get("learning_rate", 1e-2),
            max_epochs=payload.get("max_epochs", 2000),
            tolerance=payload.get("tolerance", 1e-10),
            patience=payload.get("patience", 100),
            seed=payload.get("seed", 42),
            device=payload.get("requested_device", payload.get("resolved_device", "cpu")),
            model_id=payload.get("model_id", ""),
            metadata=payload.get("metadata", {}),
        )
        model.resolved_device = str(payload.get("resolved_device", ""))
        model.torch_version = str(payload.get("torch_version", ""))
        model.cuda_version = str(payload.get("cuda_version", ""))
        model.cuda_device_name = str(payload.get("cuda_device_name", ""))
        model._coefficient = float(payload["coefficient"])
        model._intercept = float(payload["intercept"])
        model._is_fitted = True
        model.loss_history = list(payload.get("loss_history", []))
        model._fit_summary = ModelFitSummary(**payload["fit_summary"])
        return model
