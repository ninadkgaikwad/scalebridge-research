from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None


class MLflowTracker:
    """Thin wrapper around MLflow for research experiment tracking."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None):
        if mlflow is None:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_pytorch_model(self, model: Any, artifact_path: str = "model") -> None:
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)
