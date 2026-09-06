from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


def get_generated_data_root() -> Path:
    """Return the configured ScaleBridge generated-data root."""
    value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
    if not value:
        raise RuntimeError("SCALEBRIDGE_GENERATED_DATA_ROOT is not configured.")
    return Path(value).resolve()


def get_machine_id() -> str:
    """Return the ScaleBridge machine id used for run metadata."""
    return (
        os.environ.get("SCALEBRIDGE_MACHINE_ID")
        or os.environ.get("COMPUTERNAME")
        or socket.gethostname()
    )


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def configure_mlflow_tracking() -> str:
    """
    Configure MLflow tracking URI.

    Local-server default:
        http://127.0.0.1:5000
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def semantic_artifact_uri(artifact_subdir: str) -> str:
    """
    Return a semantic artifact URI under:

        SCALEBRIDGE_GENERATED_DATA_ROOT/mlflow_artifacts/<artifact_subdir>
    """
    artifact_dir = get_generated_data_root() / "mlflow_artifacts" / artifact_subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir.as_uri()


def get_or_create_semantic_experiment(
    experiment_name: str,
    artifact_subdir: str | None = None,
) -> str:
    """
    Create or select an MLflow experiment with semantic artifact organization.

    Example:
        experiment_name = "p1_ann_baselines"
        artifact_subdir = "p1_ann_baselines"

    Artifacts go to:
        Data/ScaleBridge/mlflow_artifacts/p1_ann_baselines/
    """
    configure_mlflow_tracking()

    artifact_name = artifact_subdir or experiment_name
    artifact_uri = semantic_artifact_uri(artifact_name)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def set_standard_tags(
    *,
    campaign_id: str | None = None,
    case_id: str | None = None,
    model_family: str | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> None:
    """Attach standard ScaleBridge provenance tags to the active MLflow run."""
    tags: dict[str, Any] = {
        "machine_id": get_machine_id(),
        "hostname": socket.gethostname(),
        "git_commit": get_git_commit(),
        "tracking_uri": mlflow.get_tracking_uri(),
    }

    if campaign_id is not None:
        tags["campaign_id"] = campaign_id
    if case_id is not None:
        tags["case_id"] = case_id
    if model_family is not None:
        tags["model_family"] = model_family
    if extra_tags:
        tags.update(extra_tags)

    for key, value in tags.items():
        mlflow.set_tag(key, str(value))