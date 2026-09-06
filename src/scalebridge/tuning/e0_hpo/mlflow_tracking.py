from __future__ import annotations

"""MLflow provenance layer for E0-8 studies and nested trials.

Optuna remains the optimization authority.  MLflow records study/trial/final
provenance and is never used to rank trials by run order.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import StudySpec


def _short(value: Any, limit: int = 500) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True)
class MLflowHPOConfig:
    enabled: bool = False
    experiment_name: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None
    artifact_root: str | Path | None = None
    strict: bool = False


class HPOStudyTracker:
    def __init__(self, config: MLflowHPOConfig) -> None:
        self.config = config
        self.parent_run_id: str | None = None
        self.trial_run_ids: dict[int, str] = {}
        self._active = False

    def _handle(self, exc: Exception, context: str) -> None:
        if self.config.strict:
            raise RuntimeError(f"E0-8 MLflow {context} failed: {exc}") from exc
        print(f"WARNING: E0-8 MLflow {context} failed: {type(exc).__name__}: {exc}")

    def start_study(self, spec: StudySpec) -> str | None:
        if not self.config.enabled:
            return None
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            from scalebridge.tracking.mlflow.semantic import set_standard_tags

            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)

            experiment_name = self.config.experiment_name or f"ScaleBridge_E0_8_{spec.method_family}"
            if self.config.artifact_root is not None:
                root = Path(self.config.artifact_root).resolve()
                root.mkdir(parents=True, exist_ok=True)
                client = MlflowClient()
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    client.create_experiment(experiment_name, artifact_location=root.as_uri())
                mlflow.set_experiment(experiment_name)
            else:
                from scalebridge.tracking.mlflow.semantic import get_or_create_semantic_experiment

                get_or_create_semantic_experiment(
                    experiment_name,
                    artifact_subdir=f"phase_e0/e08_hpo/{spec.method_family}",
                )

            run = mlflow.start_run(run_name=self.config.run_name or spec.study_name)
            self.parent_run_id = run.info.run_id
            self._active = True
            set_standard_tags(
                model_family=spec.method_family,
                extra_tags={
                    "pipeline_stage": "phase_e0_e08_hpo",
                    "e08_study_id": spec.study_id,
                    "e08_study_fingerprint": spec.fingerprint,
                    "method_id": spec.method_id,
                    "provider_version": spec.provider_version,
                },
            )
            mlflow.log_params(
                {
                    "study_name": spec.study_name,
                    "study_seed": spec.study_seed,
                    "sampler_name": spec.sampler_name,
                    "pruner_name": spec.pruner_name,
                    "objective_count": len(spec.objectives),
                    "data_selection_fingerprint": spec.data_selection.fingerprint,
                    "search_space_fingerprint": spec.search_space_fingerprint,
                    "objective_fingerprint": spec.objective_fingerprint,
                }
            )
            return self.parent_run_id
        except Exception as exc:
            try:
                import mlflow
                if mlflow.active_run() is not None:
                    mlflow.end_run(status="FAILED")
            except Exception:
                pass
            self._active = False
            self._handle(exc, "study start")
            return None

    def start_trial(self, trial_number: int, *, params: Mapping[str, Any], trial_seed: int) -> str | None:
        if not self.config.enabled or not self._active:
            return None
        try:
            import mlflow

            run = mlflow.start_run(run_name=f"trial_{int(trial_number):04d}", nested=True)
            self.trial_run_ids[int(trial_number)] = run.info.run_id
            mlflow.set_tag("e08_trial_number", str(int(trial_number)))
            mlflow.log_param("trial_seed", int(trial_seed))
            for key, value in params.items():
                mlflow.log_param(str(key), _short(value))
            return run.info.run_id
        except Exception as exc:
            self._handle(exc, "trial start")
            return None

    def log_trial_artifacts(
        self,
        trial_number: int,
        artifact_paths: Mapping[str, str],
    ) -> None:
        if not artifact_paths or not self.config.enabled or not self._active:
            return
        if int(trial_number) not in self.trial_run_ids:
            return
        try:
            import mlflow

            for key, raw_path in artifact_paths.items():
                path = Path(raw_path).resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Provider trial artifact does not exist: {path}")
                artifact_path = f"provider_artifacts/{str(key).strip()}"
                if path.is_dir():
                    mlflow.log_artifacts(str(path), artifact_path=artifact_path)
                else:
                    mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:
            self._handle(exc, f"trial {trial_number} artifact logging")

    def finish_trial(
        self,
        trial_number: int,
        *,
        state: str,
        objective_metrics: Mapping[str, float] | None = None,
        metrics: Mapping[str, float] | None = None,
        failure_message: str | None = None,
    ) -> None:
        if not self.config.enabled or not self._active:
            return
        if int(trial_number) not in self.trial_run_ids:
            return
        try:
            import mlflow

            if objective_metrics:
                mlflow.log_metrics({str(key): float(value) for key, value in objective_metrics.items()})
            if metrics:
                mlflow.log_metrics({str(key): float(value) for key, value in metrics.items()})
            mlflow.set_tag("e08_trial_state", str(state))
            if failure_message:
                mlflow.set_tag("e08_failure", _short(failure_message, 1000))
            mlflow.end_run(status="FINISHED" if state in {"COMPLETE", "PRUNED"} else "FAILED")
        except Exception as exc:
            self._handle(exc, f"trial {trial_number} finish")

    def finish_study(
        self,
        *,
        completed: int,
        pruned: int,
        failed: int,
        selected_trial_number: int | None,
        pareto_count: int,
        artifact_dir: str | Path | None,
        failed_study: bool = False,
    ) -> None:
        if not self.config.enabled or not self._active:
            return
        try:
            import mlflow

            mlflow.log_metrics(
                {
                    "completed_trial_count": float(completed),
                    "pruned_trial_count": float(pruned),
                    "failed_trial_count": float(failed),
                    "pareto_trial_count": float(pareto_count),
                }
            )
            if selected_trial_number is not None:
                mlflow.set_tag("e08_selected_trial_number", str(selected_trial_number))
            if artifact_dir is not None and Path(artifact_dir).is_dir():
                mlflow.log_artifacts(str(artifact_dir), artifact_path="e08_study")
            mlflow.end_run(status="FAILED" if failed_study else "FINISHED")
            self._active = False
        except Exception as exc:
            self._handle(exc, "study finish")
