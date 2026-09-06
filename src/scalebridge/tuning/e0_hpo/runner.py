from __future__ import annotations

"""Generic E0-8 Optuna orchestration, tracking, selection, and freezing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import StudyArtifactStore
from .contracts import (
    CompletedTrialView,
    FrozenHyperparameters,
    HPOContractError,
    ObjectiveSpec,
    RecoverableTrialError,
    StudySpec,
    TrialEvaluation,
)
from .data_policy import validate_train_only_selection
from .mlflow_tracking import HPOStudyTracker, MLflowHPOConfig
from .optuna_backend import OptunaStudyConfig, create_or_resume_study
from .provider import BaseHPOProvider, TrialContext, TrialSuggester
from .seeding import derive_trial_seed


@dataclass(frozen=True)
class HPOStudyConfig:
    study_name: str
    study_seed: int = 0
    n_trials: int = 50
    timeout_seconds: float | None = None
    storage_url: str | None = None
    resume: bool = False
    sampler_name: str = "auto"
    pruner_name: str = "none"


@dataclass
class StudyOutcome:
    spec: StudySpec
    optuna_study: Any
    artifact_root: Path
    completed_trial_count: int
    pruned_trial_count: int
    failed_trial_count: int
    pareto_trial_numbers: tuple[int, ...]
    frozen: FrozenHyperparameters | None
    mlflow_parent_run_id: str | None


def _objective_names(spec: StudySpec) -> list[str]:
    return [item.name for item in spec.objectives]


def _state_counts(study: Any) -> tuple[int, int, int]:
    counts = {"COMPLETE": 0, "PRUNED": 0, "FAIL": 0}
    for trial in study.trials:
        counts[trial.state.name] = counts.get(trial.state.name, 0) + 1
    return counts.get("COMPLETE", 0), counts.get("PRUNED", 0), counts.get("FAIL", 0)


def _pareto_trials(study: Any, objective_count: int) -> list[Any]:
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if objective_count == 1:
        if not complete:
            return []
        return [study.best_trial]
    return list(study.best_trials)


def _completed_view(trial: Any) -> CompletedTrialView:
    values = tuple(float(value) for value in (trial.values or ()))
    seed = int(trial.user_attrs["scalebridge_e08_trial_seed"])
    return CompletedTrialView(
        trial_number=int(trial.number),
        params=dict(trial.params),
        objective_values=values,
        trial_seed=seed,
    )


def _select_trial(provider: BaseHPOProvider, study: Any, spec: StudySpec) -> tuple[Any | None, list[Any], str]:
    pareto = _pareto_trials(study, len(spec.objectives))
    if not pareto:
        return None, [], "no_completed_trial"
    if len(spec.objectives) == 1:
        return pareto[0], pareto, "single_objective_optuna_best"

    views = [_completed_view(trial) for trial in pareto]
    selected_number = provider.select_final_multiobjective_trial(views)
    if selected_number is None:
        return None, pareto, "pareto_preserved_no_provider_selection"
    matches = [trial for trial in pareto if int(trial.number) == int(selected_number)]
    if len(matches) != 1:
        raise HPOContractError(
            "Provider multi-objective selection must choose exactly one trial from the Pareto set"
        )
    return matches[0], pareto, "provider_multiobjective_selection"


def _freeze(
    selected: Any,
    spec: StudySpec,
    *,
    selection_policy: str,
    tracker: HPOStudyTracker,
    study_run_ids: tuple[str, ...],
) -> FrozenHyperparameters:
    values = tuple(float(value) for value in (selected.values or ()))
    objective_values = {
        item.name: values[index]
        for index, item in enumerate(spec.objectives)
    }
    return FrozenHyperparameters(
        study_id=spec.study_id,
        study_fingerprint=spec.fingerprint,
        method_id=spec.method_id,
        method_family=spec.method_family,
        provider_version=spec.provider_version,
        trial_number=int(selected.number),
        hyperparameters=dict(selected.params),
        objective_values=objective_values,
        data_selection_fingerprint=spec.data_selection.fingerprint,
        search_space_fingerprint=spec.search_space_fingerprint,
        objective_fingerprint=spec.objective_fingerprint,
        selection_policy=selection_policy,
        provenance={
            "optuna_study_name": spec.study_name,
            "mlflow_study_run_ids": list(
                study_run_ids
            ),
            "mlflow_selected_trial_parent_run_id": selected.user_attrs.get(
                "scalebridge_e08_mlflow_parent_run_id"
            ),
            "mlflow_trial_run_id": selected.user_attrs.get(
                "scalebridge_e08_mlflow_run_id"
            ) or tracker.trial_run_ids.get(int(selected.number)),
        },
    )


def run_hpo_study(
    provider: BaseHPOProvider,
    phase_d_contract: Any,
    *,
    output_dir: str | Path,
    config: HPOStudyConfig,
    mlflow_config: MLflowHPOConfig | None = None,
) -> StudyOutcome:
    """Run one generic E0-8 study and materialize the standardized artifacts."""
    objectives = tuple(provider.objective_specs())
    provider.validate_hpo_configuration(
        sampler_name=str(config.sampler_name),
        pruner_name=str(config.pruner_name),
        objective_count=len(objectives),
    )
    selection = provider.data_selection(phase_d_contract)
    validate_train_only_selection(phase_d_contract, selection)

    spec = StudySpec(
        study_name=str(config.study_name),
        method_id=str(provider.method_id),
        method_family=str(provider.method_family),
        provider_version=str(provider.provider_version),
        search_space_snapshot=dict(provider.search_space_snapshot()),
        objectives=objectives,
        data_selection=selection,
        study_seed=int(config.study_seed),
        sampler_name=str(config.sampler_name),
        pruner_name=str(config.pruner_name),
    )

    # Validate execution/resume semantics before touching the artifact directory.
    optuna_config = OptunaStudyConfig(
        n_trials=int(config.n_trials),
        timeout_seconds=config.timeout_seconds,
        storage_url=config.storage_url,
        resume=bool(config.resume),
        sampler_name=str(config.sampler_name),
        pruner_name=str(config.pruner_name),
    )

    store = StudyArtifactStore(output_dir, resume=bool(config.resume))
    if config.resume:
        store.assert_resume_manifest(spec)

    study = create_or_resume_study(spec, optuna_config)

    tracker = HPOStudyTracker(mlflow_config or MLflowHPOConfig(enabled=False))
    parent_run_id = tracker.start_study(spec)
    stored_run_ids = [str(value) for value in study.user_attrs.get(
        "scalebridge_e08_mlflow_study_run_ids", []
    )]
    if parent_run_id is not None and parent_run_id not in stored_run_ids:
        stored_run_ids.append(parent_run_id)
        study.set_user_attr("scalebridge_e08_mlflow_study_run_ids", stored_run_ids)
    study_run_ids = tuple(stored_run_ids)
    store.write_static_contracts(spec, mlflow_parent_run_id=parent_run_id)

    objective_names = _objective_names(spec)

    def objective(optuna_trial: Any):
        trial_seed = derive_trial_seed(spec.study_seed, int(optuna_trial.number), spec.study_id)
        optuna_trial.set_user_attr("scalebridge_e08_trial_seed", int(trial_seed))
        suggester = TrialSuggester(optuna_trial)
        params = dict(provider.suggest_hyperparameters(suggester))
        if dict(optuna_trial.params) != params:
            raise HPOContractError(
                "Provider must return exactly the active parameters suggested through TrialSuggester"
            )
        trial_mlflow_run_id = tracker.start_trial(
            int(optuna_trial.number), params=params, trial_seed=trial_seed
        )
        if trial_mlflow_run_id is not None:
            optuna_trial.set_user_attr("scalebridge_e08_mlflow_run_id", trial_mlflow_run_id)
            if parent_run_id is not None:
                optuna_trial.set_user_attr(
                    "scalebridge_e08_mlflow_parent_run_id", parent_run_id
                )
        context = TrialContext(
            trial_number=int(optuna_trial.number),
            trial_seed=trial_seed,
            data_selection=selection,
            pruning_allowed=bool(provider.pruning_supported),
            objective_count=len(spec.objectives),
            _optuna_trial=optuna_trial,
        )
        try:
            evaluation = provider.evaluate_trial(params, context)
            if not isinstance(evaluation, TrialEvaluation):
                raise HPOContractError("Provider evaluate_trial must return TrialEvaluation")
            if len(evaluation.objective_values) != len(spec.objectives):
                raise RecoverableTrialError(
                    "Trial objective value count does not match objective contract"
                )
            for key, value in evaluation.metrics.items():
                optuna_trial.set_user_attr(f"scalebridge_e08_metric::{key}", float(value))
            for key, value in evaluation.metadata.items():
                optuna_trial.set_user_attr(f"scalebridge_e08_meta::{key}", value)
            if evaluation.artifact_paths:
                optuna_trial.set_user_attr(
                    "scalebridge_e08_artifact_paths", dict(evaluation.artifact_paths)
                )
                tracker.log_trial_artifacts(
                    int(optuna_trial.number), evaluation.artifact_paths
                )
            objective_metrics = {
                name: evaluation.objective_values[index]
                for index, name in enumerate(objective_names)
            }
            tracker.finish_trial(
                int(optuna_trial.number),
                state="COMPLETE",
                objective_metrics=objective_metrics,
                metrics=evaluation.metrics,
            )
            if len(evaluation.objective_values) == 1:
                return evaluation.objective_values[0]
            return tuple(evaluation.objective_values)
        except Exception as exc:
            try:
                import optuna
                is_pruned = isinstance(exc, optuna.TrialPruned)
            except Exception:  # pragma: no cover
                is_pruned = False
            if is_pruned:
                tracker.finish_trial(
                    int(optuna_trial.number), state="PRUNED", failure_message=str(exc)
                )
                raise
            if isinstance(exc, RecoverableTrialError):
                optuna_trial.set_user_attr("scalebridge_e08_failure", str(exc))
                tracker.finish_trial(
                    int(optuna_trial.number), state="FAILED", failure_message=str(exc)
                )
                raise
            tracker.finish_trial(
                int(optuna_trial.number), state="FAILED", failure_message=f"{type(exc).__name__}: {exc}"
            )
            raise

    study_failed = False
    try:
        study.optimize(
            objective,
            n_trials=int(optuna_config.n_trials),
            timeout=optuna_config.timeout_seconds,
            catch=(RecoverableTrialError,),
        )
        selected, pareto, selection_policy = _select_trial(provider, study, spec)
        pareto_numbers = tuple(sorted(int(trial.number) for trial in pareto))
        frozen = None if selected is None else _freeze(
            selected, spec, selection_policy=selection_policy, tracker=tracker, study_run_ids=study_run_ids
        )

        store.write_trial_tables(
            study,
            objective_names=objective_names,
            trial_run_ids=tracker.trial_run_ids,
            pareto_trial_numbers=pareto_numbers,
        )
        store.write_selection_manifest(
            study_id=spec.study_id,
            selected_trial_number=None if selected is None else int(selected.number),
            pareto_trial_numbers=pareto_numbers,
            selection_policy=selection_policy,
        )
        if frozen is not None:
            store.write_frozen(frozen)

        complete_count, pruned_count, failed_count = _state_counts(study)
        store.write_summary(
            {
                "study_id": spec.study_id,
                "study_fingerprint": spec.fingerprint,
                "trial_count": len(study.trials),
                "completed_trial_count": complete_count,
                "pruned_trial_count": pruned_count,
                "failed_trial_count": failed_count,
                "pareto_trial_numbers": list(pareto_numbers),
                "selected_trial_number": None if selected is None else int(selected.number),
                "selection_policy": selection_policy,
                "status": "COMPLETE" if complete_count > 0 else "NO_COMPLETED_TRIALS",
            }
        )
        tracker.finish_study(
            completed=complete_count,
            pruned=pruned_count,
            failed=failed_count,
            selected_trial_number=None if selected is None else int(selected.number),
            pareto_count=len(pareto_numbers),
            artifact_dir=store.root,
            failed_study=False,
        )
        return StudyOutcome(
            spec=spec,
            optuna_study=study,
            artifact_root=store.root,
            completed_trial_count=complete_count,
            pruned_trial_count=pruned_count,
            failed_trial_count=failed_count,
            pareto_trial_numbers=pareto_numbers,
            frozen=frozen,
            mlflow_parent_run_id=parent_run_id,
        )
    except Exception:
        study_failed = True
        complete_count, pruned_count, failed_count = _state_counts(study)
        tracker.finish_study(
            completed=complete_count,
            pruned=pruned_count,
            failed=failed_count,
            selected_trial_number=None,
            pareto_count=0,
            artifact_dir=store.root,
            failed_study=True,
        )
        raise
