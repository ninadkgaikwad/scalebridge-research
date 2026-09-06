from __future__ import annotations

import json
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")

from scalebridge.tuning.e0_hpo import (
    BaseHPOProvider,
    HPOContractError,
    HPODataSelection,
    HPOStudyConfig,
    IncompatibleResumeError,
    MLflowHPOConfig,
    ObjectiveSpec,
    RecoverableTrialError,
    TrialEvaluation,
    build_e07_hpo_provenance,
    create_train_only_selection,
    derive_trial_seed,
    run_hpo_study,
    validate_train_only_selection,
)
from scalebridge.tuning.e0_hpo.artifacts import StudyArtifactStore


class _FakeTemporal:
    hyperparameter_tuning_source_partitions = ("train",)

    def assert_hyperparameter_tuning_partitions(self, partitions):
        bad = set(partitions) - {"train"}
        if bad:
            raise ValueError(f"forbidden partitions: {sorted(bad)}")


class _FakePhaseDContract:
    contract_id = "phasee0_fake"
    source_manifest_sha256 = "a" * 64
    temporal = _FakeTemporal()


class _Provider(BaseHPOProvider):
    def __init__(
        self,
        *,
        direction: str = "minimize",
        provider_version: str = "v1",
        multi: bool = False,
        select_multi: bool = True,
        fail_and_prune: bool = False,
        leakage: bool = False,
    ) -> None:
        self._direction = direction
        self._provider_version = provider_version
        self._multi = multi
        self._select_multi = select_multi
        self._fail_and_prune = fail_and_prune
        self._leakage = leakage

    @property
    def method_id(self) -> str:
        return "synthetic_method"

    @property
    def method_family(self) -> str:
        return "synthetic"

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def pruning_supported(self) -> bool:
        return self._fail_and_prune

    def search_space_snapshot(self):
        return {
            "x": {"type": "float", "low": 0.0, "high": 1.0},
            "depth": {"type": "int", "low": 1, "high": 3},
            "kind": {"type": "categorical", "choices": ["A", "B"]},
        }

    def objective_specs(self):
        if self._multi:
            return (
                ObjectiveSpec("left", "minimize"),
                ObjectiveSpec("right", "minimize"),
            )
        return (ObjectiveSpec("score", self._direction),)

    def data_selection(self, phase_d_contract):
        partitions = ("validation",) if self._leakage else ("train",)
        return HPODataSelection.create(
            phase_d_contract_id=phase_d_contract.contract_id,
            phase_d_source_manifest_sha256=phase_d_contract.source_manifest_sha256,
            source_partitions=partitions,
            selection_policy="synthetic_subset",
            selection_payload={"window_ids": [1, 2, 3], "selection_seed": 7},
        )

    def suggest_hyperparameters(self, suggester):
        x = suggester.suggest_float("x", 0.0, 1.0)
        depth = suggester.suggest_int("depth", 1, 3)
        kind = suggester.suggest_categorical("kind", ["A", "B"])
        return {"x": x, "depth": depth, "kind": kind}

    def evaluate_trial(self, hyperparameters, context):
        if self._fail_and_prune and context.trial_number == 0:
            raise RecoverableTrialError("synthetic numerical failure")
        if self._fail_and_prune and context.trial_number == 1:
            context.prune("synthetic prune")

        x = float(hyperparameters["x"])
        depth = int(hyperparameters["depth"])
        kind_penalty = 0.0 if hyperparameters["kind"] == "A" else 0.01
        base = (x - 0.35) ** 2 + 0.01 * abs(depth - 2) + kind_penalty
        if self._multi:
            return TrialEvaluation(
                objective_values=(x, 1.0 - x),
                metrics={"base": base},
                metadata={"source": "synthetic"},
            )
        score = base if self._direction == "minimize" else -base
        return TrialEvaluation(
            objective_values=(score,),
            metrics={"base": base},
            metadata={"source": "synthetic"},
        )

    def select_final_multiobjective_trial(self, pareto_trials):
        if not self._select_multi:
            return None
        return min(item.trial_number for item in pareto_trials)


def _fake_parquet(monkeypatch):
    def _writer(path: Path, rows):
        Path(path).write_text(json.dumps(rows, sort_keys=True, default=str), encoding="utf-8")

    monkeypatch.setattr(StudyArtifactStore, "_write_parquet", staticmethod(_writer))


def test_train_only_selection_and_leakage_rejection():
    contract = _FakePhaseDContract()
    selection = create_train_only_selection(
        contract,
        selection_policy="all_train",
        selection_payload={"mode": "all_train"},
    )
    validate_train_only_selection(contract, selection)
    assert selection.source_partitions == ("train",)

    bad = HPODataSelection.create(
        phase_d_contract_id=contract.contract_id,
        phase_d_source_manifest_sha256=contract.source_manifest_sha256,
        source_partitions=("validation",),
        selection_policy="bad",
        selection_payload={"mode": "bad"},
    )
    with pytest.raises(HPOContractError):
        validate_train_only_selection(contract, bad)


def test_trial_seed_is_deterministic_and_trial_specific():
    a = derive_trial_seed(1234, 7, "e08_example")
    b = derive_trial_seed(1234, 7, "e08_example")
    c = derive_trial_seed(1234, 8, "e08_example")
    assert a == b
    assert a != c


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_single_objective_study_freezes_best(tmp_path, monkeypatch, direction):
    _fake_parquet(monkeypatch)
    outcome = run_hpo_study(
        _Provider(direction=direction),
        _FakePhaseDContract(),
        output_dir=tmp_path / direction,
        config=HPOStudyConfig(
            study_name=f"single_{direction}",
            study_seed=2026,
            n_trials=8,
            sampler_name="random",
        ),
    )
    assert outcome.completed_trial_count == 8
    assert outcome.frozen is not None
    assert outcome.frozen.trial_number == outcome.optuna_study.best_trial.number
    assert (outcome.artifact_root / "frozen_hyperparameters.json").is_file()
    assert (outcome.artifact_root / "trials.parquet").is_file()


def test_complete_pruned_failed_remain_distinct(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    outcome = run_hpo_study(
        _Provider(fail_and_prune=True),
        _FakePhaseDContract(),
        output_dir=tmp_path / "states",
        config=HPOStudyConfig(
            study_name="states",
            study_seed=7,
            n_trials=5,
            sampler_name="random",
            pruner_name="none",
        ),
    )
    assert outcome.failed_trial_count == 1
    assert outcome.pruned_trial_count == 1
    assert outcome.completed_trial_count == 3
    states = [trial.state.name for trial in outcome.optuna_study.trials]
    assert "FAIL" in states and "PRUNED" in states and "COMPLETE" in states


def test_multiobjective_pareto_and_provider_selection(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    outcome = run_hpo_study(
        _Provider(multi=True, select_multi=True),
        _FakePhaseDContract(),
        output_dir=tmp_path / "multi",
        config=HPOStudyConfig(
            study_name="multi",
            study_seed=99,
            n_trials=7,
            sampler_name="nsga2",
        ),
    )
    assert len(outcome.pareto_trial_numbers) >= 1
    assert outcome.frozen is not None
    assert outcome.frozen.trial_number in outcome.pareto_trial_numbers
    assert (outcome.artifact_root / "pareto_trials.parquet").is_file()


def test_multiobjective_without_provider_selection_preserves_pareto_only(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    outcome = run_hpo_study(
        _Provider(multi=True, select_multi=False),
        _FakePhaseDContract(),
        output_dir=tmp_path / "pareto_only",
        config=HPOStudyConfig(
            study_name="pareto_only",
            study_seed=99,
            n_trials=5,
            sampler_name="nsga2",
        ),
    )
    assert outcome.pareto_trial_numbers
    assert outcome.frozen is None
    assert not (outcome.artifact_root / "frozen_hyperparameters.json").exists()


def test_persistent_optuna_resume_and_incompatible_resume_rejection(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    output = tmp_path / "resume_artifacts"
    first = run_hpo_study(
        _Provider(provider_version="v1"),
        _FakePhaseDContract(),
        output_dir=output,
        config=HPOStudyConfig(
            study_name="resume_demo",
            study_seed=44,
            n_trials=2,
            storage_url=storage,
            resume=False,
            sampler_name="random",
        ),
    )
    second = run_hpo_study(
        _Provider(provider_version="v1"),
        _FakePhaseDContract(),
        output_dir=output,
        config=HPOStudyConfig(
            study_name="resume_demo",
            study_seed=44,
            n_trials=2,
            storage_url=storage,
            resume=True,
            sampler_name="random",
        ),
    )
    assert first.spec.fingerprint == second.spec.fingerprint
    assert len(second.optuna_study.trials) == 4
    first_params = [dict(trial.params) for trial in second.optuna_study.trials[:2]]
    resumed_params = [dict(trial.params) for trial in second.optuna_study.trials[2:4]]
    assert resumed_params != first_params
    segments = second.optuna_study.user_attrs["scalebridge_e08_sampler_segments"]
    assert [item["start_trial"] for item in segments] == [0, 2]
    assert segments[0]["seed"] != segments[1]["seed"]

    with pytest.raises(IncompatibleResumeError):
        run_hpo_study(
            _Provider(provider_version="v2"),
            _FakePhaseDContract(),
            output_dir=output,
            config=HPOStudyConfig(
                study_name="resume_demo",
                study_seed=44,
                n_trials=1,
                storage_url=storage,
                resume=True,
                sampler_name="random",
            ),
        )



def test_resume_requires_existing_persistent_study(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    storage = f"sqlite:///{(tmp_path / 'missing.db').as_posix()}"
    # The runner may reject first on the missing standardized resume manifest;
    # both that and a missing persistent study are E0-8 contract failures.
    with pytest.raises(HPOContractError):
        run_hpo_study(
            _Provider(provider_version="v1"),
            _FakePhaseDContract(),
            output_dir=tmp_path / "missing_resume",
            config=HPOStudyConfig(
                study_name="does_not_exist",
                study_seed=44,
                n_trials=1,
                storage_url=storage,
                resume=True,
                sampler_name="random",
            ),
        )

def test_e07_handoff_contains_study_and_frozen_hash(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    outcome = run_hpo_study(
        _Provider(),
        _FakePhaseDContract(),
        output_dir=tmp_path / "handoff",
        config=HPOStudyConfig(study_name="handoff", n_trials=3, sampler_name="random"),
    )
    record = build_e07_hpo_provenance(
        outcome.spec,
        outcome.frozen,
        mlflow_parent_run_id=outcome.mlflow_parent_run_id,
        pareto_trial_numbers=outcome.pareto_trial_numbers,
    )
    assert record["study_id"] == outcome.spec.study_id
    assert record["selected_trial_number"] == outcome.frozen.trial_number
    assert record["frozen_hyperparameters_sha256"] == outcome.frozen.content_sha256


def test_real_parquet_artifact_contract_when_pyarrow_available(tmp_path):
    pytest.importorskip("pyarrow")
    outcome = run_hpo_study(
        _Provider(),
        _FakePhaseDContract(),
        output_dir=tmp_path / "real_parquet",
        config=HPOStudyConfig(study_name="real_parquet", n_trials=3, sampler_name="random"),
    )
    import pandas as pd

    frame = pd.read_parquet(outcome.artifact_root / "trials.parquet")
    assert len(frame) == 3
    assert {"trial_number", "state", "trial_seed"}.issubset(frame.columns)


def test_mlflow_nested_study_trial_linkage_when_mlflow_available(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("pyarrow")
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifact_root = tmp_path / "mlartifacts"
    experiment_name = "ScaleBridge_E08_Pytest"

    outcome = run_hpo_study(
        _Provider(),
        _FakePhaseDContract(),
        output_dir=tmp_path / "mlflow_hpo",
        config=HPOStudyConfig(study_name="mlflow_hpo", n_trials=2, sampler_name="random"),
        mlflow_config=MLflowHPOConfig(
            enabled=True,
            strict=True,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
        ),
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 3
    parent = [run for run in runs if run.data.tags.get("pipeline_stage") == "phase_e0_e08_hpo"]
    assert len(parent) == 1
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent[0].info.run_id]
    assert len(children) == 2
    assert outcome.mlflow_parent_run_id == parent[0].info.run_id


def test_resume_requires_persistent_optuna_storage(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    with pytest.raises(HPOContractError, match="persistent Optuna storage_url"):
        run_hpo_study(
            _Provider(),
            _FakePhaseDContract(),
            output_dir=tmp_path / "bad_resume",
            config=HPOStudyConfig(
                study_name="bad_resume",
                n_trials=1,
                resume=True,
                storage_url=None,
            ),
        )


def test_provider_must_authorize_configured_pruning(tmp_path, monkeypatch):
    _fake_parquet(monkeypatch)
    with pytest.raises(HPOContractError, match="does not authorize pruning"):
        run_hpo_study(
            _Provider(fail_and_prune=False),
            _FakePhaseDContract(),
            output_dir=tmp_path / "pruner_reject",
            config=HPOStudyConfig(
                study_name="pruner_reject",
                n_trials=1,
                sampler_name="random",
                pruner_name="median",
            ),
        )


class _ArtifactProvider(_Provider):
    def __init__(self, artifact_root: Path) -> None:
        super().__init__()
        self._artifact_root = Path(artifact_root)

    def evaluate_trial(self, hyperparameters, context):
        result = super().evaluate_trial(hyperparameters, context)
        artifact = self._artifact_root / f"trial_{context.trial_number}.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"seed={context.trial_seed}\n", encoding="utf-8")
        return TrialEvaluation(
            objective_values=result.objective_values,
            metrics=result.metrics,
            metadata=result.metadata,
            artifact_paths={"evidence": str(artifact)},
        )


def test_provider_trial_artifact_references_are_persisted(tmp_path, monkeypatch):
    captured = {}

    def _capture(path: Path, rows):
        captured[Path(path).name] = list(rows)
        Path(path).write_text(json.dumps(rows, sort_keys=True, default=str), encoding="utf-8")

    monkeypatch.setattr(StudyArtifactStore, "_write_parquet", staticmethod(_capture))
    outcome = run_hpo_study(
        _ArtifactProvider(tmp_path / "provider_artifacts"),
        _FakePhaseDContract(),
        output_dir=tmp_path / "artifact_study",
        config=HPOStudyConfig(study_name="artifact_study", n_trials=2, sampler_name="random"),
    )
    assert outcome.completed_trial_count == 2
    rows = captured["trials.parquet"]
    assert all("provider_artifact_paths_json" in row for row in rows)
    refs = json.loads(rows[0]["provider_artifact_paths_json"])
    assert Path(refs["evidence"]).is_file()
