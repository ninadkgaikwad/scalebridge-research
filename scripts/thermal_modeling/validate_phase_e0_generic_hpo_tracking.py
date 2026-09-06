from __future__ import annotations

"""Authoritative laptop qualification for ScaleBridge E0-8 generic HPO."""

import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from scalebridge.data.thermal_modeling.phase_e_adapter import load_phase_e_data_contract
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
    run_hpo_study,
    validate_train_only_selection,
)


CONTROLLED_MANIFEST_REL = Path(
    "campaigns/p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3/"
    "phase_d/cases/epcase_827ca4812c0199221d031e59/aggregation_runs/"
    "aggr_20260715_114401_0002_a8695a44_smoke_l05_identity_equal/"
    "silos/ml/ind/Dining/grp_vrin/l1_h1/mdh/manifest.json"
)


class SyntheticProvider(BaseHPOProvider):
    """Framework-only fixture.  It is not an E.1-E.4 scientific method."""

    def __init__(
        self,
        *,
        data_payload: dict,
        direction: str = "minimize",
        multi: bool = False,
        select_multi: bool = True,
        states_fixture: bool = False,
        provider_version: str = "synthetic_v1",
        artifact_root: Path | None = None,
    ) -> None:
        self.data_payload = data_payload
        self.direction = direction
        self.multi = multi
        self.select_multi = select_multi
        self.states_fixture = states_fixture
        self._provider_version = provider_version
        self.artifact_root = None if artifact_root is None else Path(artifact_root)

    @property
    def method_id(self) -> str:
        return "e08_framework_synthetic_provider"

    @property
    def method_family(self) -> str:
        return "generic"

    @property
    def provider_version(self) -> str:
        return self._provider_version

    @property
    def pruning_supported(self) -> bool:
        return self.states_fixture

    def search_space_snapshot(self):
        return {
            "demo_float": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "demo_depth": {"type": "int", "low": 2, "high": 4},
            "demo_kind": {"type": "categorical", "choices": ["A", "B"]},
        }

    def objective_specs(self):
        if self.multi:
            return (
                ObjectiveSpec("demo_accuracy_loss", "minimize"),
                ObjectiveSpec("demo_complexity", "minimize"),
            )
        return (ObjectiveSpec("demo_score", self.direction),)

    def data_selection(self, phase_d_contract):
        return HPODataSelection.create(
            phase_d_contract_id=phase_d_contract.contract_id,
            phase_d_source_manifest_sha256=phase_d_contract.source_manifest_sha256,
            source_partitions=("train",),
            selection_policy="synthetic_framework_fixture_real_phase_d_train_subset",
            selection_payload=self.data_payload,
        )

    def suggest_hyperparameters(self, suggester):
        demo_float = suggester.suggest_float("demo_float", 1e-4, 1e-2, log=True)
        demo_depth = suggester.suggest_int("demo_depth", 2, 4)
        demo_kind = suggester.suggest_categorical("demo_kind", ["A", "B"])
        return {
            "demo_float": demo_float,
            "demo_depth": demo_depth,
            "demo_kind": demo_kind,
        }

    def evaluate_trial(self, hyperparameters, context):
        if self.states_fixture and context.trial_number == 0:
            raise RecoverableTrialError("synthetic recoverable numerical failure")
        if self.states_fixture and context.trial_number == 1:
            context.prune("synthetic provider-authorized prune")

        x = float(hyperparameters["demo_float"])
        depth = int(hyperparameters["demo_depth"])
        kind = hyperparameters["demo_kind"]
        loss = abs(x - 0.0025) + 0.002 * abs(depth - 3) + (0.0005 if kind == "B" else 0.0)
        if self.multi:
            return TrialEvaluation(
                objective_values=(loss, float(depth)),
                metrics={"synthetic_loss": loss},
                metadata={"framework_fixture": True},
            )
        value = loss if self.direction == "minimize" else -loss
        artifact_paths = {}
        if self.artifact_root is not None:
            artifact = self.artifact_root / f"trial_{context.trial_number:04d}.txt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                f"framework_fixture=true\ntrial={context.trial_number}\nseed={context.trial_seed}\n",
                encoding="utf-8",
            )
            artifact_paths = {"fixture_evidence": str(artifact)}
        return TrialEvaluation(
            objective_values=(value,),
            metrics={"synthetic_loss": loss},
            metadata={"framework_fixture": True},
            artifact_paths=artifact_paths,
        )

    def select_final_multiobjective_trial(self, pareto_trials):
        if not self.select_multi:
            return None
        # Deterministic fixture policy only: lexicographic objectives then trial number.
        chosen = min(
            pareto_trials,
            key=lambda item: (tuple(item.objective_values), item.trial_number),
        )
        return chosen.trial_number


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required E0-8 package is missing: {name}") from exc


def _real_train_selection(data_path: Path) -> tuple[dict, dict]:
    frame = pd.read_parquet(data_path, columns=["timestamp", "partition", "window_id"])
    counts = {str(key): int(value) for key, value in frame["partition"].value_counts().to_dict().items()}
    train = frame.loc[frame["partition"].astype(str) == "train"].head(40).copy()
    if len(train) < 40:
        raise RuntimeError("Controlled Phase-D artifact has fewer than 40 TRAIN rows")
    if set(train["partition"].astype(str)) != {"train"}:
        raise RuntimeError("Real HPO fixture selection leaked outside Phase-D TRAIN")
    record_ids = [
        f"{index}|{row.timestamp}|{row.window_id}"
        for index, row in train.iterrows()
    ]
    payload = {
        "fixture_only": True,
        "selection_kind": "first_40_real_phase_d_train_rows",
        "selected_record_ids": record_ids,
        "inner_fit_record_ids": record_ids[:30],
        "inner_score_record_ids": record_ids[30:],
        "selected_count": 40,
        "inner_fit_count": 30,
        "inner_score_count": 10,
    }
    return payload, counts


def _assert_artifacts(root: Path, *, multi: bool, frozen: bool) -> None:
    required = {
        "study_manifest.json",
        "search_space_snapshot.json",
        "objective_contract.json",
        "data_selection_manifest.json",
        "trials.parquet",
        "study_summary.json",
        "selection_manifest.json",
    }
    if multi:
        required.add("pareto_trials.parquet")
    if frozen:
        required.add("frozen_hyperparameters.json")
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"E0-8 standardized artifacts missing: {missing}")
    trials = pd.read_parquet(root / "trials.parquet")
    if trials.empty or "trial_seed" not in trials.columns or "state" not in trials.columns:
        raise RuntimeError("E0-8 trials.parquet lacks required trial identity/state fields")


def main() -> None:
    generated = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
    if not generated:
        raise RuntimeError("SCALEBRIDGE_GENERATED_DATA_ROOT is required for E0-8 qualification")
    generated_root = Path(generated).resolve()
    manifest_path = generated_root / CONTROLLED_MANIFEST_REL
    data_path = manifest_path.with_name("data.parquet")
    if not manifest_path.is_file() or not data_path.is_file():
        raise RuntimeError(
            "Controlled real Phase-D E0-8 qualification artifact is missing: "
            f"manifest={manifest_path} parquet={data_path}"
        )

    versions = {
        "optuna": _version("optuna"),
        "mlflow": _version("mlflow"),
        "pandas": _version("pandas"),
        "pyarrow": _version("pyarrow"),
    }
    contract = load_phase_e_data_contract(manifest_path)
    selection_payload, partition_counts = _real_train_selection(data_path)

    # Direct leakage rejection against the real Phase-D contract.
    bad = HPODataSelection.create(
        phase_d_contract_id=contract.contract_id,
        phase_d_source_manifest_sha256=contract.source_manifest_sha256,
        source_partitions=("validation",),
        selection_policy="intentional_leakage_fixture",
        selection_payload={"fixture_only": True},
    )
    try:
        validate_train_only_selection(contract, bad)
    except HPOContractError:
        leakage_rejected = True
    else:
        raise RuntimeError("E0-8 failed to reject Phase-D VALIDATION leakage")

    report: dict = {
        "phase": "E0-8",
        "qualified": False,
        "versions": versions,
        "real_phase_d_manifest": str(manifest_path),
        "real_phase_d_data": str(data_path),
        "phase_d_contract_id": contract.contract_id,
        "phase_d_partition_counts": partition_counts,
        "real_train_fixture_count": 40,
        "leakage_rejected": leakage_rejected,
    }

    root = Path(tempfile.mkdtemp(prefix="scalebridge_e08_validate_"))
    try:

        # Generic safety guards: active pruning requires method authorization,
        # and resume cannot use ephemeral in-memory Optuna storage.
        try:
            run_hpo_study(
                SyntheticProvider(data_payload=selection_payload),
                contract,
                output_dir=root / "unauthorized_pruner",
                config=HPOStudyConfig(
                    study_name="e08_validate_unauthorized_pruner",
                    n_trials=1,
                    pruner_name="median",
                ),
            )
        except HPOContractError:
            unauthorized_pruner_rejected = True
        else:
            raise RuntimeError("E0-8 accepted a pruner that the provider did not authorize")

        try:
            run_hpo_study(
                SyntheticProvider(data_payload=selection_payload),
                contract,
                output_dir=root / "ephemeral_resume",
                config=HPOStudyConfig(
                    study_name="e08_validate_ephemeral_resume",
                    n_trials=1,
                    resume=True,
                    storage_url=None,
                ),
            )
        except HPOContractError:
            ephemeral_resume_rejected = True
        else:
            raise RuntimeError("E0-8 accepted resume without persistent Optuna storage")
        report["generic_safety_guards"] = {
            "unauthorized_pruner_rejected": unauthorized_pruner_rejected,
            "ephemeral_resume_rejected": ephemeral_resume_rejected,
        }

        # Single-objective minimization and maximization.
        single_results = {}
        for direction in ("minimize", "maximize"):
            outcome = run_hpo_study(
                SyntheticProvider(data_payload=selection_payload, direction=direction),
                contract,
                output_dir=root / f"single_{direction}",
                config=HPOStudyConfig(
                    study_name=f"e08_validate_single_{direction}",
                    study_seed=1234,
                    n_trials=6,
                    sampler_name="random",
                ),
            )
            _assert_artifacts(outcome.artifact_root, multi=False, frozen=True)
            if outcome.frozen is None:
                raise RuntimeError("Single-objective E0-8 study did not freeze hyperparameters")
            single_results[direction] = {
                "study_id": outcome.spec.study_id,
                "selected_trial": outcome.frozen.trial_number,
                "trial_count": len(outcome.optuna_study.trials),
                "frozen_sha256": outcome.frozen.content_sha256,
            }
        report["single_objective"] = single_results

        # Terminal state separation.
        states = run_hpo_study(
            SyntheticProvider(data_payload=selection_payload, states_fixture=True),
            contract,
            output_dir=root / "states",
            config=HPOStudyConfig(
                study_name="e08_validate_states",
                study_seed=17,
                n_trials=5,
                sampler_name="random",
                pruner_name="none",
            ),
        )
        _assert_artifacts(states.artifact_root, multi=False, frozen=True)
        state_frame = pd.read_parquet(states.artifact_root / "trials.parquet")
        observed_states = set(state_frame["state"].astype(str))
        if not {"COMPLETE", "PRUNED", "FAILED"}.issubset(observed_states):
            raise RuntimeError(f"E0-8 terminal-state distinction failed: {sorted(observed_states)}")
        report["trial_states"] = {
            "completed": states.completed_trial_count,
            "pruned": states.pruned_trial_count,
            "failed": states.failed_trial_count,
        }

        # Multi-objective Pareto + provider selection and Pareto-only mode.
        multi = run_hpo_study(
            SyntheticProvider(data_payload=selection_payload, multi=True, select_multi=True),
            contract,
            output_dir=root / "multi",
            config=HPOStudyConfig(
                study_name="e08_validate_multi",
                study_seed=99,
                n_trials=8,
                sampler_name="nsga2",
            ),
        )
        _assert_artifacts(multi.artifact_root, multi=True, frozen=True)
        if multi.frozen is None or multi.frozen.trial_number not in multi.pareto_trial_numbers:
            raise RuntimeError("Provider-selected multi-objective trial is not in Pareto set")

        pareto_only = run_hpo_study(
            SyntheticProvider(data_payload=selection_payload, multi=True, select_multi=False),
            contract,
            output_dir=root / "pareto_only",
            config=HPOStudyConfig(
                study_name="e08_validate_pareto_only",
                study_seed=100,
                n_trials=6,
                sampler_name="nsga2",
            ),
        )
        _assert_artifacts(pareto_only.artifact_root, multi=True, frozen=False)
        if pareto_only.frozen is not None or not pareto_only.pareto_trial_numbers:
            raise RuntimeError("E0-8 Pareto-only behavior failed")
        report["multi_objective"] = {
            "pareto_count": len(multi.pareto_trial_numbers),
            "selected_trial": multi.frozen.trial_number,
            "pareto_only_count": len(pareto_only.pareto_trial_numbers),
        }

        # Persistent Optuna recovery and incompatible-resume rejection.
        storage_url = f"sqlite:///{(root / 'optuna_resume.sqlite3').as_posix()}"
        resume_dir = root / "resume"
        first = run_hpo_study(
            SyntheticProvider(data_payload=selection_payload),
            contract,
            output_dir=resume_dir,
            config=HPOStudyConfig(
                study_name="e08_validate_resume",
                study_seed=808,
                n_trials=2,
                storage_url=storage_url,
                sampler_name="random",
            ),
        )
        resumed = run_hpo_study(
            SyntheticProvider(data_payload=selection_payload),
            contract,
            output_dir=resume_dir,
            config=HPOStudyConfig(
                study_name="e08_validate_resume",
                study_seed=808,
                n_trials=2,
                storage_url=storage_url,
                resume=True,
                sampler_name="random",
            ),
        )
        if len(resumed.optuna_study.trials) != 4 or first.spec.fingerprint != resumed.spec.fingerprint:
            raise RuntimeError("Persistent E0-8 Optuna resume did not recover the same study")
        first_params = [dict(trial.params) for trial in resumed.optuna_study.trials[:2]]
        resumed_params = [dict(trial.params) for trial in resumed.optuna_study.trials[2:4]]
        if resumed_params == first_params:
            raise RuntimeError("Persistent E0-8 resume replayed the initial sampler segment")
        sampler_segments = resumed.optuna_study.user_attrs.get(
            "scalebridge_e08_sampler_segments", []
        )
        if [item.get("start_trial") for item in sampler_segments] != [0, 2]:
            raise RuntimeError(f"Unexpected E0-8 sampler segment provenance: {sampler_segments}")
        try:
            run_hpo_study(
                SyntheticProvider(data_payload=selection_payload, provider_version="synthetic_v2"),
                contract,
                output_dir=resume_dir,
                config=HPOStudyConfig(
                    study_name="e08_validate_resume",
                    study_seed=808,
                    n_trials=1,
                    storage_url=storage_url,
                    resume=True,
                    sampler_name="random",
                ),
            )
        except IncompatibleResumeError:
            incompatible_rejected = True
        else:
            raise RuntimeError("E0-8 incompatible persistent resume was not rejected")
        report["resume"] = {
            "recovered_trial_count": len(resumed.optuna_study.trials),
            "fingerprint": resumed.spec.fingerprint,
            "incompatible_rejected": incompatible_rejected,
            "sampler_segments": sampler_segments,
            "initial_segment_replayed": False,
        }

        # MLflow qualification on a temporary SQLite backend, not a live server.
        tracking_uri = f"sqlite:///{(root / 'mlflow.sqlite3').as_posix()}"
        experiment_name = "ScaleBridge_E08_Validation"
        tracked = run_hpo_study(
            SyntheticProvider(
                data_payload=selection_payload,
                artifact_root=root / "provider_trial_artifacts",
            ),
            contract,
            output_dir=root / "mlflow_study",
            config=HPOStudyConfig(
                study_name="e08_validate_mlflow",
                study_seed=505,
                n_trials=3,
                sampler_name="random",
            ),
            mlflow_config=MLflowHPOConfig(
                enabled=True,
                strict=True,
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                artifact_root=root / "mlflow_artifacts",
            ),
        )
        _assert_artifacts(tracked.artifact_root, multi=False, frozen=True)
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError("E0-8 MLflow experiment missing")
        runs = client.search_runs([experiment.experiment_id])
        parents = [run for run in runs if run.data.tags.get("pipeline_stage") == "phase_e0_e08_hpo"]
        if len(parents) != 1:
            raise RuntimeError(f"Expected one E0-8 parent MLflow run, got {len(parents)}")
        parent_id = parents[0].info.run_id
        children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent_id]
        if len(children) != 3:
            raise RuntimeError(f"Expected three E0-8 nested trial runs, got {len(children)}")
        for child in children:
            artifact_infos = client.list_artifacts(
                child.info.run_id, "provider_artifacts/fixture_evidence"
            )
            if not artifact_infos:
                raise RuntimeError(
                    "E0-8 MLflow child run is missing provider trial artifact evidence"
                )
        report["mlflow"] = {
            "tracking_uri": tracking_uri,
            "parent_run_id": parent_id,
            "nested_trial_run_count": len(children),
        }

        # Downstream E0-7 provenance linkage; no final model is trained here.
        handoff = build_e07_hpo_provenance(
            tracked.spec,
            tracked.frozen,
            mlflow_parent_run_id=tracked.mlflow_parent_run_id,
            pareto_trial_numbers=tracked.pareto_trial_numbers,
        )
        if handoff["frozen_hyperparameters_sha256"] != tracked.frozen.content_sha256:
            raise RuntimeError("E0-8 -> E0-7 frozen-hyperparameter lineage hash mismatch")
        report["e07_handoff"] = handoff
    finally:
        # MLflow/SQLAlchemy may retain a SQLite engine handle on Windows even
        # after all runs are ended. Qualification evidence is already copied
        # outside this temporary root, so cleanup must be best-effort only.
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass
        gc.collect()
        shutil.rmtree(root, ignore_errors=True)

    report["qualified"] = True
    output = Path("validated_artifacts/phase_e0/e08_generic_hpo_tracking_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("E0-8 GENERIC HPO + OPTUNA + MLFLOW + FROZEN CONFIGURATION VALIDATION PASSED")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
