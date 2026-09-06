from __future__ import annotations

"""Standard E0-8 artifact materialization."""

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import FrozenHyperparameters, HPOContractError, IncompatibleResumeError, StudySpec


class StudyArtifactStore:
    def __init__(self, root: str | Path, *, resume: bool = False) -> None:
        self.root = Path(root).resolve()
        if self.root.exists() and not resume and any(self.root.iterdir()):
            raise FileExistsError(f"E0-8 artifact directory already contains files: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.root / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        return path

    def write_static_contracts(self, spec: StudySpec, *, mlflow_parent_run_id: str | None) -> None:
        manifest = spec.to_dict()
        manifest["mlflow_parent_run_id"] = mlflow_parent_run_id
        self._write_json("study_manifest.json", manifest)
        self._write_json(
            "search_space_snapshot.json",
            {
                "study_id": spec.study_id,
                "fingerprint": spec.search_space_fingerprint,
                "search_space_snapshot": dict(spec.search_space_snapshot),
            },
        )
        self._write_json(
            "objective_contract.json",
            {
                "study_id": spec.study_id,
                "fingerprint": spec.objective_fingerprint,
                "objectives": [item.to_dict() for item in spec.objectives],
            },
        )
        self._write_json("data_selection_manifest.json", spec.data_selection.to_dict())

    def assert_resume_manifest(self, spec: StudySpec) -> None:
        path = self.root / "study_manifest.json"
        if not path.is_file():
            raise HPOContractError("Resume requested but E0-8 study_manifest.json is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = str(payload.get("study_fingerprint", ""))
        if stored != spec.fingerprint:
            raise IncompatibleResumeError(
                "E0-8 artifact-directory fingerprint mismatch; refusing incompatible resume"
            )

    @staticmethod
    def _trial_rows(study: Any, *, objective_names: list[str], trial_run_ids: Mapping[int, str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for trial in study.trials:
            row: dict[str, Any] = {
                "trial_number": int(trial.number),
                "state": "FAILED" if trial.state.name == "FAIL" else trial.state.name,
                "trial_seed": trial.user_attrs.get("scalebridge_e08_trial_seed"),
                "mlflow_run_id": trial.user_attrs.get("scalebridge_e08_mlflow_run_id") or trial_run_ids.get(int(trial.number)),
                "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
                "datetime_complete": None if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
                "failure_message": trial.user_attrs.get("scalebridge_e08_failure"),
            }
            if trial.duration is not None:
                row["duration_seconds"] = float(trial.duration.total_seconds())
            for key, value in trial.params.items():
                row[f"param::{key}"] = value
            values = trial.values or ()
            for index, name in enumerate(objective_names):
                row[f"objective::{name}"] = None if index >= len(values) else values[index]
            for key, value in trial.user_attrs.items():
                if key.startswith("scalebridge_e08_metric::"):
                    row[f"metric::{key.split('::', 1)[1]}"] = value
                elif key.startswith("scalebridge_e08_meta::"):
                    row[f"meta::{key.split('::', 1)[1]}"] = value
            artifact_refs = trial.user_attrs.get("scalebridge_e08_artifact_paths")
            if artifact_refs:
                row["provider_artifact_paths_json"] = json.dumps(
                    artifact_refs, sort_keys=True, ensure_ascii=True
                )
            rows.append(row)
        return rows

    @staticmethod
    def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("E0-8 trial artifacts require pandas") from exc
        if not rows:
            frame = pd.DataFrame({"trial_number": pd.Series(dtype="int64")})
        else:
            frame = pd.DataFrame(rows)
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError(
                "E0-8 standardized trials.parquet requires pyarrow/Parquet support"
            ) from exc

    def write_trial_tables(
        self,
        study: Any,
        *,
        objective_names: list[str],
        trial_run_ids: Mapping[int, str],
        pareto_trial_numbers: Iterable[int],
    ) -> None:
        rows = self._trial_rows(study, objective_names=objective_names, trial_run_ids=trial_run_ids)
        self._write_parquet(self.root / "trials.parquet", rows)
        pareto = {int(value) for value in pareto_trial_numbers}
        if len(objective_names) > 1:
            pareto_rows = [row for row in rows if int(row["trial_number"]) in pareto]
            self._write_parquet(self.root / "pareto_trials.parquet", pareto_rows)

    def write_selection_manifest(
        self,
        *,
        study_id: str,
        selected_trial_number: int | None,
        pareto_trial_numbers: Iterable[int],
        selection_policy: str,
    ) -> Path:
        return self._write_json(
            "selection_manifest.json",
            {
                "study_id": study_id,
                "selected_trial_number": selected_trial_number,
                "pareto_trial_numbers": [int(value) for value in pareto_trial_numbers],
                "selection_policy": selection_policy,
            },
        )

    def write_frozen(self, frozen: FrozenHyperparameters) -> Path:
        payload = frozen.to_dict()
        payload["content_sha256"] = frozen.content_sha256
        return self._write_json("frozen_hyperparameters.json", payload)

    def write_summary(self, payload: Mapping[str, Any]) -> Path:
        return self._write_json("study_summary.json", payload)
