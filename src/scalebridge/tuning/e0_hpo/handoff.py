from __future__ import annotations

"""Small downstream provenance handoff from E0-8 to future E.x/E0-7 exporters."""

from typing import Any

from .contracts import FrozenHyperparameters, StudySpec


def build_e07_hpo_provenance(
    spec: StudySpec,
    frozen: FrozenHyperparameters | None,
    *,
    mlflow_parent_run_id: str | None,
    pareto_trial_numbers: tuple[int, ...] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "phase_e0_e08_to_e07_provenance_v1",
        "study_id": spec.study_id,
        "study_fingerprint": spec.fingerprint,
        "method_id": spec.method_id,
        "method_family": spec.method_family,
        "provider_version": spec.provider_version,
        "data_selection_fingerprint": spec.data_selection.fingerprint,
        "search_space_fingerprint": spec.search_space_fingerprint,
        "objective_fingerprint": spec.objective_fingerprint,
        "mlflow_parent_run_id": mlflow_parent_run_id,
        "pareto_trial_numbers": [int(value) for value in pareto_trial_numbers],
    }
    if frozen is not None:
        payload.update(
            {
                "selected_trial_number": frozen.trial_number,
                "frozen_hyperparameters_sha256": frozen.content_sha256,
            }
        )
    else:
        payload["selected_trial_number"] = None
        payload["frozen_hyperparameters_sha256"] = None
    return payload
