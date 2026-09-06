from __future__ import annotations

from pathlib import Path

import pytest

from scalebridge.data.aggregation.campaign_definition import (
    AggregationCampaignDefinition,
)
from scalebridge.data.aggregation.campaign_runner import (
    _select_generation_runs,
    _safe_token,
)
from scalebridge.data.aggregation.models import GenerationRunRef


def _run_ref(case_id: str) -> GenerationRunRef:
    root = Path("/") / "tmp" / case_id
    return GenerationRunRef(
        case_id=case_id,
        run_id=f"run_{case_id}",
        status="completed",
        case_root=root,
        run_root=root / "runs" / f"run_{case_id}",
        manifest_path=root / "runs" / f"run_{case_id}" / "run_manifest.json",
    )


def _definition(**updates):
    payload = {
        "aggregation_campaign_id": "bgirs_phase_b_test_v1",
        "parent_generation_campaign_id": "generation_parent_v1",
        "machine_id": "labpc",
        "plan_requests": [{"strategy": "identity", "weight_mode": "equal"}],
        "mlflow_enabled": False,
    }
    payload.update(updates)
    return AggregationCampaignDefinition.model_validate(payload)


def test_case_selection_defaults_to_all_sorted_cases():
    selected = _select_generation_runs(
        [_run_ref("case_b"), _run_ref("case_a")],
        _definition(),
    )
    assert [item.case_id for item in selected] == ["case_a", "case_b"]


def test_case_selection_supports_explicit_subset():
    selected = _select_generation_runs(
        [_run_ref("case_a"), _run_ref("case_b")],
        _definition(case_ids=["case_b"]),
    )
    assert [item.case_id for item in selected] == ["case_b"]


def test_case_selection_rejects_missing_requested_case():
    with pytest.raises(ValueError, match="not available"):
        _select_generation_runs(
            [_run_ref("case_a")],
            _definition(case_ids=["case_missing"]),
        )


def test_case_limit_applies_after_case_selection():
    selected = _select_generation_runs(
        [_run_ref("case_b"), _run_ref("case_a")],
        _definition(case_limit=1),
    )
    assert [item.case_id for item in selected] == ["case_a"]


def test_safe_token_is_filesystem_friendly():
    assert _safe_token("A / B:C", max_length=20) == "A___B_C"
