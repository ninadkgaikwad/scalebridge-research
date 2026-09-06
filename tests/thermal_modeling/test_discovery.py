# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scalebridge.data.thermal_modeling.discovery import (
    PhaseDDiscoveryError,
    discover_phase_d_sources,
    resolve_aggregation_run,
    resolve_phase_c_child_runs,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _build_campaign(tmp_path: Path) -> tuple[Path, str, str, str]:
    campaign = tmp_path / "campaigns" / "test_campaign"
    matrix_id = "matrix_1"
    aggr_run_id = "aggr_1"
    phase_c_run_id = "phase_c_1"
    case_id = "case_1"
    aggregation_id = "smoke_l05_identity"
    weight = "equal"
    zone = "Kitchen"

    run_root = (
        campaign / "aggregation" / "cases" / case_id / "runs" / aggr_run_id
    )
    _json(
        run_root / "aggregation_manifest.json",
        {
            "case_id": case_id,
            "aggregation_id": aggregation_id,
            "weight_mode": weight,
            "strategy": "identity",
        },
    )
    _json(
        run_root / "inputs" / "aggregation_plan.json",
        {
            "aggregation_id": aggregation_id,
            "weight_mode": weight,
            "strategy": "identity",
        },
    )
    _csv(run_root / "inputs" / "zone_mapping.csv", [{"zone": zone}])
    zone_root = run_root / "zones" / zone
    _touch(zone_root / "aggregated_timeseries_wide.parquet")
    _csv(zone_root / "aggregated_timeseries_wide_preview.csv", [{"a": "1"}])

    _csv(
        campaign
        / "aggregation"
        / "matrix_runs"
        / matrix_id
        / "aggregation_matrix_outputs.csv",
        [
            {
                "case_id": case_id,
                "aggregation_id": aggregation_id,
                "weight_mode": weight,
                "aggregation_run_id": aggr_run_id,
                "run_root": str(run_root),
            }
        ],
    )

    campaign_run_root = (
        campaign
        / "heat_input_regression"
        / "campaign_runs"
        / phase_c_run_id
    )
    _json(campaign_run_root / "phase_c_campaign_plan.json", {"ok": True})
    _json(
        campaign_run_root / "phase_c_campaign_run_manifest.json",
        {
            "stages": {
                "audit": {"audit_run_id": "audit_1"},
                "features": {"feature_run_id": "features_1"},
                "splits": {"split_run_id": "splits_1"},
                "datasets": {"dataset_run_id": "datasets_1"},
                "training": {"training_run_id": "training_1"},
                "evaluation": {"evaluation_run_id": "evaluation_1"},
                "inference": {"inference_run_id": "inference_1"},
            }
        },
    )

    audit_zone = (
        campaign
        / "heat_input_regression"
        / "audit_runs"
        / "audit_1"
        / "cases"
        / case_id
        / aggregation_id
        / weight
        / zone
    )
    _csv(audit_zone / "applicable_models.csv", [{"model": "QAC"}])
    _csv(audit_zone / "unavailable_models.csv", [{"model": "QSol1"}])
    _csv(audit_zone / "heat_input_signal_catalog.csv", [{"model": "QAC"}])

    inference_zone = (
        campaign
        / "heat_input_regression"
        / "inference_runs"
        / "inference_1"
        / "cases"
        / case_id
        / aggregation_id
        / weight
        / zone
    )
    _touch(inference_zone / "predictions.parquet")
    _csv(inference_zone / "predictions_preview.csv", [{"predicted_QAC": "1"}])
    _csv(
        inference_zone / "component_prediction_summary.csv",
        [{"component": "QAC"}],
    )

    split_zone = (
        campaign
        / "heat_input_regression"
        / "split_runs"
        / "splits_1"
        / "cases"
        / case_id
        / aggregation_id
        / weight
        / zone
    )
    _touch(split_zone / "split_assignments.parquet")
    _csv(split_zone / "split_assignments_preview.csv", [{"split": "train"}])

    return campaign, matrix_id, aggr_run_id, phase_c_run_id


def test_resolve_aggregation_run_uses_matrix_lineage(tmp_path: Path) -> None:
    campaign, matrix_id, aggr_run_id, _ = _build_campaign(tmp_path)
    ref = resolve_aggregation_run(
        campaign_root=campaign,
        matrix_run_id=matrix_id,
        aggregation_run_id=aggr_run_id,
    )
    assert ref.case_id == "case_1"
    assert ref.aggregation_id == "smoke_l05_identity"
    assert ref.weight_mode == "equal"


def test_resolve_phase_c_child_runs_reads_nested_manifest(tmp_path: Path) -> None:
    campaign, _, _, phase_c_run_id = _build_campaign(tmp_path)
    refs = resolve_phase_c_child_runs(
        campaign_root=campaign,
        phase_c_campaign_run_id=phase_c_run_id,
    )
    assert refs.audit_run_id == "audit_1"
    assert refs.inference_run_id == "inference_1"
    assert refs.split_run_id == "splits_1"


def test_discover_phase_d_sources_resolves_all_required_sources(
    tmp_path: Path,
) -> None:
    campaign, matrix_id, aggr_run_id, phase_c_run_id = _build_campaign(tmp_path)
    result = discover_phase_d_sources(
        campaign_root=campaign,
        matrix_run_id=matrix_id,
        aggregation_run_id=aggr_run_id,
        phase_c_campaign_run_id=phase_c_run_id,
        aggregate_zone_id="Kitchen",
    )
    assert result.aggregation_zone.wide_parquet_path.is_file()
    assert result.phase_c_zone.predictions_parquet_path.is_file()
    assert result.phase_c_zone.applicable_models_path.is_file()
    assert result.phase_c_zone.split_assignments_parquet_path.is_file()
    assert result.to_dict()["phase_c_runs"]["inference_run_id"] == "inference_1"


def test_missing_phase_b_parquet_is_a_discovery_error(tmp_path: Path) -> None:
    campaign, matrix_id, aggr_run_id, phase_c_run_id = _build_campaign(tmp_path)
    (
        campaign
        / "aggregation"
        / "cases"
        / "case_1"
        / "runs"
        / aggr_run_id
        / "zones"
        / "Kitchen"
        / "aggregated_timeseries_wide.parquet"
    ).unlink()
    with pytest.raises(PhaseDDiscoveryError, match="Phase B wide Parquet"):
        discover_phase_d_sources(
            campaign_root=campaign,
            matrix_run_id=matrix_id,
            aggregation_run_id=aggr_run_id,
            phase_c_campaign_run_id=phase_c_run_id,
            aggregate_zone_id="Kitchen",
        )


def test_missing_child_run_id_is_rejected(tmp_path: Path) -> None:
    campaign, _, _, phase_c_run_id = _build_campaign(tmp_path)
    manifest = (
        campaign
        / "heat_input_regression"
        / "campaign_runs"
        / phase_c_run_id
        / "phase_c_campaign_run_manifest.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["stages"]["inference"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PhaseDDiscoveryError, match="inference_run_id"):
        resolve_phase_c_child_runs(
            campaign_root=campaign,
            phase_c_campaign_run_id=phase_c_run_id,
        )



def test_resolve_aggregation_zone_accepts_unique_storage_safe_directory(tmp_path: Path) -> None:
    from scalebridge.data.thermal_modeling.discovery import resolve_aggregation_zone

    campaign, matrix_id, aggr_run_id, _ = _build_campaign(tmp_path)
    run = resolve_aggregation_run(
        campaign_root=campaign,
        matrix_run_id=matrix_id,
        aggregation_run_id=aggr_run_id,
    )
    original = run.run_root / "zones" / "Kitchen"
    renamed = run.run_root / "zones" / "Kitchen_Zone"
    original.rename(renamed)

    ref = resolve_aggregation_zone(
        aggregation_run=run,
        aggregate_zone_id="Kitchen Zone",
    )
    assert ref.aggregate_zone_id == "Kitchen Zone"
    assert ref.zone_root.name == "Kitchen_Zone"


def test_resolve_aggregation_zone_rejects_ambiguous_storage_safe_directory(tmp_path: Path) -> None:
    from scalebridge.data.thermal_modeling.discovery import resolve_aggregation_zone

    campaign, matrix_id, aggr_run_id, _ = _build_campaign(tmp_path)
    run = resolve_aggregation_run(
        campaign_root=campaign,
        matrix_run_id=matrix_id,
        aggregation_run_id=aggr_run_id,
    )
    zones = run.run_root / "zones"
    (zones / "Kitchen").rename(zones / "Kitchen_Zone")
    duplicate = zones / "Kitchen-Zone"
    duplicate.mkdir(parents=True)
    _touch(duplicate / "aggregated_timeseries_wide.parquet")

    with pytest.raises(PhaseDDiscoveryError, match="ambiguous"):
        resolve_aggregation_zone(
            aggregation_run=run,
            aggregate_zone_id="Kitchen Zone",
        )


def test_discover_phase_d_sources_accepts_storage_safe_phase_c_zone_directories(tmp_path: Path) -> None:
    campaign, matrix_id, aggr_run_id, phase_c_run_id = _build_campaign(tmp_path)

    # Stage B logical/storage mismatch.
    stage_b_zones = campaign / "aggregation" / "cases" / "case_1" / "runs" / aggr_run_id / "zones"
    (stage_b_zones / "Kitchen").rename(stage_b_zones / "Kitchen_Zone")

    # Phase C audit/inference/split use the same filesystem-safe storage name
    # while the Phase D logical zone identity remains "Kitchen Zone".
    phase_c_roots = [
        campaign / "heat_input_regression" / "audit_runs" / "audit_1" / "cases" / "case_1" / "smoke_l05_identity" / "equal",
        campaign / "heat_input_regression" / "inference_runs" / "inference_1" / "cases" / "case_1" / "smoke_l05_identity" / "equal",
        campaign / "heat_input_regression" / "split_runs" / "splits_1" / "cases" / "case_1" / "smoke_l05_identity" / "equal",
    ]
    for root in phase_c_roots:
        (root / "Kitchen").rename(root / "Kitchen_Zone")

    result = discover_phase_d_sources(
        campaign_root=campaign,
        matrix_run_id=matrix_id,
        aggregation_run_id=aggr_run_id,
        phase_c_campaign_run_id=phase_c_run_id,
        aggregate_zone_id="Kitchen Zone",
    )

    assert result.aggregation_zone.aggregate_zone_id == "Kitchen Zone"
    assert result.aggregation_zone.zone_root.name == "Kitchen_Zone"
    assert result.phase_c_zone.aggregate_zone_id == "Kitchen Zone"
    assert result.phase_c_zone.metadata["audit_zone_root"].endswith("Kitchen_Zone")
    assert result.phase_c_zone.inference_zone_root.name == "Kitchen_Zone"
    assert result.phase_c_zone.split_zone_root.name == "Kitchen_Zone"


def test_discover_phase_d_sources_rejects_ambiguous_phase_c_storage_name(tmp_path: Path) -> None:
    campaign, matrix_id, aggr_run_id, phase_c_run_id = _build_campaign(tmp_path)

    stage_b_zones = campaign / "aggregation" / "cases" / "case_1" / "runs" / aggr_run_id / "zones"
    (stage_b_zones / "Kitchen").rename(stage_b_zones / "Kitchen_Zone")

    audit_parent = campaign / "heat_input_regression" / "audit_runs" / "audit_1" / "cases" / "case_1" / "smoke_l05_identity" / "equal"
    (audit_parent / "Kitchen").rename(audit_parent / "Kitchen_Zone")
    duplicate = audit_parent / "Kitchen-Zone"
    duplicate.mkdir(parents=True)

    with pytest.raises(PhaseDDiscoveryError, match="ambiguous"):
        discover_phase_d_sources(
            campaign_root=campaign,
            matrix_run_id=matrix_id,
            aggregation_run_id=aggr_run_id,
            phase_c_campaign_run_id=phase_c_run_id,
            aggregate_zone_id="Kitchen Zone",
        )
