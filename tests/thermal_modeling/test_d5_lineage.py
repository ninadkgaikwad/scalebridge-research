# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from pathlib import Path

from scalebridge.data.thermal_modeling.lineage import (
    load_aggregation_lineage,
    resolve_all_to_one_counterpart,
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


def _build_run(
    campaign: Path,
    *,
    matrix_id: str,
    case_id: str,
    run_id: str,
    aggregation_id: str,
    zones: list[tuple[str, list[str]]],
    weight: str = "equal",
    strategy: str = "custom_groups",
    created: str = "2026-01-01T00:00:00+00:00",
) -> dict[str, str]:
    run_root = campaign / "aggregation" / "cases" / case_id / "runs" / run_id
    _json(
        run_root / "aggregation_manifest.json",
        {
            "aggregation_run_id": run_id,
            "case_id": case_id,
            "plan_aggregation_id": aggregation_id,
            "source_generation_run_id": "gen_1",
            "strategy": strategy,
            "rule_set": "legacy_v1",
            "schema_version": "0.1.0",
            "created_at_utc": created,
        },
    )
    _json(
        run_root / "inputs" / "aggregation_plan.json",
        {
            "campaign_id": campaign.name,
            "source_case_id": case_id,
            "source_generation_run_id": "gen_1",
            "aggregation_id": aggregation_id,
            "building_type": "RestaurantFastFood",
            "weather_location": "Buffalo",
            "climate_zone": "5A",
            "strategy": strategy,
            "rule_set": "legacy_v1",
            "weight_mode": weight,
            "schema_version": "0.1.0",
            "system_node_name_pattern": "DIRECT AIR INLET NODE",
            "thermal_zone_filter": {"include_when": {"Part of Total Building Area": "Yes"}},
            "aggregate_zones": [
                {"aggregate_zone_id": zone, "source_zones": sources}
                for zone, sources in zones
            ],
        },
    )
    rows = []
    for zone, sources in zones:
        for source in sources:
            rows.append(
                {
                    "campaign_id": campaign.name,
                    "case_id": case_id,
                    "run_id": "gen_1",
                    "aggregation_id": aggregation_id,
                    "strategy": strategy,
                    "rule_set": "legacy_v1",
                    "weight_mode": weight,
                    "aggregate_zone_id": zone,
                    "source_zone": source,
                }
            )
    _csv(run_root / "inputs" / "zone_mapping.csv", rows)
    return {
        "case_id": case_id,
        "source_generation_run_id": "gen_1",
        "building_type": "RestaurantFastFood",
        "climate_zone": "5A",
        "weather_location": "Buffalo",
        "aggregation_id": aggregation_id,
        "weight_mode": weight,
        "aggregation_run_id": run_id,
    }


def _phase_c(campaign: Path, matrix_id: str, phase_c_id: str, inference_id: str,
             aggregation_id: str, weight: str, zones: list[str]) -> None:
    campaign_root = (
        campaign / "heat_input_regression" / "campaign_runs" / phase_c_id
    )
    _json(
        campaign_root / "phase_c_campaign_run_manifest.json",
        {
            "campaign_id": campaign.name,
            "matrix_run_id": matrix_id,
            "stages": {"inference": {"inference_run_id": inference_id}},
        },
    )
    for zone in zones:
        _json(
            campaign
            / "heat_input_regression"
            / "inference_runs"
            / inference_id
            / "cases"
            / "case_1"
            / aggregation_id
            / weight
            / zone
            / "annual_component_predictions_manifest.json",
            {"status": "completed"},
        )


def _campaign(tmp_path: Path) -> tuple[Path, str, str]:
    campaign = tmp_path / "test_campaign"
    matrix = "matrix_1"
    phase_c = "phase_c_1"
    all_one = _build_run(
        campaign,
        matrix_id=matrix,
        case_id="case_1",
        run_id="aggr_all",
        aggregation_id="l01_all",
        zones=[("All", ["DINING", "KITCHEN"])],
    )
    identity = _build_run(
        campaign,
        matrix_id=matrix,
        case_id="case_1",
        run_id="aggr_identity",
        aggregation_id="l05_identity",
        zones=[("Dining", ["DINING"]), ("Kitchen", ["KITCHEN"])],
    )
    _csv(
        campaign
        / "aggregation"
        / "matrix_runs"
        / matrix
        / "aggregation_matrix_case_runs.csv",
        [all_one, identity],
    )
    _phase_c(campaign, matrix, phase_c, "infer_1", "l01_all", "equal", ["All"])
    return campaign, matrix, phase_c


def test_load_lineage_preserves_zone_membership(tmp_path: Path) -> None:
    campaign, matrix, _ = _campaign(tmp_path)
    lineage = load_aggregation_lineage(
        campaign_root=campaign,
        matrix_run_id=matrix,
        aggregation_run_id="aggr_identity",
    )
    assert lineage.aggregate_zone_count == 2
    assert lineage.source_zone_count == 2
    assert lineage.source_zone_ids == ("DINING", "KITCHEN")
    assert not lineage.is_all_to_one


def test_all_to_one_is_matched_self(tmp_path: Path) -> None:
    campaign, matrix, phase_c = _campaign(tmp_path)
    result = resolve_all_to_one_counterpart(
        campaign_root=campaign,
        matrix_run_id=matrix,
        aggregation_run_id="aggr_all",
        phase_c_campaign_run_id=phase_c,
    )
    assert result.status == "matched_self"
    assert result.dependent_2_available is True
    assert result.selected_aggregation_run_id == "aggr_all"


def test_identity_matches_exact_all_to_one(tmp_path: Path) -> None:
    campaign, matrix, phase_c = _campaign(tmp_path)
    result = resolve_all_to_one_counterpart(
        campaign_root=campaign,
        matrix_run_id=matrix,
        aggregation_run_id="aggr_identity",
        phase_c_campaign_run_id=phase_c,
    )
    assert result.status == "matched_exact"
    assert result.dependent_2_available is True
    assert result.selected_aggregation_run_id == "aggr_all"


def test_mismatched_weight_disables_dep2(tmp_path: Path) -> None:
    campaign, matrix, phase_c = _campaign(tmp_path)
    manifest = (
        campaign / "aggregation" / "cases" / "case_1" / "runs"
        / "aggr_identity" / "inputs" / "aggregation_plan.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["weight_mode"] = "volume"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = resolve_all_to_one_counterpart(
        campaign_root=campaign,
        matrix_run_id=matrix,
        aggregation_run_id="aggr_identity",
        phase_c_campaign_run_id=phase_c,
    )
    assert result.status == "invalid_configuration_mismatch"
    assert result.dependent_2_available is False


def test_ambiguous_counterparts_select_first_matrix_record(tmp_path: Path) -> None:
    campaign, matrix, phase_c = _campaign(tmp_path)
    second = _build_run(
        campaign,
        matrix_id=matrix,
        case_id="case_1",
        run_id="aggr_all_second",
        aggregation_id="l01_all_second",
        zones=[("All2", ["DINING", "KITCHEN"])],
        created="2026-01-02T00:00:00+00:00",
    )
    matrix_path = (
        campaign / "aggregation" / "matrix_runs" / matrix
        / "aggregation_matrix_case_runs.csv"
    )
    rows = list(csv.DictReader(matrix_path.open("r", encoding="utf-8")))
    _csv(matrix_path, rows + [second])
    _phase_c(
        campaign, matrix, phase_c, "infer_1",
        "l01_all_second", "equal", ["All2"]
    )
    result = resolve_all_to_one_counterpart(
        campaign_root=campaign,
        matrix_run_id=matrix,
        aggregation_run_id="aggr_identity",
        phase_c_campaign_run_id=phase_c,
    )
    assert result.status == "ambiguous_multiple_counterparts"
    assert result.selected_aggregation_run_id == "aggr_all"


def test_dep2_counterpart_is_structural_not_name_or_strategy_based(tmp_path: Path) -> None:
    campaign = tmp_path / "test_campaign"
    matrix = "matrix_structural"
    phase_c = "phase_c_structural"

    single = _build_run(
        campaign,
        matrix_id=matrix,
        case_id="case_1",
        run_id="aggr_user_one_zone",
        aggregation_id="user_defined_whole_building_v7",
        zones=[("MyWholeBuildingZone", ["DINING", "KITCHEN"])],
        strategy="all_thermal_zones_to_one",
    )
    current = _build_run(
        campaign,
        matrix_id=matrix,
        case_id="case_1",
        run_id="aggr_user_custom_multi",
        aggregation_id="researcher_scheme_alpha",
        zones=[("DiningAgg", ["DINING"]), ("KitchenAgg", ["KITCHEN"])],
        strategy="custom_groups",
    )
    _csv(
        campaign
        / "aggregation"
        / "matrix_runs"
        / matrix
        / "aggregation_matrix_case_runs.csv",
        [single, current],
    )
    _phase_c(
        campaign,
        matrix,
        phase_c,
        "infer_structural",
        "user_defined_whole_building_v7",
        "equal",
        ["MyWholeBuildingZone"],
    )

    result = resolve_all_to_one_counterpart(
        campaign_root=campaign,
        matrix_run_id=matrix,
        aggregation_run_id="aggr_user_custom_multi",
        phase_c_campaign_run_id=phase_c,
    )

    assert result.status == "matched_exact"
    assert result.dependent_2_available is True
    assert result.selected_aggregation_run_id == "aggr_user_one_zone"
    assert result.selected_lineage is not None
    assert result.selected_lineage.is_single_zone_full_coverage is True
