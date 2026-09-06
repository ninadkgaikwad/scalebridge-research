# -*- coding: utf-8 -*-
"""D8 production campaign orchestration for Phase D final datasets."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SUCCESS_STATUSES = {"completed", "completed_with_warnings"}


class PhaseDCampaignRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatrixAggregationRun:
    case_id: str
    aggregation_run_id: str
    aggregation_id: str
    aggregation_level: str
    aggregation_level_index: int
    aggregation_family: str
    weight_mode: str
    plan_strategy: str
    rule_set: str
    building_type: str
    climate_zone: str
    weather_location: str
    aggregate_zone_count: int | None
    source_zone_count: int | None
    matrix_record_order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "aggregation_run_id": self.aggregation_run_id,
            "aggregation_id": self.aggregation_id,
            "aggregation_level": self.aggregation_level,
            "aggregation_level_index": self.aggregation_level_index,
            "aggregation_family": self.aggregation_family,
            "weight_mode": self.weight_mode,
            "plan_strategy": self.plan_strategy,
            "rule_set": self.rule_set,
            "building_type": self.building_type,
            "climate_zone": self.climate_zone,
            "weather_location": self.weather_location,
            "aggregate_zone_count": self.aggregate_zone_count,
            "source_zone_count": self.source_zone_count,
            "matrix_record_order": self.matrix_record_order,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PhaseDCampaignRunnerError(f"Required JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PhaseDCampaignRunnerError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        keys = ["note"]
        rows = [{"note": "no rows"}]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def resolve_latest_successful_matrix_run_id(campaign_root: Path) -> str:
    root = Path(campaign_root) / "aggregation" / "matrix_runs"
    manifests = sorted(
        root.glob("*/aggregation_matrix_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in manifests:
        payload = load_json(path)
        selected = int(payload.get("selected_plan_count") or 0)
        successful = int(payload.get("successful_plan_count") or 0)
        failed = int(payload.get("failed_plan_count") or 0)
        if selected > 0 and successful == selected and failed == 0:
            return str(payload.get("matrix_run_id") or path.parent.name)
    raise PhaseDCampaignRunnerError(f"No successful aggregation matrix under {root}")


def resolve_latest_successful_phase_c_run_id(
    campaign_root: Path, *, matrix_run_id: str
) -> str:
    root = Path(campaign_root) / "heat_input_regression" / "campaign_runs"
    manifests = sorted(
        root.glob("*/phase_c_campaign_run_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in manifests:
        payload = load_json(path)
        if (
            str(payload.get("status", "")).lower() == "completed"
            and str(payload.get("matrix_run_id", "")) == matrix_run_id
        ):
            return str(payload.get("phase_c_run_id") or path.parent.name)
    raise PhaseDCampaignRunnerError(
        f"No completed Phase C campaign matching matrix_run_id={matrix_run_id} under {root}"
    )


def validate_phase_c_lineage(
    campaign_root: Path, *, matrix_run_id: str, phase_c_campaign_run_id: str
) -> dict[str, Any]:
    path = (
        Path(campaign_root)
        / "heat_input_regression"
        / "campaign_runs"
        / phase_c_campaign_run_id
        / "phase_c_campaign_run_manifest.json"
    )
    payload = load_json(path)
    if str(payload.get("status", "")).lower() != "completed":
        raise PhaseDCampaignRunnerError(f"Phase C campaign is not completed: {path}")
    if str(payload.get("matrix_run_id", "")) != matrix_run_id:
        raise PhaseDCampaignRunnerError(
            f"Phase C matrix mismatch: {payload.get('matrix_run_id')} != {matrix_run_id}"
        )
    if payload.get("campaign_id") and str(payload["campaign_id"]) != Path(campaign_root).name:
        raise PhaseDCampaignRunnerError(
            f"Phase C campaign mismatch: {payload['campaign_id']} != {Path(campaign_root).name}"
        )
    return payload


def load_matrix_aggregation_runs(
    campaign_root: Path,
    *,
    matrix_run_id: str,
    aggregation_ids: set[str] | None = None,
    weight_modes: set[str] | None = None,
    case_ids: set[str] | None = None,
) -> list[MatrixAggregationRun]:
    path = (
        Path(campaign_root)
        / "aggregation"
        / "matrix_runs"
        / matrix_run_id
        / "aggregation_matrix_case_runs.csv"
    )
    if not path.is_file():
        raise PhaseDCampaignRunnerError(f"Aggregation matrix case-run CSV not found: {path}")

    items: list[MatrixAggregationRun] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for order, row in enumerate(csv.DictReader(f), start=1):
            if str(row.get("status", "")).strip().lower() not in SUCCESS_STATUSES:
                continue
            aggregation_run_id = str(row.get("aggregation_run_id", "")).strip()
            if not aggregation_run_id:
                raise PhaseDCampaignRunnerError(
                    f"Successful matrix row {order} has no aggregation_run_id"
                )
            aggregation_id = str(row.get("aggregation_id", "")).strip()
            weight_mode = str(row.get("weight_mode", "")).strip()
            case_id = str(row.get("case_id", "")).strip()
            if aggregation_ids and aggregation_id not in aggregation_ids:
                continue
            if weight_modes and weight_mode not in weight_modes:
                continue
            if case_ids and case_id not in case_ids:
                continue

            def opt_int(value: Any) -> int | None:
                text = str(value or "").strip()
                return int(float(text)) if text else None

            items.append(
                MatrixAggregationRun(
                    case_id=case_id,
                    aggregation_run_id=aggregation_run_id,
                    aggregation_id=aggregation_id,
                    aggregation_level=str(row.get("aggregation_level", "")).strip(),
                    aggregation_level_index=opt_int(row.get("aggregation_level_index")) or 0,
                    aggregation_family=str(row.get("aggregation_family", "")).strip(),
                    weight_mode=weight_mode,
                    plan_strategy=str(row.get("plan_strategy", "")).strip(),
                    rule_set=str(row.get("rule_set", "")).strip(),
                    building_type=str(row.get("building_type", "")).strip(),
                    climate_zone=str(row.get("climate_zone", "")).strip(),
                    weather_location=str(row.get("weather_location", "")).strip(),
                    aggregate_zone_count=opt_int(row.get("aggregate_zone_count")),
                    source_zone_count=opt_int(row.get("source_zone_count")),
                    matrix_record_order=order,
                )
            )
    if not items:
        raise PhaseDCampaignRunnerError("No successful aggregation matrix rows selected")
    return items


def aggregation_output_root(
    output_root: Path, *, case_id: str, aggregation_run_id: str
) -> Path:
    return (
        Path(output_root)
        / "phase_d"
        / "cases"
        / case_id
        / "aggregation_runs"
        / aggregation_run_id
    )


def is_completed_aggregation_output(
    run_root: Path,
    *,
    expected_configuration: dict[str, Any] | None = None,
) -> bool:
    manifest_path = Path(run_root) / "aggregation_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = load_json(manifest_path)
    except Exception:
        return False
    if str(payload.get("status", "")).lower() != "completed":
        return False
    if expected_configuration is not None:
        if payload.get("runner_configuration") != expected_configuration:
            return False
    declared = int(payload.get("final_dataset_count") or 0)
    parquets = list((Path(run_root) / "silos").rglob("data.parquet"))
    manifests = list((Path(run_root) / "silos").rglob("manifest.json"))
    return declared > 0 and len(parquets) == declared and len(manifests) == declared


def build_single_run_command(
    *,
    repo_root: Path,
    campaign_root: Path,
    output_root: Path,
    matrix_run_id: str,
    phase_c_campaign_run_id: str,
    item: MatrixAggregationRun,
    phase_d_calendar_year: int,
    heat_representation: str,
    qzivr_separate: bool,
    ml_policies: list[str],
    ob_policies: list[str],
    ml_input_lag: int,
    ml_target_horizon: int,
    ml_train_fraction: float,
    ml_test_fraction: float,
    ml_validation_fraction: float,
    ml_sh_train_seasons: str,
    ml_sh_test_seasons: str,
    ml_sh_validation_seasons: str,
    sd_season_offset_days: int,
    sd_train_days: int,
    sd_test_days: int,
    sbh_train_seasons: str,
    sbh_test_seasons: str,
    ci_start_datetime: str | None,
    ci_train_days: int,
    ci_test_days: int,
    cdr_train_ranges: list[str],
    cdr_test_ranges: list[str],
    parquet_compression: str,
) -> list[str]:
    script = Path(repo_root) / "scripts" / "thermal_modeling" / "build_phase_d_final_datasets.py"
    runner_configuration = {
        "phase_d_calendar_year": phase_d_calendar_year,
        "heat_representation": heat_representation,
        "qzivr_separate": qzivr_separate,
        "ml_policies": list(ml_policies),
        "ob_policies": list(ob_policies),
        "ml_input_lag": ml_input_lag,
        "ml_target_horizon": ml_target_horizon,
        "ml_train_fraction": ml_train_fraction,
        "ml_test_fraction": ml_test_fraction,
        "ml_validation_fraction": ml_validation_fraction,
        "ml_sh_train_seasons": ml_sh_train_seasons,
        "ml_sh_test_seasons": ml_sh_test_seasons,
        "ml_sh_validation_seasons": ml_sh_validation_seasons,
        "sd_season_offset_days": sd_season_offset_days,
        "sd_train_days": sd_train_days,
        "sd_test_days": sd_test_days,
        "sbh_train_seasons": sbh_train_seasons,
        "sbh_test_seasons": sbh_test_seasons,
        "ci_start_datetime": ci_start_datetime,
        "ci_train_days": ci_train_days,
        "ci_test_days": ci_test_days,
        "cdr_train_ranges": list(cdr_train_ranges),
        "cdr_test_ranges": list(cdr_test_ranges),
        "parquet_compression": parquet_compression,
    }
    cmd = [
        sys.executable, str(script),
        "--campaign-root", str(campaign_root),
        "--matrix-run-id", matrix_run_id,
        "--aggregation-run-id", item.aggregation_run_id,
        "--phase-c-campaign-run-id", phase_c_campaign_run_id,
        "--output-root", str(output_root),
        "--phase-d-calendar-year", str(phase_d_calendar_year),
        "--heat-representation", heat_representation,
        "--ml-input-lag", str(ml_input_lag),
        "--ml-target-horizon", str(ml_target_horizon),
        "--ml-train-fraction", str(ml_train_fraction),
        "--ml-test-fraction", str(ml_test_fraction),
        "--ml-validation-fraction", str(ml_validation_fraction),
        "--ml-sh-train-seasons", ml_sh_train_seasons,
        "--ml-sh-test-seasons", ml_sh_test_seasons,
        "--ml-sh-validation-seasons", ml_sh_validation_seasons,
        "--sd-season-offset-days", str(sd_season_offset_days),
        "--sd-train-days", str(sd_train_days),
        "--sd-test-days", str(sd_test_days),
        "--sbh-train-seasons", sbh_train_seasons,
        "--sbh-test-seasons", sbh_test_seasons,
        "--ci-train-days", str(ci_train_days),
        "--ci-test-days", str(ci_test_days),
        "--parquet-compression", parquet_compression,
        "--runner-configuration-json", json.dumps(runner_configuration, sort_keys=True),
        "--allow-missing-dep2",
    ]
    for policy in ml_policies:
        cmd.extend(["--ml-policy", policy])
    for policy in ob_policies:
        cmd.extend(["--ob-policy", policy])
    if ci_start_datetime:
        cmd.extend(["--ci-start-datetime", ci_start_datetime])
    for value in cdr_train_ranges:
        cmd.extend(["--cdr-train-range", value])
    for value in cdr_test_ranges:
        cmd.extend(["--cdr-test-range", value])
    if qzivr_separate:
        cmd.append("--qzivr-separate")
    return cmd


def collect_dataset_registry(run_root: Path, item: MatrixAggregationRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    silo_root = Path(run_root) / "silos"
    for data_path in sorted(silo_root.rglob("data.parquet")):
        manifest_path = data_path.with_name("manifest.json")
        payload = load_json(manifest_path)
        temporal = payload.get("temporal_config") or {}
        heat = payload.get("heat_representation") or {}
        rows.append(
            {
                **item.to_dict(),
                "data_path": str(data_path),
                "manifest_path": str(manifest_path),
                "silo": payload.get("silo"),
                "mode": payload.get("mode"),
                "independent_zone_id": payload.get("independent_zone_id"),
                "heat_representation": heat.get("representation"),
                "heat_folder": heat.get("folder_name"),
                "input_lag": temporal.get("input_lag"),
                "target_horizon": temporal.get("target_horizon"),
                "policy_name": temporal.get("policy_name"),
                "policy_token": temporal.get("policy_token"),
                "policy_realization_id": temporal.get("policy_realization_id"),
                "policy_parameters": json.dumps(
                    temporal.get("policy_parameters") or {}, sort_keys=True
                ),
                "row_count": payload.get("row_count"),
                "included_row_count": payload.get("included_row_count"),
            }
        )
    return rows


def build_phase_d_run_id() -> str:
    return "phase_d_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def run_campaign(
    *,
    repo_root: Path,
    campaign_root: Path,
    output_root: Path,
    matrix_run_id: str,
    phase_c_campaign_run_id: str,
    phase_d_run_id: str,
    items: list[MatrixAggregationRun],
    config: dict[str, Any],
    resume: bool,
    overwrite_existing: bool,
    continue_on_error: bool,
    dry_run: bool,
) -> dict[str, Any]:
    phase_root = Path(output_root) / "phase_d"
    campaign_run_root = phase_root / "campaign_runs" / phase_d_run_id
    campaign_run_root.mkdir(parents=True, exist_ok=True)
    log_root = campaign_run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    plan = {
        "schema_version": "phase_d_d8_campaign_plan_v1",
        "created_at_utc": utc_now(),
        "campaign_id": Path(campaign_root).name,
        "campaign_root": str(Path(campaign_root).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "matrix_run_id": matrix_run_id,
        "phase_c_campaign_run_id": phase_c_campaign_run_id,
        "phase_d_run_id": phase_d_run_id,
        "selected_aggregation_run_count": len(items),
        "selected_aggregation_runs": [x.to_dict() for x in items],
        "configuration": config,
        "resume": resume,
        "overwrite_existing": overwrite_existing,
        "dry_run": dry_run,
    }
    write_json(campaign_run_root / "phase_d_campaign_plan.json", plan)

    result_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seq, item in enumerate(items, start=1):
        run_root = aggregation_output_root(
            output_root, case_id=item.case_id, aggregation_run_id=item.aggregation_run_id
        )
        base_row = {**item.to_dict(), "sequence": seq, "phase_d_run_id": phase_d_run_id}

        if dry_run:
            result_rows.append({**base_row, "status": "planned", "run_root": str(run_root)})
            continue

        if run_root.exists():
            if overwrite_existing:
                shutil.rmtree(run_root)
            elif resume:
                if is_completed_aggregation_output(
                    run_root, expected_configuration=config
                ):
                    rows = collect_dataset_registry(run_root, item)
                    dataset_rows.extend(rows)
                    result_rows.append(
                        {
                            **base_row,
                            "status": "skipped_completed",
                            "run_root": str(run_root),
                            "final_dataset_count": len(rows),
                            "return_code": 0,
                            "runtime_seconds": 0.0,
                        }
                    )
                    continue
                shutil.rmtree(run_root)
            else:
                raise PhaseDCampaignRunnerError(
                    f"Phase D output already exists for {item.aggregation_run_id}: "
                    f"{run_root}. Use --resume or --overwrite-existing."
                )

        cmd = build_single_run_command(
            repo_root=repo_root,
            campaign_root=campaign_root,
            output_root=output_root,
            matrix_run_id=matrix_run_id,
            phase_c_campaign_run_id=phase_c_campaign_run_id,
            item=item,
            **config,
        )
        log_path = log_root / f"{seq:04d}_{item.aggregation_run_id}.log"
        t0 = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        runtime = time.perf_counter() - t0

        if proc.returncode == 0 and is_completed_aggregation_output(
            run_root, expected_configuration=config
        ):
            rows = collect_dataset_registry(run_root, item)
            dataset_rows.extend(rows)
            result_rows.append(
                {
                    **base_row,
                    "status": "completed",
                    "run_root": str(run_root),
                    "final_dataset_count": len(rows),
                    "return_code": proc.returncode,
                    "runtime_seconds": runtime,
                    "log_path": str(log_path),
                }
            )
        else:
            row = {
                **base_row,
                "status": "failed",
                "run_root": str(run_root),
                "final_dataset_count": 0,
                "return_code": proc.returncode,
                "runtime_seconds": runtime,
                "log_path": str(log_path),
            }
            result_rows.append(row)
            if not continue_on_error:
                break

        write_csv(campaign_run_root / "aggregation_run_registry.csv", result_rows)
        write_csv(campaign_run_root / "dataset_registry.csv", dataset_rows)

    failed = [r for r in result_rows if r["status"] == "failed"]
    completed = [r for r in result_rows if r["status"] == "completed"]
    skipped = [r for r in result_rows if r["status"] == "skipped_completed"]

    write_csv(campaign_run_root / "aggregation_run_registry.csv", result_rows)
    write_csv(campaign_run_root / "dataset_registry.csv", dataset_rows)
    write_csv(campaign_run_root / "failures.csv", failed)

    summary = {
        "schema_version": "phase_d_d8_campaign_run_manifest_v1",
        "created_at_utc": utc_now(),
        "campaign_id": Path(campaign_root).name,
        "matrix_run_id": matrix_run_id,
        "phase_c_campaign_run_id": phase_c_campaign_run_id,
        "phase_d_run_id": phase_d_run_id,
        "status": "failed" if failed else "completed",
        "selected_aggregation_run_count": len(items),
        "completed_aggregation_run_count": len(completed),
        "skipped_completed_aggregation_run_count": len(skipped),
        "failed_aggregation_run_count": len(failed),
        "dataset_count": len(dataset_rows),
        "runtime_seconds": time.perf_counter() - started,
        "intermediate_time_series_persisted": False,
        "campaign_run_root": str(campaign_run_root),
    }
    write_json(campaign_run_root / "phase_d_campaign_run_manifest.json", summary)
    return summary
