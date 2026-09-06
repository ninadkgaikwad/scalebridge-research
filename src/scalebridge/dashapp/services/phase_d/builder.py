"""Campaign Builder services for Phase D, backed by the general runner CLI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from scalebridge.data.thermal_modeling.policies import parse_datetime_range

from scalebridge.dashapp.schemas.pipeline.phase_d import (
    PhaseDCampaignDefinition,
    PhaseDRunnerConfig,
)

from .upstream_phase_c import resolve_phase_c_context, selected_aggregation_count


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "scripts").is_dir():
            return parent
    # Directed-input bundles used by tests may omit pyproject.toml; retain the
    # same repository-relative contract without changing production behavior.
    for parent in here.parents:
        if (parent / "scripts" / "thermal_modeling" / "run_phase_d_campaign.py").is_file():
            return parent
    raise FileNotFoundError("Could not resolve scalebridge-research repository root")


def runner_script() -> Path:
    path = _repo_root() / "scripts" / "thermal_modeling" / "run_phase_d_campaign.py"
    if not path.is_file():
        raise FileNotFoundError(f"Phase D runner not found: {path}")
    return path


def _tuple(values: Any) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in (values or []) if str(value).strip())


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in (values or []))


def build_definition(
    *,
    phase_d_campaign_id: str,
    parent_phase_c_run_key: str,
    machine_id: str,
    display_name: str | None = None,
    notes: str | None = None,
    values: dict[str, Any] | None = None,
) -> PhaseDCampaignDefinition:
    phase_d_campaign_id = str(phase_d_campaign_id or "").strip()
    parent_phase_c_run_key = str(parent_phase_c_run_key or "").strip()
    machine_id = str(machine_id or "").strip()
    if not parent_phase_c_run_key:
        raise ValueError("Select a completed Phase C campaign run")
    if not machine_id:
        raise ValueError("Machine ID is required")

    context = resolve_phase_c_context(parent_phase_c_run_key)
    raw = dict(values or {})
    runner = PhaseDRunnerConfig(
        campaign_root=str(context["campaign_root"]),
        output_root=str(raw.get("output_root") or context["campaign_root"]),
        matrix_run_id=str(context["matrix_run_id"]),
        phase_c_campaign_run_id=str(context["phase_c_campaign_run_id"]),
        aggregation_ids=_tuple(raw.get("aggregation_ids")),
        weight_modes=_tuple(raw.get("weight_modes")),
        case_ids=_tuple(raw.get("case_ids")),
        max_aggregation_runs=(
            int(raw["max_aggregation_runs"])
            if raw.get("max_aggregation_runs") not in (None, "")
            else None
        ),
        phase_d_calendar_year=int(raw.get("phase_d_calendar_year") or 2001),
        heat_representation=str(raw.get("heat_representation") or "grouped"),
        qzivr_separate=bool(raw.get("qzivr_separate")),
        ml_policies=_tuple(raw.get("ml_policies") or ["monthly_distributed_holdout"]),
        ml_input_lags=_int_tuple(raw.get("ml_input_lags") or [12]),
        ml_target_horizons=_int_tuple(raw.get("ml_target_horizons") or [6]),
        ml_train_fraction=float(raw.get("ml_train_fraction", 0.70)),
        ml_test_fraction=float(raw.get("ml_test_fraction", 0.15)),
        ml_validation_fraction=float(raw.get("ml_validation_fraction", 0.15)),
        ml_sh_train_seasons=_tuple(raw.get("ml_sh_train_seasons") or ["winter", "spring"]),
        ml_sh_test_seasons=_tuple(raw.get("ml_sh_test_seasons") or ["summer"]),
        ml_sh_validation_seasons=_tuple(raw.get("ml_sh_validation_seasons") or ["fall"]),
        ob_policies=_tuple(raw.get("ob_policies") or ["seasonal_distributed"]),
        sd_season_offset_days=int(raw.get("sd_season_offset_days") or 0),
        sd_train_days=int(raw.get("sd_train_days") or 21),
        sd_test_days=int(raw.get("sd_test_days") or 7),
        sbh_train_seasons=_tuple(raw.get("sbh_train_seasons") or ["winter", "spring", "fall"]),
        sbh_test_seasons=_tuple(raw.get("sbh_test_seasons") or ["summer"]),
        ci_start_datetime=str(raw.get("ci_start_datetime") or "").strip() or None,
        ci_train_days=int(raw.get("ci_train_days") or 21),
        ci_test_days=int(raw.get("ci_test_days") or 7),
        cdr_train_ranges=_tuple(raw.get("cdr_train_ranges")),
        cdr_test_ranges=_tuple(raw.get("cdr_test_ranges")),
        parquet_compression=str(raw.get("parquet_compression") or "zstd"),
        mlflow_enabled=bool(raw.get("mlflow_enabled")),
        mlflow_experiment_name=str(raw.get("mlflow_experiment_name") or "").strip() or None,
        mlflow_run_name=str(raw.get("mlflow_run_name") or "").strip() or None,
        mlflow_strict=bool(raw.get("mlflow_strict")),
    )

    # Validate the user-entered datetime syntax at Campaign Builder time using
    # the same scientific parsing semantics as Phase D. Canonical-axis membership
    # remains an execution-time scientific check because it depends on resolved data.
    if "contiguous_identification" in runner.ob_policies and runner.ci_start_datetime:
        try:
            pd.Timestamp(runner.ci_start_datetime)
        except Exception as exc:
            raise ValueError(
                "Contiguous Identification Start Datetime must be parseable, e.g. "
                "2001-01-01T00:05:00"
            ) from exc
    if "custom_datetime_ranges" in runner.ob_policies:
        parsed = []
        for partition, ranges in (("train", runner.cdr_train_ranges), ("test", runner.cdr_test_ranges)):
            for value in ranges:
                try:
                    start, end = parse_datetime_range(value)
                except Exception as exc:
                    raise ValueError(
                        f"Custom Datetime {partition} range must use START/END syntax, e.g. "
                        "2001-01-01T00:05:00/2001-01-08T00:05:00"
                    ) from exc
                parsed.append((partition, start, end))
        ordered = sorted(parsed, key=lambda item: (item[1], item[2]))
        for previous, current in zip(ordered, ordered[1:]):
            if current[1] < previous[2]:
                raise ValueError("Custom Datetime train/test ranges cannot overlap")

    matched = selected_aggregation_count(
        context,
        case_ids=runner.case_ids,
        aggregation_ids=runner.aggregation_ids,
        weight_modes=runner.weight_modes,
        max_aggregation_runs=runner.max_aggregation_runs,
    )
    if matched <= 0:
        raise ValueError("The selected Phase D scope matches zero successful aggregation runs")

    return PhaseDCampaignDefinition(
        phase_d_campaign_id=phase_d_campaign_id,
        parent_generation_campaign_id=str(context["parent_generation_campaign_id"]),
        parent_phase_c_run_key=parent_phase_c_run_key,
        machine_id=machine_id,
        display_name=str(display_name or "").strip() or None,
        notes=str(notes or "").strip() or None,
        runner_config=runner,
    )


def command_for_definition(
    definition: PhaseDCampaignDefinition,
    *,
    phase_d_run_id: str | None = None,
    resume: bool = False,
    overwrite_existing: bool = False,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Compile the exact general Phase D runner argv for Builder/Execution."""
    if resume and overwrite_existing:
        raise ValueError("resume and overwrite_existing are mutually exclusive")
    config = definition.runner_config
    cmd = [
        sys.executable,
        str(runner_script()),
        "--campaign-root", str(config.campaign_root),
        "--output-root", str(config.output_root or config.campaign_root),
        "--matrix-run-id", config.matrix_run_id,
        "--phase-c-campaign-run-id", config.phase_c_campaign_run_id,
    ]
    if phase_d_run_id:
        cmd += ["--phase-d-run-id", str(phase_d_run_id)]
    for value in config.aggregation_ids:
        cmd += ["--aggregation-id", value]
    for value in config.weight_modes:
        cmd += ["--weight-mode", value]
    for value in config.case_ids:
        cmd += ["--case-id", value]
    if config.max_aggregation_runs is not None:
        cmd += ["--max-aggregation-runs", str(config.max_aggregation_runs)]

    cmd += [
        "--phase-d-calendar-year", str(config.phase_d_calendar_year),
        "--heat-representation", config.heat_representation,
    ]
    if config.qzivr_separate:
        cmd.append("--qzivr-separate")
    for value in config.ml_policies:
        cmd += ["--ml-policy", value]
    for value in config.ml_input_lags:
        cmd += ["--ml-input-lag", str(value)]
    for value in config.ml_target_horizons:
        cmd += ["--ml-target-horizon", str(value)]
    if {"monthly_distributed_holdout", "chronological_holdout"} & set(config.ml_policies):
        cmd += [
            "--ml-train-fraction", str(config.ml_train_fraction),
            "--ml-test-fraction", str(config.ml_test_fraction),
            "--ml-validation-fraction", str(config.ml_validation_fraction),
        ]
    if "seasonal_holdout" in config.ml_policies:
        cmd += [
            "--ml-sh-train-seasons", ",".join(config.ml_sh_train_seasons),
            "--ml-sh-test-seasons", ",".join(config.ml_sh_test_seasons),
            "--ml-sh-validation-seasons", ",".join(config.ml_sh_validation_seasons),
        ]
    for value in config.ob_policies:
        cmd += ["--ob-policy", value]
    if "seasonal_distributed" in config.ob_policies:
        cmd += [
            "--sd-season-offset-days", str(config.sd_season_offset_days),
            "--sd-train-days", str(config.sd_train_days),
            "--sd-test-days", str(config.sd_test_days),
        ]
    if "seasonal_block_holdout" in config.ob_policies:
        cmd += [
            "--sbh-train-seasons", ",".join(config.sbh_train_seasons),
            "--sbh-test-seasons", ",".join(config.sbh_test_seasons),
        ]
    if "contiguous_identification" in config.ob_policies:
        if config.ci_start_datetime:
            cmd += ["--ci-start-datetime", config.ci_start_datetime]
        cmd += [
            "--ci-train-days", str(config.ci_train_days),
            "--ci-test-days", str(config.ci_test_days),
        ]
    if "custom_datetime_ranges" in config.ob_policies:
        for value in config.cdr_train_ranges:
            cmd += ["--cdr-train-range", value]
        for value in config.cdr_test_ranges:
            cmd += ["--cdr-test-range", value]
    cmd += ["--parquet-compression", config.parquet_compression]

    if resume:
        cmd.append("--resume")
    if overwrite_existing:
        cmd.append("--overwrite-existing")
    if continue_on_error:
        cmd.append("--continue-on-error")
    if dry_run:
        cmd.append("--dry-run")
    if config.mlflow_enabled:
        cmd.append("--mlflow")
        if config.mlflow_experiment_name:
            cmd += ["--mlflow-experiment-name", config.mlflow_experiment_name]
        if config.mlflow_run_name:
            cmd += ["--mlflow-run-name", config.mlflow_run_name]
        if config.mlflow_strict:
            cmd.append("--mlflow-strict")
    return cmd


def command_preview(definition: PhaseDCampaignDefinition) -> str:
    return subprocess.list2cmdline(command_for_definition(definition))


def definition_summary(definition: PhaseDCampaignDefinition) -> dict[str, Any]:
    config = definition.runner_config
    context = resolve_phase_c_context(definition.parent_phase_c_run_key)
    return {
        "phase_d_campaign_id": definition.phase_d_campaign_id,
        "parent_generation_campaign_id": definition.parent_generation_campaign_id,
        "parent_phase_c_run_key": definition.parent_phase_c_run_key,
        "matrix_run_id": config.matrix_run_id,
        "matched_aggregation_runs": selected_aggregation_count(
            context,
            case_ids=config.case_ids,
            aggregation_ids=config.aggregation_ids,
            weight_modes=config.weight_modes,
            max_aggregation_runs=config.max_aggregation_runs,
        ),
        "case_ids": list(config.case_ids),
        "aggregation_ids": list(config.aggregation_ids),
        "weight_modes": list(config.weight_modes),
        "heat_representation": config.heat_representation,
        "qzivr_separate": config.qzivr_separate,
        "ml_policies": list(config.ml_policies),
        "ml_input_lags": list(config.ml_input_lags),
        "ml_target_horizons": list(config.ml_target_horizons),
        "ob_policies": list(config.ob_policies),
        "mlflow_enabled": config.mlflow_enabled,
        "command": command_preview(definition),
    }
