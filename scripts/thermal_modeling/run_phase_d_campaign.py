# -*- coding: utf-8 -*-
"""Run ScaleBridge Phase D D8 across an authoritative aggregation matrix."""

from __future__ import annotations
import argparse
from pathlib import Path

from scalebridge.data.thermal_modeling.campaign_runner import (
    build_phase_d_run_id,
    load_matrix_aggregation_runs,
    resolve_latest_successful_matrix_run_id,
    resolve_latest_successful_phase_c_run_id,
    run_campaign,
    validate_phase_c_lineage,
)
from scalebridge.data.thermal_modeling.policies import normalize_policy_name


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-root", required=True, type=Path)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--matrix-run-id", default=None)
    p.add_argument("--phase-c-campaign-run-id", default=None)
    p.add_argument("--phase-d-run-id", default=None)

    p.add_argument("--aggregation-id", action="append", default=None)
    p.add_argument("--weight-mode", action="append", default=None)
    p.add_argument("--case-id", action="append", default=None)
    p.add_argument("--max-aggregation-runs", type=int, default=None)

    p.add_argument("--phase-d-calendar-year", type=int, default=2001)
    p.add_argument("--heat-representation", choices=("grouped", "components"), default="grouped")
    p.add_argument("--qzivr-separate", action="store_true")

    p.add_argument(
        "--ml-policy", action="append", default=None,
        help="Repeatable ML/SciML policy: mdh, ch, sh (or long name). Default: mdh.",
    )
    p.add_argument("--ml-input-lag", type=int, action="append", default=None, help="Repeatable ML/SciML input lag. Default: 12.")
    p.add_argument("--ml-target-horizon", type=int, action="append", default=None, help="Repeatable ML/SciML target horizon. Default: 6.")
    p.add_argument("--ml-train-fraction", type=float, default=0.70)
    p.add_argument("--ml-test-fraction", type=float, default=0.15)
    p.add_argument("--ml-validation-fraction", type=float, default=0.15)
    p.add_argument("--ml-sh-train-seasons", default="winter,spring")
    p.add_argument("--ml-sh-test-seasons", default="summer")
    p.add_argument("--ml-sh-validation-seasons", default="fall")

    p.add_argument(
        "--ob-policy", action="append", default=None,
        help="Repeatable Opt/Bayes policy: sd, sbh, ci, cdr (or long name). Default: sd.",
    )
    p.add_argument("--sd-season-offset-days", type=int, default=0)
    p.add_argument("--sd-train-days", type=int, default=21)
    p.add_argument("--sd-test-days", type=int, default=7)
    p.add_argument("--sbh-train-seasons", default="winter,spring,fall")
    p.add_argument("--sbh-test-seasons", default="summer")
    p.add_argument("--ci-start-datetime", default=None)
    p.add_argument("--ci-train-days", type=int, default=21)
    p.add_argument("--ci-test-days", type=int, default=7)
    p.add_argument("--cdr-train-range", action="append", default=None)
    p.add_argument("--cdr-test-range", action="append", default=None)
    p.add_argument("--parquet-compression", default="zstd")

    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--mlflow", action="store_true", help="Enable one campaign-level Phase D MLflow run.")
    p.add_argument("--mlflow-experiment-name", default=None)
    p.add_argument("--mlflow-run-name", default=None)
    p.add_argument("--mlflow-strict", action="store_true", help="Fail Phase D if MLflow setup/logging fails.")
    return p


def main() -> int:
    args = parser().parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    output_root = (args.output_root or campaign_root).expanduser().resolve()
    if args.resume and args.overwrite_existing:
        raise SystemExit("--resume and --overwrite-existing are mutually exclusive")

    matrix_run_id = args.matrix_run_id or resolve_latest_successful_matrix_run_id(campaign_root)
    phase_c_run_id = (
        args.phase_c_campaign_run_id
        or resolve_latest_successful_phase_c_run_id(
            campaign_root, matrix_run_id=matrix_run_id
        )
    )
    validate_phase_c_lineage(
        campaign_root,
        matrix_run_id=matrix_run_id,
        phase_c_campaign_run_id=phase_c_run_id,
    )

    items = load_matrix_aggregation_runs(
        campaign_root,
        matrix_run_id=matrix_run_id,
        aggregation_ids=set(args.aggregation_id) if args.aggregation_id else None,
        weight_modes=set(args.weight_mode) if args.weight_mode else None,
        case_ids=set(args.case_id) if args.case_id else None,
    )
    if args.max_aggregation_runs is not None:
        items = items[: max(0, args.max_aggregation_runs)]

    phase_d_run_id = args.phase_d_run_id or build_phase_d_run_id()
    repo_root = Path(__file__).resolve().parents[2]

    ml_policies = [normalize_policy_name(x) for x in (args.ml_policy or ["mdh"])]
    ml_input_lags = list(args.ml_input_lag or [12])
    if any(x < 1 for x in ml_input_lags):
        raise SystemExit("Every --ml-input-lag must be >= 1")
    if len(ml_input_lags) != len(set(ml_input_lags)):
        raise SystemExit("Duplicate --ml-input-lag values are not allowed")
    ml_target_horizons = list(args.ml_target_horizon or [6])
    if any(x < 1 for x in ml_target_horizons):
        raise SystemExit("Every --ml-target-horizon must be >= 1")
    if len(ml_target_horizons) != len(set(ml_target_horizons)):
        raise SystemExit("Duplicate --ml-target-horizon values are not allowed")
    ob_policies = [normalize_policy_name(x) for x in (args.ob_policy or ["sd"])]
    allowed_ml = {"monthly_distributed_holdout", "chronological_holdout", "seasonal_holdout"}
    allowed_ob = {"seasonal_distributed", "seasonal_block_holdout", "contiguous_identification", "custom_datetime_ranges"}
    if set(ml_policies) - allowed_ml:
        raise SystemExit(f"Unsupported ML/SciML policy: {sorted(set(ml_policies)-allowed_ml)}")
    if set(ob_policies) - allowed_ob:
        raise SystemExit(f"Unsupported Opt/Bayes policy: {sorted(set(ob_policies)-allowed_ob)}")
    if len(ml_policies) != len(set(ml_policies)) or len(ob_policies) != len(set(ob_policies)):
        raise SystemExit("Duplicate policy selections are not allowed")
    if "custom_datetime_ranges" in ob_policies and (not args.cdr_train_range or not args.cdr_test_range):
        raise SystemExit("cdr requires at least one --cdr-train-range and one --cdr-test-range")

    config = {
        "phase_d_calendar_year": args.phase_d_calendar_year,
        "heat_representation": args.heat_representation,
        "qzivr_separate": args.qzivr_separate,
        "ml_policies": ml_policies,
        "ob_policies": ob_policies,
        "ml_input_lags": ml_input_lags,
        "ml_target_horizons": ml_target_horizons,
        "ml_train_fraction": args.ml_train_fraction,
        "ml_test_fraction": args.ml_test_fraction,
        "ml_validation_fraction": args.ml_validation_fraction,
        "ml_sh_train_seasons": args.ml_sh_train_seasons,
        "ml_sh_test_seasons": args.ml_sh_test_seasons,
        "ml_sh_validation_seasons": args.ml_sh_validation_seasons,
        "sd_season_offset_days": args.sd_season_offset_days,
        "sd_train_days": args.sd_train_days,
        "sd_test_days": args.sd_test_days,
        "sbh_train_seasons": args.sbh_train_seasons,
        "sbh_test_seasons": args.sbh_test_seasons,
        "ci_start_datetime": args.ci_start_datetime,
        "ci_train_days": args.ci_train_days,
        "ci_test_days": args.ci_test_days,
        "cdr_train_ranges": list(args.cdr_train_range or []),
        "cdr_test_ranges": list(args.cdr_test_range or []),
        "parquet_compression": args.parquet_compression,
    }

    print("=" * 100)
    print("SCALEBRIDGE PHASE D D8 CAMPAIGN RUNNER")
    print("=" * 100)
    print(f"campaign_root: {campaign_root}")
    print(f"output_root: {output_root}")
    print(f"matrix_run_id: {matrix_run_id}")
    print(f"phase_c_campaign_run_id: {phase_c_run_id}")
    print(f"phase_d_run_id: {phase_d_run_id}")
    print(f"selected_aggregation_runs: {len(items)}")
    print(f"aggregation_ids: {sorted({x.aggregation_id for x in items})}")
    print(f"aggregation_families: {sorted({x.aggregation_family for x in items})}")
    print(f"weight_modes: {sorted({x.weight_mode for x in items})}")
    print(f"ml_policies: {ml_policies}")
    print(f"ml_input_lags: {ml_input_lags}")
    print(f"ml_target_horizons: {ml_target_horizons}")
    print(f"ob_policies: {ob_policies}")

    summary = run_campaign(
        repo_root=repo_root,
        campaign_root=campaign_root,
        output_root=output_root,
        matrix_run_id=matrix_run_id,
        phase_c_campaign_run_id=phase_c_run_id,
        phase_d_run_id=phase_d_run_id,
        items=items,
        config=config,
        resume=args.resume,
        overwrite_existing=args.overwrite_existing,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        mlflow=args.mlflow,
        mlflow_experiment_name=args.mlflow_experiment_name,
        mlflow_run_name=args.mlflow_run_name,
        mlflow_strict=args.mlflow_strict,
    )
    print("")
    print("PHASE D D8 CAMPAIGN SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
