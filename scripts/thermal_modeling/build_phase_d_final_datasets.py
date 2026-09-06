# -*- coding: utf-8 -*-
"""Build controlled/final Phase D D7 products for one aggregation run."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from scalebridge.data.thermal_modeling.builders import (
    assemble_zone_in_memory,
    build_contract,
    build_final_dataset,
    resolve_dep2_for_build,
    write_final_dataset,
)
from scalebridge.data.thermal_modeling.constants import ModelingSilo, PhaseDMode
from scalebridge.data.thermal_modeling.lineage import load_aggregation_lineage
from scalebridge.data.thermal_modeling.policies import normalize_policy_name
from scalebridge.data.thermal_modeling.silo_contracts import (
    HeatInputRepresentation,
    HeatRepresentationConfig,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build final Phase D D7 silo datasets for one aggregation run."
    )
    p.add_argument("--campaign-root", required=True, type=Path)
    p.add_argument("--matrix-run-id", required=True)
    p.add_argument("--aggregation-run-id", required=True)
    p.add_argument("--phase-c-campaign-run-id", required=True)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--phase-d-calendar-year", type=int, default=2001)

    p.add_argument("--heat-representation", choices=("grouped", "components"), default="grouped")
    p.add_argument("--qzivr-separate", action="store_true")

    p.add_argument(
        "--ml-policy", action="append", default=None,
        help="ML/SciML policy; repeat for multiple. Choices: mdh,ch,sh or long names.",
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
        help="Opt/Bayes policy; repeat for multiple. Choices: sd,sbh,ci,cdr or long names.",
    )
    p.add_argument("--sd-season-offset-days", type=int, default=0)
    p.add_argument("--sd-train-days", type=int, default=21)
    p.add_argument("--sd-test-days", type=int, default=7)
    p.add_argument("--sbh-train-seasons", default="winter,spring,fall")
    p.add_argument("--sbh-test-seasons", default="summer")
    p.add_argument("--ci-start-datetime", default=None)
    p.add_argument("--ci-train-days", type=int, default=21)
    p.add_argument("--ci-test-days", type=int, default=7)
    p.add_argument(
        "--cdr-train-range", action="append", default=None,
        help="Repeatable half-open CDR range START/END.",
    )
    p.add_argument(
        "--cdr-test-range", action="append", default=None,
        help="Repeatable half-open CDR range START/END.",
    )
    p.add_argument("--parquet-compression", default="zstd")
    p.add_argument("--runner-configuration-json", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--allow-missing-dep2", action="store_true",
        help="Build ind/dep1 and omit dep2 when D5 says no usable counterpart exists.",
    )
    return p


def main() -> int:
    args = parser().parse_args()

    heat = HeatRepresentationConfig(
        representation=(
            HeatInputRepresentation.GROUPED
            if args.heat_representation == "grouped"
            else HeatInputRepresentation.COMPONENTS
        ),
        include_visible_lighting_in_qzir=(not args.qzivr_separate),
    )

    current_lineage = load_aggregation_lineage(
        campaign_root=args.campaign_root,
        matrix_run_id=args.matrix_run_id,
        aggregation_run_id=args.aggregation_run_id,
    )

    current_zone_data = {}
    for zone in current_lineage.aggregate_zones:
        item = assemble_zone_in_memory(
            campaign_root=args.campaign_root,
            matrix_run_id=args.matrix_run_id,
            aggregation_run_id=args.aggregation_run_id,
            phase_c_campaign_run_id=args.phase_c_campaign_run_id,
            aggregate_zone_id=zone.aggregate_zone_id,
            phase_d_calendar_year=args.phase_d_calendar_year,
            include_visible_lighting_in_zir=heat.include_visible_lighting_in_qzir,
        )
        current_zone_data[zone.aggregate_zone_id] = item
        print(
            f"assembled current zone={zone.aggregate_zone_id} "
            f"rows={len(item.table)} persisted_intermediate=False"
        )

    dep2_resolution = resolve_dep2_for_build(
        campaign_root=args.campaign_root,
        matrix_run_id=args.matrix_run_id,
        aggregation_run_id=args.aggregation_run_id,
        phase_c_campaign_run_id=args.phase_c_campaign_run_id,
        require_available=(not args.allow_missing_dep2),
    )
    dep2_zone_data = None
    dep2_lineage = dep2_resolution.selected_lineage
    if dep2_resolution.dependent_2_available and dep2_lineage is not None:
        if len(dep2_lineage.aggregate_zones) != 1:
            raise RuntimeError("D7/D8 Dep2 source must contain exactly one all-to-one zone")
        dep2_zone_id = dep2_lineage.aggregate_zones[0].aggregate_zone_id

        if (
            dep2_lineage.aggregation_run_id == current_lineage.aggregation_run_id
            and dep2_zone_id in current_zone_data
        ):
            dep2_zone_data = current_zone_data[dep2_zone_id]
        else:
            dep2_zone_data = assemble_zone_in_memory(
                campaign_root=args.campaign_root,
                matrix_run_id=args.matrix_run_id,
                aggregation_run_id=dep2_lineage.aggregation_run_id,
                phase_c_campaign_run_id=args.phase_c_campaign_run_id,
                aggregate_zone_id=dep2_zone_id,
                phase_d_calendar_year=args.phase_d_calendar_year,
                include_visible_lighting_in_zir=heat.include_visible_lighting_in_qzir,
            )
            print(
                f"assembled dep2 zone={dep2_zone_id} "
                f"source_run={dep2_lineage.aggregation_run_id} "
                f"rows={len(dep2_zone_data.table)} persisted_intermediate=False"
            )
    else:
        print(
            "dep2 unavailable; building ind/dep1 only "
            f"status={dep2_resolution.status}"
        )

    current_tables = {
        zone_id: item.table for zone_id, item in current_zone_data.items()
    }
    current_availability = tuple(
        current_zone_data[zone.aggregate_zone_id].availability
        for zone in current_lineage.aggregate_zones
    )

    run_root = (
        args.output_root
        / "phase_d"
        / "cases"
        / current_lineage.case_id
        / "aggregation_runs"
        / current_lineage.aggregation_run_id
    )
    silo_root = run_root / "silos"
    silo_root.mkdir(parents=True, exist_ok=True)

    aggregation_manifest = {
        "schema_version": "phase_d_d7_run_manifest_v1",
        "campaign_id": current_lineage.campaign_id,
        "case_id": current_lineage.case_id,
        "aggregation_matrix_run_id": current_lineage.aggregation_matrix_run_id,
        "aggregation_run_id": current_lineage.aggregation_run_id,
        "aggregation_id": current_lineage.aggregation_id,
        "weight_mode": current_lineage.weight_mode,
        "aggregate_zone_ids": [
            zone.aggregate_zone_id for zone in current_lineage.aggregate_zones
        ],
        "source_zone_ids": list(current_lineage.source_zone_ids),
        "heat_representation": heat.to_dict(),
        "dependent_2": dep2_resolution.to_dict(),
        "phase_d_calendar_year": args.phase_d_calendar_year,
        "intermediate_time_series_persisted": False,
        "status": "running",
        "runner_configuration": (
            json.loads(args.runner_configuration_json)
            if args.runner_configuration_json
            else None
        ),
    }
    (run_root / "aggregation_manifest.json").write_text(
        json.dumps(aggregation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    build_count = 0
    rows = []

    ml_policies = [normalize_policy_name(x) for x in (args.ml_policy or ["mdh"])]
    ml_input_lags = list(args.ml_input_lag or [12])
    if any(x < 1 for x in ml_input_lags):
        raise RuntimeError("Every --ml-input-lag must be >= 1")
    if len(ml_input_lags) != len(set(ml_input_lags)):
        raise RuntimeError("Duplicate --ml-input-lag values are not allowed")
    ml_target_horizons = list(args.ml_target_horizon or [6])
    if any(x < 1 for x in ml_target_horizons):
        raise RuntimeError("Every --ml-target-horizon must be >= 1")
    if len(ml_target_horizons) != len(set(ml_target_horizons)):
        raise RuntimeError("Duplicate --ml-target-horizon values are not allowed")
    ob_policies = [normalize_policy_name(x) for x in (args.ob_policy or ["sd"])]
    if len(ml_policies) != len(set(ml_policies)):
        raise RuntimeError("Duplicate --ml-policy values are not allowed")
    if len(ob_policies) != len(set(ob_policies)):
        raise RuntimeError("Duplicate --ob-policy values are not allowed")

    ml_parameter_map = {
        "monthly_distributed_holdout": {
            "train_fraction": args.ml_train_fraction,
            "test_fraction": args.ml_test_fraction,
            "validation_fraction": args.ml_validation_fraction,
        },
        "chronological_holdout": {
            "train_fraction": args.ml_train_fraction,
            "test_fraction": args.ml_test_fraction,
            "validation_fraction": args.ml_validation_fraction,
        },
        "seasonal_holdout": {
            "train_seasons": args.ml_sh_train_seasons,
            "test_seasons": args.ml_sh_test_seasons,
            "validation_seasons": args.ml_sh_validation_seasons,
        },
    }
    ob_parameter_map = {
        "seasonal_distributed": {
            "season_offset_days": args.sd_season_offset_days,
            "train_days": args.sd_train_days,
            "test_days": args.sd_test_days,
        },
        "seasonal_block_holdout": {
            "train_seasons": args.sbh_train_seasons,
            "test_seasons": args.sbh_test_seasons,
        },
        "contiguous_identification": {
            "start_datetime": args.ci_start_datetime,
            "train_days": args.ci_train_days,
            "test_days": args.ci_test_days,
        },
        "custom_datetime_ranges": {
            "train_ranges": args.cdr_train_range,
            "test_ranges": args.cdr_test_range,
        },
    }
    unknown_ml = sorted(set(ml_policies) - set(ml_parameter_map))
    unknown_ob = sorted(set(ob_policies) - set(ob_parameter_map))
    if unknown_ml:
        raise RuntimeError(f"Unsupported ML/SciML policies: {unknown_ml}")
    if unknown_ob:
        raise RuntimeError(f"Unsupported Opt/Bayes policies: {unknown_ob}")

    configurations = [
        *(
            (
                ModelingSilo.ML_SCIML,
                lag,
                horizon,
                policy_name,
                ml_parameter_map[policy_name],
            )
            for policy_name in ml_policies
            for lag in ml_input_lags
            for horizon in ml_target_horizons
        ),
        *(
            (
                ModelingSilo.OPT_BAYES,
                1,
                1,
                policy_name,
                ob_parameter_map[policy_name],
            )
            for policy_name in ob_policies
        ),
    ]
    aggregation_manifest["temporal_policies"] = {
        "ml_sciml": [
            {
                "policy_name": name,
                "input_lag": lag,
                "target_horizon": horizon,
                "parameters": ml_parameter_map[name],
            }
            for name in ml_policies
            for lag in ml_input_lags
            for horizon in ml_target_horizons
        ],
        "opt_bayes": [
            {"policy_name": name, "parameters": ob_parameter_map[name]}
            for name in ob_policies
        ],
    }
    (run_root / "aggregation_manifest.json").write_text(
        json.dumps(aggregation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    provenance_base = {
        "campaign_id": current_lineage.campaign_id,
        "case_id": current_lineage.case_id,
        "aggregation_matrix_run_id": current_lineage.aggregation_matrix_run_id,
        "aggregation_run_id": current_lineage.aggregation_run_id,
        "aggregation_id": current_lineage.aggregation_id,
        "weight_mode": current_lineage.weight_mode,
        "phase_c_campaign_run_id": args.phase_c_campaign_run_id,
        "dependent_2_match_status": dep2_resolution.status,
        "dependent_2_source_aggregation_run_id": (
            dep2_resolution.selected_aggregation_run_id
        ),
    }

    for silo, lag, horizon, policy_name, policy_parameters in configurations:
        for zone in current_lineage.aggregate_zones:
            contract = build_contract(
                silo=silo,
                mode=PhaseDMode.INDEPENDENT,
                current_zones=current_availability,
                heat=heat,
                input_lag=lag,
                target_horizon=horizon,
                policy_name=policy_name,
                policy_parameters=policy_parameters,
            )
            result = build_final_dataset(
                contract,
                current_tables,
                independent_zone_id=zone.aggregate_zone_id,
                provenance=provenance_base,
            )
            data_path, manifest_path = write_final_dataset(
                result,
                silo_root=silo_root,
                contract=contract,
                independent_zone_id=zone.aggregate_zone_id,
                compression=args.parquet_compression,
            )
            print(
                f"built silo={contract.silo_folder_name} mode=ind "
                f"zone={zone.aggregate_zone_id} rows={len(result.table)} "
                f"included={result.manifest['included_row_count']} "
                f"path={data_path}"
            )
            rows.append(str(data_path))
            build_count += 1
            del result
            gc.collect()

        dep1 = build_contract(
            silo=silo,
            mode=PhaseDMode.DEPENDENT1,
            current_zones=current_availability,
            heat=heat,
            input_lag=lag,
            target_horizon=horizon,
            policy_name=policy_name,
            policy_parameters=policy_parameters,
        )
        result = build_final_dataset(
            dep1,
            current_tables,
            provenance=provenance_base,
        )
        data_path, _ = write_final_dataset(
            result,
            silo_root=silo_root,
            contract=dep1,
            compression=args.parquet_compression,
        )
        print(
            f"built silo={dep1.silo_folder_name} mode=dep1 "
            f"rows={len(result.table)} included={result.manifest['included_row_count']} "
            f"path={data_path}"
        )
        rows.append(str(data_path))
        build_count += 1
        del result
        gc.collect()

        if dep2_zone_data is not None:
            dep2 = build_contract(
                silo=silo,
                mode=PhaseDMode.DEPENDENT2,
                current_zones=current_availability,
                heat=heat,
                input_lag=lag,
                target_horizon=horizon,
                policy_name=policy_name,
                policy_parameters=policy_parameters,
                dependent_2_source_zone=dep2_zone_data.availability,
            )
            result = build_final_dataset(
                dep2,
                current_tables,
                dependent_2_source_table=dep2_zone_data.table,
                provenance=provenance_base,
            )
            data_path, _ = write_final_dataset(
                result,
                silo_root=silo_root,
                contract=dep2,
                compression=args.parquet_compression,
            )
            print(
                f"built silo={dep2.silo_folder_name} mode=dep2 "
                f"rows={len(result.table)} included={result.manifest['included_row_count']} "
                f"path={data_path}"
            )
            rows.append(str(data_path))
            build_count += 1
            del result
            gc.collect()

    expected_count = len(configurations) * (
        len(current_lineage.aggregate_zones)
        + 1
        + (1 if dep2_zone_data is not None else 0)
    )
    if build_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} final datasets, built {build_count}"
        )

    aggregation_manifest.update(
        {
            "status": "completed",
            "final_dataset_count": build_count,
            "dependent_2_built": dep2_zone_data is not None,
            "final_data_parquets": rows,
        }
    )
    (run_root / "aggregation_manifest.json").write_text(
        json.dumps(aggregation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"D7_BUILD_COMPLETE aggregation_run_id={current_lineage.aggregation_run_id} "
        f"zones={len(current_lineage.aggregate_zones)} "
        f"final_parquets={build_count} intermediate_parquets=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
