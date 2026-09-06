# -*- coding: utf-8 -*-
"""Audit Stage B aggregation outputs for Stage C heat-input regression readiness."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scalebridge.data.aggregation.writers import make_safe_name, write_csv, write_json
from scalebridge.data.heat_input_regression.discovery import discover_from_aggregation_run, discover_from_matrix_run, resolve_inputs
from scalebridge.data.heat_input_regression.signal_catalog import get_signal_definition, resolve_present_column
from scalebridge.data.heat_input_regression.hvac import build_hvac_predictor, build_hvac_target, build_phvac_features
from scalebridge.data.heat_input_regression.source_detection import build_heat_source_inventory
from scalebridge.data.heat_input_regression.validation import audit_signal_frame, evaluate_node_mapping_quality, load_provenance_sets, validate_series_pair
from scalebridge.models.heat_input_regression.registry import list_model_specifications

DEFAULT_CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
AIR_HEAT_CAPACITY_KJ_PER_KG_K = 1.005


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    p.add_argument("--campaign-root", default=None)
    p.add_argument("--generated-data-root", default=None)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--matrix-run-id")
    group.add_argument("--aggregation-run-root")
    p.add_argument("--case-id", default=None)
    p.add_argument("--aggregation-id", default=None)
    p.add_argument("--weight-mode", default=None)
    p.add_argument("--aggregate-zone-id", default=None)
    p.add_argument("--max-zones", type=int, default=None, help="Optional development truncation applied after discovery/filtering.")
    p.add_argument("--output-root", default=None)
    p.add_argument("--audit-run-id", default=None)
    p.add_argument("--minimum-sample-count", type=int, default=1000)
    p.add_argument("--internal-gain-predictor-method", choices=["aggregate_average", "contribution_sum"], default="aggregate_average")
    p.add_argument("--hvac-target-method", choices=["signed_zone_sensible", "absolute_zone_sensible"], default="signed_zone_sensible")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv); started = time.perf_counter()
    campaign_root = resolve_inputs(campaign_id=args.campaign_id, campaign_root=args.campaign_root, generated_data_root=args.generated_data_root)
    if args.matrix_run_id:
        refs, issues = discover_from_matrix_run(campaign_root=campaign_root, matrix_run_id=args.matrix_run_id, case_id=args.case_id, aggregation_id=args.aggregation_id, weight_mode=args.weight_mode, aggregate_zone_id=args.aggregate_zone_id)
    else:
        refs, issues = discover_from_aggregation_run(campaign_root=campaign_root, aggregation_run_root=Path(args.aggregation_run_root), aggregate_zone_id=args.aggregate_zone_id)
    if not refs: raise SystemExit("No valid aggregate-zone outputs were discovered.")
    if args.max_zones is not None:
        if args.max_zones <= 0:
            raise ValueError("--max-zones must be positive when supplied")
        refs = refs[: args.max_zones]
    audit_run_id = args.audit_run_id or f"heat_input_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else campaign_root / "heat_input_regression" / "audit_runs" / audit_run_id
    output_root.mkdir(parents=True, exist_ok=True)
    selected_rows = [ref.identity_dict() for ref in refs]
    write_csv(output_root / "selected_aggregation_zone_outputs.csv", selected_rows)
    write_csv(output_root / "source_discovery_issues.csv", issues)
    result_rows=[]
    print("="*100); print("SCALEBRIDGE HEAT-INPUT REGRESSION READINESS AUDIT"); print("="*100)
    print(f"campaign_root: {campaign_root}"); print(f"audit_run_id: {audit_run_id}"); print(f"selected_aggregation_zone_count: {len(refs)}")
    for index, ref in enumerate(refs, 1):
        print(f"[{index}/{len(refs)}] {ref.case_id} | {ref.aggregation_id} | {ref.weight_mode} | {ref.aggregate_zone_id}")
        try:
            zone_result = audit_one(ref, output_root, args)
            result_rows.append(zone_result)
        except Exception as exc:
            row={**ref.identity_dict(), "status":"failed", "error_type":type(exc).__name__, "error_message":str(exc), "traceback":traceback.format_exc()}
            result_rows.append(row); print(f"ERROR: {exc}")
            if not args.continue_on_error: raise
    write_csv(output_root / "audit_zone_results.csv", result_rows)
    successful = sum(r.get("status") == "completed" for r in result_rows)
    candidate_model_count = sum(int(r.get("candidate_model_count", 0) or 0) for r in result_rows)
    applicable_model_count = sum(int(r.get("applicable_model_count", 0) or 0) for r in result_rows)
    structurally_inapplicable_model_count = sum(
        int(r.get("structurally_inapplicable_model_count", 0) or 0)
        for r in result_rows
    )
    invalid_model_count = sum(int(r.get("invalid_model_count", 0) or 0) for r in result_rows)
    missing_expected_data_model_count = sum(
        int(r.get("missing_expected_data_model_count", 0) or 0)
        for r in result_rows
    )
    manifest={
        "schema_version":"0.1.0", "created_at_utc":datetime.now(timezone.utc).isoformat(), "status":"completed" if successful==len(refs) else "completed_with_failures",
        "audit_run_id":audit_run_id, "campaign_id":args.campaign_id, "campaign_root":str(campaign_root), "source_matrix_run_id":args.matrix_run_id or "",
        "selected_aggregation_run_count":len({r.aggregation_run_id for r in refs}), "selected_aggregation_zone_count":len(refs),
        "successful_zone_count": successful,
        "failed_zone_count": len(refs) - successful,
        "candidate_model_count": candidate_model_count,
        "applicable_model_count": applicable_model_count,
        "structurally_inapplicable_model_count": structurally_inapplicable_model_count,
        "invalid_model_count": invalid_model_count,
        "missing_expected_data_model_count": missing_expected_data_model_count,
        "internal_gain_predictor_method":args.internal_gain_predictor_method, "hvac_target_method":args.hvac_target_method, "minimum_sample_count":args.minimum_sample_count,
        "runtime_seconds":time.perf_counter()-started, "output_root":str(output_root),
    }
    write_json(output_root / "heat_input_regression_audit_manifest.json", manifest)
    print("="*100); print("AUDIT SUMMARY"); print("="*100); print(f"successful_zone_count: {successful}"); print(f"failed_zone_count: {len(refs)-successful}"); print(f"candidate_model_count: {candidate_model_count}"); print(f"applicable_model_count: {applicable_model_count}"); print(f"structurally_inapplicable_model_count: {structurally_inapplicable_model_count}"); print(f"invalid_model_count: {invalid_model_count}"); print(f"missing_expected_data_model_count: {missing_expected_data_model_count}"); print(f"output_root: {output_root}")
    return 0 if successful==len(refs) else 1


def audit_one(ref, output_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    zone_out = output_root / "cases" / make_safe_name(ref.case_id) / make_safe_name(ref.aggregation_run_id) / make_safe_name(ref.aggregate_zone_id)
    zone_out.mkdir(parents=True, exist_ok=True)
    try:
        frame = pd.read_parquet(ref.wide_parquet_path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading Stage B Parquet requires pyarrow or fastparquet in the active "
            "ScaleBridge environment. The aggregation environment normally includes pyarrow."
        ) from exc
    static_frame = pd.read_csv(ref.static_equipment_path)
    contrib_frame = pd.read_csv(ref.equipment_contributions_path)
    signal_rows = audit_signal_frame(frame); write_csv(zone_out / "heat_input_signal_catalog.csv", signal_rows)
    wide_columns=set(str(c) for c in frame.columns)
    inventory, static_audit = build_heat_source_inventory(wide_columns=wide_columns, static_frame=static_frame, contribution_frame=contrib_frame, predictor_method=args.internal_gain_predictor_method)
    write_csv(zone_out / "heat_source_inventory.csv", inventory); write_csv(zone_out / "static_level_audit.csv", static_audit)
    node_rows=[
        evaluate_node_mapping_quality(summary_path=ref.node_temperature_summary_path, mapping_path=ref.node_temperature_mapping_path, zone_mapping_path=ref.zone_mapping_path, variable_label="System Node Temperature"),
        evaluate_node_mapping_quality(summary_path=ref.node_mass_flow_summary_path, mapping_path=ref.node_mass_flow_mapping_path, zone_mapping_path=ref.zone_mapping_path, variable_label="System Node Mass Flow Rate"),
    ]
    write_csv(zone_out / "node_mapping_quality.csv", node_rows)
    provenance=load_provenance_sets(rdd_path=ref.rdd_intersection_path, source_manifest_path=ref.source_run_manifest_path, variable_manifest_path=ref.variable_manifest_path, loaded_variables_path=ref.loaded_variables_path)
    model_rows: list[dict[str, Any]] = []
    model_rows_by_id: dict[str, dict[str, Any]] = {}
    for spec in list_model_specifications():
        dependency_id = str(spec.dependency_model_id or "").strip()
        dependency_row = model_rows_by_id.get(dependency_id) if dependency_id else None
        if dependency_row is not None and not bool(dependency_row.get("applicable")):
            model_row = make_dependency_unavailable_row(
                spec,
                dependency_row=dependency_row,
            )
        else:
            try:
                model_row = evaluate_model(
                    spec,
                    frame,
                    inventory,
                    static_frame,
                    provenance,
                    node_rows,
                    args,
                    aggregate_zone_count=ref.aggregate_zone_count,
                )
            except KeyError as exc:
                model_row = classify_expected_key_error(spec, exc)
                if model_row is None:
                    raise
        model_rows.append(model_row)
        model_rows_by_id[spec.model_id] = model_row

    write_csv(zone_out / "candidate_models.csv", model_rows)
    write_csv(zone_out / "model_applicability.csv", model_rows)
    applicable = [row for row in model_rows if row["applicability_status"] == "applicable"]
    unavailable = [row for row in model_rows if row["applicability_status"] != "applicable"]
    structurally_inapplicable = [
        row for row in unavailable
        if row.get("applicability_class") == "structurally_inapplicable"
    ]
    invalid_models = [
        row for row in unavailable
        if row.get("applicability_class") == "invalid_data"
    ]
    missing_expected_models = [
        row for row in unavailable
        if row.get("applicability_class") == "missing_expected_data"
    ]
    write_csv(zone_out / "applicable_models.csv", applicable)
    write_csv(zone_out / "unavailable_models.csv", unavailable)
    write_csv(zone_out / "inapplicable_models.csv", structurally_inapplicable)

    result = {
        **ref.identity_dict(),
        "status": "completed",
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "candidate_model_count": len(model_rows),
        "applicable_model_count": len(applicable),
        "unavailable_model_count": len(unavailable),
        "structurally_inapplicable_model_count": len(structurally_inapplicable),
        "invalid_model_count": len(invalid_models),
        "missing_expected_data_model_count": len(missing_expected_models),
        "output_dir": str(zone_out),
    }
    write_json(
        zone_out / "zone_audit_manifest.json",
        {
            **result,
            "internal_gain_predictor_method": args.internal_gain_predictor_method,
            "hvac_target_method": args.hvac_target_method,
        },
    )
    return result


def evaluate_model(spec, frame: pd.DataFrame, inventory: list[dict[str, Any]], static_frame: pd.DataFrame, provenance: dict[str,set[str]], node_rows: list[dict[str,Any]], args: argparse.Namespace, aggregate_zone_count: int = 1) -> dict[str, Any]:
    row = {
        **spec.to_dict(),
        "predictor_columns": "",
        "target_column": "",
        "applicability_status": "",
        "applicability_class": "",
        "applicable": False,
        "reason_code": "",
        "reason": "",
        "missing_required_signals": "",
        "dependency_status": "",
        "fatal_for_zone": False,
        "aligned_sample_count": 0,
        "predictor_method": spec.default_predictor_method or "",
    }
    columns=set(str(c) for c in frame.columns)
    resolved=[]
    if spec.model_id == "QAC":
        predictor, predictor_meta = build_hvac_predictor(frame)
        target_series, target_meta = build_hvac_target(frame, method=args.hvac_target_method)
        resolved = predictor_meta["source_columns"]
        target_column = target_meta["feature_name"]
    elif spec.model_id == "PHVAC":
        qhvac_target, _ = build_hvac_target(frame, method=args.hvac_target_method)
        predictor, target_series, predictor_meta, target_meta = build_phvac_features(
            frame,
            qhvac_target=qhvac_target,
            aggregate_zone_count=aggregate_zone_count,
        )
        resolved = predictor_meta["source_columns"]
        target_column = target_meta["feature_name"]
    else:
        definition=get_signal_definition(spec.target_semantic_name); target_column=resolve_present_column(spec.target_semantic_name, columns); target_series=frame[target_column] if target_column else None
        energyplus_name=definition.canonical_column.rstrip("_").replace("_", " ")
        if energyplus_name in provenance["rdd_unavailable"]:
            return finish(row,"not_applicable_rdd_unavailable",f"target unavailable in RDD: {energyplus_name}",target_column)
        if provenance["requested"] and energyplus_name not in provenance["requested"]:
            return finish(row,"not_applicable_not_requested",f"target not requested: {energyplus_name}",target_column)
        if provenance["generated"] and energyplus_name not in provenance["generated"]:
            return finish(row,"not_applicable_not_generated",f"target not generated: {energyplus_name}",target_column)
        if provenance["loaded"] and energyplus_name not in provenance["loaded"]:
            return finish(row,"not_applicable_not_loaded_by_aggregation",f"target not loaded by aggregation: {energyplus_name}",target_column)
        if spec.predictor_kind=="ghi":
            names=spec.predictor_semantic_names; resolved=[resolve_present_column(name,columns) for name in names]
            if any(x is None for x in resolved): return finish(row,"not_applicable_predictor_absent","one or more GHI inputs are absent",target_column,resolved)
            altitude=np.deg2rad(pd.to_numeric(frame[resolved[2]],errors="coerce")); predictor=pd.to_numeric(frame[resolved[0]],errors="coerce")*np.abs(np.sin(altitude))+pd.to_numeric(frame[resolved[1]],errors="coerce")
        elif spec.predictor_kind=="corrected_schedule":
            source=next((x for x in inventory if x["equipment_type"]==spec.source_family),None)
            if not source or not source["source_present"]: return finish(row,"not_applicable_source_absent","equipment source absent",target_column)
            if args.internal_gain_predictor_method=="contribution_sum":
                return finish(row,"not_applicable_predictor_absent","contribution_sum requires source schedule time series and is reserved for the later feature-construction phase",target_column)
            schedule_column=source["schedule_column"]
            if not schedule_column: return finish(row,"not_applicable_schedule_absent","aggregate schedule absent",target_column)
            if not source["static_level_present"]: return finish(row,"not_applicable_static_level_absent","aggregate static level absent",target_column,[schedule_column])
            predictor=pd.to_numeric(frame[schedule_column],errors="coerce")*float(source["static_level_value"]); resolved=[schedule_column,"<aggregated_static_equipment.value>"]
        else:
            return finish(row,"not_applicable_predictor_absent",f"unsupported predictor kind: {spec.predictor_kind}",target_column)
    if target_series is None or target_column is None: return finish(row,"not_applicable_target_absent","target column absent",target_column,resolved)
    status, reason, n=validate_series_pair(predictor,target_series,args.minimum_sample_count)
    row.update({
        "predictor_columns": " | ".join(str(x) for x in resolved if x),
        "target_column": target_column,
        "applicability_status": status,
        "applicability_class": classify_status(status),
        "applicable": status == "applicable",
        "reason_code": status,
        "reason": reason,
        "aligned_sample_count": n,
        "predictor_method": (
            args.internal_gain_predictor_method
            if spec.predictor_kind == "corrected_schedule"
            else spec.predictor_kind
        ),
    })
    return row


def classify_status(status: str) -> str:
    if status == "applicable":
        return "applicable"
    if status.startswith("not_applicable_"):
        return "structurally_inapplicable"
    if status.startswith("invalid_"):
        return "invalid_data"
    return "missing_expected_data"


def extract_missing_signals(message: str) -> list[str]:
    match = re.search(r"missing signals:\s*\[([^]]*)\]", message)
    if not match:
        return []
    return [
        token.strip().strip("'\"")
        for token in match.group(1).split(",")
        if token.strip()
    ]


def classify_expected_key_error(
    spec,
    exc: KeyError,
) -> dict[str, Any] | None:
    message = str(exc).strip('"')
    missing_signals = extract_missing_signals(message)
    if missing_signals:
        reason_code = "not_applicable_required_predictor_signals_absent"
    elif "sensible heating and cooling targets are required" in message:
        reason_code = "not_applicable_required_hvac_target_signals_absent"
    elif "Facility HVAC electric-demand target is absent" in message:
        reason_code = "not_applicable_required_hvac_power_target_absent"
    elif "target is absent" in message.casefold():
        reason_code = "not_applicable_required_model_target_absent"
    else:
        # Unknown KeyErrors may indicate a programming/schema defect. Preserve
        # them as fatal rather than silently classifying them as structural.
        return None
    row = {
        **spec.to_dict(),
        "predictor_columns": "",
        "target_column": "",
        "applicability_status": reason_code,
        "applicability_class": "structurally_inapplicable",
        "applicable": False,
        "reason_code": reason_code,
        "reason": message,
        "missing_required_signals": " | ".join(missing_signals),
        "dependency_status": "",
        "fatal_for_zone": False,
        "aligned_sample_count": 0,
        "predictor_method": spec.default_predictor_method or spec.predictor_kind,
    }
    return row


def make_dependency_unavailable_row(
    spec,
    *,
    dependency_row: dict[str, Any],
) -> dict[str, Any]:
    dependency_id = str(spec.dependency_model_id)
    dependency_status = str(dependency_row.get("applicability_status", "unavailable"))
    reason_code = "not_applicable_dependency_model_unavailable"
    return {
        **spec.to_dict(),
        "predictor_columns": "",
        "target_column": "",
        "applicability_status": reason_code,
        "applicability_class": "structurally_inapplicable",
        "applicable": False,
        "reason_code": reason_code,
        "reason": (
            f"dependency model {dependency_id} is unavailable: "
            f"{dependency_status}"
        ),
        "missing_required_signals": "",
        "dependency_status": dependency_status,
        "fatal_for_zone": False,
        "aligned_sample_count": 0,
        "predictor_method": spec.default_predictor_method or spec.predictor_kind,
    }


def finish(
    row,
    status,
    reason,
    target_column,
    predictor_columns=None,
    *,
    reason_code: str | None = None,
    missing_required_signals: list[str] | None = None,
):
    row.update({
        "target_column": target_column or "",
        "predictor_columns": " | ".join(
            str(x) for x in (predictor_columns or []) if x
        ),
        "applicability_status": status,
        "applicability_class": classify_status(status),
        "applicable": False,
        "reason_code": reason_code or status,
        "reason": reason,
        "missing_required_signals": " | ".join(missing_required_signals or []),
        "fatal_for_zone": False,
    })
    return row

if __name__=="__main__": raise SystemExit(main(sys.argv[1:]))
