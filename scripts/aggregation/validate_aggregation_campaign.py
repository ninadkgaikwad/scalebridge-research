# -*- coding: utf-8 -*-
"""Validate a general ScaleBridge Phase B Aggregation matrix run.

This CLI contains no P1-specific campaign-size, level, building, climate, or
weighting expectations. It validates the B1/B2 generic artifact contract and
writes both JSON and text reports suitable for later BGIRS consumption.

Examples from repository root:

    python scripts/aggregation/validate_aggregation_campaign.py \
      --campaign-definition .\\bgirs_phase_b_testing_b1b2_v1.json \
      --matrix-run-id aggregation_matrix_20260812_154214

    python scripts/aggregation/validate_aggregation_campaign.py \
      --matrix-run-dir D:\\...\\aggregation\\matrix_runs\\aggregation_matrix_...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scalebridge.data.aggregation.campaign_definition import load_aggregation_campaign_definition
from scalebridge.data.aggregation.discovery import resolve_campaign_root, resolve_repo_root
from scalebridge.data.aggregation.validation import (
    AggregationValidationPolicy,
    validate_aggregation_matrix,
    write_validation_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign-definition", default=None)
    source.add_argument("--matrix-run-dir", default=None)
    parser.add_argument("--matrix-run-id", default=None)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--allow-failed-runs", action="store_true")
    parser.add_argument("--no-run-outputs", action="store_true")
    parser.add_argument("--no-shared-node-summaries", action="store_true")
    parser.add_argument("--require-legacy-pickle", action="store_true")
    parser.add_argument("--no-rule-warning-notices", action="store_true")
    parser.add_argument("--no-unmapped-node-notices", action="store_true")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--report-text", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    campaign_root: Path | None = None
    if args.matrix_run_dir:
        matrix_dir = Path(args.matrix_run_dir).expanduser().resolve()
        if args.campaign_root:
            campaign_root = Path(args.campaign_root).expanduser().resolve()
    else:
        definition_path = Path(args.campaign_definition).expanduser().resolve()
        definition = load_aggregation_campaign_definition(definition_path)
        repo_root = resolve_repo_root()
        campaign_root = resolve_campaign_root(
            repo_root=repo_root,
            campaign_id=definition.parent_generation_campaign_id,
            campaign_root=args.campaign_root or definition.parent_generation_campaign_root,
            generated_data_root=args.generated_data_root or definition.generated_data_root,
        )
        matrix_root = campaign_root / "aggregation" / "matrix_runs"
        matrix_dir = _resolve_matrix_dir(
            matrix_root=matrix_root,
            matrix_run_id=args.matrix_run_id,
            aggregation_campaign_id=definition.aggregation_campaign_id,
        )

    policy = AggregationValidationPolicy(
        require_successful_completion=not args.allow_failed_runs,
        require_run_outputs=not args.no_run_outputs,
        require_shared_node_summaries=not args.no_shared_node_summaries,
        require_legacy_pickle=True if args.require_legacy_pickle else None,
        warn_on_rule_warnings=not args.no_rule_warning_notices,
        warn_on_unmapped_system_nodes=not args.no_unmapped_node_notices,
    )
    report = validate_aggregation_matrix(
        matrix_run_dir=matrix_dir,
        campaign_root=campaign_root,
        policy=policy,
    )

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json
        else matrix_dir / "aggregation_validation_report.json"
    )
    report_text = (
        Path(args.report_text).expanduser().resolve()
        if args.report_text
        else matrix_dir / "aggregation_validation_report.txt"
    )
    write_validation_reports(report, json_path=report_json, text_path=report_text)

    print("=" * 100)
    print("GENERIC AGGREGATION VALIDATION")
    print("=" * 100)
    print(f"status: {report.status}")
    print(f"error_count: {report.error_count}")
    print(f"warning_count: {report.warning_count}")
    print(f"matrix_run_id: {report.matrix_run_id}")
    print(f"matrix_status: {report.matrix_status}")
    print(f"selected_plan_count: {report.selected_plan_count}")
    print(f"successful_plan_count: {report.successful_plan_count}")
    print(f"failed_plan_count: {report.failed_plan_count}")
    print(f"checked_run_count: {report.checked_run_count}")
    print(f"checked_zone_count: {report.checked_zone_count}")
    print(f"rule_warning_row_count: {report.rule_warning_row_count}")
    print(f"unmapped_system_node_count: {report.unmapped_system_node_count}")
    print(f"report_json: {report_json}")
    print(f"report_text: {report_text}")
    return 0 if report.status == "PASS" else 2


def _resolve_matrix_dir(
    *,
    matrix_root: Path,
    matrix_run_id: str | None,
    aggregation_campaign_id: str,
) -> Path:
    if matrix_run_id:
        path = matrix_root / matrix_run_id
        if not path.is_dir():
            raise FileNotFoundError(f"Aggregation matrix run does not exist: {path}")
        return path.resolve()

    candidates = sorted(
        [path for path in matrix_root.glob("aggregation_matrix_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        definition_path = path / "aggregation_campaign_definition.json"
        if not definition_path.is_file():
            continue
        try:
            definition = load_aggregation_campaign_definition(definition_path)
        except Exception:
            continue
        if definition.aggregation_campaign_id == aggregation_campaign_id:
            return path.resolve()
    raise FileNotFoundError(
        "No matrix run found for Aggregation campaign definition: "
        f"{aggregation_campaign_id} under {matrix_root}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
