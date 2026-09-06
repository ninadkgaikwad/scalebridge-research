# -*- coding: utf-8 -*-
"""Generic validation for ScaleBridge Phase B Aggregation campaigns.

The validator audits the portable B1 campaign definition, B2 plan/matrix
artifacts, and completed scientific Aggregation runs without encoding any
paper-specific campaign size, building, weather, level, or weighting
expectations.

Validation is intentionally read-only. Scientific warning diagnostics (for
example rule warnings or unmapped EnergyPlus system-node keys) are surfaced as
warnings and do not become validation errors unless the persisted artifact
contract is internally inconsistent.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scalebridge.data.aggregation.campaign_definition import (
    AggregationCampaignDefinition,
    load_aggregation_campaign_definition,
)
from scalebridge.data.aggregation.models import SUCCESS_STATUSES as GENERATION_SUCCESS_STATUSES


_AGGREGATION_SUCCESS_STATUSES = {"completed", "success", "succeeded"}
_REQUIRED_MATRIX_FILES = (
    "aggregation_matrix_manifest.json",
    "aggregation_campaign_definition.json",
    "selected_aggregation_plans.csv",
    "aggregation_matrix_case_runs.csv",
    "aggregation_matrix_outputs.csv",
    "missing_generation_rows.csv",
)
_REQUIRED_RUN_INPUTS = (
    "aggregation_plan.json",
    "zone_mapping.csv",
    "source_run_manifest.json",
    "source_generation_run.json",
)
_REQUIRED_DIAGNOSTICS = (
    "loaded_variables.csv",
    "rule_summary.csv",
    "rule_diagnostics.csv",
    "schedule_equipment_mapping_used.csv",
    "equipment_contributions.csv",
    "system_node_temperature_summary.csv",
    "system_node_mass_flow_summary.csv",
)
_REQUIRED_ZONE_FILES = (
    "aggregated_timeseries_wide.parquet",
    "aggregated_timeseries_wide_preview.csv",
    "aggregated_timeseries_long.parquet",
    "aggregated_timeseries_long_preview.csv",
    "aggregated_static_equipment.parquet",
    "aggregated_static_equipment.csv",
    "equipment_contributions.csv",
    "equipment_contributions.parquet",
    "zone_mapping.csv",
)


@dataclass(frozen=True)
class ValidationIssue:
    """One generic Aggregation validation finding."""

    severity: str
    code: str
    message: str
    path: str = ""
    case_id: str = ""
    aggregation_id: str = ""
    aggregation_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "case_id": self.case_id,
            "aggregation_id": self.aggregation_id,
            "aggregation_run_id": self.aggregation_run_id,
        }


@dataclass
class AggregationValidationReport:
    """Machine-readable result of one generic Phase B validation."""

    matrix_run_id: str
    matrix_run_dir: Path
    aggregation_campaign_id: str = ""
    parent_generation_campaign_id: str = ""
    matrix_status: str = ""
    dry_run: bool = False
    selected_plan_count: int = 0
    successful_plan_count: int = 0
    failed_plan_count: int = 0
    checked_run_count: int = 0
    checked_zone_count: int = 0
    rule_warning_row_count: int = 0
    unmapped_system_node_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == "warning")

    @property
    def status(self) -> str:
        return "PASS" if self.error_count == 0 else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "aggregation_campaign_id": self.aggregation_campaign_id,
            "parent_generation_campaign_id": self.parent_generation_campaign_id,
            "matrix_run_id": self.matrix_run_id,
            "matrix_run_dir": str(self.matrix_run_dir),
            "matrix_status": self.matrix_status,
            "dry_run": self.dry_run,
            "selected_plan_count": self.selected_plan_count,
            "successful_plan_count": self.successful_plan_count,
            "failed_plan_count": self.failed_plan_count,
            "checked_run_count": self.checked_run_count,
            "checked_zone_count": self.checked_zone_count,
            "rule_warning_row_count": self.rule_warning_row_count,
            "unmapped_system_node_count": self.unmapped_system_node_count,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class AggregationValidationPolicy:
    """Generic validation policy knobs suitable for CLI and future BGIRS use."""

    require_successful_completion: bool = True
    require_run_outputs: bool = True
    require_shared_node_summaries: bool = True
    require_legacy_pickle: bool | None = None
    warn_on_rule_warnings: bool = True
    warn_on_unmapped_system_nodes: bool = True


def validate_aggregation_matrix(
    *,
    matrix_run_dir: str | Path,
    campaign_root: str | Path | None = None,
    policy: AggregationValidationPolicy | None = None,
) -> AggregationValidationReport:
    """Validate one B2 matrix run without paper-specific assumptions."""
    policy = policy or AggregationValidationPolicy()
    matrix_dir = Path(matrix_run_dir).expanduser().resolve()
    report = AggregationValidationReport(
        matrix_run_id=matrix_dir.name,
        matrix_run_dir=matrix_dir,
    )

    if not matrix_dir.is_dir():
        _error(report, "matrix_dir_missing", f"Matrix directory does not exist: {matrix_dir}", matrix_dir)
        return report

    for name in _REQUIRED_MATRIX_FILES:
        path = matrix_dir / name
        if not path.is_file():
            _error(report, "matrix_file_missing", f"Required matrix artifact is missing: {name}", path)

    manifest_path = matrix_dir / "aggregation_matrix_manifest.json"
    manifest = _read_json(manifest_path, report, required=True)
    if not manifest:
        return report

    report.aggregation_campaign_id = _text(manifest.get("aggregation_campaign_id"))
    report.parent_generation_campaign_id = _text(
        manifest.get("parent_generation_campaign_id") or manifest.get("campaign_id")
    )
    report.matrix_status = _text(manifest.get("status"))
    report.dry_run = bool(manifest.get("dry_run", False))
    report.selected_plan_count = _int(manifest.get("selected_plan_count"))
    report.successful_plan_count = _int(manifest.get("successful_plan_count"))
    report.failed_plan_count = _int(manifest.get("failed_plan_count"))

    if _text(manifest.get("matrix_run_id")) and _text(manifest.get("matrix_run_id")) != matrix_dir.name:
        _error(report, "matrix_id_mismatch", "Matrix manifest matrix_run_id does not match directory name", manifest_path)

    resolved_campaign_root = _resolve_campaign_root(campaign_root, manifest, matrix_dir)
    canonical_aggregation_root = resolved_campaign_root / "aggregation" if resolved_campaign_root else None
    if resolved_campaign_root and not resolved_campaign_root.is_dir():
        _error(report, "campaign_root_missing", f"Campaign root does not exist: {resolved_campaign_root}", resolved_campaign_root)

    definition_path = matrix_dir / "aggregation_campaign_definition.json"
    definition = _load_definition(definition_path, report)
    if definition:
        _validate_definition_lineage(definition, manifest, report, definition_path)

    selected_rows = _read_csv(matrix_dir / "selected_aggregation_plans.csv", report, required=True)
    case_rows = _read_csv(matrix_dir / "aggregation_matrix_case_runs.csv", report, required=True)
    output_rows = _read_csv(matrix_dir / "aggregation_matrix_outputs.csv", report, required=True)
    missing_generation_rows = _read_csv(matrix_dir / "missing_generation_rows.csv", report, required=False)

    _validate_matrix_counts(
        report=report,
        manifest=manifest,
        selected_rows=selected_rows,
        case_rows=case_rows,
        output_rows=output_rows,
        missing_generation_rows=missing_generation_rows,
        policy=policy,
    )
    _validate_plan_build(
        manifest=manifest,
        report=report,
        definition=definition,
        selected_rows=selected_rows,
        canonical_aggregation_root=canonical_aggregation_root,
    )

    selected_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in selected_rows:
        key = (_text(row.get("case_id")), _normalized_path_text(row.get("plan_path")))
        if key in selected_by_key:
            _error(report, "duplicate_selected_plan", f"Duplicate selected plan for case/path: {key}")
        selected_by_key[key] = row
        _validate_plan_row(
            row=row,
            report=report,
            canonical_aggregation_root=canonical_aggregation_root,
        )

    case_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in case_rows:
        key = (_text(row.get("case_id")), _normalized_path_text(row.get("plan_path")))
        if key in case_by_key:
            _error(report, "duplicate_case_run", f"Duplicate matrix case-run row for case/path: {key}")
        case_by_key[key] = row
        if key not in selected_by_key:
            _error(
                report,
                "case_run_not_selected",
                "Matrix case-run row has no matching selected plan",
                case_id=_text(row.get("case_id")),
                aggregation_id=_text(row.get("aggregation_id")),
            )

    missing_case_rows = sorted(set(selected_by_key).difference(case_by_key))
    for case_id, plan_path in missing_case_rows:
        _error(report, "selected_plan_not_attempted", "Selected plan has no matrix case-run row", plan_path, case_id=case_id)

    if report.dry_run:
        for row in case_rows:
            if _text(row.get("status")).casefold() != "planned":
                _error(report, "dry_run_status_invalid", "Dry-run matrix contains a non-planned case row")
        if output_rows:
            _error(report, "dry_run_outputs_present", "Dry-run matrix unexpectedly contains completed output rows")
        return report

    for row in case_rows:
        status = _text(row.get("status")).casefold()
        if status in _AGGREGATION_SUCCESS_STATUSES:
            _validate_completed_run(
                row=row,
                report=report,
                canonical_aggregation_root=canonical_aggregation_root,
                definition=definition,
                policy=policy,
            )
        elif status == "failed":
            if not _text(row.get("error_type")) and not _text(row.get("error_message")):
                _warning(report, "failed_run_without_error_detail", "Failed matrix row has no error_type/error_message")
        else:
            _error(report, "unexpected_case_status", f"Unexpected matrix case-run status: {status!r}")

    return report


def write_validation_reports(
    report: AggregationValidationReport,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> tuple[Path, Path]:
    """Write machine-readable JSON and human-readable text validation reports."""
    json_out = Path(json_path).expanduser().resolve()
    text_out = Path(text_path).expanduser().resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    text_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text_out.write_text(render_validation_text(report), encoding="utf-8")
    return json_out, text_out


def render_validation_text(report: AggregationValidationReport) -> str:
    """Render concise deterministic human-readable validation report."""
    lines = [
        "SCALEBRIDGE GENERIC AGGREGATION VALIDATION REPORT",
        "=" * 80,
        f"status: {report.status}",
        f"error_count: {report.error_count}",
        f"warning_count: {report.warning_count}",
        f"aggregation_campaign_id: {report.aggregation_campaign_id}",
        f"parent_generation_campaign_id: {report.parent_generation_campaign_id}",
        f"matrix_run_id: {report.matrix_run_id}",
        f"matrix_run_dir: {report.matrix_run_dir}",
        f"matrix_status: {report.matrix_status}",
        f"dry_run: {report.dry_run}",
        f"selected_plan_count: {report.selected_plan_count}",
        f"successful_plan_count: {report.successful_plan_count}",
        f"failed_plan_count: {report.failed_plan_count}",
        f"checked_run_count: {report.checked_run_count}",
        f"checked_zone_count: {report.checked_zone_count}",
        f"rule_warning_row_count: {report.rule_warning_row_count}",
        f"unmapped_system_node_count: {report.unmapped_system_node_count}",
        "",
        "ISSUES",
        "-" * 80,
    ]
    if not report.issues:
        lines.append("none")
    else:
        for item in report.issues:
            context = " | ".join(
                value
                for value in (
                    f"case={item.case_id}" if item.case_id else "",
                    f"aggregation={item.aggregation_id}" if item.aggregation_id else "",
                    f"run={item.aggregation_run_id}" if item.aggregation_run_id else "",
                    f"path={item.path}" if item.path else "",
                )
                if value
            )
            suffix = f" | {context}" if context else ""
            lines.append(f"[{item.severity.upper()}] {item.code}: {item.message}{suffix}")
    return "\n".join(lines) + "\n"


def _validate_plan_build(
    *,
    manifest: dict[str, Any],
    report: AggregationValidationReport,
    definition: AggregationCampaignDefinition | None,
    selected_rows: list[dict[str, str]],
    canonical_aggregation_root: Path | None,
) -> None:
    root_text = _text(manifest.get("plan_build_root"))
    if not root_text:
        _error(report, "plan_build_root_missing", "Matrix manifest has no plan_build_root")
        return
    root = Path(root_text).expanduser().resolve()
    if not root.is_dir():
        _error(report, "plan_build_root_not_found", "Matrix plan_build_root does not exist", root)
        return
    if canonical_aggregation_root and not _is_relative_to(root, canonical_aggregation_root / "plans"):
        _error(report, "plan_build_outside_aggregation_root", "Plan build root is outside canonical aggregation/plans root", root)

    summary_path = root / "aggregation_plan_build_summary.json"
    summary = _read_json(summary_path, report, required=True)
    _read_csv(root / "aggregation_plan_index.csv", report, required=True)
    missing_rows = _read_csv(root / "missing_plan_inputs.csv", report, required=False)
    if missing_rows:
        _error(report, "plan_build_has_missing_inputs", f"Plan build reports {len(missing_rows)} missing input row(s)", root / "missing_plan_inputs.csv")
    if summary:
        if _text(summary.get("plan_build_id")) and _text(summary.get("plan_build_id")) != root.name:
            _error(report, "plan_build_id_mismatch", "Plan-build summary ID does not match plan-build directory name", summary_path)
        plan_count = _optional_int(summary.get("plan_count"))
        if plan_count is not None and plan_count != len(selected_rows):
            _error(report, "plan_build_count_mismatch", f"Plan-build summary plan_count={plan_count}, selected plans={len(selected_rows)}", summary_path)
        if definition is not None:
            if _text(summary.get("aggregation_campaign_id")) != definition.aggregation_campaign_id:
                _error(report, "plan_build_campaign_id_mismatch", "Plan-build summary aggregation_campaign_id does not match embedded definition", summary_path)
            parent = _text(summary.get("parent_generation_campaign_id") or summary.get("campaign_id"))
            if parent and parent != definition.parent_generation_campaign_id:
                _error(report, "plan_build_parent_id_mismatch", "Plan-build summary parent campaign does not match embedded definition", summary_path)


def _validate_definition_lineage(
    definition: AggregationCampaignDefinition,
    manifest: dict[str, Any],
    report: AggregationValidationReport,
    path: Path,
) -> None:
    if definition.aggregation_campaign_id != report.aggregation_campaign_id:
        _error(report, "definition_campaign_id_mismatch", "Embedded definition aggregation_campaign_id does not match matrix manifest", path)
    if definition.parent_generation_campaign_id != report.parent_generation_campaign_id:
        _error(report, "definition_parent_id_mismatch", "Embedded definition parent_generation_campaign_id does not match matrix manifest", path)
    machine_id = _text(manifest.get("machine_id"))
    if machine_id and definition.machine_id != machine_id:
        _error(report, "definition_machine_id_mismatch", "Embedded definition machine_id does not match matrix manifest", path)


def _validate_matrix_counts(
    *,
    report: AggregationValidationReport,
    manifest: dict[str, Any],
    selected_rows: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    output_rows: list[dict[str, str]],
    missing_generation_rows: list[dict[str, str]],
    policy: AggregationValidationPolicy,
) -> None:
    if len(selected_rows) != report.selected_plan_count:
        _error(report, "selected_plan_count_mismatch", f"Manifest selected_plan_count={report.selected_plan_count}, selected plan rows={len(selected_rows)}")

    successful_rows = [row for row in case_rows if _text(row.get("status")).casefold() in _AGGREGATION_SUCCESS_STATUSES]
    failed_rows = [row for row in case_rows if _text(row.get("status")).casefold() == "failed"]
    planned_rows = [row for row in case_rows if _text(row.get("status")).casefold() == "planned"]

    if len(successful_rows) != report.successful_plan_count:
        _error(report, "successful_plan_count_mismatch", f"Manifest successful_plan_count={report.successful_plan_count}, successful case rows={len(successful_rows)}")
    if len(failed_rows) != report.failed_plan_count:
        _error(report, "failed_plan_count_mismatch", f"Manifest failed_plan_count={report.failed_plan_count}, failed case rows={len(failed_rows)}")

    expected_case_rows = len(planned_rows) if report.dry_run else len(successful_rows) + len(failed_rows)
    if len(case_rows) != expected_case_rows:
        _error(report, "case_row_status_accounting_mismatch", "Matrix case-run rows are not fully accounted for by planned/success/failed statuses")

    if not report.dry_run and len(output_rows) != len(successful_rows):
        _error(report, "output_row_count_mismatch", f"Completed output rows={len(output_rows)} but successful case rows={len(successful_rows)}")

    if policy.require_successful_completion and not report.dry_run:
        if report.matrix_status != "completed":
            _error(report, "matrix_not_completed", f"Matrix status is {report.matrix_status!r}; expected 'completed'")
        if failed_rows:
            _error(report, "matrix_has_failed_runs", f"Matrix contains {len(failed_rows)} failed run(s)")
        if len(successful_rows) != len(selected_rows):
            _error(report, "not_all_selected_plans_completed", "Not every selected plan completed successfully")

    if missing_generation_rows:
        _warning(report, "missing_generation_rows", f"Matrix reports {len(missing_generation_rows)} missing/unavailable Generation row(s)")

    output_keys = {
        (_text(row.get("case_id")), _text(row.get("aggregation_run_id")))
        for row in output_rows
    }
    for row in successful_rows:
        key = (_text(row.get("case_id")), _text(row.get("aggregation_run_id")))
        if key not in output_keys:
            _error(report, "successful_run_missing_output_index", "Successful case-run has no matching aggregation_matrix_outputs row", case_id=key[0], aggregation_run_id=key[1])


def _validate_plan_row(
    *,
    row: dict[str, str],
    report: AggregationValidationReport,
    canonical_aggregation_root: Path | None,
) -> None:
    case_id = _text(row.get("case_id"))
    aggregation_id = _text(row.get("aggregation_id"))
    plan_path = Path(_text(row.get("plan_path"))).expanduser() if _text(row.get("plan_path")) else None
    if plan_path is None:
        _error(report, "plan_path_missing", "Selected plan row has empty plan_path", case_id=case_id, aggregation_id=aggregation_id)
        return
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        _error(report, "plan_file_missing", "Selected aggregation_plan.json does not exist", plan_path, case_id=case_id, aggregation_id=aggregation_id)
        return
    if canonical_aggregation_root and not _is_relative_to(plan_path, canonical_aggregation_root):
        _error(report, "plan_outside_aggregation_root", "Selected plan path is outside canonical campaign aggregation root", plan_path, case_id=case_id, aggregation_id=aggregation_id)

    payload = _read_json(plan_path, report, required=True)
    if not payload:
        return
    for key in ("aggregation_id", "strategy", "rule_set", "weight_mode", "source_case_id", "source_generation_run_id", "aggregate_zones"):
        if key not in payload:
            _error(report, "plan_field_missing", f"Aggregation plan missing required field: {key}", plan_path, case_id=case_id, aggregation_id=aggregation_id)

    comparisons = (
        ("aggregation_id", "aggregation_id"),
        ("strategy", "plan_strategy"),
        ("rule_set", "rule_set"),
        ("weight_mode", "weight_mode"),
        ("source_case_id", "case_id"),
        ("source_generation_run_id", "source_generation_run_id"),
    )
    for plan_key, row_key in comparisons:
        left = _text(payload.get(plan_key))
        right = _text(row.get(row_key))
        if left and right and left != right:
            _error(report, "plan_lineage_mismatch", f"Plan {plan_key}={left!r} does not match selected row {row_key}={right!r}", plan_path, case_id=case_id, aggregation_id=aggregation_id)

    aggregate_zones = payload.get("aggregate_zones")
    if not isinstance(aggregate_zones, list) or not aggregate_zones:
        _error(report, "plan_aggregate_zones_invalid", "Aggregation plan must contain at least one aggregate zone", plan_path, case_id=case_id, aggregation_id=aggregation_id)
        return

    expected_pairs: set[tuple[str, str]] = set()
    seen_source_zones: set[str] = set()
    seen_aggregate_ids: set[str] = set()
    for group in aggregate_zones:
        if not isinstance(group, dict):
            _error(report, "plan_group_invalid", "Aggregation plan aggregate_zones entry is not an object", plan_path, case_id=case_id, aggregation_id=aggregation_id)
            continue
        aggregate_zone_id = _text(group.get("aggregate_zone_id"))
        source_zones = group.get("source_zones")
        if not aggregate_zone_id:
            _error(report, "aggregate_zone_id_missing", "Aggregation plan group has empty aggregate_zone_id", plan_path, case_id=case_id, aggregation_id=aggregation_id)
            continue
        if aggregate_zone_id in seen_aggregate_ids:
            _error(report, "duplicate_aggregate_zone_id", f"Duplicate aggregate_zone_id: {aggregate_zone_id}", plan_path, case_id=case_id, aggregation_id=aggregation_id)
        seen_aggregate_ids.add(aggregate_zone_id)
        if not isinstance(source_zones, list) or not source_zones:
            _error(report, "aggregate_group_empty", f"Aggregate zone {aggregate_zone_id} contains no source zones", plan_path, case_id=case_id, aggregation_id=aggregation_id)
            continue
        for source_zone in source_zones:
            source = _text(source_zone)
            if source in seen_source_zones:
                _error(report, "source_zone_duplicate_assignment", f"Source zone appears in more than one aggregate group: {source}", plan_path, case_id=case_id, aggregation_id=aggregation_id)
            seen_source_zones.add(source)
            expected_pairs.add((aggregate_zone_id, source))

    zone_mapping_path = plan_path.parent / "zone_mapping.csv"
    mapping_rows = _read_csv(zone_mapping_path, report, required=True)
    actual_pairs = {(_text(row.get("aggregate_zone_id")), _text(row.get("source_zone"))) for row in mapping_rows}
    if expected_pairs != actual_pairs:
        _error(report, "zone_mapping_plan_mismatch", "zone_mapping.csv source-zone assignments do not exactly match aggregation_plan.json", zone_mapping_path, case_id=case_id, aggregation_id=aggregation_id)

    expected_aggregate_count = _optional_int(row.get("aggregate_zone_count"))
    expected_source_count = _optional_int(row.get("source_zone_count"))
    if expected_aggregate_count is not None and expected_aggregate_count != len(seen_aggregate_ids):
        _error(report, "aggregate_zone_count_mismatch", f"Selected row aggregate_zone_count={expected_aggregate_count}, plan has {len(seen_aggregate_ids)}", plan_path, case_id=case_id, aggregation_id=aggregation_id)
    if expected_source_count is not None and expected_source_count != len(seen_source_zones):
        _error(report, "source_zone_count_mismatch", f"Selected row source_zone_count={expected_source_count}, plan has {len(seen_source_zones)}", plan_path, case_id=case_id, aggregation_id=aggregation_id)

    missing_inputs = _read_csv(plan_path.parent / "missing_plan_inputs.csv", report, required=False)
    if missing_inputs:
        _error(report, "plan_has_missing_inputs", f"Plan reports {len(missing_inputs)} missing input row(s)", plan_path.parent / "missing_plan_inputs.csv", case_id=case_id, aggregation_id=aggregation_id)


def _validate_completed_run(
    *,
    row: dict[str, str],
    report: AggregationValidationReport,
    canonical_aggregation_root: Path | None,
    definition: AggregationCampaignDefinition | None,
    policy: AggregationValidationPolicy,
) -> None:
    case_id = _text(row.get("case_id"))
    aggregation_id = _text(row.get("aggregation_id"))
    aggregation_run_id = _text(row.get("aggregation_run_id"))
    run_root_text = _text(row.get("run_root"))
    if not run_root_text:
        _error(report, "completed_run_root_missing", "Completed matrix row has empty run_root", case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
        return
    run_root = Path(run_root_text).expanduser().resolve()
    report.checked_run_count += 1
    if not run_root.is_dir():
        _error(report, "run_root_missing", "Completed Aggregation run_root does not exist", run_root, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
        return
    if canonical_aggregation_root and not _is_relative_to(run_root, canonical_aggregation_root / "cases"):
        _error(report, "run_outside_aggregation_root", "Completed run_root is outside canonical campaign aggregation/cases root", run_root, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)

    manifest_path = run_root / "aggregation_manifest.json"
    manifest = _read_json(manifest_path, report, required=True)
    if not manifest:
        return

    expected = {
        "case_id": case_id,
        "source_generation_run_id": _text(row.get("source_generation_run_id")),
        "aggregation_run_id": aggregation_run_id,
        "plan_aggregation_id": aggregation_id,
        "strategy": _text(row.get("loaded_plan_strategy") or row.get("plan_strategy")),
        "rule_set": _text(row.get("loaded_plan_rule_set") or row.get("rule_set")),
        "weight_mode": _text(row.get("loaded_plan_weight_mode") or row.get("weight_mode")),
    }
    for key, expected_value in expected.items():
        actual = _text(manifest.get(key))
        if expected_value and actual != expected_value:
            _error(report, "run_manifest_lineage_mismatch", f"Run manifest {key}={actual!r}, expected {expected_value!r}", manifest_path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
    if _text(manifest.get("status")).casefold() not in _AGGREGATION_SUCCESS_STATUSES:
        _error(report, "run_manifest_not_completed", f"Run manifest status is {_text(manifest.get('status'))!r}", manifest_path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)

    plan_path = Path(_text(manifest.get("plan_path"))).expanduser().resolve() if _text(manifest.get("plan_path")) else None
    selected_plan_path = Path(_text(row.get("plan_path"))).expanduser().resolve() if _text(row.get("plan_path")) else None
    if plan_path and selected_plan_path and plan_path != selected_plan_path:
        _error(report, "run_manifest_plan_path_mismatch", "Run manifest plan_path does not match matrix selected plan_path", manifest_path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)

    for name in _REQUIRED_RUN_INPUTS:
        path = run_root / "inputs" / name
        if not path.is_file():
            _error(report, "run_input_missing", f"Required run provenance artifact is missing: {name}", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)

    _validate_source_generation_provenance(
        run_root / "inputs" / "source_generation_run.json",
        report,
        case_id,
        _text(row.get("source_generation_run_id")),
        aggregation_id,
        aggregation_run_id,
    )

    diagnostics_root = run_root / "diagnostics"
    for name in _REQUIRED_DIAGNOSTICS:
        if not policy.require_shared_node_summaries and name.startswith("system_node_"):
            continue
        path = diagnostics_root / name
        if not path.is_file():
            _error(report, "diagnostic_missing", f"Required diagnostic artifact is missing: {name}", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)

    _validate_persisted_counts(row, manifest, diagnostics_root, report, case_id, aggregation_id, aggregation_run_id)
    _collect_rule_warnings(diagnostics_root, report, case_id, aggregation_id, aggregation_run_id, policy)
    _validate_shared_node_summary(diagnostics_root / "system_node_temperature_summary.csv", report, case_id, aggregation_id, aggregation_run_id, policy)
    _validate_shared_node_summary(diagnostics_root / "system_node_mass_flow_summary.csv", report, case_id, aggregation_id, aggregation_run_id, policy)

    if policy.require_run_outputs:
        zones_root = run_root / "zones"
        zone_dirs = sorted([path for path in zones_root.iterdir() if path.is_dir()]) if zones_root.is_dir() else []
        expected_zone_count = _int(manifest.get("aggregate_zone_count"))
        if len(zone_dirs) != expected_zone_count:
            _error(report, "zone_directory_count_mismatch", f"Run manifest aggregate_zone_count={expected_zone_count}, zone directories={len(zone_dirs)}", zones_root, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
        report.checked_zone_count += len(zone_dirs)
        for zone_dir in zone_dirs:
            for name in _REQUIRED_ZONE_FILES:
                path = zone_dir / name
                if not path.is_file():
                    _error(report, "zone_output_missing", f"Required aggregate-zone artifact is missing: {name}", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
        _validate_long_parquet_row_count(zone_dirs, manifest, report, case_id, aggregation_id, aggregation_run_id)

    require_legacy = policy.require_legacy_pickle
    if require_legacy is None and definition is not None:
        require_legacy = definition.write_legacy_pickle
    if require_legacy:
        legacy = run_root / "legacy" / "Aggregation_Dict_1Zone.pickle"
        if not legacy.is_file():
            _error(report, "legacy_pickle_missing", "Campaign requested legacy pickle but run output is missing it", legacy, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)


def _validate_source_generation_provenance(
    path: Path,
    report: AggregationValidationReport,
    case_id: str,
    source_generation_run_id: str,
    aggregation_id: str,
    aggregation_run_id: str,
) -> None:
    payload = _read_json(path, report, required=True)
    if not payload:
        return
    if _text(payload.get("case_id")) != case_id:
        _error(report, "source_generation_case_mismatch", "source_generation_run.json case_id does not match Aggregation run", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
    if _text(payload.get("run_id")) != source_generation_run_id:
        _error(report, "source_generation_run_mismatch", "source_generation_run.json run_id does not match Aggregation lineage", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
    status = _text(payload.get("status")).casefold()
    if status and status not in GENERATION_SUCCESS_STATUSES:
        _error(
            report,
            "source_generation_not_successful",
            f"Source Generation run status is {status!r}",
            path,
            case_id=case_id,
            aggregation_id=aggregation_id,
            aggregation_run_id=aggregation_run_id,
        )
    elif status == "completed_with_warnings":
        _warning(
            report,
            "source_generation_completed_with_warnings",
            "Source Generation run completed with warnings; this is an accepted upstream success status",
            path,
            case_id=case_id,
            aggregation_id=aggregation_id,
            aggregation_run_id=aggregation_run_id,
        )
    run_root_text = _text(payload.get("run_root"))
    manifest_text = _text(payload.get("manifest_path"))
    if run_root_text and not Path(run_root_text).expanduser().is_dir():
        _error(report, "source_generation_root_missing", "Source Generation run_root recorded in provenance does not exist", run_root_text, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)
    if manifest_text and not Path(manifest_text).expanduser().is_file():
        _error(report, "source_generation_manifest_missing", "Source Generation manifest recorded in provenance does not exist", manifest_text, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=aggregation_run_id)


def _validate_persisted_counts(
    row: dict[str, str],
    manifest: dict[str, Any],
    diagnostics_root: Path,
    report: AggregationValidationReport,
    case_id: str,
    aggregation_id: str,
    run_id: str,
) -> None:
    comparisons = (
        ("aggregate_zone_count", "aggregate_zone_count"),
        ("loaded_variable_count", "loaded_variable_count"),
        ("aggregated_long_rows", "aggregated_long_rows"),
        ("static_equipment_rows", "static_equipment_rows"),
        ("equipment_contribution_rows", "equipment_contribution_rows"),
        ("diagnostic_rows", "diagnostic_rows"),
        ("rule_summary_rows", "rule_summary_rows"),
    )
    for row_key, manifest_key in comparisons:
        row_value = _optional_int(row.get(row_key))
        manifest_value = _optional_int(manifest.get(manifest_key))
        if row_value is not None and manifest_value is not None and row_value != manifest_value:
            _error(report, "matrix_manifest_count_mismatch", f"Matrix {row_key}={row_value}, run manifest {manifest_key}={manifest_value}", diagnostics_root.parent / "aggregation_manifest.json", case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)

    actual_files = (
        ("diagnostic_rows", diagnostics_root / "rule_diagnostics.csv"),
        ("rule_summary_rows", diagnostics_root / "rule_summary.csv"),
        ("equipment_contribution_rows", diagnostics_root / "equipment_contributions.csv"),
    )
    for manifest_key, path in actual_files:
        if path.is_file():
            actual = len(_read_csv_raw(path))
            expected = _optional_int(manifest.get(manifest_key))
            if expected is not None and actual != expected:
                _error(report, "diagnostic_row_count_mismatch", f"Run manifest {manifest_key}={expected}, persisted CSV rows={actual}", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)


def _collect_rule_warnings(
    diagnostics_root: Path,
    report: AggregationValidationReport,
    case_id: str,
    aggregation_id: str,
    run_id: str,
    policy: AggregationValidationPolicy,
) -> None:
    path = diagnostics_root / "rule_summary.csv"
    if not path.is_file():
        return
    rows = _read_csv_raw(path)
    warning_count = sum(1 for row in rows if _text(row.get("status")).casefold() == "warning")
    report.rule_warning_row_count += warning_count
    if warning_count and policy.warn_on_rule_warnings:
        _warning(report, "rule_warning_rows", f"Scientific rule summary contains {warning_count} warning row(s); warnings are preserved and do not fail generic validation", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)


def _validate_shared_node_summary(
    path: Path,
    report: AggregationValidationReport,
    case_id: str,
    aggregation_id: str,
    run_id: str,
    policy: AggregationValidationPolicy,
) -> None:
    if not path.is_file():
        return
    rows = _read_csv_raw(path)
    for row in rows:
        source = _optional_int(row.get("source_key_count"))
        mapped = _optional_int(row.get("mapped_key_count"))
        unmapped = _optional_int(row.get("unmapped_key_count"))
        mapped_rows = _optional_int(row.get("mapped_row_count"))
        skipped_rows = _optional_int(row.get("skipped_row_count"))
        values = (source, mapped, unmapped, mapped_rows, skipped_rows)
        if any(value is not None and value < 0 for value in values):
            _error(report, "shared_node_negative_count", "Shared-node summary contains a negative count", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)
        if source is not None and mapped is not None and unmapped is not None and mapped + unmapped != source:
            _error(report, "shared_node_key_accounting_mismatch", f"mapped_key_count + unmapped_key_count != source_key_count ({mapped}+{unmapped}!={source})", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)
        if unmapped:
            report.unmapped_system_node_count += unmapped
            if policy.warn_on_unmapped_system_nodes:
                _warning(report, "unmapped_system_nodes", f"Shared-node diagnostic reports {unmapped} unmapped key(s); this is a scientific diagnostic, not automatically a validation failure", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)


def _validate_long_parquet_row_count(
    zone_dirs: Iterable[Path],
    manifest: dict[str, Any],
    report: AggregationValidationReport,
    case_id: str,
    aggregation_id: str,
    run_id: str,
) -> None:
    expected = _optional_int(manifest.get("aggregated_long_rows"))
    if expected is None:
        return
    try:
        import pyarrow.parquet as pq
    except Exception:
        _warning(report, "parquet_metadata_unavailable", "pyarrow is unavailable; skipped efficient aggregated_long_rows verification", case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)
        return
    actual = 0
    for zone_dir in zone_dirs:
        path = zone_dir / "aggregated_timeseries_long.parquet"
        if path.is_file():
            try:
                actual += int(pq.ParquetFile(path).metadata.num_rows)
            except Exception as exc:
                _error(report, "parquet_unreadable", f"Could not read Parquet metadata: {exc}", path, case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)
    if actual != expected:
        _error(report, "aggregated_long_row_count_mismatch", f"Run manifest aggregated_long_rows={expected}, zone Parquet row total={actual}", case_id=case_id, aggregation_id=aggregation_id, aggregation_run_id=run_id)


def _resolve_campaign_root(campaign_root: str | Path | None, manifest: dict[str, Any], matrix_dir: Path) -> Path | None:
    if campaign_root is not None:
        return Path(campaign_root).expanduser().resolve()
    from_manifest = _text(manifest.get("campaign_root"))
    if from_manifest:
        return Path(from_manifest).expanduser().resolve()
    # Canonical fallback: <campaign>/aggregation/matrix_runs/<matrix>
    try:
        if matrix_dir.parent.name == "matrix_runs" and matrix_dir.parent.parent.name == "aggregation":
            return matrix_dir.parent.parent.parent.resolve()
    except Exception:
        pass
    return None


def _load_definition(path: Path, report: AggregationValidationReport) -> AggregationCampaignDefinition | None:
    if not path.is_file():
        _error(report, "campaign_definition_missing", "Embedded Aggregation campaign definition is missing", path)
        return None
    try:
        return load_aggregation_campaign_definition(path)
    except Exception as exc:
        _error(report, "campaign_definition_invalid", f"Embedded Aggregation campaign definition is invalid: {exc}", path)
        return None


def _read_json(path: Path, report: AggregationValidationReport, *, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            _error(report, "json_missing", "Required JSON file is missing", path)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        _error(report, "json_unreadable", f"Could not parse JSON: {exc}", path)
        return {}
    if not isinstance(payload, dict):
        _error(report, "json_not_object", "Expected JSON object", path)
        return {}
    return payload


def _read_csv(path: Path, report: AggregationValidationReport, *, required: bool) -> list[dict[str, str]]:
    if not path.is_file():
        if required:
            _error(report, "csv_missing", "Required CSV file is missing", path)
        return []
    try:
        return _read_csv_raw(path)
    except Exception as exc:
        _error(report, "csv_unreadable", f"Could not parse CSV: {exc}", path)
        return []


def _read_csv_raw(path: Path) -> list[dict[str, str]]:
    """Read CSV rows while honoring ScaleBridge's canonical empty-row sentinel.

    ``scalebridge.data.aggregation.writers.write_csv`` intentionally writes
    an empty logical row set as::

        note
        no rows

    That two-line CSV is an artifact-format sentinel, not a scientific/data
    record. Normalize only that exact one-row sentinel to ``[]``; all other
    CSV rows, including arbitrary ``note`` columns, remain untouched.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) == 1:
        normalized = {
            str(key).strip().casefold(): str(value).strip().casefold()
            for key, value in rows[0].items()
            if key is not None
        }
        if normalized == {"note": "no rows"}:
            return []

    return rows


def _error(
    report: AggregationValidationReport,
    code: str,
    message: str,
    path: str | Path = "",
    *,
    case_id: str = "",
    aggregation_id: str = "",
    aggregation_run_id: str = "",
) -> None:
    report.issues.append(ValidationIssue("error", code, message, str(path) if path else "", case_id, aggregation_id, aggregation_run_id))


def _warning(
    report: AggregationValidationReport,
    code: str,
    message: str,
    path: str | Path = "",
    *,
    case_id: str = "",
    aggregation_id: str = "",
    aggregation_run_id: str = "",
) -> None:
    report.issues.append(ValidationIssue("warning", code, message, str(path) if path else "", case_id, aggregation_id, aggregation_run_id))


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int(value: Any) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def _optional_int(value: Any) -> int | None:
    text = _text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _normalized_path_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
