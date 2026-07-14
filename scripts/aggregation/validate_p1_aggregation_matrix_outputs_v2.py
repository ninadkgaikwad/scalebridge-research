# -*- coding: utf-8 -*-
"""Validate a completed ScaleBridge P1 aggregation matrix run.

Version 2 accepts both normalized and legacy trailing-underscore node
output column names and uses per-run aggregation manifests as fallback
for aggregation level / weight-mode coverage when matrix CSV fields are blank.

Read-only post-run audit for the P1 4-building x 4-climate aggregation matrix.
It verifies matrix summary files, all completed rows, per-run manifests,
diagnostics, zone outputs, legacy pickles, and shared node-mapping diagnostics.

Typical use from repo root:

    python scripts\aggregation\validate_p1_aggregation_matrix_outputs.py `
      --campaign-id p1_compact_4b4c_labpc_1w_v1 `
      --matrix-run-id aggregation_matrix_20260712_215839 `
      --expect-plan-count 240 `
      --expect-legacy-pickle

If --matrix-run-id is omitted, the latest aggregation_matrix_* folder is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CAMPAIGN_ID = "p1_compact_4b4c_labpc_1w_v1"
EXPECTED_AGGREGATION_IDS = (
    "p1_l01_all_to_one",
    "p1_l02_functional",
    "p1_l03_intermediate",
    "p1_l04_spatial_detailed",
    "p1_l05_identity",
)
EXPECTED_WEIGHT_MODES = ("equal", "floor_area", "volume")
EXPECTED_BUILDINGS = (
    "ApartmentMidRise",
    "OfficeSmall",
    "RestaurantFastFood",
    "RetailStripmall",
)
EXPECTED_WEATHERS = ("Buffalo", "Seattle", "Tampa", "Tucson")
# Accept both the normalized special-output names and the older legacy-v1
# trailing-underscore convention used for System Node Temperature in the
# shared-node aggregation patch.
REQUIRED_NODE_OUTPUT_COLUMN_ALIASES = {
    "System Node Temperature": (
        "System_Node_Temperature",
        "System_Node_Temperature_",
    ),
    "System Node Mass Flow Rate": (
        "System_Node_Mass_Flow_Rate",
        "System_Node_Mass_Flow_Rate_",
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = resolve_repo_root()
    campaign_root = resolve_campaign_root(
        repo_root=repo_root,
        campaign_id=args.campaign_id,
        campaign_root=args.campaign_root,
        generated_data_root=args.generated_data_root,
    )
    matrix_run_dir = resolve_matrix_run_dir(campaign_root, args.matrix_run_id)
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else repo_root / f"p1_aggregation_validation_{matrix_run_dir.name}.txt"
    )

    status, error_count, warning_count = audit(
        repo_root=repo_root,
        campaign_id=args.campaign_id,
        campaign_root=campaign_root,
        matrix_run_dir=matrix_run_dir,
        report_path=report_path,
        expect_plan_count=args.expect_plan_count,
        expect_legacy_pickle=args.expect_legacy_pickle,
        sample_zone_outputs=args.sample_zone_outputs,
    )

    print("=" * 100)
    print("P1 AGGREGATION MATRIX VALIDATION")
    print("=" * 100)
    print(f"status: {status}")
    print(f"error_count: {error_count}")
    print(f"warning_count: {warning_count}")
    print(f"report_path: {report_path}")
    return 0 if status == "PASS" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--matrix-run-id", default=None)
    parser.add_argument("--expect-plan-count", type=int, default=240)
    parser.add_argument("--expect-legacy-pickle", action="store_true")
    parser.add_argument("--sample-zone-outputs", type=int, default=50)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args(argv)


def audit(
    *,
    repo_root: Path,
    campaign_id: str,
    campaign_root: Path,
    matrix_run_dir: Path,
    report_path: Path,
    expect_plan_count: int,
    expect_legacy_pickle: bool,
    sample_zone_outputs: int,
) -> tuple[str, int, int]:
    errors: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    def warn(msg: str) -> None:
        warnings.append(msg)

    lines += [
        "SCALEBRIDGE P1 AGGREGATION MATRIX VALIDATION REPORT",
        "=" * 80,
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"repo_root: {repo_root}",
        f"campaign_id: {campaign_id}",
        f"campaign_root: {campaign_root}",
        f"matrix_run_dir: {matrix_run_dir}",
        f"expect_plan_count: {expect_plan_count}",
        f"expect_legacy_pickle: {expect_legacy_pickle}",
        "",
    ]

    matrix_manifest_path = matrix_run_dir / "aggregation_matrix_manifest.json"
    case_runs_path = matrix_run_dir / "aggregation_matrix_case_runs.csv"
    outputs_path = matrix_run_dir / "aggregation_matrix_outputs.csv"
    failures_path = matrix_run_dir / "aggregation_matrix_failures.csv"

    lines += ["1. MATRIX-LEVEL FILES", "-" * 80]
    for label, path in (
        ("aggregation_matrix_manifest", matrix_manifest_path),
        ("aggregation_matrix_case_runs", case_runs_path),
        ("aggregation_matrix_outputs", outputs_path),
    ):
        lines.append(f"{label}: {'FOUND' if path.is_file() else 'MISSING'} | {path}")
        if not path.is_file():
            err(f"missing required matrix file: {path}")
    lines.append(f"aggregation_matrix_failures: {'FOUND' if failures_path.is_file() else 'not present'} | {failures_path}")

    manifest = load_json(matrix_manifest_path)
    case_rows = read_csv(case_runs_path)
    output_rows = read_csv(outputs_path)
    failure_rows = read_csv(failures_path)

    lines += ["", "2. MATRIX SUMMARY", "-" * 80]
    selected = int_or_none(manifest.get("selected_plan_count"))
    success = int_or_none(manifest.get("successful_plan_count"))
    failed = int_or_none(manifest.get("failed_plan_count"))
    lines += [
        f"manifest.selected_plan_count: {selected}",
        f"manifest.successful_plan_count: {success}",
        f"manifest.failed_plan_count: {failed}",
        f"case_run_rows: {len(case_rows)}",
        f"output_rows: {len(output_rows)}",
        f"failure_rows: {len(failure_rows)}",
    ]

    if selected is not None and selected != expect_plan_count:
        err(f"selected_plan_count {selected} != expected {expect_plan_count}")
    if success is not None and success != expect_plan_count:
        err(f"successful_plan_count {success} != expected {expect_plan_count}")
    if failed is not None and failed != 0:
        err(f"failed_plan_count is not zero: {failed}")
    if len(case_rows) != expect_plan_count:
        err(f"case_run_rows {len(case_rows)} != expected {expect_plan_count}")
    if failure_rows:
        err(f"failure file contains rows: {len(failure_rows)}")

    status_counts = Counter(str(row.get("status", "")).strip() for row in case_rows)
    lines.append(f"status_counts: {dict(status_counts)}")
    if any(str(row.get("status", "")).strip().casefold() != "completed" for row in case_rows):
        err("one or more matrix rows are not completed")

    lines += ["", "3. COVERAGE", "-" * 80]
    # Matrix CSVs from older runner versions may leave plan_aggregation_id and
    # plan_weight_mode blank. Use per-run aggregation_manifest.json as the
    # authoritative fallback for coverage.
    enriched_rows = []
    for row in case_rows:
        enriched = dict(row)
        run_root_text = str(row.get("run_root", "")).strip()
        manifest = load_json(Path(run_root_text) / "aggregation_manifest.json") if run_root_text else {}
        if not str(enriched.get("plan_aggregation_id", "")).strip():
            enriched["plan_aggregation_id"] = str(manifest.get("plan_aggregation_id", ""))
        if not str(enriched.get("plan_weight_mode", "")).strip():
            enriched["plan_weight_mode"] = str(manifest.get("weight_mode", ""))
        enriched_rows.append(enriched)

    building_counts = Counter(str(row.get("building_type", "")) for row in enriched_rows)
    weather_counts = Counter(str(row.get("weather_location", "")) for row in enriched_rows)
    aggregation_counts = Counter(str(row.get("plan_aggregation_id", "")) for row in enriched_rows)
    weight_counts = Counter(str(row.get("plan_weight_mode", "")) for row in enriched_rows)
    lines += [
        f"building_counts: {dict(sorted(building_counts.items()))}",
        f"weather_counts: {dict(sorted(weather_counts.items()))}",
        f"aggregation_id_counts: {dict(sorted(aggregation_counts.items()))}",
        f"weight_mode_counts: {dict(sorted(weight_counts.items()))}",
    ]
    for item in EXPECTED_BUILDINGS:
        if building_counts.get(item, 0) == 0:
            warn(f"expected building missing from matrix rows: {item}")
    for item in EXPECTED_WEATHERS:
        if weather_counts.get(item, 0) == 0:
            warn(f"expected weather missing from matrix rows: {item}")
    for item in EXPECTED_AGGREGATION_IDS:
        if aggregation_counts.get(item, 0) == 0:
            warn(f"expected aggregation_id missing from matrix rows: {item}")
    for item in EXPECTED_WEIGHT_MODES:
        if weight_counts.get(item, 0) == 0:
            warn(f"expected weight_mode missing from matrix rows: {item}")

    combo_counts = Counter(
        (
            str(row.get("case_id", "")),
            str(row.get("plan_aggregation_id", "")),
            str(row.get("plan_weight_mode", "")),
        )
        for row in enriched_rows
    )
    duplicates = {combo: count for combo, count in combo_counts.items() if count > 1}
    lines.append(f"unique_case_level_weight_combos: {len(combo_counts)}")
    if duplicates:
        warn(f"duplicate case/level/weight rows found: {duplicates}")

    lines += ["", "4. PER-RUN OUTPUT CHECKS", "-" * 80]
    run_roots = [Path(str(row.get("run_root", ""))) for row in case_rows if str(row.get("run_root", "")).strip()]
    missing_roots = 0
    missing_manifests = 0
    manifest_status_counts = Counter()
    aggregate_zone_counts = Counter()
    missing_diagnostics = Counter()
    missing_zone_files = Counter()
    missing_pickles = 0

    required_diag = (
        "loaded_variables.csv",
        "rule_summary.csv",
        "rule_diagnostics.csv",
        "schedule_equipment_mapping_used.csv",
        "equipment_contributions.csv",
        "system_node_temperature_summary.csv",
        "system_node_temperature_mapping.csv",
        "system_node_temperature_unmapped_nodes.csv",
        "system_node_mass_flow_summary.csv",
        "system_node_mass_flow_mapping.csv",
        "system_node_mass_flow_unmapped_nodes.csv",
    )
    required_zone = (
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

    for root in run_roots:
        if not root.is_dir():
            missing_roots += 1
            continue
        run_manifest = root / "aggregation_manifest.json"
        if not run_manifest.is_file():
            missing_manifests += 1
        else:
            payload = load_json(run_manifest)
            manifest_status_counts[str(payload.get("status", ""))] += 1
            aggregate_zone_counts[str(payload.get("aggregate_zone_count", ""))] += 1
        diag_root = root / "diagnostics"
        for name in required_diag:
            if not (diag_root / name).is_file():
                missing_diagnostics[name] += 1
        zones_root = root / "zones"
        zone_dirs = [p for p in zones_root.iterdir() if p.is_dir()] if zones_root.is_dir() else []
        if not zone_dirs:
            missing_zone_files["zone_directory"] += 1
        for zone_dir in zone_dirs:
            for name in required_zone:
                if not (zone_dir / name).is_file():
                    missing_zone_files[name] += 1
        if expect_legacy_pickle and not (root / "legacy" / "Aggregation_Dict_1Zone.pickle").is_file():
            missing_pickles += 1

    lines += [
        f"run_roots_seen: {len(run_roots)}",
        f"missing_run_roots: {missing_roots}",
        f"missing_manifests: {missing_manifests}",
        f"manifest_status_counts: {dict(manifest_status_counts)}",
        f"aggregate_zone_counts: {dict(sorted(aggregate_zone_counts.items()))}",
        f"missing_diagnostics: {dict(missing_diagnostics)}",
        f"missing_zone_files: {dict(missing_zone_files)}",
        f"missing_legacy_pickles: {missing_pickles}",
    ]
    if missing_roots:
        err(f"missing run roots: {missing_roots}")
    if missing_manifests:
        err(f"missing aggregation manifests: {missing_manifests}")
    if missing_zone_files:
        err(f"required zone output files missing: {dict(missing_zone_files)}")
    if expect_legacy_pickle and missing_pickles:
        err(f"legacy pickle missing in {missing_pickles} runs")
    if missing_diagnostics:
        warn(f"some diagnostic files missing: {dict(missing_diagnostics)}")

    lines += ["", "5. SHARED NODE-MAPPING CHECKS", "-" * 80]
    for prefix, title in (
        ("system_node_temperature", "System Node Temperature"),
        ("system_node_mass_flow", "System Node Mass Flow Rate"),
    ):
        summary = summarize_node(run_roots, prefix)
        lines += [
            f"{title}:",
            f"  summary_file_count: {summary['summary_file_count']}",
            f"  mapping_file_count: {summary['mapping_file_count']}",
            f"  unmapped_file_count: {summary['unmapped_file_count']}",
            f"  total_source_key_count: {summary['total_source_key_count']}",
            f"  total_mapped_key_count: {summary['total_mapped_key_count']}",
            f"  total_unmapped_key_count: {summary['total_unmapped_key_count']}",
            f"  total_mapped_row_count: {summary['total_mapped_row_count']}",
            f"  total_skipped_row_count: {summary['total_skipped_row_count']}",
            f"  runs_with_zero_mapped_keys: {summary['runs_with_zero_mapped_keys']}",
            f"  matched_suffix_counts: {summary['matched_suffix_counts']}",
        ]
        if summary["summary_file_count"] != len(run_roots):
            warn(f"{title} summary files {summary['summary_file_count']} != runs {len(run_roots)}")
        if summary["runs_with_zero_mapped_keys"]:
            err(f"{title} has runs with zero mapped keys: {summary['runs_with_zero_mapped_keys']}")
        if summary["total_mapped_key_count"] <= 0:
            err(f"{title} has no mapped keys")

    lines += ["", "6. WIDE PARQUET SAMPLE COLUMN CHECK", "-" * 80]
    sample_errors, sample_warnings, sample_lines = inspect_parquet_samples(run_roots, sample_zone_outputs)
    lines += sample_lines
    for msg in sample_errors:
        err(msg)
    for msg in sample_warnings:
        warn(msg)

    status = "PASS" if not errors else "FAIL"
    lines += ["", "7. RESULT", "-" * 80, f"status: {status}", f"error_count: {len(errors)}", f"warning_count: {len(warnings)}"]
    if errors:
        lines += ["", "ERRORS", "-" * 80]
        lines += [f"{i}. {m}" for i, m in enumerate(errors, 1)]
    if warnings:
        lines += ["", "WARNINGS", "-" * 80]
        lines += [f"{i}. {m}" for i, m in enumerate(warnings, 1)]
    lines += ["", "END OF REPORT"]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status, len(errors), len(warnings)


def summarize_node(run_roots: list[Path], prefix: str) -> dict[str, Any]:
    summary_file_count = 0
    mapping_file_count = 0
    unmapped_file_count = 0
    total_source_key_count = 0
    total_mapped_key_count = 0
    total_unmapped_key_count = 0
    total_mapped_row_count = 0
    total_skipped_row_count = 0
    runs_with_zero_mapped_keys = 0
    suffix_counts = Counter()
    for root in run_roots:
        diag = root / "diagnostics"
        summary_path = diag / f"{prefix}_summary.csv"
        mapping_path = diag / f"{prefix}_mapping.csv"
        unmapped_path = diag / f"{prefix}_unmapped_nodes.csv"
        current_mapped = 0
        if summary_path.is_file():
            summary_file_count += 1
            for row in read_csv(summary_path):
                total_source_key_count += int_or_zero(row.get("source_key_count"))
                mapped = int_or_zero(row.get("mapped_key_count"))
                total_mapped_key_count += mapped
                current_mapped += mapped
                total_unmapped_key_count += int_or_zero(row.get("unmapped_key_count"))
                total_mapped_row_count += int_or_zero(row.get("mapped_row_count"))
                total_skipped_row_count += int_or_zero(row.get("skipped_row_count"))
        if summary_path.is_file() and current_mapped == 0:
            runs_with_zero_mapped_keys += 1
        if mapping_path.is_file():
            mapping_file_count += 1
            for row in read_csv(mapping_path):
                if str(row.get("match_status", "")).casefold() == "mapped":
                    suffix = str(row.get("matched_suffix_pattern", "")).strip()
                    if suffix:
                        suffix_counts[suffix] += 1
        if unmapped_path.is_file():
            unmapped_file_count += 1
    return {
        "summary_file_count": summary_file_count,
        "mapping_file_count": mapping_file_count,
        "unmapped_file_count": unmapped_file_count,
        "total_source_key_count": total_source_key_count,
        "total_mapped_key_count": total_mapped_key_count,
        "total_unmapped_key_count": total_unmapped_key_count,
        "total_mapped_row_count": total_mapped_row_count,
        "total_skipped_row_count": total_skipped_row_count,
        "runs_with_zero_mapped_keys": runs_with_zero_mapped_keys,
        "matched_suffix_counts": dict(sorted(suffix_counts.items())),
    }


def inspect_parquet_samples(run_roots: list[Path], sample_count: int) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []
    if sample_count <= 0:
        return errors, warnings, ["sample inspection skipped"]
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        return errors, [f"pyarrow not available for sample inspection: {exc}"], ["sample inspection skipped"]
    paths: list[Path] = []
    for root in run_roots:
        paths.extend(sorted((root / "zones").glob("*/aggregated_timeseries_wide.parquet")))
    sampled = sorted(paths)[:sample_count]
    missing_counts = Counter()
    lines += [f"wide_parquet_total_found: {len(paths)}", f"wide_parquet_sample_count: {len(sampled)}"]
    for path in sampled:
        names = set(pq.ParquetFile(path).schema.names)
        for logical_name, aliases in REQUIRED_NODE_OUTPUT_COLUMN_ALIASES.items():
            if not any(alias in names for alias in aliases):
                missing_counts[logical_name] += 1
    lines.append(f"accepted_node_column_aliases: {REQUIRED_NODE_OUTPUT_COLUMN_ALIASES}")
    lines.append(f"missing_required_node_columns_in_sample: {dict(missing_counts)}")
    if missing_counts:
        errors.append(f"sampled wide parquets missing node columns: {dict(missing_counts)}")
    return errors, warnings, lines


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src" / "scalebridge").is_dir():
            return candidate
    return cwd


def resolve_campaign_root(repo_root: Path, campaign_id: str, campaign_root: str | None, generated_data_root: str | None) -> Path:
    if campaign_root:
        return Path(campaign_root).expanduser().resolve()
    if generated_data_root:
        root = Path(generated_data_root).expanduser().resolve()
    else:
        import os
        env_value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT", "").strip()
        root = Path(env_value).expanduser().resolve() if env_value else (repo_root / ".." / ".." / "Data" / "ScaleBridge").resolve()
    return root / "campaigns" / campaign_id


def resolve_matrix_run_dir(campaign_root: Path, matrix_run_id: str | None) -> Path:
    root = campaign_root / "aggregation" / "matrix_runs"
    if matrix_run_id:
        out = root / matrix_run_id
        if not out.is_dir():
            raise SystemExit(f"matrix run folder does not exist: {out}")
        return out
    candidates = sorted([p for p in root.glob("aggregation_matrix_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"no aggregation_matrix_* folders found under {root}")
    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def int_or_zero(value: Any) -> int:
    value = int_or_none(value)
    return value if value is not None else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
