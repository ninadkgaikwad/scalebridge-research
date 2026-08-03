#!/usr/bin/env python
"""Canonical-aware compatibility validator for ScaleBridge C2 feature outputs.

The existing validator is still run with its original arguments. If it fails
only because it recomputes the pre-canonical row count, this wrapper verifies
that the difference is exactly explained by documented parsed-timestamp
deduplication.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROW_COUNT_PATTERN = re.compile(
    r"row-count mismatch:\s*stored=(?P<stored>\d+),\s*recomputed=(?P<recomputed>\d+)",
    re.IGNORECASE,
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--campaign-root", type=Path, default=None)
    parser.add_argument("--generated-data-root", type=Path, default=None)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--audit-run-id", required=True)
    parser.add_argument("--feature-run-id", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--aggregation-id", default=None)
    parser.add_argument("--weight-mode", default=None)
    parser.add_argument("--aggregate-zone-id", default=None)
    parser.add_argument("--minimum-sample-count", type=int, default=1000)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--expected-canonical-row-count", type=int, default=105120)
    args = parser.parse_args()

    feature_root = args.feature_root.expanduser().resolve()
    if not feature_root.is_dir():
        raise FileNotFoundError(f"Feature root does not exist: {feature_root}")

    legacy_script = (
        Path(__file__).resolve().parent
        / "validate_heat_input_regression_features.py"
    )
    if not legacy_script.is_file():
        raise FileNotFoundError(f"Legacy validator not found: {legacy_script}")

    # Forward the complete provenance and validation configuration required by
    # the legacy deterministic validator. --feature-root is wrapper-only because
    # the legacy validator reconstructs it from campaign_root + feature_run_id.
    legacy_command = [
        sys.executable,
        str(legacy_script),
        "--matrix-run-id",
        args.matrix_run_id,
        "--audit-run-id",
        args.audit_run_id,
        "--feature-run-id",
        args.feature_run_id,
        "--minimum-sample-count",
        str(args.minimum_sample_count),
        "--absolute-tolerance",
        str(args.absolute_tolerance),
        "--relative-tolerance",
        str(args.relative_tolerance),
    ]

    if args.campaign_id:
        legacy_command.extend(["--campaign-id", args.campaign_id])
    if args.campaign_root is not None:
        legacy_command.extend(["--campaign-root", str(args.campaign_root)])
    if args.generated_data_root is not None:
        legacy_command.extend(
            ["--generated-data-root", str(args.generated_data_root)]
        )
    if args.case_id:
        legacy_command.extend(["--case-id", args.case_id])
    if args.aggregation_id:
        legacy_command.extend(["--aggregation-id", args.aggregation_id])
    if args.weight_mode:
        legacy_command.extend(["--weight-mode", args.weight_mode])
    if args.aggregate_zone_id:
        legacy_command.extend(
            ["--aggregate-zone-id", args.aggregate_zone_id]
        )

    print("=" * 100)
    print("SCALEBRIDGE C2 STANDARD + CANONICAL-AWARE VALIDATION")
    print("=" * 100)
    print(f"feature_root: {feature_root}")
    print("Running legacy standard validator with original arguments...")
    print()

    legacy_result = subprocess.run(legacy_command, check=False)
    if legacy_result.returncode == 0:
        print("Legacy standard validation passed.")
        return 0

    results_path = feature_root / "feature_validation_results.csv"
    if not results_path.is_file():
        print(
            f"ERROR: Missing legacy validation results: {results_path}",
            file=sys.stderr,
        )
        return 1

    results = pd.read_csv(results_path)
    status_col = (
        "validation_status"
        if "validation_status" in results.columns
        else "status"
        if "status" in results.columns
        else None
    )
    if status_col is None:
        print("ERROR: No validation status column found.", file=sys.stderr)
        return 1

    failed = results[
        results[status_col].astype(str).str.lower().isin(
            {"failed", "fail", "false", "invalid", "error"}
        )
    ].copy()

    if failed.empty:
        print(
            "ERROR: Legacy validator returned nonzero without failed result rows.",
            file=sys.stderr,
        )
        return 1

    audit_rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []

    for _, row in failed.iterrows():
        zone_id = str(row.get("aggregate_zone_id", ""))
        reason = str(row.get("reason", "")).strip()
        match = ROW_COUNT_PATTERN.fullmatch(reason)

        if match is None:
            hard_failures.append(f"{zone_id}: {reason}")
            continue

        stored = int(match.group("stored"))
        recomputed = int(match.group("recomputed"))
        feature_path = Path(str(row.get("feature_path", "")))
        manifest_path = feature_path.parent / "zone_feature_manifest.json"

        if not manifest_path.is_file():
            hard_failures.append(f"{zone_id}: missing manifest {manifest_path}")
            continue

        manifest = _load_json(manifest_path)
        canonical = manifest.get("canonical_timestamp_metadata", {})

        checks = {
            "stored_matches_manifest":
                stored == int(manifest.get("row_count", -1)),
            "stored_matches_canonical_output":
                stored == int(canonical.get("output_row_count", -1)),
            "recomputed_matches_canonical_input":
                recomputed == int(canonical.get("input_row_count", -1)),
            "removed_count_matches_difference":
                int(canonical.get("removed_duplicate_row_count", -1))
                == recomputed - stored,
            "no_remaining_parsed_duplicates":
                int(canonical.get(
                    "output_duplicate_parsed_timestamp_count", -1
                )) == 0,
            "timestamps_monotonic":
                _as_bool(canonical.get("output_timestamp_monotonic", False)),
            "cadence_300_seconds":
                float(canonical.get("canonical_cadence_seconds", -1)) == 300.0,
            "zero_noncanonical_gaps":
                int(canonical.get("noncanonical_gap_count", -1)) == 0,
            "expected_annual_row_count":
                stored == args.expected_canonical_row_count,
        }

        failed_checks = [name for name, ok in checks.items() if not ok]
        accepted = not failed_checks

        audit_rows.append({
            "aggregate_zone_id": zone_id,
            "legacy_reason": reason,
            "stored_row_count": stored,
            "recomputed_precanonical_row_count": recomputed,
            "removed_duplicate_row_count":
                canonical.get("removed_duplicate_row_count"),
            "canonical_aware_status": "passed" if accepted else "failed",
            "failed_canonical_checks": "; ".join(failed_checks),
            "zone_feature_manifest": str(manifest_path),
            "feature_path": str(feature_path),
        })

        if not accepted:
            hard_failures.append(
                f"{zone_id}: failed canonical checks: {', '.join(failed_checks)}"
            )

    audit = pd.DataFrame(audit_rows)
    results_out = feature_root / "canonical_aware_feature_validation_results.csv"
    audit.to_csv(results_out, index=False)

    passed_count = int(
        (audit.get("canonical_aware_status", pd.Series(dtype=str)) == "passed").sum()
    )
    failed_count = int(
        (audit.get("canonical_aware_status", pd.Series(dtype=str)) == "failed").sum()
    )

    manifest_out = feature_root / "canonical_aware_feature_validation_manifest.json"
    manifest_out.write_text(
        json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_root": str(feature_root),
            "legacy_validator_return_code": legacy_result.returncode,
            "legacy_failed_zone_count": int(len(failed)),
            "canonical_aware_passed_zone_count": passed_count,
            "canonical_aware_failed_zone_count": failed_count,
            "status": "passed" if not hard_failures else "failed",
            "hard_failures": hard_failures,
            "results": str(results_out),
        }, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("CANONICAL-AWARE VALIDATION SUMMARY")
    print("=" * 100)
    print(f"legacy_failed_zone_count: {len(failed)}")
    print(f"canonical_aware_passed_zone_count: {passed_count}")
    print(f"canonical_aware_failed_zone_count: {failed_count}")
    print(f"results: {results_out}")
    print(f"manifest: {manifest_out}")

    if hard_failures:
        for failure in hard_failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("validation_status: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
