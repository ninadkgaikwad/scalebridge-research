#!/usr/bin/env python
"""Validate C2 timestamp coalescence metadata and dense derived features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--expected-row-count", type=int, default=105120)
    parser.add_argument(
        "--fail-on-conflicting-source-values",
        action="store_true",
    )
    args = parser.parse_args()

    feature_root = Path(args.feature_root).expanduser().resolve()
    manifests = sorted(feature_root.rglob("zone_feature_manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No zone_feature_manifest.json files found under {feature_root}"
        )

    rows = []
    failed = 0

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("canonical_timestamp_metadata", {})
        parquet_path = Path(
            manifest["outputs"]["derived_features_parquet"]
        )
        frame = pd.read_parquet(parquet_path)
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")

        source_columns = [
            c for c in frame.columns
            if c not in {
                "timestamp_raw",
                "timestamp",
                "case_id",
                "aggregation_id",
                "aggregation_run_id",
                "aggregate_zone_id",
                "weight_mode",
            }
        ]
        missing_source_value_count = int(
            frame[source_columns].isna().sum().sum()
        )

        checks = {
            "row_count": len(frame) == args.expected_row_count,
            "parsed": timestamp.notna().all(),
            "unique": not timestamp.duplicated().any(),
            "monotonic": timestamp.is_monotonic_increasing,
            "cadence": metadata.get("canonical_cadence_seconds") == 300.0,
            "no_gaps": metadata.get("noncanonical_gap_count") == 0,
            "dense_features": missing_source_value_count == 0,
            "coalescence_policy": metadata.get(
                "duplicate_resolution_policy"
            ) == "column_wise_coalescence_prefer_densest_then_earliest",
        }

        if args.fail_on_conflicting_source_values:
            checks["no_conflicts"] = (
                int(metadata.get("conflicting_source_value_count", 0)) == 0
            )

        status = "passed" if all(checks.values()) else "failed"
        failed += status == "failed"
        rows.append({
            "aggregate_zone_id": manifest.get("aggregate_zone_id", ""),
            "row_count": len(frame),
            "missing_source_value_count": missing_source_value_count,
            "coalesced_source_value_count": metadata.get(
                "coalesced_source_value_count", 0
            ),
            "complementary_duplicate_timestamp_count": metadata.get(
                "complementary_duplicate_timestamp_count", 0
            ),
            "conflicting_source_value_count": metadata.get(
                "conflicting_source_value_count", 0
            ),
            "status": status,
            "failed_checks": ";".join(
                name for name, ok in checks.items() if not ok
            ),
            "manifest": str(manifest_path),
        })

    output = feature_root / "timestamp_coalescence_validation.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    print("=" * 100)
    print("C2 TIMESTAMP COALESCENCE VALIDATION")
    print("=" * 100)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"passed_zone_count: {len(rows) - failed}")
    print(f"failed_zone_count: {failed}")
    print(f"output: {output}")
    print(f"validation_status: {'passed' if failed == 0 else 'failed'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
