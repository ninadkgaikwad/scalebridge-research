"""
Build EIO-only zone feature tables for ScaleBridge aggregation suggestions.

Example:
    python scripts/aggregation/build_zone_features.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3

Optional single-case run:
    python scripts/aggregation/build_zone_features.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3 ^
      --case-id epcase_827ca4812c0199221d031e59
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.data.aggregation.zone_features import (
    build_zone_features_from_eio,
    find_latest_variable_manifest_path,
    timestamp_id,
    write_zone_feature_case_outputs,
)
from scalebridge.data.aggregation.zone_inventory import (
    find_case_dirs,
    find_latest_eio_tables_path,
    resolve_campaign_root,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EIO-only zone feature tables for aggregation suggestions."
    )

    parser.add_argument(
        "--campaign-id",
        required=True,
        help="ScaleBridge campaign ID.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional single case_id. If omitted, all campaign cases are processed.",
    )
    parser.add_argument(
        "--generated-data-root",
        default=None,
        help=(
            "Optional generated data root. If omitted, uses "
            "SCALEBRIDGE_GENERATED_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help="Optional explicit campaign root. Overrides --generated-data-root.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional output root. If omitted, writes under "
            "<campaign_root>/aggregation/zone_features/zone_features_<timestamp>."
        ),
    )

    return parser.parse_args()


def write_campaign_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "status",
        "eio_tables_path",
        "variable_manifest_path",
        "case_output_dir",
        "zone_feature_count",
        "approved_zone_count",
        "excluded_zone_count",
        "default_max_aggregate_zones",
        "approved_zones",
        "excluded_zones",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    campaign_root = resolve_campaign_root(
        campaign_id=args.campaign_id,
        generated_data_root=args.generated_data_root,
        campaign_root=args.campaign_root,
    )

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = (
            campaign_root
            / "aggregation"
            / "zone_features"
            / timestamp_id("zone_features")
        )

    cases_output_root = output_root / "cases"
    cases_output_root.mkdir(parents=True, exist_ok=True)

    case_dirs = find_case_dirs(campaign_root, case_id=args.case_id)

    campaign_rows: list[dict[str, Any]] = []
    success_count = 0
    failed_count = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        case_output_dir = cases_output_root / case_id

        try:
            eio_tables_path = find_latest_eio_tables_path(case_dir)
            variable_manifest_path = find_latest_variable_manifest_path(case_dir)

            result = build_zone_features_from_eio(
                case_id=case_id,
                eio_tables_path=eio_tables_path,
                variable_manifest_path=variable_manifest_path,
            )

            manifest = write_zone_feature_case_outputs(
                result=result,
                case_output_dir=case_output_dir,
            )

            success_count += 1
            campaign_rows.append(
                {
                    "case_id": case_id,
                    "status": "success",
                    "eio_tables_path": str(eio_tables_path),
                    "variable_manifest_path": (
                        str(variable_manifest_path) if variable_manifest_path else ""
                    ),
                    "case_output_dir": str(case_output_dir),
                    "zone_feature_count": manifest["zone_feature_count"],
                    "approved_zone_count": manifest["approved_zone_count"],
                    "excluded_zone_count": manifest["excluded_zone_count"],
                    "default_max_aggregate_zones": manifest[
                        "default_max_aggregate_zones"
                    ],
                    "approved_zones": "|".join(manifest["approved_zones"]),
                    "excluded_zones": "|".join(manifest["excluded_zones"]),
                    "error": "",
                }
            )

            print(
                f"[OK] {case_id}: "
                f"features={manifest['zone_feature_count']} "
                f"approved={manifest['approved_zone_count']} "
                f"excluded={manifest['excluded_zone_count']} "
                f"default_max_aggregate_zones={manifest['default_max_aggregate_zones']}"
            )

        except Exception as exc:
            failed_count += 1
            campaign_rows.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "eio_tables_path": "",
                    "variable_manifest_path": "",
                    "case_output_dir": str(case_output_dir),
                    "zone_feature_count": "",
                    "approved_zone_count": "",
                    "excluded_zone_count": "",
                    "default_max_aggregate_zones": "",
                    "approved_zones": "",
                    "excluded_zones": "",
                    "error": repr(exc),
                }
            )
            print(f"[FAILED] {case_id}: {exc}")

    write_campaign_cases_csv(output_root / "zone_features_cases.csv", campaign_rows)

    campaign_manifest = {
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "output_root": str(output_root),
        "case_count": len(case_dirs),
        "successful_case_count": success_count,
        "failed_case_count": failed_count,
        "eligibility_rule": "Part of Total Building Area == Yes",
        "feature_source": "EnergyPlus EIO tables",
        "geometry_source": "EnergyPlus EIO Zone Information table",
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "outputs": {
            "zone_features_cases_csv": str(output_root / "zone_features_cases.csv"),
            "cases_root": str(cases_output_root),
        },
    }

    write_json(output_root / "zone_features_manifest.json", campaign_manifest)

    print("=" * 100)
    print("Zone features complete")
    print(f"output_root: {output_root}")
    print(f"case_count: {len(case_dirs)}")
    print(f"successful_case_count: {success_count}")
    print(f"failed_case_count: {failed_count}")
    print("=" * 100)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())