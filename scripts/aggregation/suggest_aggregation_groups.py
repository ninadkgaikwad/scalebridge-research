"""
Generate complete-partition aggregation grouping suggestions.

Example:
    python scripts/aggregation/suggest_aggregation_groups.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3

Optional single-case run:
    python scripts/aggregation/suggest_aggregation_groups.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3 ^
      --case-id epcase_827ca4812c0199221d031e59

Optional max aggregate-zone ceiling:
    python scripts/aggregation/suggest_aggregation_groups.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3 ^
      --max-aggregate-zones 1
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

from scalebridge.data.aggregation.zone_inventory import (
    find_case_dirs,
    resolve_campaign_root,
    write_json,
)
from scalebridge.data.aggregation.grouping_suggestions import (
    GroupingSuggestionConfig,
    build_grouping_suggestions,
    find_approved_zone_features_path,
    find_zone_pairwise_similarity_path,
    timestamp_id,
    write_grouping_suggestion_case_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate complete-partition aggregation grouping suggestions."
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
        "--zone-features-root",
        default=None,
        help=(
            "Optional specific zone_features_<timestamp> root. "
            "If omitted, latest zone_features_* folder is used."
        ),
    )
    parser.add_argument(
        "--zone-similarity-root",
        default=None,
        help=(
            "Optional specific zone_similarity_<timestamp> root. "
            "If omitted, latest zone_similarity_* folder is used."
        ),
    )
    parser.add_argument(
        "--max-aggregate-zones",
        type=int,
        default=None,
        help=(
            "Maximum aggregate zones allowed in suggestions. "
            "If omitted, defaults to approved_zone_count."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional output root. If omitted, writes under "
            "<campaign_root>/aggregation/grouping_suggestions/grouping_suggestions_<timestamp>."
        ),
    )

    return parser.parse_args()


def write_campaign_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "status",
        "approved_zone_features_path",
        "zone_pairwise_similarity_path",
        "case_output_dir",
        "approved_zone_count",
        "effective_max_aggregate_zones",
        "suggestion_count",
        "suggestion_row_count",
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
            / "grouping_suggestions"
            / timestamp_id("grouping_suggestions")
        )

    cases_output_root = output_root / "cases"
    cases_output_root.mkdir(parents=True, exist_ok=True)

    case_dirs = find_case_dirs(campaign_root, case_id=args.case_id)

    config = GroupingSuggestionConfig(
        max_aggregate_zones=args.max_aggregate_zones,
    )

    campaign_rows: list[dict[str, Any]] = []
    success_count = 0
    failed_count = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        case_output_dir = cases_output_root / case_id

        try:
            approved_zone_features_path = find_approved_zone_features_path(
                campaign_root=campaign_root,
                case_id=case_id,
                zone_features_root=args.zone_features_root,
            )
            zone_pairwise_similarity_path = find_zone_pairwise_similarity_path(
                campaign_root=campaign_root,
                case_id=case_id,
                zone_similarity_root=args.zone_similarity_root,
            )

            result = build_grouping_suggestions(
                case_id=case_id,
                approved_zone_features_path=approved_zone_features_path,
                zone_pairwise_similarity_path=zone_pairwise_similarity_path,
                config=config,
            )

            manifest = write_grouping_suggestion_case_outputs(
                result=result,
                case_output_dir=case_output_dir,
            )

            success_count += 1
            campaign_rows.append(
                {
                    "case_id": case_id,
                    "status": "success",
                    "approved_zone_features_path": str(approved_zone_features_path),
                    "zone_pairwise_similarity_path": str(zone_pairwise_similarity_path),
                    "case_output_dir": str(case_output_dir),
                    "approved_zone_count": manifest["approved_zone_count"],
                    "effective_max_aggregate_zones": manifest[
                        "effective_max_aggregate_zones"
                    ],
                    "suggestion_count": manifest["suggestion_count"],
                    "suggestion_row_count": manifest["suggestion_row_count"],
                    "error": "",
                }
            )

            print(
                f"[OK] {case_id}: "
                f"approved_zones={manifest['approved_zone_count']} "
                f"effective_max={manifest['effective_max_aggregate_zones']} "
                f"suggestions={manifest['suggestion_count']} "
                f"rows={manifest['suggestion_row_count']}"
            )

        except Exception as exc:
            failed_count += 1
            campaign_rows.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "approved_zone_features_path": "",
                    "zone_pairwise_similarity_path": "",
                    "case_output_dir": str(case_output_dir),
                    "approved_zone_count": "",
                    "effective_max_aggregate_zones": "",
                    "suggestion_count": "",
                    "suggestion_row_count": "",
                    "error": repr(exc),
                }
            )
            print(f"[FAILED] {case_id}: {exc}")

    write_campaign_cases_csv(
        output_root / "grouping_suggestions_cases.csv",
        campaign_rows,
    )

    campaign_manifest = {
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "zone_features_root": str(args.zone_features_root or ""),
        "zone_similarity_root": str(args.zone_similarity_root or ""),
        "output_root": str(output_root),
        "requested_max_aggregate_zones": args.max_aggregate_zones,
        "case_count": len(case_dirs),
        "successful_case_count": success_count,
        "failed_case_count": failed_count,
        "partition_rule": {
            "covers_all_approved_zones": True,
            "uses_each_approved_zone_exactly_once": True,
            "uses_excluded_zones": False,
        },
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "outputs": {
            "grouping_suggestions_cases_csv": str(
                output_root / "grouping_suggestions_cases.csv"
            ),
            "cases_root": str(cases_output_root),
        },
    }

    write_json(output_root / "grouping_suggestions_manifest.json", campaign_manifest)

    print("=" * 100)
    print("Grouping suggestions complete")
    print(f"output_root: {output_root}")
    print(f"case_count: {len(case_dirs)}")
    print(f"successful_case_count: {success_count}")
    print(f"failed_case_count: {failed_count}")
    print("=" * 100)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())