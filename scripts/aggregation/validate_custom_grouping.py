"""
Validate a user-defined custom grouping CSV.

Supports multiple aggregation_id blocks in one CSV.

Example:
    python scripts/aggregation/validate_custom_grouping.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3 ^
      --case-id epcase_827ca4812c0199221d031e59 ^
      --custom-zone-groups path\\to\\approved_custom_groups.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.data.aggregation.zone_inventory import resolve_campaign_root
from scalebridge.data.aggregation.grouping_suggestions import (
    find_approved_zone_features_path,
)
from scalebridge.data.aggregation.custom_groups import (
    timestamp_id,
    validate_custom_grouping_file,
    write_validation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a user-defined custom aggregation grouping CSV."
    )

    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--custom-zone-groups",
        required=True,
        help="Path to custom grouping CSV.",
    )
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument(
        "--zone-features-root",
        default=None,
        help="Optional zone_features_<timestamp> root. Defaults to latest.",
    )
    parser.add_argument(
        "--max-aggregate-zones",
        type=int,
        default=None,
        help="Optional validation ceiling. Defaults to approved_zone_count.",
    )
    parser.add_argument("--output-root", default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    campaign_root = resolve_campaign_root(
        campaign_id=args.campaign_id,
        generated_data_root=args.generated_data_root,
        campaign_root=args.campaign_root,
    )

    approved_zone_features_path = find_approved_zone_features_path(
        campaign_root=campaign_root,
        case_id=args.case_id,
        zone_features_root=args.zone_features_root,
    )

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = (
            campaign_root
            / "aggregation"
            / "custom_grouping_validation"
            / timestamp_id("custom_grouping_validation")
            / "cases"
            / args.case_id
        )

    validation = validate_custom_grouping_file(
        custom_grouping_path=args.custom_zone_groups,
        approved_zone_features_path=approved_zone_features_path,
        max_aggregate_zones=args.max_aggregate_zones,
    )

    write_validation_outputs(
        validation=validation,
        output_root=output_root,
    )

    print("=" * 100)
    print("Custom grouping validation complete")
    print(f"case_id: {args.case_id}")
    print(f"custom_zone_groups: {args.custom_zone_groups}")
    print(f"valid: {validation.valid}")
    print(f"aggregation_count: {validation.aggregation_count}")
    print(f"row_count: {validation.row_count}")
    print(f"approved_zones: {validation.approved_zones}")
    print(f"output_root: {output_root}")

    if validation.errors:
        print("errors:")
        for error in validation.errors:
            print(f"  - {error}")

    if validation.warnings:
        print("warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")

    print("=" * 100)

    return 0 if validation.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())