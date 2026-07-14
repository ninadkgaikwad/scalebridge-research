"""
Approve one or more grouping suggestions and save them as user-approved custom groupings.

Examples:

Approve all generated suggestions:
    python scripts/aggregation/approve_grouping_suggestion.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3 ^
      --case-id epcase_827ca4812c0199221d031e59 ^
      --approve-all ^
      --aggregation-id-prefix rff_user_approved ^
      --approved-by Ninad ^
      --approval-notes "Approved all generated grouping styles."

Approve selected suggestions:
    python scripts/aggregation/approve_grouping_suggestion.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3 ^
      --case-id epcase_827ca4812c0199221d031e59 ^
      --suggestion-id epcase_827ca4812c0199221d031e59_k1_all_to_one ^
      --suggestion-id epcase_827ca4812c0199221d031e59_k2_identity ^
      --aggregation-id-prefix rff_user_approved

Approve exactly one suggestion with a custom aggregation_id:
    python scripts/aggregation/approve_grouping_suggestion.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3 ^
      --case-id epcase_827ca4812c0199221d031e59 ^
      --suggestion-id epcase_827ca4812c0199221d031e59_k1_all_to_one ^
      --aggregation-id rff_user_approved_all_to_one_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.data.aggregation.zone_inventory import (
    resolve_campaign_root,
    write_json,
)
from scalebridge.data.aggregation.grouping_suggestions import (
    find_approved_zone_features_path,
    find_latest_root,
)
from scalebridge.data.aggregation.custom_groups import (
    approve_multiple_suggestion_rows,
    timestamp_id,
    write_approved_grouping_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approve grouping suggestions as custom groupings."
    )

    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--case-id", required=True)

    parser.add_argument(
        "--suggestion-id",
        action="append",
        default=[],
        help=(
            "Suggestion ID to approve. Can be repeated. "
            "Use --approve-all to approve all suggestion_id blocks."
        ),
    )
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="Approve all suggestion_id blocks in suggested_groupings.csv.",
    )
    parser.add_argument(
        "--aggregation-id",
        default=None,
        help=(
            "Optional custom aggregation_id. Allowed only when approving exactly "
            "one suggestion."
        ),
    )
    parser.add_argument(
        "--aggregation-id-prefix",
        default=None,
        help=(
            "Optional prefix for generated aggregation IDs when approving multiple "
            "suggestions. Example: rff_user_approved"
        ),
    )
    parser.add_argument("--approved-by", default="user")
    parser.add_argument("--approval-notes", default="")
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument(
        "--grouping-suggestions-root",
        default=None,
        help="Optional grouping_suggestions_<timestamp> root. Defaults to latest.",
    )
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

    if not args.approve_all and not args.suggestion_id:
        raise ValueError("Provide at least one --suggestion-id or use --approve-all.")

    if args.aggregation_id and (args.approve_all or len(args.suggestion_id) > 1):
        raise ValueError(
            "--aggregation-id is only allowed when approving exactly one suggestion. "
            "Use --aggregation-id-prefix for multiple suggestions."
        )

    campaign_root = resolve_campaign_root(
        campaign_id=args.campaign_id,
        generated_data_root=args.generated_data_root,
        campaign_root=args.campaign_root,
    )

    grouping_root = (
        Path(args.grouping_suggestions_root).expanduser().resolve()
        if args.grouping_suggestions_root
        else find_latest_root(
            campaign_root,
            "aggregation/grouping_suggestions",
            "grouping_suggestions",
        )
    )

    suggested_groupings_path = (
        grouping_root
        / "cases"
        / args.case_id
        / "suggested_groupings.csv"
    )
    if not suggested_groupings_path.exists():
        raise FileNotFoundError(f"Suggested groupings not found: {suggested_groupings_path}")

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
            / "user_groupings"
            / timestamp_id("user_grouping")
        )

    case_output_root = output_root / "cases" / args.case_id

    approved_rows = approve_multiple_suggestion_rows(
        suggested_groupings_path=suggested_groupings_path,
        suggestion_ids=args.suggestion_id,
        approve_all=args.approve_all,
        aggregation_id=args.aggregation_id,
        aggregation_id_prefix=args.aggregation_id_prefix,
        approved_by=args.approved_by,
        approval_notes=args.approval_notes,
    )

    payload = write_approved_grouping_outputs(
        approved_rows=approved_rows,
        approved_zone_features_path=approved_zone_features_path,
        output_root=case_output_root,
        max_aggregate_zones=args.max_aggregate_zones,
        notes_title="Approved Grouping Suggestions",
    )

    campaign_manifest = {
        "campaign_id": args.campaign_id,
        "case_id": args.case_id,
        "approved_suggestion_ids": args.suggestion_id,
        "approve_all": args.approve_all,
        "aggregation_id": args.aggregation_id or "",
        "aggregation_id_prefix": args.aggregation_id_prefix or "",
        "grouping_suggestions_root": str(grouping_root),
        "suggested_groupings_path": str(suggested_groupings_path),
        "approved_zone_features_path": str(approved_zone_features_path),
        "output_root": str(output_root),
        "case_output_root": str(case_output_root),
        "valid": payload["valid"],
        "aggregation_count": payload["aggregation_count"],
        "row_count": payload["row_count"],
        "approved_zones": payload["approved_zones"],
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "outputs": {
            "approved_custom_groups_csv": str(
                case_output_root / "approved_custom_groups.csv"
            ),
            "approved_custom_groups_json": str(
                case_output_root / "approved_custom_groups.json"
            ),
            "user_grouping_notes_md": str(
                case_output_root / "user_grouping_notes.md"
            ),
        },
    }

    write_json(output_root / "user_grouping_manifest.json", campaign_manifest)

    print("=" * 100)
    print("Grouping suggestions approved")
    print(f"case_id: {args.case_id}")
    print(f"approve_all: {args.approve_all}")
    print(f"selected_suggestion_ids: {args.suggestion_id}")
    print(f"aggregation_count: {payload['aggregation_count']}")
    print(f"row_count: {payload['row_count']}")
    print(f"valid: {payload['valid']}")
    print(f"output_root: {output_root}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())