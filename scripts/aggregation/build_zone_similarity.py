"""
Build EIO-feature-based zone similarity tables for ScaleBridge aggregation suggestions.

Example:
    python scripts/aggregation/build_zone_similarity.py ^
      --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3

Optional single-case run:
    python scripts/aggregation/build_zone_similarity.py ^
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

from scalebridge.data.aggregation.zone_inventory import (
    find_case_dirs,
    resolve_campaign_root,
    write_json,
)
from scalebridge.data.aggregation.zone_similarity import (
    build_zone_similarity_from_features,
    find_approved_zone_features_path,
    timestamp_id,
    write_zone_similarity_case_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pairwise approved-zone similarity tables for aggregation suggestions."
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
        "--output-root",
        default=None,
        help=(
            "Optional output root. If omitted, writes under "
            "<campaign_root>/aggregation/zone_similarity/zone_similarity_<timestamp>."
        ),
    )

    return parser.parse_args()


def write_campaign_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "status",
        "approved_zone_features_path",
        "case_output_dir",
        "approved_zone_count",
        "pair_count",
        "token_row_count",
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
            / "zone_similarity"
            / timestamp_id("zone_similarity")
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
            approved_zone_features_path = find_approved_zone_features_path(
                campaign_root=campaign_root,
                case_id=case_id,
                zone_features_root=args.zone_features_root,
            )

            result = build_zone_similarity_from_features(
                case_id=case_id,
                approved_zone_features_path=approved_zone_features_path,
            )

            manifest = write_zone_similarity_case_outputs(
                result=result,
                case_output_dir=case_output_dir,
            )

            success_count += 1
            campaign_rows.append(
                {
                    "case_id": case_id,
                    "status": "success",
                    "approved_zone_features_path": str(approved_zone_features_path),
                    "case_output_dir": str(case_output_dir),
                    "approved_zone_count": manifest["approved_zone_count"],
                    "pair_count": manifest["pair_count"],
                    "token_row_count": manifest["token_row_count"],
                    "error": "",
                }
            )

            print(
                f"[OK] {case_id}: "
                f"approved_zones={manifest['approved_zone_count']} "
                f"pairs={manifest['pair_count']} "
                f"tokens={manifest['token_row_count']}"
            )

        except Exception as exc:
            failed_count += 1
            campaign_rows.append(
                {
                    "case_id": case_id,
                    "status": "failed",
                    "approved_zone_features_path": "",
                    "case_output_dir": str(case_output_dir),
                    "approved_zone_count": "",
                    "pair_count": "",
                    "token_row_count": "",
                    "error": repr(exc),
                }
            )
            print(f"[FAILED] {case_id}: {exc}")

    write_campaign_cases_csv(output_root / "zone_similarity_cases.csv", campaign_rows)

    campaign_manifest = {
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "zone_features_root": str(args.zone_features_root or ""),
        "output_root": str(output_root),
        "case_count": len(case_dirs),
        "successful_case_count": success_count,
        "failed_case_count": failed_count,
        "similarity_scope": "approved_zones_only",
        "feature_source": "EIO-only zone_features.csv",
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "outputs": {
            "zone_similarity_cases_csv": str(output_root / "zone_similarity_cases.csv"),
            "cases_root": str(cases_output_root),
        },
    }

    write_json(output_root / "zone_similarity_manifest.json", campaign_manifest)

    print("=" * 100)
    print("Zone similarity complete")
    print(f"output_root: {output_root}")
    print(f"case_count: {len(case_dirs)}")
    print(f"successful_case_count: {success_count}")
    print(f"failed_case_count: {failed_count}")
    print("=" * 100)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())