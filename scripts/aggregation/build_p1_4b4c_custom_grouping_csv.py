# -*- coding: utf-8 -*-
"""Build paper-ready custom aggregation grouping CSV for P1 4-building x 4-climate campaign.

This script creates a campaign-level custom grouping CSV with common aggregation
level IDs that exist for every generated case. Each case gets building-specific
zone group definitions under the same level IDs, which lets run_p1_aggregation.py
execute campaign-wide custom aggregation by --aggregation-id.

Intended campaign:
    p1_compact_4b4c_labpc_1w_v1

Output:
    <campaign_root>/aggregation/custom_grouping_levels/p1_4b4c_custom_groups.csv
    <campaign_root>/aggregation/custom_grouping_levels/p1_4b4c_custom_groups_manifest.json

Usage:
    python scripts\\aggregation\\build_p1_4b4c_custom_grouping_csv.py `
      --campaign-id p1_compact_4b4c_labpc_1w_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.discovery import (
    DEFAULT_CAMPAIGN_ID,
    discover_generation_runs,
    load_json,
    resolve_campaign_root,
    resolve_repo_root,
)


OUTPUT_FOLDER_NAME = "custom_grouping_levels"
OUTPUT_CSV_NAME = "p1_4b4c_custom_groups.csv"
OUTPUT_MANIFEST_NAME = "p1_4b4c_custom_groups_manifest.json"


AGGREGATION_LEVELS = [
    "p1_l01_all_to_one",
    "p1_l02_functional",
    "p1_l03_intermediate",
    "p1_l04_spatial_detailed",
    "p1_l05_identity",
]


def restaurant_groups() -> dict[str, list[tuple[str, list[str]]]]:
    zones = ["Dining", "Kitchen"]
    identity = [(zone, [zone]) for zone in zones]
    return {
        "p1_l01_all_to_one": [
            ("Restaurant_All", zones),
        ],
        "p1_l02_functional": [
            ("Restaurant_Dining", ["Dining"]),
            ("Restaurant_Kitchen", ["Kitchen"]),
        ],
        "p1_l03_intermediate": identity,
        "p1_l04_spatial_detailed": identity,
        "p1_l05_identity": identity,
    }


def office_groups() -> dict[str, list[tuple[str, list[str]]]]:
    core = ["Core_ZN"]
    p1 = ["Perimeter_ZN_1"]
    p2 = ["Perimeter_ZN_2"]
    p3 = ["Perimeter_ZN_3"]
    p4 = ["Perimeter_ZN_4"]
    all_zones = core + p1 + p2 + p3 + p4
    identity = [(zone, [zone]) for zone in all_zones]
    return {
        "p1_l01_all_to_one": [
            ("Office_All", all_zones),
        ],
        "p1_l02_functional": [
            ("Office_Core", core),
            ("Office_Perimeter", p1 + p2 + p3 + p4),
        ],
        "p1_l03_intermediate": [
            ("Office_Core", core),
            ("Office_Perimeter_1_2", p1 + p2),
            ("Office_Perimeter_3_4", p3 + p4),
        ],
        "p1_l04_spatial_detailed": identity,
        "p1_l05_identity": identity,
    }


def retail_groups() -> dict[str, list[tuple[str, list[str]]]]:
    # Geometry order from west/left to east/right:
    # LGstore1, SMstore1-4, LGstore2, SMstore5-8.
    zones = [
        "LGstore1",
        "SMstore1",
        "SMstore2",
        "SMstore3",
        "SMstore4",
        "LGstore2",
        "SMstore5",
        "SMstore6",
        "SMstore7",
        "SMstore8",
    ]
    identity = [(zone, [zone]) for zone in zones]
    return {
        "p1_l01_all_to_one": [
            ("Retail_All", zones),
        ],
        "p1_l02_functional": [
            ("Retail_Left_Block", ["LGstore1", "SMstore1", "SMstore2", "SMstore3", "SMstore4"]),
            ("Retail_Right_Block", ["LGstore2", "SMstore5", "SMstore6", "SMstore7", "SMstore8"]),
        ],
        "p1_l03_intermediate": [
            ("Retail_Pair_01", ["LGstore1", "SMstore1"]),
            ("Retail_Pair_02", ["SMstore2", "SMstore3"]),
            ("Retail_Pair_03", ["SMstore4", "LGstore2"]),
            ("Retail_Pair_04", ["SMstore5", "SMstore6"]),
            ("Retail_Pair_05", ["SMstore7", "SMstore8"]),
        ],
        "p1_l04_spatial_detailed": identity,
        "p1_l05_identity": identity,
    }


def apartment_groups() -> dict[str, list[tuple[str, list[str]]]]:
    office = ["Office"]
    corridors = ["G Corridor", "M Corridor", "T Corridor"]

    g_south = ["G SW Apartment", "G S1 Apartment", "G S2 Apartment"]
    g_north = ["G NW Apartment", "G N1 Apartment", "G N2 Apartment", "G NE Apartment"]

    m_south = ["M SW Apartment", "M S1 Apartment", "M S2 Apartment", "M SE Apartment"]
    m_north = ["M NW Apartment", "M N1 Apartment", "M N2 Apartment", "M NE Apartment"]

    t_south = ["T SW Apartment", "T S1 Apartment", "T S2 Apartment", "T SE Apartment"]
    t_north = ["T NW Apartment", "T N1 Apartment", "T N2 Apartment", "T NE Apartment"]

    ground_apartments = g_south + g_north
    middle_apartments = m_south + m_north
    top_apartments = t_south + t_north

    apartment_zones = ground_apartments + middle_apartments + top_apartments
    all_zones = apartment_zones + office + corridors

    identity_order = [
        "G SW Apartment",
        "G NW Apartment",
        "Office",
        "G NE Apartment",
        "G N1 Apartment",
        "G N2 Apartment",
        "G S1 Apartment",
        "G S2 Apartment",
        "M SW Apartment",
        "M NW Apartment",
        "M SE Apartment",
        "M NE Apartment",
        "M N1 Apartment",
        "M N2 Apartment",
        "M S1 Apartment",
        "M S2 Apartment",
        "T SW Apartment",
        "T NW Apartment",
        "T SE Apartment",
        "T NE Apartment",
        "T N1 Apartment",
        "T N2 Apartment",
        "T S1 Apartment",
        "T S2 Apartment",
        "T Corridor",
        "G Corridor",
        "M Corridor",
    ]

    return {
        "p1_l01_all_to_one": [
            ("Apartment_All", identity_order),
        ],
        "p1_l02_functional": [
            ("Apartment_Residential", apartment_zones),
            ("Apartment_NonResidential_Common", office + corridors),
        ],
        "p1_l03_intermediate": [
            ("Apartment_Office", office),
            ("Apartment_Corridors", corridors),
            ("Apartment_Ground_Residential", ground_apartments),
            ("Apartment_Middle_Residential", middle_apartments),
            ("Apartment_Top_Residential", top_apartments),
        ],
        "p1_l04_spatial_detailed": [
            ("Apartment_Office", office),
            ("Apartment_G_Corridor", ["G Corridor"]),
            ("Apartment_M_Corridor", ["M Corridor"]),
            ("Apartment_T_Corridor", ["T Corridor"]),
            ("Apartment_G_South_Row", g_south),
            ("Apartment_G_North_Row", g_north),
            ("Apartment_M_South_Row", m_south),
            ("Apartment_M_North_Row", m_north),
            ("Apartment_T_South_Row", t_south),
            ("Apartment_T_North_Row", t_north),
        ],
        "p1_l05_identity": [(zone, [zone]) for zone in identity_order],
    }


BUILDING_GROUPS = {
    "RestaurantFastFood": restaurant_groups(),
    "OfficeSmall": office_groups(),
    "RetailStripmall": retail_groups(),
    "ApartmentMidRise": apartment_groups(),
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

    cases_root = campaign_root / "generation" / "cases"
    if not cases_root.is_dir():
        raise SystemExit(f"Generation cases folder does not exist: {cases_root}")

    run_refs, missing_rows = discover_generation_runs(
        cases_root=cases_root,
        case_id=args.case_id,
        include_failed=False,
    )
    if not run_refs:
        raise SystemExit(f"No successful generation runs found under {cases_root}")

    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for run_ref in run_refs:
        manifest = load_json(run_ref.manifest_path)
        case_spec = manifest.get("case_spec", {})
        building_type = str(case_spec.get("building_type", "")).strip()

        if building_type not in BUILDING_GROUPS:
            raise ValueError(
                f"Unsupported building_type={building_type!r} for case_id={run_ref.case_id}."
            )

        level_map = BUILDING_GROUPS[building_type]

        for aggregation_id in AGGREGATION_LEVELS:
            groups = level_map[aggregation_id]
            for aggregate_zone_name, source_zones in groups:
                for source_zone_name in source_zones:
                    rows.append(
                        {
                            "case_id": run_ref.case_id,
                            "run_id": run_ref.run_id,
                            "campaign_id": args.campaign_id,
                            "building_type": building_type,
                            "weather_location": case_spec.get("weather_location", ""),
                            "climate_zone": case_spec.get("climate_zone", ""),
                            "aggregation_id": aggregation_id,
                            "aggregation_level": aggregation_id,
                            "aggregate_zone_name": aggregate_zone_name,
                            "source_zone_name": source_zone_name,
                            "created_by": "build_p1_4b4c_custom_grouping_csv.py",
                            "notes": "paper_ready_4b4c_aggregation_ladder",
                        }
                    )

            case_summaries.append(
                {
                    "case_id": run_ref.case_id,
                    "run_id": run_ref.run_id,
                    "building_type": building_type,
                    "weather_location": case_spec.get("weather_location", ""),
                    "climate_zone": case_spec.get("climate_zone", ""),
                    "aggregation_id": aggregation_id,
                    "aggregate_zone_count": len(groups),
                    "source_zone_count": sum(len(source_zones) for _, source_zones in groups),
                }
            )

    output_root = resolve_output_root(
        campaign_root=campaign_root,
        output_root=args.output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / OUTPUT_CSV_NAME
    summary_csv_path = output_root / "p1_4b4c_custom_groups_case_summary.csv"
    manifest_path = output_root / OUTPUT_MANIFEST_NAME

    write_csv(csv_path, rows)
    write_csv(summary_csv_path, case_summaries)

    manifest = {
        "schema_version": "0.1.0",
        "created_at_local": datetime.now().isoformat(),
        "campaign_id": args.campaign_id,
        "campaign_root": str(campaign_root),
        "output_root": str(output_root),
        "custom_grouping_csv": str(csv_path),
        "case_summary_csv": str(summary_csv_path),
        "case_count": len(run_refs),
        "row_count": len(rows),
        "missing_generation_rows": missing_rows,
        "aggregation_levels": AGGREGATION_LEVELS,
        "building_types": sorted(BUILDING_GROUPS.keys()),
        "intended_use": (
            "Use this CSV with build_p1_aggregation_plan.py --strategy custom_groups "
            "and --custom-aggregation-id <aggregation level>."
        ),
    }
    write_json(manifest_path, manifest)

    print("=" * 100)
    print("P1 4B4C CUSTOM GROUPING CSV BUILDER")
    print("=" * 100)
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"case_count: {len(run_refs)}")
    print(f"row_count: {len(rows)}")
    print(f"output_root: {output_root}")
    print(f"custom_grouping_csv: {csv_path}")
    print(f"case_summary_csv: {summary_csv_path}")
    print(f"manifest_path: {manifest_path}")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help=f"Campaign ID. Default: {DEFAULT_CAMPAIGN_ID}",
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help="Explicit campaign root.",
    )
    parser.add_argument(
        "--generated-data-root",
        default=None,
        help="Explicit SCALEBRIDGE_GENERATED_DATA_ROOT.",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional case_id filter for debugging.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output folder. Default: "
            "<campaign_root>/aggregation/custom_grouping_levels."
        ),
    )
    return parser.parse_args(argv)


def resolve_output_root(*, campaign_root: Path, output_root: str | None) -> Path:
    if output_root:
        return Path(output_root).expanduser().resolve()
    return campaign_root / "aggregation" / OUTPUT_FOLDER_NAME


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    if not fieldnames:
        fieldnames = ["note"]
        rows = [{"note": "no rows"}]

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
