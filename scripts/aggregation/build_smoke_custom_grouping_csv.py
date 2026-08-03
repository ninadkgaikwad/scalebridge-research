# -*- coding: utf-8 -*-
"""Build custom aggregation groups for the old P1 RDD smoke campaign.

Creates two equal-weight smoke aggregation levels:
  - smoke_l01_all_to_one
  - smoke_l05_identity

This v2 script is robust to old smoke-generation metadata where latest_run.json
may not contain building_type/weather fields. It attempts to infer metadata from:
  1. latest_run.json
  2. generation/cases/<case_id>/runs/<run_id>/generation_manifest.json
  3. generation/cases/<case_id>/runs/<run_id>/canonical/metadata.json
  4. generation/cases/<case_id>/runs/<run_id>/canonical/source_case.json
  5. optional --default-building-type, default RestaurantFastFood

Typical use from repo root:
  python scripts\\aggregation\\build_smoke_custom_grouping_csv.py `
    --campaign-id p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"

APPROVED_ZONES_BY_BUILDING = {
    "RestaurantFastFood": ["Dining", "Kitchen"],
    "OfficeSmall": ["Core_ZN", "Perimeter_ZN_1", "Perimeter_ZN_2", "Perimeter_ZN_3", "Perimeter_ZN_4"],
    "RetailStripmall": ["LGstore1", "SMstore1", "SMstore2", "SMstore3", "SMstore4", "LGstore2", "SMstore5", "SMstore6", "SMstore7", "SMstore8"],
    "ApartmentMidRise": [
        "G SW Apartment", "G NW Apartment", "Office", "G NE Apartment",
        "G N1 Apartment", "G N2 Apartment", "G S1 Apartment", "G S2 Apartment",
        "M SW Apartment", "M NW Apartment", "M SE Apartment", "M NE Apartment",
        "M N1 Apartment", "M N2 Apartment", "M S1 Apartment", "M S2 Apartment",
        "T SW Apartment", "T NW Apartment", "T SE Apartment", "T NE Apartment",
        "T N1 Apartment", "T N2 Apartment", "T S1 Apartment", "T S2 Apartment",
        "T Corridor", "G Corridor", "M Corridor",
    ],
}

BUILDING_ALIASES = {
    "restaurantfastfood": "RestaurantFastFood",
    "restaurant_fast_food": "RestaurantFastFood",
    "restaurant fast food": "RestaurantFastFood",
    "fastfoodrestaurant": "RestaurantFastFood",
    "office small": "OfficeSmall",
    "officesmall": "OfficeSmall",
    "retailstripmall": "RetailStripmall",
    "retail stripmall": "RetailStripmall",
    "retail strip mall": "RetailStripmall",
    "apartmentmidrise": "ApartmentMidRise",
    "apartment midrise": "ApartmentMidRise",
    "apartment mid rise": "ApartmentMidRise",
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

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else campaign_root / "aggregation" / "custom_grouping_smoke"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metadata_debug_rows: list[dict[str, Any]] = []

    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if args.case_id and case_dir.name != args.case_id:
            continue

        latest_path = case_dir / "latest_run.json"
        if not latest_path.is_file():
            print(f"Skipping {case_dir.name}: missing latest_run.json")
            continue

        latest = load_json(latest_path)
        case_id = case_dir.name
        run_id = str(first_nonempty(latest, ("run_id", "source_generation_run_id")) or "").strip()
        if not run_id:
            print(f"Skipping {case_id}: latest_run.json has no run_id")
            continue

        run_root = case_dir / "runs" / run_id
        metadata_sources = collect_metadata_sources(latest=latest, run_root=run_root)

        campaign_id = str(
            find_first_metadata_value(metadata_sources, ("campaign_id",))
            or args.campaign_id
        ).strip()

        raw_building_type = find_first_metadata_value(
            metadata_sources,
            (
                "building_type",
                "building",
                "prototype_building",
                "prototype",
                "building_name",
                "building_id",
            ),
        )
        building_type = normalize_building_type(raw_building_type)
        metadata_source_note = "metadata"

        if not building_type:
            building_type = normalize_building_type(args.default_building_type)
            metadata_source_note = "default_building_type"

        if not building_type:
            raise SystemExit(
                f"Could not infer building_type for case_id={case_id}. "
                "Rerun with --default-building-type RestaurantFastFood or inspect metadata_debug CSV."
            )

        weather_location = str(
            find_first_metadata_value(
                metadata_sources,
                (
                    "weather_location",
                    "weather_city",
                    "city",
                    "weather_name",
                    "weather",
                ),
            )
            or args.default_weather_location
            or ""
        ).strip()

        climate_zone = str(
            find_first_metadata_value(
                metadata_sources,
                (
                    "climate_zone",
                    "ashrae_climate_zone",
                    "climate",
                ),
            )
            or ""
        ).strip()

        approved_zones = APPROVED_ZONES_BY_BUILDING.get(building_type)
        if not approved_zones:
            raise SystemExit(
                f"No approved-zone template is available for building_type={building_type!r}. "
                "Add it to APPROVED_ZONES_BY_BUILDING in this script."
            )

        metadata_debug_rows.append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "raw_building_type": raw_building_type or "",
                "resolved_building_type": building_type,
                "metadata_source_note": metadata_source_note,
                "weather_location": weather_location,
                "climate_zone": climate_zone,
                "metadata_source_count": len(metadata_sources),
                "run_root": str(run_root),
            }
        )

        group_specs = [
            ("smoke_l01_all_to_one", "L01_all_to_one", [(f"{building_type}_All", approved_zones)]),
            ("smoke_l05_identity", "L05_identity", [(zone, [zone]) for zone in approved_zones]),
        ]

        for aggregation_id, aggregation_level, groups in group_specs:
            for aggregate_zone_name, source_zones in groups:
                for source_zone in source_zones:
                    rows.append(
                        {
                            "case_id": case_id,
                            "run_id": run_id,
                            "campaign_id": campaign_id,
                            "building_type": building_type,
                            "weather_location": weather_location,
                            "climate_zone": climate_zone,
                            "aggregation_id": aggregation_id,
                            "aggregation_level": aggregation_level,
                            "aggregate_zone_name": aggregate_zone_name,
                            "source_zone_name": source_zone,
                            "created_by": "build_smoke_custom_grouping_csv.py",
                            "notes": "old RDD smoke campaign grouping for updated shared-node aggregation logic",
                        }
                    )

            summary_rows.append(
                {
                    "case_id": case_id,
                    "run_id": run_id,
                    "building_type": building_type,
                    "weather_location": weather_location,
                    "aggregation_id": aggregation_id,
                    "aggregate_zone_count": len(groups),
                    "source_zone_count": len(approved_zones),
                }
            )

    if not rows:
        raise SystemExit("No grouping rows were created. Check campaign/case filters.")

    csv_path = output_root / "smoke_custom_groups.csv"
    summary_path = output_root / "smoke_custom_groups_summary.csv"
    metadata_debug_path = output_root / "smoke_custom_groups_metadata_debug.csv"
    manifest_path = output_root / "smoke_custom_groups_manifest.json"

    write_csv(csv_path, rows)
    write_csv(summary_path, summary_rows)
    write_csv(metadata_debug_path, metadata_debug_rows)
    write_json(
        manifest_path,
        {
            "schema_version": "0.2.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": args.campaign_id,
            "campaign_root": str(campaign_root),
            "output_root": str(output_root),
            "custom_groups_csv": str(csv_path),
            "summary_csv": str(summary_path),
            "metadata_debug_csv": str(metadata_debug_path),
            "row_count": len(rows),
            "case_count": len({row["case_id"] for row in rows}),
            "aggregation_ids": sorted({row["aggregation_id"] for row in rows}),
        },
    )

    print("=" * 100)
    print("SMOKE CUSTOM GROUPING CSV CREATED")
    print("=" * 100)
    print(f"campaign_id: {args.campaign_id}")
    print(f"campaign_root: {campaign_root}")
    print(f"custom_groups_csv: {csv_path}")
    print(f"summary_csv: {summary_path}")
    print(f"metadata_debug_csv: {metadata_debug_path}")
    print(f"manifest_path: {manifest_path}")
    print(f"row_count: {len(rows)}")
    print(f"case_count: {len({row['case_id'] for row in rows})}")
    print(f"aggregation_ids: {sorted({row['aggregation_id'] for row in rows})}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID)
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--generated-data-root", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--default-building-type",
        default="RestaurantFastFood",
        help="Fallback building type for old smoke metadata. Default: RestaurantFastFood.",
    )
    parser.add_argument(
        "--default-weather-location",
        default="",
        help="Optional fallback weather location if old metadata omits it.",
    )
    return parser.parse_args(argv)


def collect_metadata_sources(*, latest: dict[str, Any], run_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    sources.append(flatten_dict(latest))

    candidate_paths = [
        run_root / "generation_manifest.json",
        run_root / "manifest.json",
        run_root / "run_manifest.json",
        run_root / "canonical" / "metadata.json",
        run_root / "canonical" / "source_case.json",
        run_root / "canonical" / "case_metadata.json",
        run_root / "inputs" / "source_case.json",
        run_root / "inputs" / "case_metadata.json",
    ]

    for path in candidate_paths:
        if path.is_file():
            try:
                payload = load_json(path)
            except Exception:
                continue
            sources.append(flatten_dict(payload))

    return sources


def flatten_dict(payload: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            full_key = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(value, dict):
                flat.update(flatten_dict(value, full_key))
            else:
                flat[full_key] = value
                flat[key_text] = value
    return flat


def find_first_metadata_value(sources: list[dict[str, Any]], keys: tuple[str, ...]) -> Any:
    normalized_keys = {normalize_key(key): key for key in keys}

    for source in sources:
        for key, value in source.items():
            if value is None or str(value).strip() == "":
                continue
            key_norm = normalize_key(key)
            if key_norm in normalized_keys:
                return value

        for key, value in source.items():
            if value is None or str(value).strip() == "":
                continue
            key_norm = normalize_key(key)
            for target_norm in normalized_keys:
                if key_norm.endswith("." + target_norm):
                    return value

    return None


def normalize_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_building_type(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if text in APPROVED_ZONES_BY_BUILDING:
        return text

    normalized = text.strip().lower().replace("-", "_")
    normalized_space = normalized.replace("_", " ")
    normalized_compact = normalized.replace("_", "").replace(" ", "")

    for candidate in (normalized, normalized_space, normalized_compact):
        if candidate in BUILDING_ALIASES:
            return BUILDING_ALIASES[candidate]

    for building in APPROVED_ZONES_BY_BUILDING:
        if building.lower() in normalized_compact:
            return building

    return ""


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src" / "scalebridge").is_dir():
            return candidate
    return cwd


def resolve_campaign_root(
    *,
    repo_root: Path,
    campaign_id: str,
    campaign_root: str | None,
    generated_data_root: str | None,
) -> Path:
    if campaign_root:
        return Path(campaign_root).expanduser().resolve()
    if generated_data_root:
        root = Path(generated_data_root).expanduser().resolve()
    else:
        import os

        env_value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT", "").strip()
        if env_value:
            root = Path(env_value).expanduser().resolve()
        else:
            root = (repo_root / ".." / ".." / "Data" / "ScaleBridge").resolve()
    return root / "campaigns" / campaign_id


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
