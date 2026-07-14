"""
EIO-only zone feature extraction for ScaleBridge aggregation suggestions.

Increment 2 purpose:
    - Build per-zone features from generated eio_tables.json.
    - Use Increment 1 zone eligibility logic.
    - Do not parse raw IDF files.
    - Do not use opyplus.
    - Prepare approved-zone features for later similarity and grouping suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

from scalebridge.data.aggregation.zone_inventory import (
    ZoneInventoryRow,
    build_zone_inventory_from_eio,
    find_case_dirs,
    find_latest_eio_tables_path,
    load_json,
    parse_float,
    resolve_campaign_root,
    row_to_dict as inventory_row_to_dict,
    write_json,
)


@dataclass(frozen=True)
class ZoneFeatureResult:
    case_id: str
    eio_tables_path: Path
    variable_manifest_path: Path | None
    zone_feature_rows: list[dict[str, Any]]

    @property
    def approved_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.zone_feature_rows if bool(row.get("include_flag"))]

    @property
    def excluded_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.zone_feature_rows if not bool(row.get("include_flag"))]

    @property
    def approved_zone_count(self) -> int:
        return len(self.approved_rows)

    @property
    def excluded_zone_count(self) -> int:
        return len(self.excluded_rows)

    @property
    def default_max_aggregate_zones(self) -> int:
        return self.approved_zone_count


def timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def normalize_zone_name(name: Any) -> str:
    return str(name).strip().upper()


def normalize_schedule_list(values: Iterable[Any]) -> str:
    cleaned = sorted({str(v).strip() for v in values if str(v).strip()})
    return "|".join(cleaned)


def get_eio_tables(eio_payload: dict[str, Any]) -> dict[str, Any]:
    tables = eio_payload.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("Expected eio_tables.json to contain dict field 'tables'.")
    return tables


def get_table_rows(eio_payload: dict[str, Any], table_name: str) -> list[dict[str, str]]:
    tables = get_eio_tables(eio_payload)
    table = tables.get(table_name)
    if not table:
        return []

    columns = table.get("columns")
    rows = table.get("rows")

    if not isinstance(columns, list) or not isinstance(rows, list):
        return []

    out: list[dict[str, str]] = []
    for row in rows:
        item: dict[str, str] = {}
        for idx, col in enumerate(columns):
            item[str(col)] = "" if idx >= len(row) or row[idx] is None else str(row[idx])
        out.append(item)
    return out


def sum_by_zone(
    rows: list[dict[str, str]],
    *,
    zone_column: str,
    value_column: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        zone = normalize_zone_name(row.get(zone_column, ""))
        if not zone:
            continue
        value = parse_float(row.get(value_column))
        if value is None:
            continue
        out[zone] = out.get(zone, 0.0) + value
    return out


def schedules_by_zone(
    rows: list[dict[str, str]],
    *,
    zone_column: str = "Zone Name",
    schedule_column: str = "Schedule Name",
) -> dict[str, str]:
    tmp: dict[str, list[str]] = {}
    for row in rows:
        zone = normalize_zone_name(row.get(zone_column, ""))
        schedule = str(row.get(schedule_column, "")).strip()
        if not zone or not schedule:
            continue
        tmp.setdefault(zone, []).append(schedule)
    return {zone: normalize_schedule_list(vals) for zone, vals in tmp.items()}


def count_objects_by_zone(
    rows: list[dict[str, str]],
    *,
    zone_column: str = "Zone Name",
) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        zone = normalize_zone_name(row.get(zone_column, ""))
        if not zone:
            continue
        out[zone] = out.get(zone, 0) + 1
    return out


def first_numeric_by_zone(
    rows: list[dict[str, str]],
    *,
    zone_column: str,
    value_column: str,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for row in rows:
        zone = normalize_zone_name(row.get(zone_column, ""))
        if not zone or zone in out:
            continue
        out[zone] = parse_float(row.get(value_column))
    return out


def find_latest_variable_manifest_path(case_dir: Path) -> Path | None:
    candidates = list(case_dir.glob("runs/*/canonical/variable_manifest.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def collect_variable_names_from_payload(payload: Any) -> set[str]:
    names: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = str(key).lower()
                if key_lower in {
                    "variable_name",
                    "energyplus_variable_name",
                    "requested_variable_name",
                    "name",
                }:
                    if isinstance(value, str) and value.strip():
                        names.add(value.strip())
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(payload)
    return names


def load_variable_manifest_features(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "variable_manifest_path": "",
            "generated_variable_count": 0,
            "has_schedule_value": False,
            "has_zone_air_temperature": False,
            "has_zone_hvac_power_or_rate": False,
            "has_internal_gain_signal": False,
        }

    try:
        payload = load_json(path)
        names = collect_variable_names_from_payload(payload)
    except Exception:
        names = set()

    normalized = {name.lower() for name in names}

    def contains_any(patterns: list[str]) -> bool:
        return any(any(pattern in name for pattern in patterns) for name in normalized)

    return {
        "variable_manifest_path": str(path),
        "generated_variable_count": len(names),
        "has_schedule_value": contains_any(["schedule value"]),
        "has_zone_air_temperature": contains_any(["zone air temperature"]),
        "has_zone_hvac_power_or_rate": contains_any(
            [
                "heating rate",
                "cooling rate",
                "heating energy",
                "cooling energy",
                "hvac",
            ]
        ),
        "has_internal_gain_signal": contains_any(
            [
                "people",
                "lights",
                "electric equipment",
                "gas equipment",
                "internal",
            ]
        ),
    }


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def build_zone_features_from_eio(
    *,
    case_id: str,
    eio_tables_path: str | Path,
    variable_manifest_path: str | Path | None = None,
) -> ZoneFeatureResult:
    eio_path = Path(eio_tables_path).expanduser().resolve()
    variable_manifest = (
        Path(variable_manifest_path).expanduser().resolve()
        if variable_manifest_path
        else None
    )

    eio_payload = load_json(eio_path)

    inventory = build_zone_inventory_from_eio(
        case_id=case_id,
        eio_tables_path=eio_path,
    )

    zone_internal_rows = get_table_rows(eio_payload, "Zone Internal Gains Nominal")
    people_rows = get_table_rows(eio_payload, "People Internal Gains Nominal")
    lights_rows = get_table_rows(eio_payload, "Lights Internal Gains Nominal")
    electric_rows = get_table_rows(eio_payload, "ElectricEquipment Internal Gains Nominal")
    gas_rows = get_table_rows(eio_payload, "GasEquipment Internal Gains Nominal")

    zone_internal_occupants = first_numeric_by_zone(
        zone_internal_rows,
        zone_column="Zone Name",
        value_column="# Occupants",
    )
    zone_internal_people_per_m2 = first_numeric_by_zone(
        zone_internal_rows,
        zone_column="Zone Name",
        value_column="Occupant per Area {person/m2}",
    )
    zone_internal_lights_w_per_m2 = first_numeric_by_zone(
        zone_internal_rows,
        zone_column="Zone Name",
        value_column="Interior Lighting {W/m2}",
    )
    zone_internal_electric_w_per_m2 = first_numeric_by_zone(
        zone_internal_rows,
        zone_column="Zone Name",
        value_column="Electric Load {W/m2}",
    )
    zone_internal_gas_w_per_m2 = first_numeric_by_zone(
        zone_internal_rows,
        zone_column="Zone Name",
        value_column="Gas Load {W/m2}",
    )
    zone_internal_sum_loads_w_per_m2 = first_numeric_by_zone(
        zone_internal_rows,
        zone_column="Zone Name",
        value_column="Sum Loads per Area {W/m2}",
    )

    people_level = sum_by_zone(
        people_rows,
        zone_column="Zone Name",
        value_column="Number of People {}",
    )
    lights_level_w = sum_by_zone(
        lights_rows,
        zone_column="Zone Name",
        value_column="Lighting Level {W}",
    )
    electric_level_w = sum_by_zone(
        electric_rows,
        zone_column="Zone Name",
        value_column="Equipment Level {W}",
    )
    gas_level_w = sum_by_zone(
        gas_rows,
        zone_column="Zone Name",
        value_column="Equipment Level {W}",
    )

    people_schedule = schedules_by_zone(people_rows)
    lights_schedule = schedules_by_zone(lights_rows)
    electric_schedule = schedules_by_zone(electric_rows)
    gas_schedule = schedules_by_zone(gas_rows)

    people_object_count = count_objects_by_zone(people_rows)
    lights_object_count = count_objects_by_zone(lights_rows)
    electric_object_count = count_objects_by_zone(electric_rows)
    gas_object_count = count_objects_by_zone(gas_rows)

    variable_features = load_variable_manifest_features(variable_manifest)

    feature_rows: list[dict[str, Any]] = []

    for inv_row in inventory.zone_inventory_rows:
        zone = normalize_zone_name(inv_row.zone_name)
        inv = inventory_row_to_dict(inv_row)

        floor_area = inv_row.floor_area_m2
        exterior_gross_wall_area = inv_row.exterior_gross_wall_area_m2
        exterior_window_area = inv_row.exterior_window_area_m2
        exterior_net_wall_area = inv_row.exterior_net_wall_area_m2

        people_total = people_level.get(zone, 0.0)
        lights_total = lights_level_w.get(zone, 0.0)
        electric_total = electric_level_w.get(zone, 0.0)
        gas_total = gas_level_w.get(zone, 0.0)

        feature = {
            **inv,
            "zone_name_normalized": zone,
            "default_max_aggregate_zones": inventory.default_max_aggregate_zones,
            "geometry_source": "EnergyPlus EIO Zone Information table",
            "uses_raw_idf": False,
            "uses_opyplus": False,
            "window_to_gross_wall_area_ratio": safe_divide(
                exterior_window_area, exterior_gross_wall_area
            ),
            "window_to_net_wall_area_ratio": safe_divide(
                exterior_window_area, exterior_net_wall_area
            ),
            "has_exterior_window_area": bool(
                exterior_window_area is not None and exterior_window_area > 0
            ),
            "has_exterior_wall_area": bool(
                exterior_gross_wall_area is not None and exterior_gross_wall_area > 0
            ),
            "people_object_count": people_object_count.get(zone, 0),
            "people_level": people_total,
            "people_per_m2_from_people_table": safe_divide(people_total, floor_area),
            "people_per_m2_from_zone_internal_gains": zone_internal_people_per_m2.get(zone),
            "people_schedule_names": people_schedule.get(zone, ""),
            "lights_object_count": lights_object_count.get(zone, 0),
            "lights_level_w": lights_total,
            "lights_w_per_m2_from_lights_table": safe_divide(lights_total, floor_area),
            "lights_w_per_m2_from_zone_internal_gains": zone_internal_lights_w_per_m2.get(zone),
            "lights_schedule_names": lights_schedule.get(zone, ""),
            "electric_equipment_object_count": electric_object_count.get(zone, 0),
            "electric_equipment_level_w": electric_total,
            "electric_equipment_w_per_m2_from_equipment_table": safe_divide(
                electric_total, floor_area
            ),
            "electric_w_per_m2_from_zone_internal_gains": zone_internal_electric_w_per_m2.get(zone),
            "electric_equipment_schedule_names": electric_schedule.get(zone, ""),
            "gas_equipment_object_count": gas_object_count.get(zone, 0),
            "gas_equipment_level_w": gas_total,
            "gas_equipment_w_per_m2_from_equipment_table": safe_divide(
                gas_total, floor_area
            ),
            "gas_w_per_m2_from_zone_internal_gains": zone_internal_gas_w_per_m2.get(zone),
            "gas_equipment_schedule_names": gas_schedule.get(zone, ""),
            "zone_internal_occupants": zone_internal_occupants.get(zone),
            "zone_internal_sum_loads_w_per_m2": zone_internal_sum_loads_w_per_m2.get(zone),
            "total_static_internal_load_w": lights_total + electric_total + gas_total,
            "total_static_internal_load_w_per_m2": safe_divide(
                lights_total + electric_total + gas_total,
                floor_area,
            ),
            **variable_features,
        }

        feature_rows.append(feature)

    return ZoneFeatureResult(
        case_id=case_id,
        eio_tables_path=eio_path,
        variable_manifest_path=variable_manifest,
        zone_feature_rows=feature_rows,
    )


ZONE_FEATURE_FIELDNAMES = [
    "case_id",
    "zone_name",
    "zone_name_normalized",
    "include_flag",
    "exclude_reason",
    "part_of_total_building_area",
    "default_max_aggregate_zones",
    "geometry_source",
    "uses_raw_idf",
    "uses_opyplus",
    "north_axis_deg",
    "origin_x_m",
    "origin_y_m",
    "origin_z_m",
    "centroid_x_m",
    "centroid_y_m",
    "centroid_z_m",
    "minimum_x_m",
    "maximum_x_m",
    "minimum_y_m",
    "maximum_y_m",
    "minimum_z_m",
    "maximum_z_m",
    "ceiling_height_m",
    "volume_m3",
    "floor_area_m2",
    "exterior_gross_wall_area_m2",
    "exterior_net_wall_area_m2",
    "exterior_window_area_m2",
    "window_to_gross_wall_area_ratio",
    "window_to_net_wall_area_ratio",
    "has_exterior_window_area",
    "has_exterior_wall_area",
    "number_of_surfaces",
    "number_of_subsurfaces",
    "people_object_count",
    "people_level",
    "people_per_m2_from_people_table",
    "people_per_m2_from_zone_internal_gains",
    "people_schedule_names",
    "lights_object_count",
    "lights_level_w",
    "lights_w_per_m2_from_lights_table",
    "lights_w_per_m2_from_zone_internal_gains",
    "lights_schedule_names",
    "electric_equipment_object_count",
    "electric_equipment_level_w",
    "electric_equipment_w_per_m2_from_equipment_table",
    "electric_w_per_m2_from_zone_internal_gains",
    "electric_equipment_schedule_names",
    "gas_equipment_object_count",
    "gas_equipment_level_w",
    "gas_equipment_w_per_m2_from_equipment_table",
    "gas_w_per_m2_from_zone_internal_gains",
    "gas_equipment_schedule_names",
    "zone_internal_occupants",
    "zone_internal_sum_loads_w_per_m2",
    "total_static_internal_load_w",
    "total_static_internal_load_w_per_m2",
    "variable_manifest_path",
    "generated_variable_count",
    "has_schedule_value",
    "has_zone_air_temperature",
    "has_zone_hvac_power_or_rate",
    "has_internal_gain_signal",
]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_zone_feature_case_outputs(
    *,
    result: ZoneFeatureResult,
    case_output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(case_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "zone_features.csv",
        result.zone_feature_rows,
        ZONE_FEATURE_FIELDNAMES,
    )
    write_csv(
        out_dir / "approved_zone_features.csv",
        result.approved_rows,
        ZONE_FEATURE_FIELDNAMES,
    )
    write_csv(
        out_dir / "excluded_zone_features.csv",
        result.excluded_rows,
        ZONE_FEATURE_FIELDNAMES,
    )

    approved_zones = [str(row["zone_name"]) for row in result.approved_rows]
    excluded_zones = [str(row["zone_name"]) for row in result.excluded_rows]

    manifest = {
        "case_id": result.case_id,
        "eio_tables_path": str(result.eio_tables_path),
        "variable_manifest_path": (
            str(result.variable_manifest_path) if result.variable_manifest_path else ""
        ),
        "zone_feature_count": len(result.zone_feature_rows),
        "approved_zone_count": result.approved_zone_count,
        "excluded_zone_count": result.excluded_zone_count,
        "default_max_aggregate_zones": result.default_max_aggregate_zones,
        "approved_zones": approved_zones,
        "excluded_zones": excluded_zones,
        "eligibility_rule": "Part of Total Building Area == Yes",
        "feature_source": "EnergyPlus EIO tables",
        "geometry_source": "EnergyPlus EIO Zone Information table",
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "outputs": {
            "zone_features_csv": str(out_dir / "zone_features.csv"),
            "approved_zone_features_csv": str(out_dir / "approved_zone_features.csv"),
            "excluded_zone_features_csv": str(out_dir / "excluded_zone_features.csv"),
        },
    }

    write_json(out_dir / "zone_feature_manifest.json", manifest)
    return manifest