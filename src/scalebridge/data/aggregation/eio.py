# -*- coding: utf-8 -*-
"""EIO table helpers for ScaleBridge aggregation."""

from __future__ import annotations

from typing import Any


EQUIPMENT_TYPES = (
    "People",
    "Lights",
    "ElectricEquipment",
    "GasEquipment",
    "OtherEquipment",
    "HotWaterEquipment",
    "SteamEquipment",
)

EQUIPMENT_LEVEL_COLUMNS = {
    "People": "Number of People {}",
    "Lights": "Lighting Level {W}",
    "ElectricEquipment": "Equipment Level {W}",
    "GasEquipment": "Equipment Level {W}",
    "OtherEquipment": "Equipment Level {W}",
    "HotWaterEquipment": "Equipment Level {W}",
    "SteamEquipment": "Equipment Level {W}",
}


def get_eio_table(
    eio_payload: dict[str, Any],
    table_name: str,
) -> dict[str, Any] | None:
    """Return one EIO table payload by table name."""
    tables = eio_payload.get("tables", {})
    if not isinstance(tables, dict):
        return None

    direct = tables.get(table_name)
    if isinstance(direct, dict):
        return direct

    expected = table_name.casefold()
    for key, value in tables.items():
        if str(key).casefold() == expected and isinstance(value, dict):
            return value

    return None


def row_to_dict(columns: list[Any], row: list[Any]) -> dict[str, Any]:
    """Convert an EIO row list into a dictionary."""
    result: dict[str, Any] = {}
    for index, column in enumerate(columns):
        result[str(column)] = row[index] if index < len(row) else ""
    return result


def zone_information_rows(
    *,
    eio_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return normalized Zone Information rows."""
    table = get_eio_table(eio_payload, "Zone Information")
    if table is None:
        return []

    columns = list(table.get("columns", []))
    rows = list(table.get("rows", []))

    output_rows: list[dict[str, Any]] = []

    for raw_row in rows:
        row_dict = row_to_dict(columns, raw_row)
        zone_name = str(row_dict.get("Zone Name", "")).strip()
        part_of_total = str(row_dict.get("Part of Total Building Area", "")).strip()
        included = part_of_total.casefold() == "yes"

        output_rows.append(
            {
                "zone_name": zone_name,
                "included_thermal_zone": "true" if included else "false",
                "exclusion_reason": "" if included else "Part of Total Building Area != Yes",
                "part_of_total_building_area": part_of_total,
                "floor_area_m2": row_dict.get("Floor Area {m2}", ""),
                "volume_m3": row_dict.get("Volume {m3}", ""),
                "zone_multiplier": row_dict.get("Zone Multiplier", ""),
                "zone_list_multiplier": row_dict.get("Zone List Multiplier", ""),
                "ceiling_height_m": row_dict.get("Ceiling Height {m}", ""),
                "number_of_surfaces": row_dict.get("Number of Surfaces", ""),
                "number_of_subsurfaces": row_dict.get("Number of SubSurfaces", ""),
                "centroid_x_m": row_dict.get("Centroid X-Coordinate {m}", ""),
                "centroid_y_m": row_dict.get("Centroid Y-Coordinate {m}", ""),
                "centroid_z_m": row_dict.get("Centroid Z-Coordinate {m}", ""),
            }
        )

    return output_rows


def schedule_equipment_mapping_rows(
    *,
    eio_payload: dict[str, Any],
    included_zone_names: set[str],
) -> list[dict[str, Any]]:
    """Return equipment schedule mappings for included thermal zones."""
    output_rows: list[dict[str, Any]] = []

    for equipment_type in EQUIPMENT_TYPES:
        table_name = f"{equipment_type} Internal Gains Nominal"
        table = get_eio_table(eio_payload, table_name)
        if table is None:
            continue

        columns = list(table.get("columns", []))
        rows = list(table.get("rows", []))
        level_column = EQUIPMENT_LEVEL_COLUMNS[equipment_type]

        for raw_row in rows:
            row_dict = row_to_dict(columns, raw_row)
            zone_name = str(row_dict.get("Zone Name", "")).strip()

            if zone_name not in included_zone_names:
                continue

            output_rows.append(
                {
                    "equipment_type": equipment_type,
                    "eio_table": table_name,
                    "zone_name": zone_name,
                    "schedule_name": row_dict.get("Schedule Name", ""),
                    "equipment_level_column": level_column,
                    "equipment_level": row_dict.get(level_column, ""),
                    "included_thermal_zone": "true",
                }
            )

    return output_rows