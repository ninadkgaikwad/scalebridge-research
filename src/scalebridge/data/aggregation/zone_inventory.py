"""
EIO-based zone inventory utilities for ScaleBridge aggregation.

Increment 1 purpose:
    - Read generated EnergyPlus eio_tables.json.
    - Extract the Zone Information table.
    - Approve zones only when "Part of Total Building Area" == "Yes".
    - Write per-case inventory outputs for later grouping suggestions and custom plans.

Important design rule:
    This module does not parse raw IDF files and does not use opyplus.
    EIO is the source of truth for zone eligibility and available geometry metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


ZONE_INFORMATION_TABLE_NAME = "Zone Information"
PART_OF_TOTAL_BUILDING_AREA_COLUMN = "Part of Total Building Area"


@dataclass(frozen=True)
class ZoneInventoryRow:
    case_id: str
    zone_name: str
    include_flag: bool
    exclude_reason: str
    part_of_total_building_area: str

    north_axis_deg: float | None = None
    origin_x_m: float | None = None
    origin_y_m: float | None = None
    origin_z_m: float | None = None
    centroid_x_m: float | None = None
    centroid_y_m: float | None = None
    centroid_z_m: float | None = None
    minimum_x_m: float | None = None
    maximum_x_m: float | None = None
    minimum_y_m: float | None = None
    maximum_y_m: float | None = None
    minimum_z_m: float | None = None
    maximum_z_m: float | None = None
    ceiling_height_m: float | None = None
    volume_m3: float | None = None
    floor_area_m2: float | None = None
    exterior_gross_wall_area_m2: float | None = None
    exterior_net_wall_area_m2: float | None = None
    exterior_window_area_m2: float | None = None
    number_of_surfaces: int | None = None
    number_of_subsurfaces: int | None = None


@dataclass(frozen=True)
class ZoneInventoryResult:
    case_id: str
    eio_tables_path: Path
    zone_inventory_rows: list[ZoneInventoryRow]

    @property
    def approved_rows(self) -> list[ZoneInventoryRow]:
        return [row for row in self.zone_inventory_rows if row.include_flag]

    @property
    def excluded_rows(self) -> list[ZoneInventoryRow]:
        return [row for row in self.zone_inventory_rows if not row.include_flag]

    @property
    def approved_zone_count(self) -> int:
        return len(self.approved_rows)

    @property
    def excluded_zone_count(self) -> int:
        return len(self.excluded_rows)

    @property
    def default_max_aggregate_zones(self) -> int:
        # Per current design decision:
        # if user does not override, suggestions may go up to identity/native approved-zone count.
        return self.approved_zone_count


def resolve_generated_data_root(explicit_generated_data_root: str | Path | None = None) -> Path:
    if explicit_generated_data_root is not None:
        return Path(explicit_generated_data_root).expanduser().resolve()

    env_value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
    if not env_value:
        raise RuntimeError(
            "SCALEBRIDGE_GENERATED_DATA_ROOT is not set. "
            "Pass --generated-data-root or set the environment variable."
        )

    return Path(env_value).expanduser().resolve()


def resolve_campaign_root(
    campaign_id: str,
    generated_data_root: str | Path | None = None,
    campaign_root: str | Path | None = None,
) -> Path:
    if campaign_root is not None:
        return Path(campaign_root).expanduser().resolve()

    root = resolve_generated_data_root(generated_data_root)
    return root / "campaigns" / campaign_id


def timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")

    return payload


def find_case_dirs(campaign_root: Path, case_id: str | None = None) -> list[Path]:
    cases_root = campaign_root / "generation" / "cases"
    if not cases_root.exists():
        raise FileNotFoundError(f"Generation cases root not found: {cases_root}")

    if case_id:
        case_dir = cases_root / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Requested case_id not found: {case_dir}")
        return [case_dir]

    return sorted([p for p in cases_root.iterdir() if p.is_dir()])


def find_latest_eio_tables_path(case_dir: Path) -> Path:
    """
    Robust first implementation.

    We avoid depending on exact latest_run.json schema by searching under:
        generation/cases/<case_id>/runs/*/canonical/eio_tables.json

    If multiple exist, choose the most recently modified file.
    """
    candidates = list(case_dir.glob("runs/*/canonical/eio_tables.json"))
    if not candidates:
        raise FileNotFoundError(f"No canonical/eio_tables.json found under {case_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def get_eio_table(eio_payload: dict[str, Any], table_name: str) -> dict[str, Any]:
    tables = eio_payload.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("Expected eio_tables.json to contain a dict field named 'tables'.")

    table = tables.get(table_name)
    if table is None:
        available = ", ".join(sorted(tables.keys())[:20])
        raise KeyError(
            f"Required EIO table '{table_name}' not found. "
            f"First available tables: {available}"
        )

    if not isinstance(table, dict):
        raise ValueError(f"EIO table '{table_name}' is not a dict.")

    columns = table.get("columns")
    rows = table.get("rows")

    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError(
            f"EIO table '{table_name}' must contain list fields 'columns' and 'rows'."
        )

    return table


def normalize_column_name(column: str) -> str:
    return (
        column.strip()
        .replace("{", "")
        .replace("}", "")
        .replace("#", "n")
        .replace("/", "_per_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
        .lower()
    )


def table_rows_as_dicts(table: dict[str, Any]) -> list[dict[str, str]]:
    columns = [str(c) for c in table["columns"]]
    rows = table["rows"]

    out: list[dict[str, str]] = []
    for row in rows:
        values = ["" if v is None else str(v) for v in row]
        item = {}
        for idx, col in enumerate(columns):
            item[col] = values[idx] if idx < len(values) else ""
        out.append(item)

    return out


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(round(number))


def truthy_yes(value: Any) -> bool:
    return str(value).strip().lower() == "yes"


def get_first(row: dict[str, str], *column_names: str) -> str:
    for col in column_names:
        if col in row:
            return row[col]
    return ""


def build_zone_inventory_from_eio(
    *,
    case_id: str,
    eio_tables_path: str | Path,
) -> ZoneInventoryResult:
    eio_path = Path(eio_tables_path).expanduser().resolve()
    payload = load_json(eio_path)
    table = get_eio_table(payload, ZONE_INFORMATION_TABLE_NAME)

    raw_rows = table_rows_as_dicts(table)
    inventory_rows: list[ZoneInventoryRow] = []

    for raw in raw_rows:
        zone_name = get_first(raw, "Zone Name").strip()
        if not zone_name:
            continue

        part_of_total = get_first(raw, PART_OF_TOTAL_BUILDING_AREA_COLUMN).strip()
        include_flag = truthy_yes(part_of_total)
        exclude_reason = "" if include_flag else "Part of Total Building Area != Yes"

        inventory_rows.append(
            ZoneInventoryRow(
                case_id=case_id,
                zone_name=zone_name.upper(),
                include_flag=include_flag,
                exclude_reason=exclude_reason,
                part_of_total_building_area=part_of_total,
                north_axis_deg=parse_float(get_first(raw, "North Axis {deg}")),
                origin_x_m=parse_float(get_first(raw, "Origin X-Coordinate {m}")),
                origin_y_m=parse_float(get_first(raw, "Origin Y-Coordinate {m}")),
                origin_z_m=parse_float(get_first(raw, "Origin Z-Coordinate {m}")),
                centroid_x_m=parse_float(get_first(raw, "Centroid X-Coordinate {m}")),
                centroid_y_m=parse_float(get_first(raw, "Centroid Y-Coordinate {m}")),
                centroid_z_m=parse_float(get_first(raw, "Centroid Z-Coordinate {m}")),
                minimum_x_m=parse_float(get_first(raw, "Minimum X {m}")),
                maximum_x_m=parse_float(get_first(raw, "Maximum X {m}")),
                minimum_y_m=parse_float(get_first(raw, "Minimum Y {m}")),
                maximum_y_m=parse_float(get_first(raw, "Maximum Y {m}")),
                minimum_z_m=parse_float(get_first(raw, "Minimum Z {m}")),
                maximum_z_m=parse_float(get_first(raw, "Maximum Z {m}")),
                ceiling_height_m=parse_float(get_first(raw, "Ceiling Height {m}")),
                volume_m3=parse_float(get_first(raw, "Volume {m3}")),
                floor_area_m2=parse_float(get_first(raw, "Floor Area {m2}")),
                exterior_gross_wall_area_m2=parse_float(
                    get_first(raw, "Exterior Gross Wall Area {m2}")
                ),
                exterior_net_wall_area_m2=parse_float(
                    get_first(raw, "Exterior Net Wall Area {m2}")
                ),
                exterior_window_area_m2=parse_float(
                    get_first(raw, "Exterior Window Area {m2}")
                ),
                number_of_surfaces=parse_int(get_first(raw, "Number of Surfaces")),
                number_of_subsurfaces=parse_int(get_first(raw, "Number of SubSurfaces")),
            )
        )

    return ZoneInventoryResult(
        case_id=case_id,
        eio_tables_path=eio_path,
        zone_inventory_rows=inventory_rows,
    )


ZONE_INVENTORY_FIELDNAMES = [
    "case_id",
    "zone_name",
    "include_flag",
    "exclude_reason",
    "part_of_total_building_area",
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
    "number_of_surfaces",
    "number_of_subsurfaces",
]


def row_to_dict(row: ZoneInventoryRow) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "zone_name": row.zone_name,
        "include_flag": row.include_flag,
        "exclude_reason": row.exclude_reason,
        "part_of_total_building_area": row.part_of_total_building_area,
        "north_axis_deg": row.north_axis_deg,
        "origin_x_m": row.origin_x_m,
        "origin_y_m": row.origin_y_m,
        "origin_z_m": row.origin_z_m,
        "centroid_x_m": row.centroid_x_m,
        "centroid_y_m": row.centroid_y_m,
        "centroid_z_m": row.centroid_z_m,
        "minimum_x_m": row.minimum_x_m,
        "maximum_x_m": row.maximum_x_m,
        "minimum_y_m": row.minimum_y_m,
        "maximum_y_m": row.maximum_y_m,
        "minimum_z_m": row.minimum_z_m,
        "maximum_z_m": row.maximum_z_m,
        "ceiling_height_m": row.ceiling_height_m,
        "volume_m3": row.volume_m3,
        "floor_area_m2": row.floor_area_m2,
        "exterior_gross_wall_area_m2": row.exterior_gross_wall_area_m2,
        "exterior_net_wall_area_m2": row.exterior_net_wall_area_m2,
        "exterior_window_area_m2": row.exterior_window_area_m2,
        "number_of_surfaces": row.number_of_surfaces,
        "number_of_subsurfaces": row.number_of_subsurfaces,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_zone_inventory_case_outputs(
    *,
    result: ZoneInventoryResult,
    case_output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(case_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = [row_to_dict(row) for row in result.zone_inventory_rows]
    approved_rows = [row_to_dict(row) for row in result.approved_rows]
    excluded_rows = [row_to_dict(row) for row in result.excluded_rows]

    write_csv(out_dir / "zone_inventory.csv", all_rows, ZONE_INVENTORY_FIELDNAMES)
    write_csv(out_dir / "approved_zones.csv", approved_rows, ZONE_INVENTORY_FIELDNAMES)
    write_csv(out_dir / "excluded_zones.csv", excluded_rows, ZONE_INVENTORY_FIELDNAMES)

    manifest = {
        "case_id": result.case_id,
        "eio_tables_path": str(result.eio_tables_path),
        "zone_count": len(result.zone_inventory_rows),
        "approved_zone_count": result.approved_zone_count,
        "excluded_zone_count": result.excluded_zone_count,
        "default_max_aggregate_zones": result.default_max_aggregate_zones,
        "approved_zones": [row.zone_name for row in result.approved_rows],
        "excluded_zones": [row.zone_name for row in result.excluded_rows],
        "eligibility_rule": "Part of Total Building Area == Yes",
        "geometry_source": "EnergyPlus EIO Zone Information table",
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "outputs": {
            "zone_inventory_csv": str(out_dir / "zone_inventory.csv"),
            "approved_zones_csv": str(out_dir / "approved_zones.csv"),
            "excluded_zones_csv": str(out_dir / "excluded_zones.csv"),
        },
    }

    write_json(out_dir / "zone_inventory_manifest.json", manifest)
    return manifest