import os
import json

# ---------------------------------------------------------------------
# CONFIG: EDIT THESE TWO VARIABLES
# ---------------------------------------------------------------------
IDF_PATH = r"C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\Commercial_Prototypes\ASHRAE\90_1_2013\ASHRAE901_RetailStripmall_STD2013_Buffalo.idf"
OUTPUT_DIR = r"C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge\building_geometry\retail_stripmall"
# ---------------------------------------------------------------------


def parse_blocks(idf_text: str, prefix: str):
    """
    Extract blocks starting with a line whose left-stripped text
    begins with `prefix`, ending at the line containing ';'.
    Example prefixes: 'Zone,', 'BuildingSurface:Detailed,'.[file:1][web:28]
    """
    blocks = []
    current = []
    in_block = False

    for ln in idf_text.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith(prefix):
            if current:
                blocks.append("\n".join(current))
            current = [ln]
            in_block = True
        elif in_block:
            current.append(ln)
            if ";" in ln:
                blocks.append("\n".join(current))
                current = []
                in_block = False

    if current:
        blocks.append("\n".join(current))

    return blocks


def parse_zones(idf_text: str):
    """
    Parse Zone objects and return dict:
      name -> {x_origin, y_origin, z_origin}
    by searching for lines containing '!- Name', '!- X Origin', etc.[file:1][web:74]
    """
    blocks = parse_blocks(idf_text, "Zone,")
    zones = {}

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue

        # Helper to extract the value before ', !-'
        def value_str(line):
            if "!- " in line:
                line = line.split("!-", 1)[0]
            line = line.replace(";", "")
            return line.split(",")[0].strip()

        # Find name
        name = None
        for ln in lines:
            if "!- Name" in ln:
                name = value_str(ln)
                break
        if not name:
            continue

        # Find X, Y, Z Origin lines (if present)
        x_origin = 0.0
        y_origin = 0.0
        z_origin = 0.0

        for ln in lines:
            if "!- X Origin" in ln:
                try:
                    x_origin = float(value_str(ln))
                except ValueError:
                    x_origin = 0.0
            elif "!- Y Origin" in ln:
                try:
                    y_origin = float(value_str(ln))
                except ValueError:
                    y_origin = 0.0
            elif "!- Z Origin" in ln:
                try:
                    z_origin = float(value_str(ln))
                except ValueError:
                    z_origin = 0.0

        zones[name] = {
            "x_origin": x_origin,
            "y_origin": y_origin,
            "z_origin": z_origin,
        }

    return zones


def parse_buildingsurface_detailed(idf_text: str):
    """
    Parse BuildingSurface:Detailed objects and return list of dicts:
    {
      'name', 'surface_type', 'construction_name',
      'zone_name', 'number_of_vertices', 'vertices': [[x,y,z], ...]
    }[web:28]
    """
    blocks = parse_blocks(idf_text, "BuildingSurface:Detailed,")
    surfaces = []

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        if len(lines) < 5:
            continue

        def value(line: str):
            if "!--" in line or "!-" in line:
                line = line.split("!-", 1)[0]
            line = line.replace(";", "")
            return line.strip()

        # Line indices per EnergyPlus canonical order:[web:28]
        # 0: object type, 1: Name, 2: Surface Type, 3: Construction Name, 4: Zone Name
        name = value(lines[1]).split(",")[0].strip()
        surface_type = value(lines[2]).split(",")[0].strip()
        construction_name = value(lines[3]).split(",")[0].strip()
        zone_name = value(lines[4]).split(",")[0].strip()

        # Find number_of_vertices: first integer after the view-factor line.
        number_of_vertices = None
        verts_start_idx = None
        for idx, line in enumerate(lines):
            val = value(line)
            try:
                n = int(val.split(",")[0].strip())
                if idx >= 6:
                    number_of_vertices = n
                    verts_start_idx = idx + 1
                    break
            except ValueError:
                continue

        vertices = []
        if number_of_vertices is not None and verts_start_idx is not None:
            for i in range(verts_start_idx,
                           min(len(lines), verts_start_idx + number_of_vertices)):
                val = value(lines[i])
                coords = [c.strip() for c in val.split(",") if c.strip()]
                if len(coords) < 3:
                    continue
                try:
                    x = float(coords[0])
                    y = float(coords[1])
                    z = float(coords[2])
                    vertices.append([x, y, z])
                except ValueError:
                    continue

        surfaces.append({
            "name": name,
            "surface_type": surface_type,
            "construction_name": construction_name,
            "zone_name": zone_name,
            "number_of_vertices": number_of_vertices or len(vertices),
            "vertices": vertices,
        })

    return surfaces


def extract_geometry_with_origins(idf_path: str, output_dir: str) -> str:
    """
    Extract exact 3D geometry + zone origins from an EnergyPlus IDF
    and save as geometry.json in output_dir.
    """
    with open(idf_path, "r", encoding="utf-8", errors="ignore") as f:
        idf_text = f.read()

    # Zone origins
    zone_origins = parse_zones(idf_text)

    # All BuildingSurface:Detailed surfaces
    surfaces = parse_buildingsurface_detailed(idf_text)

    # Group surfaces by zone_name
    zone_surface_map = {}
    for s in surfaces:
        zn = s["zone_name"]
        zone_surface_map.setdefault(zn, []).append(s)

    # Build final JSON structure
    zone_names = list(zone_origins.keys())
    data = {
        "idf_path": os.path.abspath(idf_path),
        "zones": []
    }

    # Include all declared Zones (Dining, Kitchen, attic, etc.)[file:1]
    for zn in zone_names:
        data["zones"].append({
            "name": zn,
            "origin": zone_origins.get(zn, {
                "x_origin": 0.0,
                "y_origin": 0.0,
                "z_origin": 0.0,
            }),
            "surfaces": zone_surface_map.get(zn, []),
        })

    # Include any zones that appear only in surfaces
    for zn in zone_surface_map:
        if zn not in zone_origins:
            data["zones"].append({
                "name": zn,
                "origin": {
                    "x_origin": 0.0,
                    "y_origin": 0.0,
                    "z_origin": 0.0,
                },
                "surfaces": zone_surface_map[zn],
            })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "geometry.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Exported geometry + origins for {len(data['zones'])} zones to {out_path}")
    return out_path


if __name__ == "__main__":
    extract_geometry_with_origins(IDF_PATH, OUTPUT_DIR)