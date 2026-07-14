import os
import json

# ---------------------------------------------------------------------
# CONFIG: EDIT THESE TWO VARIABLES
# ---------------------------------------------------------------------
IDF_PATH = r"C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\Commercial_Prototypes\ASHRAE\90_1_2013\ASHRAE901_ApartmentMidRise_STD2013_Buffalo.idf"
OUTPUT_DIR = r"C:\Users\ninad\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge\building_geometry\apartment_mid_rise"
# ---------------------------------------------------------------------



def parse_blocks(idf_text: str, prefix: str):
    """
    Extract blocks starting with a line whose left-stripped text
    begins with `prefix`, and ending at the line containing ';'.
    Example prefixes: 'Zone,', 'BuildingSurface:Detailed,'.[file:1][web:30]
    """
    blocks = []
    current = []
    in_block = False

    for ln in idf_text.splitlines():
        stripped = ln.lstrip()
        if stripped.startswith(prefix):
            # start new block
            if current:
                blocks.append("\n".join(current))
            current = [ln]
            in_block = True
        elif in_block:
            current.append(ln)
            if ";" in ln:
                # end of block
                blocks.append("\n".join(current))
                current = []
                in_block = False

    if current:
        blocks.append("\n".join(current))

    return blocks


def parse_zone_blocks(idf_text: str):
    """Parse Zone objects and return a list of zone names.[file:1]"""
    blocks = parse_blocks(idf_text, "Zone,")
    zone_names = []

    for blk in blocks:
        lines = blk.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        # e.g. "Zone," then on next line "Dining, !- Name"
        # So we look at the second line for the actual name.[file:1]
        if len(lines) > 1:
            second_line = lines[1].strip()
            # Strip comments and semicolon
            if "!- " in second_line:
                second_line = second_line.split("!-", 1)[0]
            second_line = second_line.replace(";", "")
            parts = [p.strip() for p in second_line.split(",") if p.strip()]
            if parts:
                zone_names.append(parts[0])

    return zone_names


def parse_buildingsurface_detailed(idf_text: str):
    """
    Parse BuildingSurface:Detailed objects and return a list of dicts:
    {
        'name': ...,
        'surface_type': ...,
        'construction_name': ...,
        'zone_name': ...,
        'number_of_vertices': int,
        'vertices': [[x,y,z], ...]
    }[web:30][web:33]
    """
    blocks = parse_blocks(idf_text, "BuildingSurface:Detailed,")
    surfaces = []

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]

        if len(lines) < 5:
            continue

        def value_from_line(line: str):
            # remove comment starting with "!-"
            if "!--" in line or "!-" in line:
                line = line.split("!-", 1)[0]
            # remove trailing semicolon
            line = line.replace(";", "")
            return line.strip()

        # First line is just the object type, so Name is on line 1.[web:30]
        # Line 1: "Name, !- Name"
        name_line = value_from_line(lines[1])
        name = name_line.split(",")[0].strip()

        # Line 2: "Surface Type, !- Surface Type"
        surf_type_line = value_from_line(lines[2])
        surface_type = surf_type_line.split(",")[0].strip()

        # Line 3: "Construction Name, !- Construction Name"
        construction_line = value_from_line(lines[3])
        construction_name = construction_line.split(",")[0].strip()

        # Line 4: "Zone Name, !- Zone Name"
        zone_line = value_from_line(lines[4])
        zone_name = zone_line.split(",")[0].strip()

        # Find number_of_vertices: first integer after the view-factor line
        number_of_vertices = None
        verts_start_idx = None
        for idx, line in enumerate(lines):
            val = value_from_line(line)
            # Try to parse as integer
            try:
                n = int(val.split(",")[0].strip())
                if idx >= 6:  # heuristic: vertices count appears a bit later
                    number_of_vertices = n
                    verts_start_idx = idx + 1
                    break
            except ValueError:
                continue

        vertices = []
        if number_of_vertices is not None and verts_start_idx is not None:
            for i in range(verts_start_idx,
                           min(len(lines), verts_start_idx + number_of_vertices)):
                val = value_from_line(lines[i])
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


def extract_geometry(idf_path: str, output_dir: str) -> str:
    """
    Extract exact 3D geometry for all zones from an EnergyPlus IDF
    (parsing text directly) and save as geometry.json in output_dir.[file:1]
    """
    with open(idf_path, "r", encoding="utf-8", errors="ignore") as f:
        idf_text = f.read()

    # All Zone names
    zone_names = parse_zone_blocks(idf_text)

    # All BuildingSurface:Detailed surfaces
    surfaces = parse_buildingsurface_detailed(idf_text)

    # Group surfaces by zone_name
    zone_surface_map = {}
    for s in surfaces:
        zn = s["zone_name"]
        zone_surface_map.setdefault(zn, []).append(s)

    data = {
        "idf_path": os.path.abspath(idf_path),
        "zones": []
    }

    # Include every declared Zone
    for zn in zone_names:
        data["zones"].append({
            "name": zn,
            "surfaces": zone_surface_map.get(zn, [])
        })

    # Include any zones that appear only in surfaces (helper/pseudo zones)
    for zn in zone_surface_map:
        if zn not in zone_names:
            data["zones"].append({
                "name": zn,
                "surfaces": zone_surface_map[zn]
            })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "geometry.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Exported geometry for {len(data['zones'])} zones to {out_path}")
    return out_path


if __name__ == "__main__":
    extract_geometry(IDF_PATH, OUTPUT_DIR)