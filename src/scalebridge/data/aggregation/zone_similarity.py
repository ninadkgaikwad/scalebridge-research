"""
EIO-feature-based zone similarity utilities for aggregation suggestions.

Increment 3 purpose:
    - Read approved_zone_features.csv from Increment 2.
    - Compute case-local zone-name tokens.
    - Compute pairwise similarity only across approved zones.
    - Do not generate candidate aggregations yet.
    - Do not parse raw IDF files.
    - Do not use opyplus.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


EPS = 1e-9


@dataclass(frozen=True)
class ZoneSimilarityResult:
    case_id: str
    zone_features_path: Path
    approved_zone_count: int
    token_rows: list[dict[str, Any]]
    pairwise_rows: list[dict[str, Any]]

    @property
    def pair_count(self) -> int:
        return len(self.pairwise_rows)


def timestamp_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def normalize_zone_name(value: Any) -> str:
    return str(value).strip().upper()


def tokenize_zone_name(zone_name: str) -> list[str]:
    """
    Case-local tokenization only. No hardcoded function-name dictionary.

    Examples:
        PERIMETER_ZN_1 -> PERIMETER, ZN
        CORE-ZN        -> CORE, ZN
        Apt201East     -> APT, 201, EAST before numeric removal
    """
    text = normalize_zone_name(zone_name)
    text = re.sub(r"([A-Z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)

    raw_tokens = [tok.strip() for tok in text.split() if tok.strip()]
    tokens = [tok for tok in raw_tokens if not tok.isdigit()]
    return tokens


def build_case_local_token_rows(
    *,
    case_id: str,
    zone_names: list[str],
    common_token_threshold: float = 0.80,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    tokenized = {zone: tokenize_zone_name(zone) for zone in zone_names}

    zone_count = len(zone_names)
    token_zone_counts: dict[str, int] = {}
    for zone, tokens in tokenized.items():
        for token in set(tokens):
            token_zone_counts[token] = token_zone_counts.get(token, 0) + 1

    token_rows: list[dict[str, Any]] = []
    informative_by_zone: dict[str, set[str]] = {}

    for zone in zone_names:
        tokens = tokenized[zone]
        informative: set[str] = set()

        for position, token in enumerate(tokens):
            frequency = token_zone_counts.get(token, 0)
            frequency_ratio = frequency / max(zone_count, 1)

            # Only remove very common tokens when there are enough zones for
            # "commonness" to mean something. For two-zone cases, keep both
            # DINING and KITCHEN as informative unique tokens.
            is_case_common = zone_count >= 3 and frequency_ratio >= common_token_threshold
            is_numeric = token.isdigit()
            is_informative = (not is_numeric) and (not is_case_common)

            if is_informative:
                informative.add(token)

            token_rows.append(
                {
                    "case_id": case_id,
                    "zone_name": zone,
                    "normalized_zone_name": normalize_zone_name(zone),
                    "token": token,
                    "token_position": position,
                    "token_zone_count": frequency,
                    "token_frequency_ratio": frequency_ratio,
                    "token_is_numeric": is_numeric,
                    "token_is_case_common": is_case_common,
                    "token_is_informative": is_informative,
                }
            )

        informative_by_zone[zone] = informative

    return token_rows, informative_by_zone


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def numeric_feature_similarity(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None

    denom = max(abs(a), abs(b), EPS)
    value = 1.0 - abs(a - b) / denom
    return max(0.0, min(1.0, value))


def average_available(values: Iterable[float | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def bool_similarity(a: bool | None, b: bool | None) -> float | None:
    if a is None or b is None:
        return None
    return 1.0 if a == b else 0.0


def schedule_set(value: Any) -> set[str]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()
    return {part.strip() for part in text.split("|") if part.strip()}


def schedule_similarity(a: Any, b: Any) -> float:
    return jaccard_similarity(schedule_set(a), schedule_set(b))


def euclidean_distance(coords_a: list[float | None], coords_b: list[float | None]) -> float | None:
    pairs = [(a, b) for a, b in zip(coords_a, coords_b) if a is not None and b is not None]
    if not pairs:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs))


def compute_case_building_diagonal(rows: list[dict[str, str]]) -> float | None:
    mins_maxes = []
    for min_col, max_col in [
        ("minimum_x_m", "maximum_x_m"),
        ("minimum_y_m", "maximum_y_m"),
        ("minimum_z_m", "maximum_z_m"),
    ]:
        mins = [parse_float(row.get(min_col)) for row in rows]
        maxes = [parse_float(row.get(max_col)) for row in rows]
        mins = [v for v in mins if v is not None]
        maxes = [v for v in maxes if v is not None]
        if not mins or not maxes:
            return None
        mins_maxes.append((min(mins), max(maxes)))

    return math.sqrt(sum((hi - lo) ** 2 for lo, hi in mins_maxes))


def position_similarity(
    row_a: dict[str, str],
    row_b: dict[str, str],
    building_diagonal: float | None,
) -> tuple[float | None, float | None]:
    coords_a = [
        parse_float(row_a.get("centroid_x_m")),
        parse_float(row_a.get("centroid_y_m")),
        parse_float(row_a.get("centroid_z_m")),
    ]
    coords_b = [
        parse_float(row_b.get("centroid_x_m")),
        parse_float(row_b.get("centroid_y_m")),
        parse_float(row_b.get("centroid_z_m")),
    ]

    distance = euclidean_distance(coords_a, coords_b)
    if distance is None or building_diagonal is None or building_diagonal <= EPS:
        return None, distance

    sim = 1.0 - distance / building_diagonal
    return max(0.0, min(1.0, sim)), distance


def feature_similarity_average(
    row_a: dict[str, str],
    row_b: dict[str, str],
    columns: list[str],
) -> float | None:
    values = [
        numeric_feature_similarity(parse_float(row_a.get(col)), parse_float(row_b.get(col)))
        for col in columns
    ]
    return average_available(values)


def exposure_similarity(row_a: dict[str, str], row_b: dict[str, str]) -> float | None:
    numeric_columns = [
        "exterior_gross_wall_area_m2",
        "exterior_net_wall_area_m2",
        "exterior_window_area_m2",
        "window_to_gross_wall_area_ratio",
        "window_to_net_wall_area_ratio",
        "number_of_subsurfaces",
    ]
    bool_columns = [
        "has_exterior_window_area",
        "has_exterior_wall_area",
    ]

    numeric_values = [
        numeric_feature_similarity(parse_float(row_a.get(col)), parse_float(row_b.get(col)))
        for col in numeric_columns
    ]
    bool_values = [
        bool_similarity(parse_bool(row_a.get(col)), parse_bool(row_b.get(col)))
        for col in bool_columns
    ]

    return average_available(numeric_values + bool_values)


def internal_load_similarity(row_a: dict[str, str], row_b: dict[str, str]) -> float | None:
    columns = [
        "people_per_m2_from_people_table",
        "people_per_m2_from_zone_internal_gains",
        "lights_w_per_m2_from_lights_table",
        "lights_w_per_m2_from_zone_internal_gains",
        "electric_equipment_w_per_m2_from_equipment_table",
        "electric_w_per_m2_from_zone_internal_gains",
        "gas_equipment_w_per_m2_from_equipment_table",
        "gas_w_per_m2_from_zone_internal_gains",
        "total_static_internal_load_w_per_m2",
    ]
    return feature_similarity_average(row_a, row_b, columns)


def schedule_similarity_average(row_a: dict[str, str], row_b: dict[str, str]) -> float | None:
    columns = [
        "people_schedule_names",
        "lights_schedule_names",
        "electric_equipment_schedule_names",
        "gas_equipment_schedule_names",
    ]
    return average_available([schedule_similarity(row_a.get(col), row_b.get(col)) for col in columns])


def combined_similarity_score(
    *,
    name_similarity: float | None,
    geometry_size_similarity: float | None,
    geometry_position_similarity: float | None,
    exposure_similarity_value: float | None,
    internal_load_similarity_value: float | None,
    schedule_similarity_value: float | None,
) -> float | None:
    weighted_values = [
        (name_similarity, 0.15),
        (geometry_size_similarity, 0.20),
        (geometry_position_similarity, 0.15),
        (exposure_similarity_value, 0.20),
        (internal_load_similarity_value, 0.25),
        (schedule_similarity_value, 0.05),
    ]

    available = [(v, w) for v, w in weighted_values if v is not None]
    if not available:
        return None

    numerator = sum(v * w for v, w in available)
    denominator = sum(w for _, w in available)
    return numerator / denominator if denominator > 0 else None


def build_zone_similarity_from_features(
    *,
    case_id: str,
    approved_zone_features_path: str | Path,
) -> ZoneSimilarityResult:
    features_path = Path(approved_zone_features_path).expanduser().resolve()
    rows = read_csv_dicts(features_path)

    approved_rows = [
        row for row in rows
        if str(row.get("include_flag", "")).strip().lower() in {"true", "1", "yes"}
    ]

    zone_names = [normalize_zone_name(row.get("zone_name")) for row in approved_rows]
    for row, zone_name in zip(approved_rows, zone_names):
        row["zone_name"] = zone_name

    token_rows, informative_tokens = build_case_local_token_rows(
        case_id=case_id,
        zone_names=zone_names,
    )

    building_diagonal = compute_case_building_diagonal(approved_rows)

    pairwise_rows: list[dict[str, Any]] = []
    n = len(approved_rows)

    for i in range(n):
        for j in range(i + 1, n):
            row_i = approved_rows[i]
            row_j = approved_rows[j]
            zone_i = normalize_zone_name(row_i.get("zone_name"))
            zone_j = normalize_zone_name(row_j.get("zone_name"))

            name_sim = jaccard_similarity(
                informative_tokens.get(zone_i, set()),
                informative_tokens.get(zone_j, set()),
            )

            size_sim = feature_similarity_average(
                row_i,
                row_j,
                [
                    "floor_area_m2",
                    "volume_m3",
                    "exterior_gross_wall_area_m2",
                ],
            )

            pos_sim, centroid_distance_m = position_similarity(
                row_i,
                row_j,
                building_diagonal,
            )

            exp_sim = exposure_similarity(row_i, row_j)
            load_sim = internal_load_similarity(row_i, row_j)
            sched_sim = schedule_similarity_average(row_i, row_j)

            combined = combined_similarity_score(
                name_similarity=name_sim,
                geometry_size_similarity=size_sim,
                geometry_position_similarity=pos_sim,
                exposure_similarity_value=exp_sim,
                internal_load_similarity_value=load_sim,
                schedule_similarity_value=sched_sim,
            )

            pairwise_rows.append(
                {
                    "case_id": case_id,
                    "zone_i": zone_i,
                    "zone_j": zone_j,
                    "zone_i_index": i + 1,
                    "zone_j_index": j + 1,
                    "n_approved_zones": n,
                    "building_diagonal_m": building_diagonal,
                    "centroid_distance_m": centroid_distance_m,
                    "zone_i_informative_tokens": "|".join(sorted(informative_tokens.get(zone_i, set()))),
                    "zone_j_informative_tokens": "|".join(sorted(informative_tokens.get(zone_j, set()))),
                    "name_similarity": name_sim,
                    "geometry_size_similarity": size_sim,
                    "geometry_position_similarity": pos_sim,
                    "exposure_similarity": exp_sim,
                    "internal_load_similarity": load_sim,
                    "schedule_similarity": sched_sim,
                    "combined_similarity": combined,
                    "similarity_weight_name": 0.15,
                    "similarity_weight_geometry_size": 0.20,
                    "similarity_weight_geometry_position": 0.15,
                    "similarity_weight_exposure": 0.20,
                    "similarity_weight_internal_load": 0.25,
                    "similarity_weight_schedule": 0.05,
                }
            )

    return ZoneSimilarityResult(
        case_id=case_id,
        zone_features_path=features_path,
        approved_zone_count=n,
        token_rows=token_rows,
        pairwise_rows=pairwise_rows,
    )


TOKEN_FIELDNAMES = [
    "case_id",
    "zone_name",
    "normalized_zone_name",
    "token",
    "token_position",
    "token_zone_count",
    "token_frequency_ratio",
    "token_is_numeric",
    "token_is_case_common",
    "token_is_informative",
]


PAIRWISE_FIELDNAMES = [
    "case_id",
    "zone_i",
    "zone_j",
    "zone_i_index",
    "zone_j_index",
    "n_approved_zones",
    "building_diagonal_m",
    "centroid_distance_m",
    "zone_i_informative_tokens",
    "zone_j_informative_tokens",
    "name_similarity",
    "geometry_size_similarity",
    "geometry_position_similarity",
    "exposure_similarity",
    "internal_load_similarity",
    "schedule_similarity",
    "combined_similarity",
    "similarity_weight_name",
    "similarity_weight_geometry_size",
    "similarity_weight_geometry_position",
    "similarity_weight_exposure",
    "similarity_weight_internal_load",
    "similarity_weight_schedule",
]


def write_zone_similarity_case_outputs(
    *,
    result: ZoneSimilarityResult,
    case_output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(case_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "zone_name_tokens.csv", result.token_rows, TOKEN_FIELDNAMES)
    write_csv(
        out_dir / "zone_pairwise_similarity.csv",
        result.pairwise_rows,
        PAIRWISE_FIELDNAMES,
    )

    manifest = {
        "case_id": result.case_id,
        "zone_features_path": str(result.zone_features_path),
        "approved_zone_count": result.approved_zone_count,
        "pair_count": result.pair_count,
        "token_row_count": len(result.token_rows),
        "similarity_scope": "approved_zones_only",
        "feature_source": "EIO-only zone_features.csv",
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "weights": {
            "name_similarity": 0.15,
            "geometry_size_similarity": 0.20,
            "geometry_position_similarity": 0.15,
            "exposure_similarity": 0.20,
            "internal_load_similarity": 0.25,
            "schedule_similarity": 0.05,
        },
        "outputs": {
            "zone_name_tokens_csv": str(out_dir / "zone_name_tokens.csv"),
            "zone_pairwise_similarity_csv": str(out_dir / "zone_pairwise_similarity.csv"),
        },
    }

    write_json(out_dir / "zone_similarity_manifest.json", manifest)
    return manifest


def find_latest_zone_features_root(campaign_root: str | Path) -> Path:
    root = Path(campaign_root) / "aggregation" / "zone_features"
    candidates = [p for p in root.glob("zone_features_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No zone_features_* folder found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_approved_zone_features_path(
    *,
    campaign_root: str | Path,
    case_id: str,
    zone_features_root: str | Path | None = None,
) -> Path:
    if zone_features_root is None:
        zone_features_root = find_latest_zone_features_root(campaign_root)

    path = (
        Path(zone_features_root)
        / "cases"
        / case_id
        / "approved_zone_features.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"approved_zone_features.csv not found: {path}")

    return path