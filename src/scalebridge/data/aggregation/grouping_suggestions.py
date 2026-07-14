"""
Complete-partition grouping suggestions for ScaleBridge aggregation.

Increment 4 purpose:
    - Read approved zone features from Increment 2.
    - Read pairwise zone similarity from Increment 3.
    - Generate complete candidate partitions of approved zones.
    - Do not build formal aggregation plans yet.
    - Do not run aggregation.
    - Do not parse raw IDF files.
    - Do not use opyplus.

Design rules:
    - Every suggestion covers all approved zones exactly once.
    - No excluded zones are used.
    - Default max aggregate zones equals approved zone count.
    - User-provided max aggregate zones is a ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GroupingSuggestionConfig:
    max_aggregate_zones: int | None = None
    max_candidates_per_k: int = 1
    max_candidates_per_case: int | None = None


@dataclass(frozen=True)
class GroupingSuggestionResult:
    case_id: str
    approved_zone_features_path: Path
    zone_pairwise_similarity_path: Path
    approved_zones: list[str]
    effective_max_aggregate_zones: int
    suggestion_rows: list[dict[str, Any]]
    suggestion_payload: dict[str, Any]
    rationale_markdown: str

    @property
    def approved_zone_count(self) -> int:
        return len(self.approved_zones)

    @property
    def suggestion_count(self) -> int:
        return len(self.suggestion_payload.get("suggestions", []))


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_zone_name(value: Any) -> str:
    return str(value).strip().upper()


def safe_slug(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "zone"


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def infer_building_type(feature_rows: list[dict[str, str]]) -> str:
    # The current approved_zone_features.csv may not yet include building_type.
    # Keep this robust and update later when case manifest joins are added.
    for row in feature_rows:
        value = row.get("building_type")
        if value:
            return str(value)
    return ""


def get_approved_zones(feature_rows: list[dict[str, str]]) -> list[str]:
    zones: list[str] = []
    seen: set[str] = set()

    for row in feature_rows:
        include_text = str(row.get("include_flag", "")).strip().lower()
        if include_text not in {"true", "1", "yes"}:
            continue

        zone = normalize_zone_name(row.get("zone_name"))
        if not zone or zone in seen:
            continue

        zones.append(zone)
        seen.add(zone)

    return zones


def build_similarity_lookup(pairwise_rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}

    for row in pairwise_rows:
        zone_i = normalize_zone_name(row.get("zone_i"))
        zone_j = normalize_zone_name(row.get("zone_j"))
        if not zone_i or not zone_j:
            continue

        sim = parse_float(row.get("combined_similarity"), default=0.0)
        lookup[(zone_i, zone_j)] = sim
        lookup[(zone_j, zone_i)] = sim

    return lookup


def pair_similarity(
    zone_a: str,
    zone_b: str,
    similarity_lookup: dict[tuple[str, str], float],
) -> float:
    if zone_a == zone_b:
        return 1.0
    return similarity_lookup.get((zone_a, zone_b), 0.0)


def group_similarity(
    group_a: list[str],
    group_b: list[str],
    similarity_lookup: dict[tuple[str, str], float],
) -> float:
    values: list[float] = []
    for zone_a in group_a:
        for zone_b in group_b:
            values.append(pair_similarity(zone_a, zone_b, similarity_lookup))

    if not values:
        return 0.0

    return sum(values) / len(values)


def average_within_group_similarity(
    group: list[str],
    similarity_lookup: dict[tuple[str, str], float],
) -> float | None:
    if len(group) <= 1:
        return None

    values: list[float] = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            values.append(pair_similarity(group[i], group[j], similarity_lookup))

    if not values:
        return None
    return sum(values) / len(values)


def average_candidate_within_similarity(
    groups: list[list[str]],
    similarity_lookup: dict[tuple[str, str], float],
) -> float | None:
    values = [
        sim for group in groups
        if (sim := average_within_group_similarity(group, similarity_lookup)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def build_all_to_one_partition(approved_zones: list[str]) -> list[list[str]]:
    return [list(approved_zones)]


def build_identity_partition(approved_zones: list[str]) -> list[list[str]]:
    return [[zone] for zone in approved_zones]


def build_agglomerative_partition(
    *,
    approved_zones: list[str],
    target_k: int,
    similarity_lookup: dict[tuple[str, str], float],
) -> list[list[str]]:
    """
    Deterministic agglomerative merging.

    Start with one group per approved zone.
    Repeatedly merge the two groups with highest average cross-group similarity.
    Stop when target_k groups remain.

    Tie-breaking is deterministic using sorted group signatures.
    """
    groups: list[list[str]] = [[zone] for zone in approved_zones]

    while len(groups) > target_k:
        best_pair: tuple[int, int] | None = None
        best_score: float | None = None
        best_tiebreak: tuple[str, str] | None = None

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                score = group_similarity(groups[i], groups[j], similarity_lookup)
                sig_i = "|".join(sorted(groups[i]))
                sig_j = "|".join(sorted(groups[j]))
                tiebreak = tuple(sorted([sig_i, sig_j]))

                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (i, j)
                    best_tiebreak = tiebreak
                elif score == best_score and best_tiebreak is not None and tiebreak < best_tiebreak:
                    best_pair = (i, j)
                    best_tiebreak = tiebreak

        if best_pair is None:
            break

        i, j = best_pair
        merged = sorted(groups[i] + groups[j])

        new_groups: list[list[str]] = []
        for idx, group in enumerate(groups):
            if idx not in {i, j}:
                new_groups.append(group)
        new_groups.append(merged)

        groups = sorted(new_groups, key=lambda g: (len(g), "|".join(g)))

    return sorted(groups, key=lambda g: (len(g), "|".join(g)))


def validate_complete_partition(
    *,
    approved_zones: list[str],
    groups: list[list[str]],
    effective_max_aggregate_zones: int,
) -> dict[str, Any]:
    approved = set(approved_zones)
    used_flat = [zone for group in groups for zone in group]
    used = set(used_flat)

    duplicate_zones = sorted(
        {
            zone for zone in used_flat
            if used_flat.count(zone) > 1
        }
    )
    missing_zones = sorted(approved - used)
    extra_zones = sorted(used - approved)

    valid = (
        not duplicate_zones
        and not missing_zones
        and not extra_zones
        and len(groups) <= effective_max_aggregate_zones
        and len(groups) >= 1
        and all(len(group) >= 1 for group in groups)
    )

    return {
        "valid": valid,
        "missing_zones": missing_zones,
        "extra_zones": extra_zones,
        "duplicate_zones": duplicate_zones,
        "n_aggregate_zones": len(groups),
        "effective_max_aggregate_zones": effective_max_aggregate_zones,
    }


def candidate_family_for_k(k: int, n: int) -> str:
    if k == 1:
        return "all_to_one"
    if k == n:
        return "identity"
    return "similarity_agglomerative"


def suggestion_id_for_k(
    *,
    case_slug: str,
    k: int,
    n: int,
) -> str:
    if k == 1:
        return f"{case_slug}_k1_all_to_one"
    if k == n:
        return f"{case_slug}_k{k}_identity"
    return f"{case_slug}_k{k}_similarity"


def aggregate_zone_name(
    *,
    k: int,
    n: int,
    group_index: int,
    group: list[str],
) -> str:
    if k == 1:
        return "Aggregated_Zone_1"

    if k == n and len(group) == 1:
        return f"Aggregated_Zone_{safe_slug(group[0]).upper()}"

    return f"Aggregated_Zone_{group_index}"


def reason_for_candidate(
    *,
    k: int,
    n: int,
    group: list[str],
    group_avg_similarity: float | None,
) -> str:
    if k == 1:
        return "All approved zones collapsed into one equivalent aggregate zone."

    if k == n:
        return "Each approved zone retained as its own aggregate zone."

    if group_avg_similarity is None:
        return (
            "Intermediate complete partition generated by deterministic similarity "
            "merging; singleton group retained."
        )

    return (
        "Intermediate complete partition generated by deterministic similarity "
        f"merging; within-group average similarity={group_avg_similarity:.4f}."
    )


def build_suggestion_rows_and_payload(
    *,
    case_id: str,
    building_type: str,
    approved_zones: list[str],
    similarity_lookup: dict[tuple[str, str], float],
    config: GroupingSuggestionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    n = len(approved_zones)
    if n == 0:
        raise ValueError(f"Case {case_id} has zero approved zones.")

    user_max = config.max_aggregate_zones
    effective_max = min(user_max if user_max is not None else n, n)

    if effective_max < 1:
        raise ValueError(
            f"effective_max_aggregate_zones must be >= 1, got {effective_max}"
        )

    case_slug = safe_slug(case_id)
    suggestion_rows: list[dict[str, Any]] = []
    suggestions_json: list[dict[str, Any]] = []
    rationale_lines: list[str] = [
        f"# Grouping Suggestions for `{case_id}`",
        "",
        f"- Approved zone count: `{n}`",
        f"- Requested max aggregate zones: `{user_max if user_max is not None else 'default=approved_zone_count'}`",
        f"- Effective max aggregate zones: `{effective_max}`",
        "- Rule: every suggestion is a complete partition of all approved zones.",
        "",
    ]

    rank = 0

    for k in range(1, effective_max + 1):
        rank += 1

        if k == 1:
            groups = build_all_to_one_partition(approved_zones)
        elif k == n:
            groups = build_identity_partition(approved_zones)
        else:
            groups = build_agglomerative_partition(
                approved_zones=approved_zones,
                target_k=k,
                similarity_lookup=similarity_lookup,
            )

        validation = validate_complete_partition(
            approved_zones=approved_zones,
            groups=groups,
            effective_max_aggregate_zones=effective_max,
        )
        if not validation["valid"]:
            raise ValueError(
                f"Generated invalid partition for case={case_id}, k={k}: {validation}"
            )

        suggestion_id = suggestion_id_for_k(case_slug=case_slug, k=k, n=n)
        family = candidate_family_for_k(k, n)
        candidate_within_similarity = average_candidate_within_similarity(
            groups,
            similarity_lookup,
        )

        aggregate_payloads: list[dict[str, Any]] = []

        rationale_lines.append(f"## `{suggestion_id}`")
        rationale_lines.append("")
        rationale_lines.append(f"- Candidate family: `{family}`")
        rationale_lines.append(f"- Aggregate zone count: `{k}`")
        rationale_lines.append(
            "- Candidate within-group average similarity: "
            f"`{candidate_within_similarity if candidate_within_similarity is not None else 'NA'}`"
        )
        rationale_lines.append("")

        for group_idx, group in enumerate(groups, start=1):
            agg_name = aggregate_zone_name(
                k=k,
                n=n,
                group_index=group_idx,
                group=group,
            )
            group_avg_sim = average_within_group_similarity(group, similarity_lookup)
            reason = reason_for_candidate(
                k=k,
                n=n,
                group=group,
                group_avg_similarity=group_avg_sim,
            )

            aggregate_payloads.append(
                {
                    "aggregate_zone_name": agg_name,
                    "source_zones": list(group),
                    "source_zone_count": len(group),
                    "within_group_average_similarity": group_avg_sim,
                    "reason": reason,
                }
            )

            rationale_lines.append(f"### {agg_name}")
            for zone in group:
                rationale_lines.append(f"- {zone}")
            rationale_lines.append("")

            for zone in group:
                source_zone_index = approved_zones.index(zone) + 1
                suggestion_rows.append(
                    {
                        "case_id": case_id,
                        "building_type": building_type,
                        "suggestion_id": suggestion_id,
                        "candidate_family": family,
                        "n_approved_zones": n,
                        "n_aggregate_zones": k,
                        "effective_max_aggregate_zones": effective_max,
                        "aggregate_zone_name": agg_name,
                        "source_zone_name": zone,
                        "source_zone_index": source_zone_index,
                        "rank": rank,
                        "candidate_within_group_average_similarity": candidate_within_similarity,
                        "aggregate_zone_within_group_average_similarity": group_avg_sim,
                        "reason": reason,
                    }
                )

        suggestions_json.append(
            {
                "case_id": case_id,
                "building_type": building_type,
                "suggestion_id": suggestion_id,
                "candidate_family": family,
                "rank": rank,
                "n_approved_zones": n,
                "n_aggregate_zones": k,
                "effective_max_aggregate_zones": effective_max,
                "candidate_within_group_average_similarity": candidate_within_similarity,
                "validation": validation,
                "aggregate_zones": aggregate_payloads,
            }
        )

    payload = {
        "case_id": case_id,
        "building_type": building_type,
        "approved_zones": approved_zones,
        "approved_zone_count": n,
        "requested_max_aggregate_zones": user_max,
        "effective_max_aggregate_zones": effective_max,
        "suggestion_count": len(suggestions_json),
        "partition_rule": {
            "covers_all_approved_zones": True,
            "uses_each_approved_zone_exactly_once": True,
            "uses_excluded_zones": False,
        },
        "suggestions": suggestions_json,
    }

    return suggestion_rows, payload, "\n".join(rationale_lines)


def build_grouping_suggestions(
    *,
    case_id: str,
    approved_zone_features_path: str | Path,
    zone_pairwise_similarity_path: str | Path,
    config: GroupingSuggestionConfig | None = None,
) -> GroupingSuggestionResult:
    config = config or GroupingSuggestionConfig()

    features_path = Path(approved_zone_features_path).expanduser().resolve()
    pairwise_path = Path(zone_pairwise_similarity_path).expanduser().resolve()

    feature_rows = read_csv_dicts(features_path)
    pairwise_rows = read_csv_dicts(pairwise_path)

    approved_zones = get_approved_zones(feature_rows)
    building_type = infer_building_type(feature_rows)
    similarity_lookup = build_similarity_lookup(pairwise_rows)

    suggestion_rows, payload, rationale = build_suggestion_rows_and_payload(
        case_id=case_id,
        building_type=building_type,
        approved_zones=approved_zones,
        similarity_lookup=similarity_lookup,
        config=config,
    )

    return GroupingSuggestionResult(
        case_id=case_id,
        approved_zone_features_path=features_path,
        zone_pairwise_similarity_path=pairwise_path,
        approved_zones=approved_zones,
        effective_max_aggregate_zones=payload["effective_max_aggregate_zones"],
        suggestion_rows=suggestion_rows,
        suggestion_payload=payload,
        rationale_markdown=rationale,
    )


SUGGESTION_FIELDNAMES = [
    "case_id",
    "building_type",
    "suggestion_id",
    "candidate_family",
    "n_approved_zones",
    "n_aggregate_zones",
    "effective_max_aggregate_zones",
    "aggregate_zone_name",
    "source_zone_name",
    "source_zone_index",
    "rank",
    "candidate_within_group_average_similarity",
    "aggregate_zone_within_group_average_similarity",
    "reason",
]


def write_grouping_suggestion_case_outputs(
    *,
    result: GroupingSuggestionResult,
    case_output_dir: str | Path,
) -> dict[str, Any]:
    out_dir = Path(case_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "suggested_groupings.csv",
        result.suggestion_rows,
        SUGGESTION_FIELDNAMES,
    )
    write_json(out_dir / "suggested_groupings.json", result.suggestion_payload)
    write_text(out_dir / "suggested_grouping_rationale.md", result.rationale_markdown)

    manifest = {
        "case_id": result.case_id,
        "approved_zone_features_path": str(result.approved_zone_features_path),
        "zone_pairwise_similarity_path": str(result.zone_pairwise_similarity_path),
        "approved_zone_count": result.approved_zone_count,
        "approved_zones": result.approved_zones,
        "effective_max_aggregate_zones": result.effective_max_aggregate_zones,
        "suggestion_count": result.suggestion_count,
        "suggestion_row_count": len(result.suggestion_rows),
        "partition_rule": {
            "covers_all_approved_zones": True,
            "uses_each_approved_zone_exactly_once": True,
            "uses_excluded_zones": False,
        },
        "uses_raw_idf": False,
        "uses_opyplus": False,
        "outputs": {
            "suggested_groupings_csv": str(out_dir / "suggested_groupings.csv"),
            "suggested_groupings_json": str(out_dir / "suggested_groupings.json"),
            "suggested_grouping_rationale_md": str(out_dir / "suggested_grouping_rationale.md"),
        },
    }

    write_json(out_dir / "grouping_suggestion_manifest.json", manifest)
    return manifest


def find_latest_root(campaign_root: str | Path, relative_root: str, prefix: str) -> Path:
    root = Path(campaign_root) / relative_root
    candidates = [p for p in root.glob(f"{prefix}_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_* folder found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_approved_zone_features_path(
    *,
    campaign_root: str | Path,
    case_id: str,
    zone_features_root: str | Path | None = None,
) -> Path:
    if zone_features_root is None:
        zone_features_root = find_latest_root(
            campaign_root,
            "aggregation/zone_features",
            "zone_features",
        )

    path = Path(zone_features_root) / "cases" / case_id / "approved_zone_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"approved_zone_features.csv not found: {path}")
    return path


def find_zone_pairwise_similarity_path(
    *,
    campaign_root: str | Path,
    case_id: str,
    zone_similarity_root: str | Path | None = None,
) -> Path:
    if zone_similarity_root is None:
        zone_similarity_root = find_latest_root(
            campaign_root,
            "aggregation/zone_similarity",
            "zone_similarity",
        )

    path = Path(zone_similarity_root) / "cases" / case_id / "zone_pairwise_similarity.csv"
    if not path.exists():
        raise FileNotFoundError(f"zone_pairwise_similarity.csv not found: {path}")
    return path