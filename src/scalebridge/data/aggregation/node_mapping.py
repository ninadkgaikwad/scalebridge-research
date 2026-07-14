# -*- coding: utf-8 -*-
"""Shared EnergyPlus system-node to source-zone mapping utilities.

This module centralizes the node-name matching logic used by both:

    - System Node Temperature
    - System Node Mass Flow Rate

The first production mapping intentionally uses only zone delivery/inlet node
families that were confirmed in the P1 compact audit:

    <source zone> DIRECT AIR INLET NODE NAME
    <source zone> ZONE EQUIP INLET

The matcher is conservative by design. Return nodes, ERV outlet nodes, water
nodes, and broad AIR NODE / OUTLET NODE families are excluded unless explicitly
added in a future run after diagnostics justify them.
"""

from __future__ import annotations

from typing import Any


DEFAULT_ZONE_DELIVERY_NODE_SUFFIXES = (
    "DIRECT AIR INLET NODE NAME",
    "ZONE EQUIP INLET",
)


class SourceZoneNodeMatcher:
    """Longest-prefix matcher from EnergyPlus node key_value to source zone.

    Matching rule
    -------------
    A key maps to a source zone if the normalized key starts with the normalized
    source-zone name and the remaining suffix starts with one approved node
    suffix pattern.

    Examples
    --------
    CORE_ZN DIRECT AIR INLET NODE NAME -> Core_ZN
    LGSTORE1 DIRECT AIR INLET NODE NAME -> LGstore1
    DINING DIRECT AIR INLET NODE NAME -> Dining
    G N1 APARTMENT ZONE EQUIP INLET -> G N1 Apartment
    """

    def __init__(
        self,
        *,
        source_zones: tuple[str, ...],
        suffix_patterns: tuple[str, ...] = DEFAULT_ZONE_DELIVERY_NODE_SUFFIXES,
    ) -> None:
        self.zone_pairs = sorted(
            [
                (normalize_identifier(zone), zone)
                for zone in source_zones
                if str(zone).strip()
            ],
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self.suffix_pairs = [
            (normalize_identifier(suffix), suffix)
            for suffix in suffix_patterns
            if str(suffix).strip()
        ]

    def match_source_zone(self, key_value: str) -> str | None:
        """Return the matched source zone for one node key, if any."""
        key_norm = normalize_identifier(key_value)

        for zone_norm, source_zone in self.zone_pairs:
            if not key_norm.startswith(zone_norm + " "):
                continue

            remainder = key_norm[len(zone_norm):].strip()
            if any(
                remainder.startswith(suffix_norm)
                for suffix_norm, _ in self.suffix_pairs
            ):
                return source_zone

        return None

    def match_suffix_pattern(
        self,
        key_value: str,
        source_zone: str | None,
    ) -> str:
        """Return the matched suffix pattern for diagnostics."""
        if not source_zone:
            return ""

        key_norm = normalize_identifier(key_value)
        zone_norm = normalize_identifier(source_zone)

        if not key_norm.startswith(zone_norm + " "):
            return ""

        remainder = key_norm[len(zone_norm):].strip()
        for suffix_norm, suffix_original in self.suffix_pairs:
            if remainder.startswith(suffix_norm):
                return suffix_original

        return ""


def extract_aggregate_groups(plan: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Extract aggregate zone groups from aggregation_plan.json."""
    groups: dict[str, tuple[str, ...]] = {}

    for group in plan.get("aggregate_zones", []):
        aggregate_zone_id = str(group.get("aggregate_zone_id", "")).strip()
        source_zones = tuple(
            str(item).strip()
            for item in group.get("source_zones", [])
            if str(item).strip()
        )
        if aggregate_zone_id:
            groups[aggregate_zone_id] = source_zones

    return groups


def build_source_zone_to_aggregate_zone(
    aggregate_groups: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """Build source zone -> aggregate zone mapping from a complete plan."""
    mapping: dict[str, str] = {}
    for aggregate_zone_id, source_zones in aggregate_groups.items():
        for source_zone in source_zones:
            mapping[source_zone] = aggregate_zone_id
    return mapping


def build_zone_metadata(
    zone_mapping_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """Build source-zone metadata for area/volume weighting."""
    metadata: dict[str, dict[str, float | None]] = {}

    for row in zone_mapping_rows:
        source_zone = str(row.get("source_zone", "")).strip()
        if not source_zone:
            continue

        metadata[source_zone] = {
            "floor_area": optional_float(row.get("floor_area_m2")),
            "volume": optional_float(row.get("volume_m3")),
        }

    return metadata


def zone_weight(
    *,
    source_zone: str,
    weight_mode: str,
    zone_metadata: dict[str, dict[str, float | None]],
) -> float | None:
    """Return the requested source-zone weight, or None if unavailable."""
    normalized_weight_mode = weight_mode.casefold()

    if normalized_weight_mode == "equal":
        return 1.0

    if normalized_weight_mode == "floor_area":
        return zone_metadata.get(source_zone, {}).get("floor_area")

    if normalized_weight_mode == "volume":
        return zone_metadata.get(source_zone, {}).get("volume")

    raise ValueError(f"Unsupported weight_mode: {weight_mode}")


def normalize_identifier(value: str) -> str:
    """Normalize an EnergyPlus identifier for case/space-insensitive matching."""
    return " ".join(str(value).strip().upper().split())


def optional_float(value: Any) -> float | None:
    """Convert optional float-like value."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None
