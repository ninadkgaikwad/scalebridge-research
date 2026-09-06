# -*- coding: utf-8 -*-
"""Serializable D1 manifest contracts for Phase D products."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import PHASE_D_SCHEMA_VERSION
from .identities import PhaseDDatasetIdentity, PhaseDSourceLineage
from .models import ZoneSignalRecord


@dataclass(frozen=True)
class PhaseDZoneManifest:
    """Signal applicability and storage contract for one aggregate zone."""

    aggregate_zone_id: str
    row_count: int
    signal_records: tuple[ZoneSignalRecord, ...]
    include_visible_lighting_in_zir: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError("row_count cannot be negative")
        names = [record.signal_name for record in self.signal_records]
        if len(names) != len(set(names)):
            raise ValueError("signal_records must have unique signal_name values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_D_SCHEMA_VERSION,
            "aggregate_zone_id": self.aggregate_zone_id,
            "row_count": self.row_count,
            "include_visible_lighting_in_zir": self.include_visible_lighting_in_zir,
            "signal_records": [record.to_dict() for record in self.signal_records],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PhaseDDatasetManifest:
    """Top-level D1 manifest contract for one canonical Phase D dataset."""

    identity: PhaseDDatasetIdentity
    lineage: PhaseDSourceLineage
    zone_manifests: tuple[PhaseDZoneManifest, ...]
    canonical_columns: tuple[str, ...]
    split_strategy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        zone_ids = tuple(zone.aggregate_zone_id for zone in self.zone_manifests)
        if set(zone_ids) != set(self.identity.aggregate_zone_ids):
            raise ValueError("zone_manifests must match identity.aggregate_zone_ids")
        if len(self.canonical_columns) != len(set(self.canonical_columns)):
            raise ValueError("canonical_columns must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_D_SCHEMA_VERSION,
            "identity": self.identity.to_dict(),
            "lineage": self.lineage.to_dict(),
            "zone_manifests": [zone.to_dict() for zone in self.zone_manifests],
            "canonical_columns": list(self.canonical_columns),
            "split_strategy": self.split_strategy,
            "metadata": self.metadata,
        }
