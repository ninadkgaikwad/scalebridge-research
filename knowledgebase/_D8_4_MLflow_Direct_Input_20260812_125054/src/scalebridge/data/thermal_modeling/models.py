# -*- coding: utf-8 -*-
"""Core immutable Phase D data-contract models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import NullableReason, PhaseDSignalStatus


@dataclass(frozen=True)
class ZoneSignalRecord:
    """Manifest-ready record for one canonical signal in one aggregate zone."""

    signal_name: str
    source_phase: str
    source_name: str | None
    units: str | None
    phase_d_status: PhaseDSignalStatus
    nullable: bool
    nullable_reason: NullableReason
    included_in_group: bool = False
    group_name: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    constant_value: float | None = None
    finite_count: int = 0
    missing_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.nullable and self.nullable_reason is NullableReason.NONE:
            raise ValueError("nullable signals must provide a nullable_reason")
        if not self.nullable and self.nullable_reason is not NullableReason.NONE:
            raise ValueError("non-nullable signals cannot provide a nullable_reason")
        if self.included_in_group and self.group_name not in {"zic", "zir"}:
            raise ValueError("included grouped signals must use group_name 'zic' or 'zir'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "source_phase": self.source_phase,
            "source_name": self.source_name,
            "units": self.units,
            "phase_d_status": self.phase_d_status.value,
            "nullable": self.nullable,
            "nullable_reason": self.nullable_reason.value,
            "included_in_group": self.included_in_group,
            "group_name": self.group_name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "constant_value": self.constant_value,
            "finite_count": self.finite_count,
            "missing_count": self.missing_count,
            "metadata": self.metadata,
        }
