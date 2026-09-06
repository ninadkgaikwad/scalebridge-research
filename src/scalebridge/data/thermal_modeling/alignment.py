# -*- coding: utf-8 -*-
"""Phase D D3 timestamp normalization, cleanup, and source alignment."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import calendar
import re
from typing import Any

import pandas as pd

PHASE_B_TIMESTAMP_COLUMN = "timestamp_raw"
PHASE_B_ZONE_TEMPERATURE_COLUMN = "Zone_Air_Temperature_"
PHASE_B_OUTDOOR_TEMPERATURE_COLUMN = "Site_Outdoor_Air_Drybulb_Temperature_"
PHASE_C_TIMESTAMP_COLUMN = "timestamp"
SPLIT_TIMESTAMP_COLUMN = "timestamp"

_EP_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})\s*$")


class PhaseDAlignmentError(RuntimeError):
    """Raised when Phase B, Phase C, and split sources cannot align safely."""


@dataclass(frozen=True)
class TimestampNormalizationConfig:
    phase_d_calendar_year: int = 2001
    reject_leap_year_for_non_leap_source: bool = True

    def __post_init__(self) -> None:
        if self.phase_d_calendar_year < 1 or self.phase_d_calendar_year > 9999:
            raise ValueError("phase_d_calendar_year must be in [1, 9999]")
        if (
            self.reject_leap_year_for_non_leap_source
            and calendar.isleap(self.phase_d_calendar_year)
        ):
            raise ValueError(
                "Current Phase B/Phase C annual sources are non-leap calendars; "
                "choose a non-leap phase_d_calendar_year"
            )


@dataclass(frozen=True)
class AlignmentDiagnostics:
    phase_d_calendar_year: int
    raw_phase_b_rows: int
    phase_b_duplicate_timestamp_groups: int
    phase_b_simple_null_remnant_groups: int
    phase_b_complementary_duplicate_groups_merged: int
    phase_b_identical_duplicate_groups_collapsed: int
    phase_b_duplicate_rows_removed: int
    phase_b_conflicting_duplicate_groups: int
    cleaned_phase_b_rows: int
    phase_c_rows: int
    split_rows: int
    aligned_rows: int
    phase_b_only_timestamps: int
    phase_c_only_timestamps: int
    split_only_timestamps: int

    @property
    def phase_b_null_duplicate_rows_removed(self) -> int:
        """Backward-compatible alias for the previous D3 diagnostic name."""
        return self.phase_b_duplicate_rows_removed

    @property
    def phase_b_conflicting_nonnull_duplicates(self) -> int:
        """Backward-compatible alias for the previous D3 diagnostic name."""
        return self.phase_b_conflicting_duplicate_groups

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase_b_null_duplicate_rows_removed"] = (
            self.phase_b_duplicate_rows_removed
        )
        payload["phase_b_conflicting_nonnull_duplicates"] = (
            self.phase_b_conflicting_duplicate_groups
        )
        return payload


def parse_energyplus_timestamp(value: Any, year: int) -> pd.Timestamp:
    """Parse EnergyPlus MM/DD HH:MM:SS, including 24:00 rollover."""
    if value is None or pd.isna(value):
        return pd.NaT
    match = _EP_RE.match(str(value))
    if not match:
        return pd.NaT
    month, day, hour, minute, second = map(int, match.groups())
    if hour == 24:
        if minute != 0 or second != 0:
            return pd.NaT
        try:
            return pd.Timestamp(year=year, month=month, day=day) + pd.Timedelta(days=1)
        except ValueError:
            return pd.NaT
    if hour > 23:
        return pd.NaT
    try:
        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
        )
    except ValueError:
        return pd.NaT


def rewrite_placeholder_year(series: pd.Series, year: int) -> pd.Series:
    """Rebuild placeholder timestamps on the selected Phase D annual calendar.

    Relative source-year offsets are preserved. Therefore, the annual endpoint
    ``2002-01-01 00:00`` of a source calendar beginning in 2001 maps to
    ``<selected year + 1>-01-01 00:00`` rather than colliding with the first day.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise PhaseDAlignmentError(
            f"Could not parse {int(parsed.isna().sum())} source timestamps"
        )
    source_base_year = int(parsed.dt.year.min())
    values: list[pd.Timestamp] = []
    for ts in parsed:
        target_year = year + (int(ts.year) - source_base_year)
        try:
            values.append(ts.replace(year=target_year))
        except ValueError as exc:
            raise PhaseDAlignmentError(
                f"Timestamp {ts} cannot be represented in Phase D year {target_year}"
            ) from exc
    return pd.Series(values, index=series.index, dtype="datetime64[ns]")


def _values_conflict(values: pd.Series) -> bool:
    """Return True when one required column has multiple distinct non-null values."""
    non_null = values.dropna()
    if len(non_null) <= 1:
        return False
    return non_null.nunique(dropna=True) > 1


def _coalesce_duplicate_group(
    group: pd.DataFrame,
    required_columns: list[str],
) -> tuple[pd.DataFrame | None, str]:
    """Resolve one duplicate timestamp group safely.

    Resolution categories:
    - ``simple_null_remnant``: one complete row plus less-complete/null remnants;
    - ``identical``: repeated equivalent values across required columns;
    - ``complementary``: required values are split across multiple rows;
    - ``conflict``: a required column has differing non-null values;
    - ``incomplete``: coalescing still leaves a required value null.
    """
    for column in required_columns:
        if _values_conflict(group[column]):
            return None, "conflict"

    merged_values: dict[str, Any] = {}
    for column in required_columns:
        non_null = group[column].dropna()
        merged_values[column] = non_null.iloc[0] if len(non_null) else pd.NA

    if any(pd.isna(merged_values[column]) for column in required_columns):
        return None, "incomplete"

    complete_mask = group[required_columns].notna().all(axis=1)
    required_distinct = group[required_columns].drop_duplicates()

    if complete_mask.sum() == 1:
        category = "simple_null_remnant"
    elif complete_mask.sum() > 1 and required_distinct.shape[0] == 1:
        category = "identical"
    else:
        category = "complementary"

    # Preserve non-required metadata from the first row, then overwrite the
    # required physical values with the safely coalesced values.
    merged = group.iloc[[0]].copy()
    for column, value in merged_values.items():
        merged.loc[:, column] = value
    return merged, category


def clean_phase_b(
    frame: pd.DataFrame,
    config: TimestampNormalizationConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {
        PHASE_B_TIMESTAMP_COLUMN,
        PHASE_B_ZONE_TEMPERATURE_COLUMN,
        PHASE_B_OUTDOOR_TEMPERATURE_COLUMN,
    }
    missing = required - set(frame.columns)
    if missing:
        raise PhaseDAlignmentError(f"Phase B missing columns: {sorted(missing)}")

    work = frame.copy()
    work["timestamp"] = work[PHASE_B_TIMESTAMP_COLUMN].map(
        lambda x: parse_energyplus_timestamp(x, config.phase_d_calendar_year)
    )
    if work["timestamp"].isna().any():
        raise PhaseDAlignmentError(
            f"Phase B has {int(work['timestamp'].isna().sum())} unparseable timestamps"
        )

    required_columns = [
        PHASE_B_ZONE_TEMPERATURE_COLUMN,
        PHASE_B_OUTDOOR_TEMPERATURE_COLUMN,
    ]
    duplicate_groups = 0
    simple_groups = 0
    complementary_groups = 0
    identical_groups = 0
    removed_rows = 0
    conflicts = 0
    incomplete = 0
    kept: list[pd.DataFrame] = []

    for _, group in work.groupby("timestamp", sort=False):
        if len(group) == 1:
            if group[required_columns].isna().any(axis=None):
                incomplete += 1
            else:
                kept.append(group.iloc[[0]])
            continue

        duplicate_groups += 1
        merged, category = _coalesce_duplicate_group(group, required_columns)
        if merged is None:
            if category == "conflict":
                conflicts += 1
            else:
                incomplete += 1
            continue

        if category == "simple_null_remnant":
            simple_groups += 1
        elif category == "complementary":
            complementary_groups += 1
        elif category == "identical":
            identical_groups += 1

        removed_rows += len(group) - 1
        kept.append(merged)

    if conflicts:
        raise PhaseDAlignmentError(
            f"Phase B has {conflicts} conflicting duplicate timestamp groups"
        )
    if incomplete:
        raise PhaseDAlignmentError(
            f"Phase B has {incomplete} timestamp groups with incomplete required temperatures"
        )

    cleaned = (
        pd.concat(kept, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if cleaned["timestamp"].duplicated().any():
        raise PhaseDAlignmentError("Phase B duplicate timestamps remain after cleanup")
    if cleaned[required_columns].isna().any(axis=None):
        raise PhaseDAlignmentError(
            "Phase B required temperatures remain null after duplicate cleanup"
        )

    return cleaned, {
        "duplicate_timestamp_groups": duplicate_groups,
        "simple_null_remnant_groups": simple_groups,
        "complementary_duplicate_groups_merged": complementary_groups,
        "identical_duplicate_groups_collapsed": identical_groups,
        "duplicate_rows_removed": removed_rows,
        "conflicting_duplicate_groups": conflicts,
    }


def align_sources(
    phase_b: pd.DataFrame,
    phase_c: pd.DataFrame,
    splits: pd.DataFrame,
    config: TimestampNormalizationConfig,
) -> tuple[pd.DataFrame, AlignmentDiagnostics]:
    raw_b = len(phase_b)
    clean_b, cleanup = clean_phase_b(phase_b, config)
    for label, source, column in (
        ("Phase C", phase_c, PHASE_C_TIMESTAMP_COLUMN),
        ("split", splits, SPLIT_TIMESTAMP_COLUMN),
    ):
        if column not in source.columns:
            raise PhaseDAlignmentError(
                f"{label} missing timestamp column '{column}'"
            )

    c = phase_c.copy()
    s = splits.copy()
    c["timestamp"] = rewrite_placeholder_year(
        c[PHASE_C_TIMESTAMP_COLUMN], config.phase_d_calendar_year
    )
    s["timestamp"] = rewrite_placeholder_year(
        s[SPLIT_TIMESTAMP_COLUMN], config.phase_d_calendar_year
    )
    for label, source in (("Phase C", c), ("split", s)):
        if source["timestamp"].duplicated().any():
            raise PhaseDAlignmentError(f"{label} contains duplicate timestamps")

    bset = set(clean_b.timestamp)
    cset = set(c.timestamp)
    sset = set(s.timestamp)
    if bset != cset or bset != sset:
        raise PhaseDAlignmentError(
            "Timestamp sets do not match after canonical-year normalization: "
            f"phase_b_only={len(bset - cset - sset)}, "
            f"phase_c_only={len(cset - bset)}, "
            f"split_only={len(sset - bset)}"
        )

    b = clean_b[
        [
            "timestamp",
            PHASE_B_ZONE_TEMPERATURE_COLUMN,
            PHASE_B_OUTDOOR_TEMPERATURE_COLUMN,
        ]
    ].rename(
        columns={
            PHASE_B_ZONE_TEMPERATURE_COLUMN: "zone_temperature",
            PHASE_B_OUTDOOR_TEMPERATURE_COLUMN: "outdoor_temperature",
        }
    )
    c_payload = c.drop(columns=["timestamp_raw"], errors="ignore")
    s_payload = s.drop(columns=["timestamp_raw"], errors="ignore")
    aligned = b.merge(
        c_payload,
        on="timestamp",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_phase_c"),
    )
    aligned = aligned.merge(
        s_payload,
        on="timestamp",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_split"),
    )
    aligned = aligned.sort_values("timestamp").reset_index(drop=True)

    diagnostics = AlignmentDiagnostics(
        phase_d_calendar_year=config.phase_d_calendar_year,
        raw_phase_b_rows=raw_b,
        phase_b_duplicate_timestamp_groups=cleanup["duplicate_timestamp_groups"],
        phase_b_simple_null_remnant_groups=cleanup["simple_null_remnant_groups"],
        phase_b_complementary_duplicate_groups_merged=cleanup[
            "complementary_duplicate_groups_merged"
        ],
        phase_b_identical_duplicate_groups_collapsed=cleanup[
            "identical_duplicate_groups_collapsed"
        ],
        phase_b_duplicate_rows_removed=cleanup["duplicate_rows_removed"],
        phase_b_conflicting_duplicate_groups=cleanup[
            "conflicting_duplicate_groups"
        ],
        cleaned_phase_b_rows=len(clean_b),
        phase_c_rows=len(c),
        split_rows=len(s),
        aligned_rows=len(aligned),
        phase_b_only_timestamps=len(bset - cset - sset),
        phase_c_only_timestamps=len(cset - bset),
        split_only_timestamps=len(sset - bset),
    )
    return aligned, diagnostics


def load_and_align_paths(
    phase_b_path: Path,
    phase_c_path: Path,
    split_path: Path,
    config: TimestampNormalizationConfig,
    *,
    phase_c_columns: tuple[str, ...] | list[str] | None = None,
    split_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, AlignmentDiagnostics]:
    """Read only required Parquet columns and align one zone.

    D4 campaign execution should pass projected Phase C and split columns to
    minimize I/O and peak memory. Timestamp columns are added automatically.
    """
    phase_b_columns = [
        PHASE_B_TIMESTAMP_COLUMN,
        PHASE_B_ZONE_TEMPERATURE_COLUMN,
        PHASE_B_OUTDOOR_TEMPERATURE_COLUMN,
    ]
    projected_phase_c = None
    if phase_c_columns is not None:
        projected_phase_c = list(dict.fromkeys(
            [PHASE_C_TIMESTAMP_COLUMN, *phase_c_columns]
        ))
    projected_splits = None
    if split_columns is not None:
        projected_splits = list(dict.fromkeys(
            [SPLIT_TIMESTAMP_COLUMN, *split_columns]
        ))

    phase_b = pd.read_parquet(phase_b_path, columns=phase_b_columns)
    phase_c = pd.read_parquet(phase_c_path, columns=projected_phase_c)
    splits = pd.read_parquet(split_path, columns=projected_splits)
    return align_sources(phase_b, phase_c, splits, config)
