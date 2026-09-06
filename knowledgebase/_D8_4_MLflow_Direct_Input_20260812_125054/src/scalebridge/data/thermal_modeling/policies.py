# -*- coding: utf-8 -*-
"""Phase D D7 deterministic temporal partition/selection policies."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .silo_contracts import D6ContractError, Partition


DEFAULT_MDH_TRAIN_FRACTION = 0.70
DEFAULT_MDH_TEST_FRACTION = 0.15
DEFAULT_MDH_VALIDATION_FRACTION = 0.15

DEFAULT_SD_SEASON_OFFSET_DAYS = 0
DEFAULT_SD_TRAIN_DAYS = 21
DEFAULT_SD_TEST_DAYS = 7

METEOROLOGICAL_SEASONS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "fall": (9, 10, 11),
}


@dataclass(frozen=True)
class PolicyAssignmentDiagnostics:
    policy_name: str
    row_count: int
    included_count: int
    excluded_count: int
    partition_counts: dict[str, int]
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "row_count": self.row_count,
            "included_count": self.included_count,
            "excluded_count": self.excluded_count,
            "partition_counts": dict(self.partition_counts),
            "parameters": dict(self.parameters),
        }


def _validate_timestamps(timestamps: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamps, errors="raise")
    if ts.isna().any():
        raise D6ContractError("policy timestamps contain missing values")
    if ts.duplicated().any():
        raise D6ContractError("policy timestamps contain duplicates")
    if not ts.is_monotonic_increasing:
        raise D6ContractError("policy timestamps must be monotonic increasing")
    return ts.reset_index(drop=True)


def _base_policy_frame(timestamps: pd.Series) -> pd.DataFrame:
    ts = _validate_timestamps(timestamps)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "included": False,
            "partition": Partition.EXCLUDED.value,
            "window_id": pd.Series([pd.NA] * len(ts), dtype="string"),
            "season": pd.Series([pd.NA] * len(ts), dtype="string"),
        }
    )


def assign_monthly_distributed_holdout(
    timestamps: pd.Series,
    *,
    train_fraction: float = DEFAULT_MDH_TRAIN_FRACTION,
    test_fraction: float = DEFAULT_MDH_TEST_FRACTION,
    validation_fraction: float = DEFAULT_MDH_VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Apply contiguous Train/Test/Validation percentages within every month.

    EnergyPlus interval-ending midnight rows are associated with the preceding
    day/month by subtracting one nanosecond only for month membership.
    """

    fractions = (train_fraction, test_fraction, validation_fraction)
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise D6ContractError("MDH fractions must each be strictly between 0 and 1")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1.0e-12):
        raise D6ContractError("MDH train/test/validation fractions must sum to 1")

    frame = _base_policy_frame(timestamps)
    interval_month = (frame["timestamp"] - pd.Timedelta(nanoseconds=1)).dt.to_period("M")

    for _, index in frame.groupby(interval_month, sort=True).groups.items():
        positions = np.asarray(list(index), dtype=int)
        count = len(positions)
        train_count = int(np.floor(count * train_fraction))
        test_count = int(np.floor(count * test_fraction))
        validation_count = count - train_count - test_count
        if min(train_count, test_count, validation_count) < 1:
            raise D6ContractError(
                "MDH month is too short for the requested nonzero fractions"
            )

        train_positions = positions[:train_count]
        test_positions = positions[train_count : train_count + test_count]
        val_positions = positions[train_count + test_count :]

        frame.loc[train_positions, ["included", "partition"]] = [
            True,
            Partition.TRAIN.value,
        ]
        frame.loc[test_positions, ["included", "partition"]] = [
            True,
            Partition.TEST.value,
        ]
        frame.loc[val_positions, ["included", "partition"]] = [
            True,
            Partition.VALIDATION.value,
        ]

    diagnostics = _diagnostics(
        frame,
        "monthly_distributed_holdout",
        {
            "train_fraction": train_fraction,
            "test_fraction": test_fraction,
            "validation_fraction": validation_fraction,
            "partition_order": ["train", "test", "validation"],
            "month_assignment": "interval_ending_timestamp_previous_nanosecond",
        },
    )
    return frame, diagnostics


def _season_and_day(timestamp: pd.Timestamp) -> tuple[str, int]:
    """Return meteorological season and zero-based day within cyclic season.

    Winter ordering is Dec -> Jan -> Feb. This provides a deterministic
    within-season coordinate for a canonical Jan-Dec year while preserving
    standard meteorological season membership.
    """

    month = int(timestamp.month)
    day = int(timestamp.day)

    for season, months in METEOROLOGICAL_SEASONS.items():
        if month not in months:
            continue
        offset = 0
        for season_month in months:
            if season_month == month:
                return season, offset + day - 1
            # Canonical Phase D is non-leap; use a non-leap reference year.
            offset += calendar.monthrange(2001, season_month)[1]

    raise D6ContractError(f"Unable to resolve season for timestamp {timestamp}")


def assign_seasonal_distributed(
    timestamps: pd.Series,
    *,
    season_offset_days: int = DEFAULT_SD_SEASON_OFFSET_DAYS,
    train_days: int = DEFAULT_SD_TRAIN_DAYS,
    test_days: int = DEFAULT_SD_TEST_DAYS,
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Select contiguous seasonal Train/Test windows from the full annual axis."""

    if season_offset_days < 0:
        raise D6ContractError("season_offset_days must be >= 0")
    if train_days < 1 or test_days < 1:
        raise D6ContractError("SD train_days and test_days must be >= 1")

    shortest_season_days = min(
        sum(calendar.monthrange(2001, month)[1] for month in months)
        for months in METEOROLOGICAL_SEASONS.values()
    )
    if season_offset_days + train_days + test_days > shortest_season_days:
        raise D6ContractError(
            "SD offset + train_days + test_days must fit every season"
        )

    frame = _base_policy_frame(timestamps)
    season_values: list[str] = []
    season_days: list[int] = []
    for timestamp in frame["timestamp"]:
        # Treat interval-ending midnight as belonging to the preceding day.
        effective = timestamp - pd.Timedelta(nanoseconds=1)
        season, day_index = _season_and_day(effective)
        season_values.append(season)
        season_days.append(day_index)

    frame["season"] = pd.Series(season_values, dtype="string")
    season_day = np.asarray(season_days, dtype=int)

    train_start = season_offset_days
    train_end = train_start + train_days
    test_end = train_end + test_days

    for season in METEOROLOGICAL_SEASONS:
        season_mask = frame["season"].eq(season).to_numpy()
        train_mask = season_mask & (season_day >= train_start) & (season_day < train_end)
        test_mask = season_mask & (season_day >= train_end) & (season_day < test_end)

        frame.loc[train_mask, "included"] = True
        frame.loc[train_mask, "partition"] = Partition.TRAIN.value
        frame.loc[train_mask, "window_id"] = f"{season}_train_01"

        frame.loc[test_mask, "included"] = True
        frame.loc[test_mask, "partition"] = Partition.TEST.value
        frame.loc[test_mask, "window_id"] = f"{season}_test_01"

    diagnostics = _diagnostics(
        frame,
        "seasonal_distributed",
        {
            "season_months": {
                name: list(months)
                for name, months in METEOROLOGICAL_SEASONS.items()
            },
            "season_offset_days": season_offset_days,
            "train_days": train_days,
            "test_days": test_days,
            "window_order": ["train", "test"],
            "nonselected_partition": "excluded",
        },
    )
    return frame, diagnostics


def _diagnostics(
    frame: pd.DataFrame,
    policy_name: str,
    parameters: dict[str, Any],
) -> PolicyAssignmentDiagnostics:
    counts = frame["partition"].value_counts(dropna=False).to_dict()
    counts = {str(key): int(value) for key, value in counts.items()}
    included_count = int(frame["included"].sum())
    return PolicyAssignmentDiagnostics(
        policy_name=policy_name,
        row_count=len(frame),
        included_count=included_count,
        excluded_count=len(frame) - included_count,
        partition_counts=counts,
        parameters=parameters,
    )


# Compact CLI aliases. Long-form names remain authoritative in manifests.
POLICY_ALIASES = {
    "mdh": "monthly_distributed_holdout",
    "ch": "chronological_holdout",
    "sh": "seasonal_holdout",
    "sd": "seasonal_distributed",
    "sbh": "seasonal_block_holdout",
    "ci": "contiguous_identification",
    "cdr": "custom_datetime_ranges",
}


def normalize_policy_name(policy_name: str) -> str:
    value = str(policy_name).strip().lower()
    return POLICY_ALIASES.get(value, value)


def _normalize_seasons(values: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raw = [item.strip().lower() for item in values.split(",") if item.strip()]
    else:
        raw = [str(item).strip().lower() for item in values]
    if not raw:
        raise D6ContractError(f"{field_name} must contain at least one season")
    invalid = sorted(set(raw) - set(METEOROLOGICAL_SEASONS))
    if invalid:
        raise D6ContractError(f"{field_name} contains unsupported seasons: {invalid}")
    if len(raw) != len(set(raw)):
        raise D6ContractError(f"{field_name} cannot contain duplicate seasons")
    return tuple(raw)


def _assign_season_column(frame: pd.DataFrame) -> np.ndarray:
    season_values: list[str] = []
    season_days: list[int] = []
    for timestamp in frame["timestamp"]:
        effective = timestamp - pd.Timedelta(nanoseconds=1)
        season, day_index = _season_and_day(effective)
        season_values.append(season)
        season_days.append(day_index)
    frame["season"] = pd.Series(season_values, dtype="string")
    return np.asarray(season_days, dtype=int)


def assign_chronological_holdout(
    timestamps: pd.Series,
    *,
    train_fraction: float = DEFAULT_MDH_TRAIN_FRACTION,
    test_fraction: float = DEFAULT_MDH_TEST_FRACTION,
    validation_fraction: float = DEFAULT_MDH_VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Partition the complete ordered axis once as Train -> Test -> Validation."""
    fractions = (train_fraction, test_fraction, validation_fraction)
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise D6ContractError("CH fractions must each be strictly between 0 and 1")
    if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1.0e-12):
        raise D6ContractError("CH train/test/validation fractions must sum to 1")

    frame = _base_policy_frame(timestamps)
    count = len(frame)
    train_count = int(np.floor(count * train_fraction))
    test_count = int(np.floor(count * test_fraction))
    validation_count = count - train_count - test_count
    if min(train_count, test_count, validation_count) < 1:
        raise D6ContractError("CH axis is too short for the requested fractions")

    train_end = train_count
    test_end = train_count + test_count
    frame.loc[: train_end - 1, ["included", "partition"]] = [True, Partition.TRAIN.value]
    frame.loc[train_end : test_end - 1, ["included", "partition"]] = [True, Partition.TEST.value]
    frame.loc[test_end:, ["included", "partition"]] = [True, Partition.VALIDATION.value]

    return frame, _diagnostics(
        frame,
        "chronological_holdout",
        {
            "train_fraction": train_fraction,
            "test_fraction": test_fraction,
            "validation_fraction": validation_fraction,
            "partition_order": ["train", "test", "validation"],
        },
    )


def assign_seasonal_holdout(
    timestamps: pd.Series,
    *,
    train_seasons: Any = ("winter", "spring"),
    test_seasons: Any = ("summer",),
    validation_seasons: Any = ("fall",),
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Assign complete meteorological seasons to ML train/test/validation."""
    train = _normalize_seasons(train_seasons, field_name="train_seasons")
    test = _normalize_seasons(test_seasons, field_name="test_seasons")
    validation = _normalize_seasons(validation_seasons, field_name="validation_seasons")
    if set(train) & set(test) or set(train) & set(validation) or set(test) & set(validation):
        raise D6ContractError("SH train/test/validation seasons must be disjoint")

    frame = _base_policy_frame(timestamps)
    _assign_season_column(frame)
    for seasons, partition in (
        (train, Partition.TRAIN.value),
        (test, Partition.TEST.value),
        (validation, Partition.VALIDATION.value),
    ):
        mask = frame["season"].isin(seasons)
        frame.loc[mask, "included"] = True
        frame.loc[mask, "partition"] = partition
        frame.loc[mask, "window_id"] = frame.loc[mask, "season"].astype(str) + f"_{partition}_01"

    return frame, _diagnostics(
        frame,
        "seasonal_holdout",
        {
            "season_months": {name: list(months) for name, months in METEOROLOGICAL_SEASONS.items()},
            "train_seasons": list(train),
            "test_seasons": list(test),
            "validation_seasons": list(validation),
            "unassigned_partition": "excluded",
        },
    )


def assign_seasonal_block_holdout(
    timestamps: pd.Series,
    *,
    train_seasons: Any = ("winter", "spring", "fall"),
    test_seasons: Any = ("summer",),
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Use complete seasonal blocks for Opt/Bayes identification/evaluation."""
    train = _normalize_seasons(train_seasons, field_name="train_seasons")
    test = _normalize_seasons(test_seasons, field_name="test_seasons")
    if set(train) & set(test):
        raise D6ContractError("SBH train and test seasons must be disjoint")

    frame = _base_policy_frame(timestamps)
    _assign_season_column(frame)
    for seasons, partition in ((train, Partition.TRAIN.value), (test, Partition.TEST.value)):
        mask = frame["season"].isin(seasons)
        frame.loc[mask, "included"] = True
        frame.loc[mask, "partition"] = partition
        frame.loc[mask, "window_id"] = frame.loc[mask, "season"].astype(str) + f"_{partition}_01"

    return frame, _diagnostics(
        frame,
        "seasonal_block_holdout",
        {
            "season_months": {name: list(months) for name, months in METEOROLOGICAL_SEASONS.items()},
            "train_seasons": list(train),
            "test_seasons": list(test),
            "unassigned_partition": "excluded",
        },
    )


def _infer_timestep(ts: pd.Series) -> pd.Timedelta:
    if len(ts) < 2:
        raise D6ContractError("At least two timestamps are required")
    deltas = ts.diff().dropna()
    step = deltas.mode().iloc[0]
    if step <= pd.Timedelta(0):
        raise D6ContractError("Unable to infer a positive timestep")
    return step


def assign_contiguous_identification(
    timestamps: pd.Series,
    *,
    start_datetime: str | pd.Timestamp | None = None,
    train_days: int = DEFAULT_SD_TRAIN_DAYS,
    test_days: int = DEFAULT_SD_TEST_DAYS,
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Select one contiguous train block immediately followed by one test block."""
    if train_days < 1 or test_days < 1:
        raise D6ContractError("CI train_days and test_days must be >= 1")
    frame = _base_policy_frame(timestamps)
    ts = frame["timestamp"]
    step = _infer_timestep(ts)
    start = ts.iloc[0] if start_datetime is None else pd.Timestamp(start_datetime)
    timestamp_set = set(ts.tolist())
    if start not in timestamp_set:
        raise D6ContractError(f"CI start_datetime is not on the canonical timestamp axis: {start}")
    train_end = start + pd.Timedelta(days=train_days)
    test_end = train_end + pd.Timedelta(days=test_days)
    if test_end > ts.iloc[-1] + step:
        raise D6ContractError("CI train/test window extends beyond the canonical timestamp axis")

    train_mask = (ts >= start) & (ts < train_end)
    test_mask = (ts >= train_end) & (ts < test_end)
    if not train_mask.any() or not test_mask.any():
        raise D6ContractError("CI produced an empty train or test window")
    frame.loc[train_mask, ["included", "partition", "window_id"]] = [True, Partition.TRAIN.value, "ci_train_01"]
    frame.loc[test_mask, ["included", "partition", "window_id"]] = [True, Partition.TEST.value, "ci_test_01"]

    return frame, _diagnostics(
        frame,
        "contiguous_identification",
        {
            "start_datetime": start.isoformat(),
            "train_days": train_days,
            "test_days": test_days,
            "window_order": ["train", "test"],
            "nonselected_partition": "excluded",
        },
    )


def parse_datetime_range(value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse a half-open CLI range formatted as START/END."""
    text = str(value).strip()
    if "/" not in text:
        raise D6ContractError("Datetime range must use START/END syntax")
    start_text, end_text = text.split("/", 1)
    start = pd.Timestamp(start_text.strip())
    end = pd.Timestamp(end_text.strip())
    if start >= end:
        raise D6ContractError(f"Datetime range start must precede end: {value}")
    return start, end


def _normalize_datetime_ranges(values: Any, *, field_name: str) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    if not values:
        raise D6ContractError(f"{field_name} must contain at least one datetime range")
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for item in values:
        if isinstance(item, str):
            ranges.append(parse_datetime_range(item))
        else:
            start, end = item
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            if start >= end:
                raise D6ContractError(f"{field_name} contains start >= end")
            ranges.append((start, end))
    ordered = sorted(ranges, key=lambda x: (x[0], x[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise D6ContractError(f"{field_name} contains overlapping ranges")
    return tuple(ranges)


def assign_custom_datetime_ranges(
    timestamps: pd.Series,
    *,
    train_ranges: Any,
    test_ranges: Any,
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Assign explicit half-open [start,end) train/test datetime ranges."""
    frame = _base_policy_frame(timestamps)
    ts = frame["timestamp"]
    step = _infer_timestep(ts)
    train = _normalize_datetime_ranges(train_ranges, field_name="train_ranges")
    test = _normalize_datetime_ranges(test_ranges, field_name="test_ranges")

    all_ranges = [("train", *r) for r in train] + [("test", *r) for r in test]
    all_ranges_sorted = sorted(all_ranges, key=lambda x: (x[1], x[2]))
    for previous, current in zip(all_ranges_sorted, all_ranges_sorted[1:]):
        if current[1] < previous[2]:
            raise D6ContractError("CDR train/test ranges cannot overlap")

    axis_start, axis_end = ts.iloc[0], ts.iloc[-1] + step
    timestamp_set = set(ts.tolist())
    for partition, start, end in all_ranges:
        if start < axis_start or end > axis_end:
            raise D6ContractError(
                f"CDR {partition} range {start}/{end} lies outside canonical timestamp axis"
            )
        if start not in timestamp_set:
            raise D6ContractError(f"CDR range start is not on canonical timestamp axis: {start}")

    for partition, ranges in ((Partition.TRAIN.value, train), (Partition.TEST.value, test)):
        for idx, (start, end) in enumerate(ranges, start=1):
            mask = (ts >= start) & (ts < end)
            if not mask.any():
                raise D6ContractError(f"CDR {partition} range produced no samples: {start}/{end}")
            frame.loc[mask, "included"] = True
            frame.loc[mask, "partition"] = partition
            frame.loc[mask, "window_id"] = f"cdr_{partition}_{idx:02d}"

    return frame, _diagnostics(
        frame,
        "custom_datetime_ranges",
        {
            "range_semantics": "half_open_start_inclusive_end_exclusive",
            "train_ranges": [[a.isoformat(), b.isoformat()] for a, b in train],
            "test_ranges": [[a.isoformat(), b.isoformat()] for a, b in test],
            "nonselected_partition": "excluded",
        },
    )
