# -*- coding: utf-8 -*-
"""Validation helpers for deterministic heat-input regression splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scalebridge.data.heat_input_regression.alignment import build_timestamp_frame
from scalebridge.data.heat_input_regression.splitting import (
    SplitConfig,
    build_split_assignments,
    split_counts,
)


def validate_split_assignments(
    assignments: pd.DataFrame,
    *,
    config: SplitConfig,
    minimum_split_samples: int,
    fraction_tolerance: float,
) -> list[dict[str, Any]]:
    """Validate structural, count, fraction, and seasonal split properties."""
    rows: list[dict[str, Any]] = []

    required = {
        config.raw_timestamp_column,
        config.timestamp_column,
        "split",
        "split_index",
        "included",
        "exclusion_reason",
        "source_row_index",
    }
    missing = sorted(required.difference(assignments.columns))
    rows.append(_check("required_columns_present", not missing, missing, "", ""))
    if missing:
        return rows

    timestamps = pd.to_datetime(assignments[config.timestamp_column], errors="coerce")
    included = assignments[assignments["included"].astype(bool)].copy()
    included_timestamps = pd.to_datetime(included[config.timestamp_column], errors="coerce")

    rows.append(_check(
        "included_timestamps_parse",
        included_timestamps.notna().all(),
        int(included_timestamps.isna().sum()),
        0,
        "Included rows must have valid timestamps.",
    ))
    rows.append(_check(
        "included_timestamps_unique",
        not included_timestamps.duplicated().any(),
        int(included_timestamps.duplicated().sum()),
        0,
        "Included timestamps must be unique.",
    ))
    rows.append(_check(
        "all_included_rows_assigned",
        included["split"].isin(["train", "validation", "test"]).all(),
        int((~included["split"].isin(["train", "validation", "test"])).sum()),
        0,
        "Every included row must belong to exactly one standard split.",
    ))
    rows.append(_check(
        "excluded_rows_labeled",
        assignments.loc[~assignments["included"].astype(bool), "split"].eq("excluded").all(),
        int((~assignments.loc[~assignments["included"].astype(bool), "split"].eq("excluded")).sum()),
        0,
        "Every excluded row must use split='excluded'.",
    ))

    counts = split_counts(assignments)
    included_count = counts["train"] + counts["validation"] + counts["test"]
    expected_fractions = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    for split_name in ("train", "validation", "test"):
        observed_fraction = counts[split_name] / included_count if included_count else 0.0
        rows.append(_check(
            f"{split_name}_minimum_samples",
            counts[split_name] >= minimum_split_samples,
            counts[split_name],
            minimum_split_samples,
            "Split sample count threshold.",
        ))
        rows.append(_check(
            f"{split_name}_fraction_within_tolerance",
            abs(observed_fraction - expected_fractions[split_name]) <= fraction_tolerance,
            observed_fraction,
            expected_fractions[split_name],
            f"Allowed absolute tolerance: {fraction_tolerance}",
        ))

    if config.strategy == "monthly_distributed_holdout" and not included.empty:
        month_series = included_timestamps.dt.to_period("M")
        requested_split_count = sum(
            fraction > 0.0
            for fraction in (
                config.train_fraction,
                config.validation_fraction,
                config.test_fraction,
            )
        )

        month_counts = month_series.value_counts(dropna=True).sort_index()
        feasible_months = {
            str(month)
            for month, count in month_counts.items()
            if int(count) >= requested_split_count
        }
        undersized_months = {
            str(month): int(count)
            for month, count in month_counts.items()
            if int(count) < requested_split_count
        }

        rows.append(_check(
            "monthly_coverage_feasibility",
            True,
            (
                " | ".join(
                    f"{month}:{count}"
                    for month, count in sorted(undersized_months.items())
                )
                if undersized_months
                else "all months can populate every requested split"
            ),
            f">= {requested_split_count} rows per fully-covered month",
            (
                "Months with fewer rows than requested non-zero splits are "
                "retained, but full train/validation/test coverage is not "
                "mathematically required for those boundary months."
            ),
        ))

        for split_name in ("train", "validation", "test"):
            split_mask = included["split"].eq(split_name)
            split_months = set(
                included_timestamps.loc[split_mask]
                .dt.to_period("M")
                .dropna()
                .astype(str)
            )
            missing_months = sorted(feasible_months.difference(split_months))
            rows.append(_check(
                f"all_feasible_months_present_in_{split_name}",
                not missing_months,
                " | ".join(missing_months),
                "",
                (
                    "Monthly distributed splitting should represent every "
                    "month that has enough rows to populate all requested splits."
                ),
            ))

    rows.append(_check(
        "source_row_index_unique",
        not assignments["source_row_index"].duplicated().any(),
        int(assignments["source_row_index"].duplicated().sum()),
        0,
        "Each source row must have one assignment.",
    ))
    rows.append(_check(
        "assignment_row_count_positive",
        len(assignments) > 0,
        len(assignments),
        "> 0",
        "Assignment file must not be empty.",
    ))
    return rows


def validate_timestamp_coverage(
    *,
    feature_frame: pd.DataFrame,
    stage_b_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    timestamp_column: str = "timestamp",
    raw_timestamp_column: str = "timestamp_raw",
) -> list[dict[str, Any]]:
    """Validate C2, Stage B, and C3 timestamp correspondence."""
    feature_ts = _timestamp_set(feature_frame, timestamp_column, raw_timestamp_column)
    stage_b_ts = _timestamp_set(stage_b_frame, timestamp_column, raw_timestamp_column)
    assignment_ts = _timestamp_set(assignments, timestamp_column, raw_timestamp_column)

    return [
        _check(
            "assignment_matches_feature_timestamp_set",
            assignment_ts == feature_ts,
            len(assignment_ts.symmetric_difference(feature_ts)),
            0,
            "Saved assignments must cover the C2 feature timestamp set exactly.",
        ),
        _check(
            "feature_timestamps_available_in_stage_b",
            feature_ts.issubset(stage_b_ts),
            len(feature_ts.difference(stage_b_ts)),
            0,
            "Every C2 timestamp must be available for later Stage B target joining.",
        ),
    ]


def validate_reproducibility(
    *,
    feature_frame: pd.DataFrame,
    saved_assignments: pd.DataFrame,
    config: SplitConfig,
) -> list[dict[str, Any]]:
    """Recompute assignments and verify exact deterministic equality."""
    recomputed = build_split_assignments(feature_frame, config=config)
    compare_columns = [
        config.raw_timestamp_column,
        config.timestamp_column,
        "split",
        "split_index",
        "included",
        "exclusion_reason",
        "source_row_index",
    ]
    left = saved_assignments[compare_columns].copy().reset_index(drop=True)
    right = recomputed[compare_columns].copy().reset_index(drop=True)
    left[config.timestamp_column] = pd.to_datetime(left[config.timestamp_column], errors="coerce")
    right[config.timestamp_column] = pd.to_datetime(right[config.timestamp_column], errors="coerce")
    equal = left.equals(right)
    mismatch_count = 0
    if not equal and len(left) == len(right):
        mismatch_count = int(
            np.logical_not(
                np.logical_and.reduce(
                    [left[column].eq(right[column]).fillna(False).to_numpy() for column in compare_columns]
                )
            ).sum()
        )
    elif not equal:
        mismatch_count = abs(len(left) - len(right))
    return [
        _check(
            "saved_assignments_match_recomputation",
            equal,
            mismatch_count,
            0,
            "The split algorithm must be exactly reproducible.",
        )
    ]


def validation_passed(rows: list[dict[str, Any]]) -> bool:
    return all(str(row.get("status", "")).casefold() == "passed" for row in rows)


def _timestamp_set(
    frame: pd.DataFrame,
    timestamp_column: str,
    raw_timestamp_column: str,
) -> set[pd.Timestamp]:
    if timestamp_column in frame.columns:
        series = pd.to_datetime(frame[timestamp_column], errors="coerce")
    elif raw_timestamp_column in frame.columns:
        timestamp_frame, _ = build_timestamp_frame(
            frame, timestamp_column=raw_timestamp_column
        )
        series = timestamp_frame[timestamp_column]
    else:
        raise ValueError(
            f"Frame missing both {timestamp_column!r} and {raw_timestamp_column!r}"
        )
    return set(series.dropna().tolist())


def _check(
    check_name: str,
    passed: bool,
    observed_value: Any,
    expected_value: Any,
    message: str,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": "passed" if bool(passed) else "failed",
        "observed_value": observed_value,
        "expected_value": expected_value,
        "message": message,
    }
