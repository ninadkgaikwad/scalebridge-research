# -*- coding: utf-8 -*-
"""Deterministic timestamp splitting for heat-input regression datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_SPLIT_STRATEGIES = (
    "monthly_distributed_holdout",
    "chronological_fraction",
)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for one deterministic split run."""

    strategy: str = "monthly_distributed_holdout"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    timestamp_column: str = "timestamp"
    raw_timestamp_column: str = "timestamp_raw"
    random_seed: int = 42

    def validate(self) -> None:
        if self.strategy not in SUPPORTED_SPLIT_STRATEGIES:
            raise ValueError(
                f"Unsupported split strategy: {self.strategy}. "
                f"Supported: {SUPPORTED_SPLIT_STRATEGIES}"
            )
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("Split fractions must be between 0 and 1")
        if not np.isclose(sum(fractions), 1.0, atol=1e-12):
            raise ValueError(
                "train_fraction + validation_fraction + test_fraction must equal 1"
            )


def build_split_assignments(
    frame: pd.DataFrame,
    *,
    config: SplitConfig,
) -> pd.DataFrame:
    """Build deterministic split assignments for a zone-level feature frame."""
    config.validate()
    prepared = _prepare_timestamp_frame(frame=frame, config=config)

    assignments = prepared[
        [config.raw_timestamp_column, config.timestamp_column]
    ].copy()
    assignments["split"] = "excluded"
    assignments["split_index"] = -1
    assignments["included"] = False
    assignments["exclusion_reason"] = prepared["_exclusion_reason"]

    eligible = prepared[prepared["_exclusion_reason"] == ""].copy()
    if eligible.empty:
        return assignments

    if config.strategy == "monthly_distributed_holdout":
        labels = _monthly_distributed_labels(eligible, config=config)
    elif config.strategy == "chronological_fraction":
        labels = _chronological_fraction_labels(eligible, config=config)
    else:  # pragma: no cover - guarded by config.validate()
        raise ValueError(f"Unsupported split strategy: {config.strategy}")

    assignments.loc[labels.index, "split"] = labels
    split_index_map = {"train": 0, "validation": 1, "test": 2}
    assignments.loc[labels.index, "split_index"] = labels.map(split_index_map).astype(int)
    assignments.loc[labels.index, "included"] = True
    assignments.loc[labels.index, "exclusion_reason"] = ""
    assignments["source_row_index"] = np.arange(len(assignments), dtype=np.int64)
    return assignments


def build_split_summary(assignments: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize one split-assignment table."""
    rows: list[dict[str, Any]] = []
    total_included = int(assignments["included"].astype(bool).sum())
    for split_name in ("train", "validation", "test", "excluded"):
        current = assignments[assignments["split"] == split_name].copy()
        timestamps = pd.to_datetime(current["timestamp"], errors="coerce")
        rows.append(
            {
                "split": split_name,
                "row_count": int(len(current)),
                "fraction_of_included": (
                    float(len(current) / total_included)
                    if total_included and split_name != "excluded"
                    else 0.0
                ),
                "first_timestamp": (
                    timestamps.min().isoformat() if timestamps.notna().any() else ""
                ),
                "last_timestamp": (
                    timestamps.max().isoformat() if timestamps.notna().any() else ""
                ),
                "month_count": int(timestamps.dt.to_period("M").nunique())
                if timestamps.notna().any()
                else 0,
                "day_count": int(timestamps.dt.date.nunique())
                if timestamps.notna().any()
                else 0,
            }
        )
    return rows


def split_counts(assignments: pd.DataFrame) -> dict[str, int]:
    """Return standard split counts."""
    counts = assignments["split"].value_counts().to_dict()
    return {
        "train": int(counts.get("train", 0)),
        "validation": int(counts.get("validation", 0)),
        "test": int(counts.get("test", 0)),
        "excluded": int(counts.get("excluded", 0)),
    }


def _prepare_timestamp_frame(
    *,
    frame: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    if config.timestamp_column not in frame.columns:
        raise ValueError(
            f"Feature frame missing timestamp column: {config.timestamp_column}"
        )

    prepared = frame.copy().reset_index(drop=True)
    if config.raw_timestamp_column not in prepared.columns:
        prepared[config.raw_timestamp_column] = prepared[config.timestamp_column].astype(str)

    prepared[config.timestamp_column] = pd.to_datetime(
        prepared[config.timestamp_column], errors="coerce"
    )
    prepared["_exclusion_reason"] = ""
    invalid_mask = prepared[config.timestamp_column].isna()
    prepared.loc[invalid_mask, "_exclusion_reason"] = "invalid_timestamp"

    valid = prepared[~invalid_mask]
    duplicate_mask = valid.duplicated(subset=[config.timestamp_column], keep="first")
    duplicate_indices = valid.index[duplicate_mask]
    prepared.loc[duplicate_indices, "_exclusion_reason"] = "duplicate_timestamp"
    return prepared


def _monthly_distributed_labels(
    eligible: pd.DataFrame,
    *,
    config: SplitConfig,
) -> pd.Series:
    labels = pd.Series(index=eligible.index, dtype="object")
    months = eligible[config.timestamp_column].dt.to_period("M")

    for _, group in eligible.groupby(months, sort=True):
        ordered = group.sort_values(config.timestamp_column, kind="mergesort")
        n_rows = len(ordered)
        n_train = int(np.floor(config.train_fraction * n_rows))
        n_validation = int(np.floor(config.validation_fraction * n_rows))
        n_test = n_rows - n_train - n_validation

        # Ensure each non-zero requested split receives at least one sample when possible.
        requested = [
            config.train_fraction > 0,
            config.validation_fraction > 0,
            config.test_fraction > 0,
        ]
        if n_rows >= sum(requested):
            counts = [n_train, n_validation, n_test]
            for index, is_requested in enumerate(requested):
                if is_requested and counts[index] == 0:
                    donor = max(range(3), key=lambda item: counts[item])
                    if counts[donor] > 1:
                        counts[donor] -= 1
                        counts[index] += 1
            n_train, n_validation, n_test = counts

        ordered_indices = ordered.index.to_numpy()
        train_end = n_train
        validation_end = n_train + n_validation
        labels.loc[ordered_indices[:train_end]] = "train"
        labels.loc[ordered_indices[train_end:validation_end]] = "validation"
        labels.loc[ordered_indices[validation_end:]] = "test"

    return labels


def _chronological_fraction_labels(
    eligible: pd.DataFrame,
    *,
    config: SplitConfig,
) -> pd.Series:
    ordered = eligible.sort_values(config.timestamp_column, kind="mergesort")
    n_rows = len(ordered)
    n_train = int(np.floor(config.train_fraction * n_rows))
    n_validation = int(np.floor(config.validation_fraction * n_rows))
    ordered_indices = ordered.index.to_numpy()

    labels = pd.Series(index=eligible.index, dtype="object")
    labels.loc[ordered_indices[:n_train]] = "train"
    labels.loc[ordered_indices[n_train:n_train + n_validation]] = "validation"
    labels.loc[ordered_indices[n_train + n_validation:]] = "test"
    return labels
