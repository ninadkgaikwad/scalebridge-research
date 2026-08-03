# -*- coding: utf-8 -*-
"""Independent validation helpers for Stage C4 regression-pair datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scalebridge.data.heat_input_regression.datasets import (
    PAIR_COLUMNS,
    RegressionPairDataset,
    build_regression_pair_dataset,
)


def validate_regression_pair_dataset(
    dataset: RegressionPairDataset,
    *,
    minimum_split_samples: int,
) -> list[dict[str, Any]]:
    """Validate structure, split counts, finite values, and nonconstant pairs."""
    frame = dataset.frame
    rows: list[dict[str, Any]] = []

    rows.append(_check(
        "required_columns_present",
        set(PAIR_COLUMNS).issubset(frame.columns),
        observed=" | ".join(str(column) for column in frame.columns),
        expected=" | ".join(PAIR_COLUMNS),
    ))
    rows.append(_check(
        "source_row_index_unique",
        not frame["source_row_index"].duplicated().any(),
        observed=int(frame["source_row_index"].duplicated().sum()),
        expected=0,
    ))
    rows.append(_check(
        "timestamp_raw_unique",
        not frame["timestamp_raw"].duplicated().any(),
        observed=int(frame["timestamp_raw"].duplicated().sum()),
        expected=0,
    ))

    valid = frame[frame["pair_valid"]].copy()
    finite_x = bool(np.isfinite(pd.to_numeric(valid["x"], errors="coerce")).all())
    finite_y = bool(np.isfinite(pd.to_numeric(valid["y"], errors="coerce")).all())
    rows.append(_check("valid_predictor_values_are_finite", finite_x, finite_x, True))
    rows.append(_check("valid_target_values_are_finite", finite_y, finite_y, True))
    rows.append(_check(
        "valid_rows_are_split_included",
        bool(valid["split_included"].all()),
        observed=int((~valid["split_included"]).sum()),
        expected=0,
    ))
    rows.append(_check(
        "invalid_rows_have_reason",
        bool(
            frame.loc[~frame["pair_valid"], "pair_exclusion_reason"]
            .astype(str).str.strip().ne("").all()
        ),
        observed=int(
            frame.loc[~frame["pair_valid"], "pair_exclusion_reason"]
            .astype(str).str.strip().eq("").sum()
        ),
        expected=0,
    ))

    for split in ("train", "validation", "test"):
        current = valid[valid["split"] == split]
        rows.append(_check(
            f"minimum_{split}_samples",
            len(current) >= minimum_split_samples,
            observed=int(len(current)),
            expected=f">={minimum_split_samples}",
        ))

    rows.append(_check(
        "train_predictor_nonconstant",
        valid.loc[valid["split"] == "train", "x"].nunique(dropna=True) > 1,
        observed=int(valid.loc[valid["split"] == "train", "x"].nunique(dropna=True)),
        expected=">1",
    ))
    rows.append(_check(
        "train_target_nonconstant",
        valid.loc[valid["split"] == "train", "y"].nunique(dropna=True) > 1,
        observed=int(valid.loc[valid["split"] == "train", "y"].nunique(dropna=True)),
        expected=">1",
    ))
    rows.append(_check(
        "pair_valid_matches_rule",
        _pair_valid_rule_matches(frame),
        observed="saved pair_valid",
        expected="split_included and finite x/y",
    ))
    return rows


def compare_saved_to_recomputed(
    *,
    saved_frame: pd.DataFrame,
    model_id: str,
    feature_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    stage_b_frame: pd.DataFrame,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> list[dict[str, Any]]:
    """Recompute a C4 dataset and compare every structural/numeric field."""
    recomputed = build_regression_pair_dataset(
        model_id=model_id,
        feature_frame=feature_frame,
        split_frame=split_frame,
        stage_b_frame=stage_b_frame,
    ).frame

    rows: list[dict[str, Any]] = []
    rows.append(_check(
        "saved_row_count_matches_recomputed",
        len(saved_frame) == len(recomputed),
        observed=int(len(saved_frame)),
        expected=int(len(recomputed)),
    ))
    if len(saved_frame) != len(recomputed):
        return rows

    for column in (
        "timestamp_raw", "source_row_index", "split", "split_index",
        "split_included", "pair_valid", "pair_exclusion_reason",
    ):
        equal = saved_frame[column].reset_index(drop=True).equals(
            recomputed[column].reset_index(drop=True)
        )
        rows.append(_check(
            f"saved_{column}_matches_recomputed",
            equal,
            observed=equal,
            expected=True,
        ))

    for column in ("x", "y"):
        saved = pd.to_numeric(saved_frame[column], errors="coerce").to_numpy()
        expected = pd.to_numeric(recomputed[column], errors="coerce").to_numpy()
        close = np.isclose(
            saved,
            expected,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            equal_nan=True,
        )
        max_abs = _max_abs_difference(saved, expected)
        rows.append(_check(
            f"saved_{column}_matches_recomputed",
            bool(close.all()),
            observed=max_abs,
            expected=(
                f"allclose(atol={absolute_tolerance}, "
                f"rtol={relative_tolerance})"
            ),
        ))
    return rows


def validation_passed(rows: list[dict[str, Any]]) -> bool:
    return all(row.get("status") == "passed" for row in rows)


def _pair_valid_rule_matches(frame: pd.DataFrame) -> bool:
    x = pd.to_numeric(frame["x"], errors="coerce")
    y = pd.to_numeric(frame["y"], errors="coerce")
    expected = frame["split_included"].astype(bool) & np.isfinite(x) & np.isfinite(y)
    return bool((frame["pair_valid"].astype(bool) == expected).all())


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _check(
    check_name: str,
    passed: bool,
    observed: Any,
    expected: Any,
    message: str = "",
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": "passed" if passed else "failed",
        "observed_value": observed,
        "expected_value": expected,
        "message": message,
    }
