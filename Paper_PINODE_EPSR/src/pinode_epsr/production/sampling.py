from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.common import RolloutWindow, contiguous_segments
from ..data.phase_d import PhaseDTrajectory


HPO_SAMPLING_PROTOCOL_VERSION = "month_balanced_floor_budget_v3"


@dataclass(frozen=True)
class SampleBlock:
    month: str
    role: str
    start_index: int
    stop_index_exclusive: int
    rows: int
    source_segment_start_index: int
    source_segment_stop_index_exclusive: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MonthBalancedHPOSample:
    requested_train_percentage: float
    actual_train_percentage: float
    requested_holdout_percentage: float
    actual_holdout_percentage: float
    fit_indices: np.ndarray
    holdout_indices: np.ndarray
    blocks: tuple[SampleBlock, ...]
    monthly_counts: dict[str, dict[str, int]]
    dt_seconds: float
    conservative_context_steps: int
    conservative_rollout_steps: int

    @property
    def holdout_percentage(self) -> float:
        """Backward-compatible alias for the requested inner holdout fraction."""
        return self.requested_holdout_percentage

    def conservative_context_support_indices(self) -> np.ndarray:
        support: set[int] = set()
        context = max(1, int(self.conservative_context_steps))
        for block in self.blocks:
            first_prediction_start = int(block.start_index)
            context_start = max(
                int(block.source_segment_start_index),
                first_prediction_start - context + 1,
            )
            support.update(range(context_start, first_prediction_start))
        support.difference_update(int(i) for i in self.fit_indices)
        support.difference_update(int(i) for i in self.holdout_indices)
        return np.asarray(sorted(support), dtype=int)

    def to_dict(self) -> dict[str, object]:
        context_support = self.conservative_context_support_indices()
        return {
            "requested_train_percentage": self.requested_train_percentage,
            "actual_train_percentage": self.actual_train_percentage,
            "requested_holdout_percentage": self.requested_holdout_percentage,
            "actual_holdout_percentage": self.actual_holdout_percentage,
            "holdout_percentage": self.requested_holdout_percentage,
            "fit_indices": self.fit_indices.tolist(),
            "holdout_indices": self.holdout_indices.tolist(),
            "blocks": [b.to_dict() for b in self.blocks],
            "monthly_counts": self.monthly_counts,
            "dt_seconds": self.dt_seconds,
            "conservative_context_steps": self.conservative_context_steps,
            "conservative_rollout_steps": self.conservative_rollout_steps,
            "conservative_context_support_rows": int(len(context_support)),
            "selected_target_rows": int(len(self.fit_indices) + len(self.holdout_indices)),
            "target_plus_context_upper_bound_rows": int(len(self.fit_indices) + len(self.holdout_indices) + len(context_support)),
            "sampling_semantics": (
                "Per-month target rows=floor(month_train_rows*train_percentage/100). "
                "Per-month holdout target rows=floor(selected_month_targets*holdout_percentage/100). "
                "Neither fraction is rounded up or silently inflated. Causal encoder context may "
                "come from earlier authoritative TRAIN rows inside the same monthly contiguous "
                "segment and is never counted as a target row."
            ),
        }

    def row_segments(self, role: str) -> list[np.ndarray]:
        indices = self.fit_indices if role == "fit" else self.holdout_indices
        return _indices_to_segments(indices)

    def rollout_windows(self, role: str, *, N_r: int, L_e: int, rc_order: int) -> list[RolloutWindow]:
        """Build legal rollout windows whose prediction span stays inside selected targets.

        A 2C encoder may use causal history immediately before a selected target block,
        but only from the same authoritative monthly TRAIN segment.  This lets very
        small HPO percentages remain percentage-faithful instead of silently inflating
        the selected target budget merely to pay for context rows.
        """
        if role not in {"fit", "holdout"}:
            raise ValueError("role must be 'fit' or 'holdout'")
        if N_r < 1 or L_e < 1:
            raise ValueError("N_r and L_e must be >=1")
        context = int(L_e) if int(rc_order) == 2 else 1
        windows: list[RolloutWindow] = []
        segment_id = 0
        for block in self.blocks:
            if block.role != role:
                continue
            earliest_start = max(
                int(block.start_index),
                int(block.source_segment_start_index) + context - 1,
            )
            latest_start = int(block.stop_index_exclusive) - 1 - int(N_r)
            for start in range(earliest_start, latest_start + 1):
                context_start = start - context + 1
                stop = start + int(N_r)
                if context_start < int(block.source_segment_start_index):
                    continue
                if stop >= int(block.stop_index_exclusive):
                    continue
                windows.append(
                    RolloutWindow(
                        segment_id=segment_id,
                        context_start=int(context_start),
                        start=int(start),
                        stop=int(stop),
                        partition=f"hpo_{role}",
                    )
                )
            segment_id += 1
        return windows


def _indices_to_segments(indices: Sequence[int]) -> list[np.ndarray]:
    x = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    if x.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(x) != 1) + 1
    return [seg for seg in np.split(x, cuts) if len(seg)]


def _place_interval_in_range(
    seg: np.ndarray,
    *,
    length: int,
    center_fraction: float,
    local_min: int,
    local_max_exclusive: int,
    occupied: list[tuple[int, int]],
) -> tuple[int, int]:
    """Place one target interval inside a bounded local region without overlap."""
    n = len(seg)
    local_min = max(0, int(local_min))
    local_max_exclusive = min(n, int(local_max_exclusive))
    available = local_max_exclusive - local_min
    length = int(length)
    if length < 1 or available < length:
        raise RuntimeError("Could not place HPO block inside requested monthly region")
    desired_center = local_min + center_fraction * max(0, available - 1)
    desired_start = int(round(desired_center - length / 2.0))
    candidates = sorted(
        range(local_min, local_max_exclusive - length + 1),
        key=lambda s: abs(s - desired_start),
    )
    for start in candidates:
        stop = start + length
        if all(stop <= a or start >= b for a, b in occupied):
            return start, stop
    raise RuntimeError("Could not place non-overlapping HPO block inside monthly TRAIN segment")


def _split_even(total: int, parts: int, minimum: int) -> list[int]:
    if parts < 1:
        raise ValueError("parts must be positive")
    if total < parts * minimum:
        raise ValueError("total cannot satisfy requested minimum per part")
    base = total // parts
    rem = total % parts
    values = [base + (1 if i < rem else 0) for i in range(parts)]
    if min(values) < minimum:
        raise RuntimeError("internal split violated minimum block size")
    return values


def select_month_balanced_hpo_sample(
    trajectory: PhaseDTrajectory,
    *,
    train_percentage: float,
    holdout_percentage: float = 20.0,
    dt_seconds: float = 300.0,
    conservative_N_r: int = 12,
    conservative_L_e: int = 12,
    blocks_per_month: int = 4,
) -> MonthBalancedHPOSample:
    """Select a deterministic, percentage-faithful HPO target budget from every month.

    Scientific contract
    -------------------
    * ``train_percentage`` is applied independently to every contiguous monthly
      authoritative Phase-D TRAIN segment.
    * The selected fit+holdout **target rows** equal the requested monthly
      budget after flooring to whole rows: floor(N_train_month * p / 100).
      The sampler never rounds up or silently increases 0.5%, 2%, 5%, etc.
    * Dynamic models may use earlier causal TRAIN rows from the same monthly
      segment as encoder/initialization context.  Those support rows are context,
      not HPO target rows, and are not counted against the requested percentage.
    * Fit target blocks occur before the holdout target block, so training never
      uses future HPO-holdout observations as causal context.  Holdout may use
      preceding TRAIN/fit observations as causal history, which is the standard
      time-series validation contract.
    * If the requested percentage is mathematically too small to provide at least
      one conservative rollout target window for both fit and holdout in a month,
      the sampler raises a clear error instead of oversampling.
    """
    if not 0.0 < train_percentage <= 100.0:
        raise ValueError("train_percentage must be in (0,100]")
    if not 0.0 < holdout_percentage < 100.0:
        raise ValueError("holdout_percentage must be in (0,100)")
    if blocks_per_month < 2:
        raise ValueError("blocks_per_month must be >=2")

    segments = contiguous_segments(
        trajectory.timestamp,
        trajectory.partition,
        trajectory.included,
        partition_name="train",
        dt_seconds=dt_seconds,
    )
    if not segments:
        raise ValueError("No authoritative Phase-D TRAIN segments")

    ts = pd.to_datetime(trajectory.timestamp, errors="raise")
    context = max(1, int(conservative_L_e))
    rollout = max(1, int(conservative_N_r))
    # A selected target block must contain y_k ... y_{k+N_r} for at least one
    # complete conservative rollout.  Causal encoder rows may precede the block.
    minimum_target_block_rows = rollout + 1

    fit_all: list[int] = []
    hold_all: list[int] = []
    blocks: list[SampleBlock] = []
    monthly_counts: dict[str, dict[str, int]] = {}

    for seg in segments:
        month = ts.iloc[int(seg[0])].strftime("%Y-%m")
        n = len(seg)
        requested = int(math.floor(n * train_percentage / 100.0))
        requested = min(n, requested)
        if requested < 1:
            minimum_pct_one_row = 100.0 / n
            raise ValueError(
                f"{month}: requested HPO train_percentage={train_percentage:g}% floors to "
                f"0 target rows. Use >= {minimum_pct_one_row:.6f}% for at least one row. "
                "The sampler never rounds up."
            )

        # Inner holdout is also a hard, non-exceeding target-row fraction.  It is
        # floored to whole rows and is never inflated merely to pay for rollout
        # geometry. If the requested percentage/holdout/rollout combination is
        # impossible, fail explicitly so the caller can choose a smaller HPO
        # rollout geometry (micro qualification) or a larger HPO data percentage.
        hold_budget = int(math.floor(requested * holdout_percentage / 100.0))
        fit_budget = requested - hold_budget
        minimum_total = 2 * minimum_target_block_rows
        if (
            requested < minimum_total
            or fit_budget < minimum_target_block_rows
            or hold_budget < minimum_target_block_rows
        ):
            minimum_targets_for_holdout = int(
                math.ceil(minimum_target_block_rows * 100.0 / holdout_percentage)
            )
            minimum_targets_for_fit = int(
                math.ceil(minimum_target_block_rows * 100.0 / (100.0 - holdout_percentage))
            )
            minimum_targets = max(
                minimum_total,
                minimum_targets_for_holdout,
                minimum_targets_for_fit,
            )
            minimum_pct = 100.0 * minimum_targets / n
            raise ValueError(
                f"{month}: requested HPO train_percentage={train_percentage:g}% floors to "
                f"{requested} target rows and holdout_percentage={holdout_percentage:g}% "
                f"floors to {hold_budget} holdout rows, but conservative N_r={rollout} "
                f"requires at least {minimum_target_block_rows} target rows in both fit "
                f"and holdout. Use approximately >= {minimum_pct:.4f}% for this month "
                "or explicitly reduce the HPO rollout geometry. The sampler will not "
                "round up, oversample, or inflate the holdout."
            )

        source_start = int(seg[0])
        source_stop = int(seg[-1]) + 1
        month_blocks: list[SampleBlock] = []

        if train_percentage > 55.0:
            # High-coverage mode: use one chronological fit/holdout split and
            # consume exactly the requested target-row budget.  At 100%, all
            # monthly TRAIN rows are selected as either fit or holdout targets.
            outer_start = max(0, (n - requested) // 2)
            fit_local_start = outer_start
            fit_local_stop = fit_local_start + fit_budget
            hold_local_start = fit_local_stop
            hold_local_stop = hold_local_start + hold_budget
            placements = [
                ("fit", fit_local_start, fit_local_stop),
                ("holdout", hold_local_start, hold_local_stop),
            ]
        else:
            # Fast-paper mode: spread fit targets through the earlier 2/3 of the
            # month and put HPO holdout later in time.  The number of fit blocks
            # adapts downward when the percentage is tiny, preserving the budget.
            max_fit_blocks = max(1, int(blocks_per_month) - 1)
            fit_block_count = min(max_fit_blocks, fit_budget // minimum_target_block_rows)
            fit_block_count = max(1, fit_block_count)
            fit_lengths = _split_even(fit_budget, fit_block_count, minimum_target_block_rows)

            occupied: list[tuple[int, int]] = []
            fit_region_stop = max(minimum_target_block_rows, int(math.floor(0.67 * n)))
            centers = np.linspace(0.18, 0.82, fit_block_count)
            placements = []
            for length, center in zip(fit_lengths, centers):
                fs, fe = _place_interval_in_range(
                    seg,
                    length=length,
                    center_fraction=float(center),
                    local_min=context,
                    local_max_exclusive=fit_region_stop,
                    occupied=occupied,
                )
                occupied.append((fs, fe))
                placements.append(("fit", fs, fe))

            # Holdout is always later than every fit target block, preventing
            # future holdout observations from entering fit-window context.
            earliest_hold = max(max(b for _, _, b in placements), int(math.floor(0.72 * n)))
            hs, he = _place_interval_in_range(
                seg,
                length=hold_budget,
                center_fraction=0.65,
                local_min=earliest_hold,
                local_max_exclusive=n,
                occupied=occupied,
            )
            placements.append(("holdout", hs, he))

        for role, local_start, local_stop in placements:
            idx = seg[local_start:local_stop]
            if len(idx) != local_stop - local_start or len(idx) == 0:
                raise RuntimeError(f"{month}: invalid HPO block placement")
            block = SampleBlock(
                month=month,
                role=role,
                start_index=int(idx[0]),
                stop_index_exclusive=int(idx[-1]) + 1,
                rows=len(idx),
                source_segment_start_index=source_start,
                source_segment_stop_index_exclusive=source_stop,
            )
            blocks.append(block)
            month_blocks.append(block)
            if role == "fit":
                fit_all.extend(int(i) for i in idx)
            else:
                hold_all.extend(int(i) for i in idx)

        monthly_counts.setdefault(
            month,
            {
                "train_available": 0,
                "requested_targets": 0,
                "requested_holdout_targets": 0,
                "fit": 0,
                "holdout": 0,
            },
        )
        monthly_counts[month]["train_available"] += n
        monthly_counts[month]["requested_targets"] += requested
        monthly_counts[month]["requested_holdout_targets"] += hold_budget
        monthly_counts[month]["fit"] += sum(b.rows for b in month_blocks if b.role == "fit")
        monthly_counts[month]["holdout"] += sum(b.rows for b in month_blocks if b.role == "holdout")
        if monthly_counts[month]["fit"] + monthly_counts[month]["holdout"] != monthly_counts[month]["requested_targets"]:
            raise RuntimeError(f"{month}: selected HPO target rows do not equal requested monthly budget")
        if monthly_counts[month]["holdout"] != monthly_counts[month]["requested_holdout_targets"]:
            raise RuntimeError(f"{month}: selected HPO holdout rows do not equal floored requested holdout budget")

    fit = np.asarray(sorted(set(fit_all)), dtype=int)
    hold = np.asarray(sorted(set(hold_all)), dtype=int)
    if np.intersect1d(fit, hold).size:
        raise RuntimeError("HPO fit/holdout target overlap detected")
    train_mask = trajectory.mask("train", included_only=True)
    if not np.all(train_mask[fit]) or not np.all(train_mask[hold]):
        raise RuntimeError("HPO selection escaped authoritative Phase-D TRAIN")

    total_train = int(train_mask.sum())
    selected = len(fit) + len(hold)
    actual = 100.0 * selected / total_train
    actual_holdout = 100.0 * len(hold) / selected

    # Verify the conservative 2C/N_r geometry used to guarantee all later trial
    # search-space combinations have at least one legal fit and holdout window.
    provisional = MonthBalancedHPOSample(
        requested_train_percentage=float(train_percentage),
        actual_train_percentage=float(actual),
        requested_holdout_percentage=float(holdout_percentage),
        actual_holdout_percentage=float(actual_holdout),
        fit_indices=fit,
        holdout_indices=hold,
        blocks=tuple(blocks),
        monthly_counts=monthly_counts,
        dt_seconds=float(dt_seconds),
        conservative_context_steps=context,
        conservative_rollout_steps=rollout,
    )
    if not provisional.rollout_windows("fit", N_r=rollout, L_e=context, rc_order=2):
        raise RuntimeError("No conservative legal HPO fit rollout windows")
    if not provisional.rollout_windows("holdout", N_r=rollout, L_e=context, rc_order=2):
        raise RuntimeError("No conservative legal HPO holdout rollout windows")
    return provisional
