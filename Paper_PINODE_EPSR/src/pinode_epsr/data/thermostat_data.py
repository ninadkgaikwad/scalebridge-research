from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd

from ..core.config import ALL_TO_ONE_RUN, CASE_ID, IDENTITY_RUN, PaperConfig
from .phase_d import PhaseDTrajectory, load_case
from ..evaluation.thermostat import (
    COOL_COL,
    HEAT_COL,
    MDOT_COL,
    TIME_COL,
    TS_COL,
    TZ_COL,
    ThermostatCalibration,
    calibrate_thermostat,
)

_PHASEB_ZONE_RUN = {
    "RestaurantFastFood_All": (ALL_TO_ONE_RUN, "all_to_one"),
    "Dining": (IDENTITY_RUN, "identity_ind"),
    "Kitchen": (IDENTITY_RUN, "identity_ind"),
}

_RAW_TIME_RE = re.compile(
    r"^\s*(?P<m>\d{1,2})/(?P<d>\d{1,2})\s+(?P<h>\d{1,2}):(?P<mi>\d{2}):(?P<s>\d{2})\s*$"
)
_CORE_THERMOSTAT_COLUMNS = (TZ_COL, TS_COL, MDOT_COL, HEAT_COL, COOL_COL)
_CANONICAL_TS_COL = "_phase_d_timestamp"


@dataclass(frozen=True)
class TrainingAlignmentDiagnostics:
    raw_row_count: int
    normalized_unique_timestamp_count: int
    normalized_duplicate_group_count: int
    normalized_duplicate_row_count: int
    duplicate_conflict_group_count: int
    phase_d_train_included_count: int
    strict_exact_train_row_count: int
    exact_train_count_match: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_row_count": self.raw_row_count,
            "normalized_unique_timestamp_count": self.normalized_unique_timestamp_count,
            "normalized_duplicate_group_count": self.normalized_duplicate_group_count,
            "normalized_duplicate_row_count": self.normalized_duplicate_row_count,
            "duplicate_conflict_group_count": self.duplicate_conflict_group_count,
            "phase_d_train_included_count": self.phase_d_train_included_count,
            "strict_exact_train_row_count": self.strict_exact_train_row_count,
            "exact_train_count_match": self.exact_train_count_match,
        }


def phase_b_zone_wide_path(config: PaperConfig, zone_id: str) -> Path:
    if zone_id not in _PHASEB_ZONE_RUN:
        raise KeyError(f"Unsupported controlled thermostat zone {zone_id!r}")
    run_id, _ = _PHASEB_ZONE_RUN[zone_id]
    return (
        config.campaign_root
        / "aggregation"
        / "cases"
        / CASE_ID
        / "runs"
        / run_id
        / "zones"
        / zone_id
        / "aggregated_timeseries_wide.parquet"
    )


def _trajectory_base_year(trajectory: PhaseDTrajectory) -> int:
    """Return the simulation base year used by raw EnergyPlus month/day stamps.

    A one-year interval-ending trajectory can legitimately include Jan 1 of the
    following year because Dec 31 24:00:00 is that physical instant. Therefore
    the first authoritative Phase-D timestamp defines the raw timestamp base
    year; 24:00 normalization is allowed to roll into base_year+1.
    """
    ts = pd.to_datetime(trajectory.timestamp, errors="raise")
    if len(ts) == 0:
        raise RuntimeError("Phase-D trajectory has no timestamps")
    base_year = int(ts.iloc[0].year)
    allowed_years = {base_year, base_year + 1}
    found_years = {int(x.year) for x in ts}
    if not found_years.issubset(allowed_years):
        raise RuntimeError(
            "Controlled Phase-D thermostat alignment spans unexpected years: "
            f"{sorted(found_years)}"
        )
    return base_year


def _normalized_raw_timestamp(value: object, *, year: int) -> pd.Timestamp:
    """Convert EnergyPlus month/day interval-ending time onto the Phase-D year.

    EnergyPlus can emit 24:00:00. That is the same physical instant as the
    following day's 00:00:00, so duplicate normalized timestamps are coalesced
    before partition ownership is checked.
    """
    match = _RAW_TIME_RE.match(str(value))
    if not match:
        return pd.NaT
    month, day, hour, minute, second = [
        int(match.group(k)) for k in ("m", "d", "h", "mi", "s")
    ]
    try:
        base = pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT
    if hour == 24:
        if minute != 0 or second != 0:
            return pd.NaT
        return base + pd.Timedelta(days=1)
    if hour > 23:
        return pd.NaT
    return base + pd.Timedelta(hours=hour, minutes=minute, seconds=second)


def _coalesce_duplicate_group(group: pd.DataFrame) -> pd.Series:
    """Coalesce one normalized timestamp without inventing thermostat values.

    The wide aggregation writer can contain sparse duplicate rows at a physical
    timestamp. For each core thermostat signal, overlapping finite values must
    agree; complementary sparse values are coalesced. Non-core columns are
    filled from the first non-null occurrence.
    """
    result = group.iloc[0].copy()
    for col in group.columns:
        if col == _CANONICAL_TS_COL:
            result[col] = group[col].iloc[0]
            continue
        nonnull = group[col].dropna()
        if nonnull.empty:
            result[col] = np.nan
            continue
        if col in _CORE_THERMOSTAT_COLUMNS:
            numeric = pd.to_numeric(nonnull, errors="coerce").dropna().to_numpy(float)
            if numeric.size > 1 and not np.allclose(numeric, numeric[0], rtol=0.0, atol=1.0e-9):
                raise RuntimeError(
                    f"Conflicting Phase-B duplicate thermostat values at "
                    f"{group[_CANONICAL_TS_COL].iloc[0]} for {col}"
                )
        result[col] = nonnull.iloc[0]
    return result


def _canonicalize_phase_b_timestamps(
    frame: pd.DataFrame,
    trajectory: PhaseDTrajectory,
) -> tuple[pd.DataFrame, TrainingAlignmentDiagnostics]:
    if TIME_COL not in frame.columns:
        raise KeyError(f"Phase-B thermostat frame missing {TIME_COL!r}")

    year = _trajectory_base_year(trajectory)
    work = frame.copy()
    work[_CANONICAL_TS_COL] = work[TIME_COL].map(
        lambda value: _normalized_raw_timestamp(value, year=year)
    )
    if work[_CANONICAL_TS_COL].isna().any():
        bad = work.loc[work[_CANONICAL_TS_COL].isna(), TIME_COL].head(10).tolist()
        raise RuntimeError(f"Unparseable Phase-B thermostat timestamps: {bad}")

    duplicated = work[_CANONICAL_TS_COL].duplicated(keep=False)
    duplicate_rows = work.loc[duplicated]
    duplicate_groups = int(duplicate_rows[_CANONICAL_TS_COL].nunique())

    if duplicate_groups:
        nondup = work.loc[~duplicated]
        merged = [
            _coalesce_duplicate_group(group)
            for _, group in duplicate_rows.groupby(_CANONICAL_TS_COL, sort=True)
        ]
        canonical = pd.concat(
            [nondup, pd.DataFrame(merged)],
            axis=0,
            ignore_index=True,
        )
    else:
        canonical = work

    canonical = canonical.sort_values(_CANONICAL_TS_COL).reset_index(drop=True)

    train_mask = trajectory.mask("train", included_only=True)
    allowed = set(pd.to_datetime(trajectory.timestamp.loc[train_mask]).tolist())
    out = canonical.loc[canonical[_CANONICAL_TS_COL].isin(allowed)].copy().reset_index(drop=True)

    diag = TrainingAlignmentDiagnostics(
        raw_row_count=int(len(frame)),
        normalized_unique_timestamp_count=int(canonical[_CANONICAL_TS_COL].nunique()),
        normalized_duplicate_group_count=duplicate_groups,
        normalized_duplicate_row_count=int(len(duplicate_rows)),
        duplicate_conflict_group_count=0,
        phase_d_train_included_count=int(train_mask.sum()),
        strict_exact_train_row_count=int(len(out)),
        exact_train_count_match=bool(len(out) == int(train_mask.sum())),
    )
    if out.empty:
        raise RuntimeError("Phase-B/Phase-D TRAIN timestamp alignment produced zero rows")
    if not diag.exact_train_count_match:
        raise RuntimeError(
            "Phase-B thermostat calibration does not exactly match authoritative "
            f"Phase-D TRAIN+included ownership: aligned={len(out)} "
            f"expected={int(train_mask.sum())}"
        )
    out.attrs["training_alignment"] = diag.to_dict()
    return out, diag


def filter_phase_b_to_phase_d_training(
    frame: pd.DataFrame,
    trajectory: PhaseDTrajectory,
) -> pd.DataFrame:
    """Restrict Phase-B sequences to the exact authoritative Phase-D TRAIN axis.

    This supersedes the old month/day/time key membership. Raw EnergyPlus
    24:00:00 timestamps are first normalized onto the actual Phase-D year,
    sparse duplicate physical timestamps are coalesced, and only exact
    Phase-D ``train`` + ``included`` timestamps are retained.
    """
    out, _ = _canonicalize_phase_b_timestamps(frame, trajectory)
    return out


def load_phase_b_training_frame(
    config: PaperConfig,
    zone_id: str,
    *,
    trajectory: PhaseDTrajectory | None = None,
) -> pd.DataFrame:
    path = phase_b_zone_wide_path(config, zone_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase-B aggregated thermostat source: {path}")
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Phase-B thermostat calibration requires PyArrow/FastParquet "
            "in the ScaleBridge environment"
        ) from exc
    if trajectory is None:
        _, case_name = _PHASEB_ZONE_RUN[zone_id]
        trajectory = load_case(config, case_name)
    return filter_phase_b_to_phase_d_training(frame, trajectory)


def calibrate_controlled_thermostats(
    config: PaperConfig,
    *,
    deadband_overrides_C: Mapping[str, float] | None = None,
    heating_mode_deadband_overrides_C: Mapping[str, float] | None = None,
    setpoint_quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
    setpoint_min_separation_C: float = 0.50,
) -> dict[str, ThermostatCalibration]:
    deadband_overrides_C = dict(deadband_overrides_C or {})
    heating_mode_deadband_overrides_C = dict(heating_mode_deadband_overrides_C or {})
    trajectories = {
        "RestaurantFastFood_All": load_case(config, "all_to_one"),
        "Dining": load_case(config, "identity_ind"),
        "Kitchen": load_case(config, "identity_ind"),
    }
    out: dict[str, ThermostatCalibration] = {}
    for zone_id, trajectory in trajectories.items():
        frame = load_phase_b_training_frame(config, zone_id, trajectory=trajectory)
        cal = calibrate_thermostat(
            frame,
            zone_id=zone_id,
            deadband_override_C=deadband_overrides_C.get(zone_id),
            heating_mode_deadband_override_C=heating_mode_deadband_overrides_C.get(zone_id),
            setpoint_quantiles=setpoint_quantiles,
            setpoint_min_separation_C=setpoint_min_separation_C,
        )
        prov = dict(cal.provenance)
        prov.update(
            {
                "phase_b_source": str(phase_b_zone_wide_path(config, zone_id)),
                "phase_d_training_case": trajectory.case_name,
                "training_row_count_after_timestamp_alignment": int(len(frame)),
                "training_alignment": dict(frame.attrs.get("training_alignment", {})),
            }
        )
        out[zone_id] = ThermostatCalibration(
            **{**cal.to_dict(), "provenance": prov}
        )
    return out
