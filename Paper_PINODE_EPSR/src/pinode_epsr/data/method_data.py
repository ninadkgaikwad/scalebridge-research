from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..core.common import TensorStandardizer
from .phase_d import PhaseDTrajectory


@dataclass(frozen=True)
class MethodArrays:
    case_name: str
    y: np.ndarray
    v: np.ndarray | dict[str, np.ndarray]
    y_names: tuple[str, ...]
    v_names: tuple[str, ...] | dict[str, tuple[str, ...]]
    row_indices: np.ndarray


def _norm_token(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def resolve_signal_column(frame: pd.DataFrame, *, zone: str | None, signal: str, lag: int = 0) -> str:
    """Resolve a physical Phase-D signal without inventing absent channels."""

    signal_norm = _norm_token(signal)
    zone_norm = _norm_token(zone or "")
    candidates: list[tuple[int, str]] = []
    for col in frame.columns:
        norm = _norm_token(str(col))
        if signal_norm not in norm:
            continue
        if "target" in norm or "horizon" in norm:
            continue
        score = 0
        if zone_norm:
            if zone_norm not in norm:
                continue
            score += 4
        if f"lag{lag}" in norm:
            score += 3
        if norm.endswith(signal_norm):
            score += 1
        candidates.append((score, str(col)))
    if not candidates:
        zone_msg = f" for zone {zone}" if zone else ""
        raise KeyError(f"Phase-D signal {signal!r}{zone_msg} is not present")
    candidates.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return candidates[0][1]


def _series(frame: pd.DataFrame, zone: str | None, signal: str) -> np.ndarray:
    col = resolve_signal_column(frame, zone=zone, signal=signal)
    return pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)


def _air_temperature(frame: pd.DataFrame, zone: str) -> np.ndarray:
    return _series(frame, zone, "zone_temperature")


def _outdoor(frame: pd.DataFrame) -> np.ndarray:
    try:
        return _series(frame, None, "outdoor_temperature")
    except KeyError:
        # Some dependent layouts preserve the aggregate/zone prefix.
        cols = [c for c in frame.columns if "outdoor" in str(c).lower() and "temperature" in str(c).lower()]
        if not cols:
            raise
        return pd.to_numeric(frame[cols[0]], errors="raise").to_numpy(dtype=float)


def node_method_arrays(trajectory: PhaseDTrajectory, *, row_indices: Sequence[int] | None = None) -> MethodArrays:
    """Part-3 raw forcing vectors, exactly architecture specific."""

    frame = trajectory.frame
    rows = np.arange(len(frame), dtype=int) if row_indices is None else np.asarray(row_indices, dtype=int)
    To = _outdoor(frame)

    if trajectory.case_name == "all_to_one":
        A = trajectory.zone_ids[0]
        y_names = ("T_A",)
        v_names = ("T_o", "Q_AC,A", "Q_ZIC,A", "Q_ZIR,A", "Q_Sol1,A", "Q_Sol2,A")
        y = _air_temperature(frame, A)[:, None]
        v = np.column_stack(
            [
                To,
                _series(frame, A, "qac"),
                _series(frame, A, "zic"),
                _series(frame, A, "zir"),
                _series(frame, A, "qsol1"),
                _series(frame, A, "qsol2"),
            ]
        )
    elif trajectory.case_name == "identity_ind":
        D, K = trajectory.zone_ids
        y_names = ("T_D", "T_K")
        y = np.column_stack([_air_temperature(frame, D), _air_temperature(frame, K)])
        v = {
            D: np.column_stack(
                [To, _series(frame, D, "qac"), _series(frame, D, "zic"), _series(frame, D, "zir"), _series(frame, D, "qsol1"), _series(frame, D, "qsol2")]
            ),
            K: np.column_stack(
                [To, _series(frame, K, "qac"), _series(frame, K, "zic"), _series(frame, K, "zir")]
            ),
        }
        v_names = {
            D: ("T_o", "Q_AC,D", "Q_ZIC,D", "Q_ZIR,D", "Q_Sol1,D", "Q_Sol2,D"),
            K: ("T_o", "Q_AC,K", "Q_ZIC,K", "Q_ZIR,K"),
        }
    elif trajectory.case_name == "identity_dep1":
        D, K = trajectory.zone_ids
        y_names = ("T_D", "T_K")
        v_names = (
            "T_o", "Q_AC,D", "Q_AC,K", "Q_ZIC,D", "Q_ZIR,D", "Q_Sol1,D", "Q_Sol2,D", "Q_ZIC,K", "Q_ZIR,K"
        )
        y = np.column_stack([_air_temperature(frame, D), _air_temperature(frame, K)])
        v = np.column_stack(
            [
                To,
                _series(frame, D, "qac"), _series(frame, K, "qac"),
                _series(frame, D, "zic"), _series(frame, D, "zir"), _series(frame, D, "qsol1"), _series(frame, D, "qsol2"),
                _series(frame, K, "zic"), _series(frame, K, "zir"),
            ]
        )
    elif trajectory.case_name == "identity_dep2":
        D, K = trajectory.zone_ids
        A = "RestaurantFastFood_All"
        y_names = ("T_D", "T_K")
        v_names = ("T_o", "Q_AC,D", "Q_AC,K", "Q_ZIC,A", "Q_ZIR,A", "Q_Sol1,A", "Q_Sol2,A")
        y = np.column_stack([_air_temperature(frame, D), _air_temperature(frame, K)])
        v = np.column_stack(
            [
                To,
                _series(frame, D, "qac"), _series(frame, K, "qac"),
                _series(frame, A, "zic"), _series(frame, A, "zir"), _series(frame, A, "qsol1"), _series(frame, A, "qsol2"),
            ]
        )
    else:
        raise ValueError(f"Unknown paper case {trajectory.case_name!r}")

    if isinstance(v, dict):
        v_out = {key: value[rows] for key, value in v.items()}
    else:
        v_out = v[rows]
    return MethodArrays(
        case_name=trajectory.case_name,
        y=y[rows],
        v=v_out,
        y_names=tuple(y_names),
        v_names=v_names,
        row_indices=rows,
    )


def inverse_pinn_forcing(trajectory: PhaseDTrajectory, *, row_indices: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Part-2 physical forcing dictionary used by the RC residuals."""

    frame = trajectory.frame
    rows = np.arange(len(frame), dtype=int) if row_indices is None else np.asarray(row_indices, dtype=int)
    out: dict[str, np.ndarray] = {"T_o": _outdoor(frame)[rows]}
    if trajectory.case_name == "all_to_one":
        A = trajectory.zone_ids[0]
        for signal, key in (("qac", "Q_AC,A"), ("zic", "Q_ZIC,A"), ("zir", "Q_ZIR,A"), ("qsol1", "Q_Sol1,A"), ("qsol2", "Q_Sol2,A")):
            out[key] = _series(frame, A, signal)[rows]
    elif trajectory.case_name in {"identity_ind", "identity_dep1"}:
        D, K = trajectory.zone_ids
        for zone, suffix in ((D, "D"), (K, "K")):
            for signal, key in (("qac", "Q_AC"), ("zic", "Q_ZIC"), ("zir", "Q_ZIR")):
                out[f"{key},{suffix}"] = _series(frame, zone, signal)[rows]
            # Kitchen solar is intentionally absent in the controlled identity data.
            for signal, key in (("qsol1", "Q_Sol1"), ("qsol2", "Q_Sol2")):
                try:
                    out[f"{key},{suffix}"] = _series(frame, zone, signal)[rows]
                except KeyError:
                    pass
    elif trajectory.case_name == "identity_dep2":
        D, K = trajectory.zone_ids
        A = "RestaurantFastFood_All"
        out["Q_AC,D"] = _series(frame, D, "qac")[rows]
        out["Q_AC,K"] = _series(frame, K, "qac")[rows]
        out["Qbar_c_nh"] = (_series(frame, A, "zic") + _series(frame, A, "qsol1"))[rows]
        out["Qbar_r"] = (_series(frame, A, "zir") + _series(frame, A, "qsol2"))[rows]
    else:
        raise ValueError(f"Unknown paper case {trajectory.case_name!r}")
    return out


def training_normalizers(arrays: MethodArrays, train_row_mask: np.ndarray, *, dtype=torch.float64):
    local = np.asarray(train_row_mask, dtype=bool)[arrays.row_indices]
    y_scaler = TensorStandardizer.fit(arrays.y[local], names=arrays.y_names, dtype=dtype)
    if isinstance(arrays.v, dict):
        v_scaler = {
            key: TensorStandardizer.fit(values[local], names=arrays.v_names[key], dtype=dtype)
            for key, values in arrays.v.items()
        }
    else:
        v_scaler = TensorStandardizer.fit(arrays.v[local], names=arrays.v_names, dtype=dtype)
    return y_scaler, v_scaler
