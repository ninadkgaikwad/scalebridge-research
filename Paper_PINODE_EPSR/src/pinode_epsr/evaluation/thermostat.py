from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.phase_c import Q_HVAC_X, PhaseCModelBundle

FlowChoice = Literal["nominal", "max"]
UnobservedModePolicy = Literal["fallback", "error"]

TZ_COL = "Zone_Air_Temperature_"
TS_COL = "System_Node_Temperature_"
MDOT_COL = "System_Node_Mass_Flow_Rate"
COOL_COL = "Zone_Air_System_Sensible_Cooling_Rate_"
HEAT_COL = "Zone_Air_System_Sensible_Heating_Rate_"
TIME_COL = "timestamp_raw"


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float); weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values, weights = values[good], weights[good]
    if values.size == 0:
        return float("nan")
    if float(weights.sum()) <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())])


def _quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p:02d}": float("nan") for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    return {f"p{p:02d}": float(np.quantile(x, p / 100.0)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}


@dataclass(frozen=True)
class ThermostatCalibration:
    zone_id: str
    row_count: int
    mode_counts: dict[str, int]
    tz_quantiles_C: dict[str, float]
    ts_cooling_C: float
    ts_heating_C: float
    ts_off_C: float
    mdot_off_kg_s: float
    mdot_cooling_nominal_kg_s: float
    mdot_cooling_max_kg_s: float
    mdot_heating_nominal_kg_s: float
    mdot_heating_max_kg_s: float
    mdot_cooling_quantiles_kg_s: dict[str, float]
    mdot_heating_quantiles_kg_s: dict[str, float]
    deadband_data_C: float
    heating_mode_deadband_data_C: float
    deadband_used_C: float
    heating_mode_deadband_used_C: float
    setpoint_low_C: float
    setpoint_medium_C: float
    setpoint_high_C: float
    transition_temperatures_C: dict[str, list[float]]
    qac_activity_threshold_W: float
    mdot_activity_threshold_kg_s: float
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def supports_mode(self, mode: Literal["cooling", "heating"]) -> bool:
        if mode == "cooling":
            return bool(
                self.mode_counts.get("cooling", 0) > 0
                and np.isfinite(self.ts_cooling_C)
                and np.isfinite(self.mdot_cooling_nominal_kg_s)
                and np.isfinite(self.mdot_cooling_max_kg_s)
            )
        return bool(
            self.mode_counts.get("heating", 0) > 0
            and np.isfinite(self.ts_heating_C)
            and np.isfinite(self.mdot_heating_nominal_kg_s)
            and np.isfinite(self.mdot_heating_max_kg_s)
        )

    def active_mdot(self, mode: Literal["cooling", "heating"], choice: FlowChoice) -> float:
        if not self.supports_mode(mode):
            raise RuntimeError(f"{self.zone_id}: no data-derived {mode} operating regime is available")
        if mode == "cooling":
            value = self.mdot_cooling_nominal_kg_s if choice == "nominal" else self.mdot_cooling_max_kg_s
        else:
            value = self.mdot_heating_nominal_kg_s if choice == "nominal" else self.mdot_heating_max_kg_s
        if not np.isfinite(value):
            raise RuntimeError(f"{self.zone_id}: no data-derived {mode} mass flow is available")
        return float(value)


def classify_hvac_modes(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    """Classify effective aggregate HVAC mode from signed net sensible QAC and Ts-Tz.

    QAC = heating - cooling. This intentionally handles all-to-one rows where
    source zones can heat and cool simultaneously: the effective aggregate mode
    is determined by the signed net effect, not by requiring one raw component
    to be exactly zero.
    """
    tz = pd.to_numeric(frame[TZ_COL], errors="coerce").to_numpy(float)
    ts = pd.to_numeric(frame[TS_COL], errors="coerce").to_numpy(float)
    mdot = pd.to_numeric(frame[MDOT_COL], errors="coerce").to_numpy(float)
    heat = pd.to_numeric(frame[HEAT_COL], errors="coerce").fillna(0).to_numpy(float)
    cool = pd.to_numeric(frame[COOL_COL], errors="coerce").fillna(0).to_numpy(float)
    qac = heat - cool
    finite = np.isfinite(tz) & np.isfinite(ts) & np.isfinite(mdot) & np.isfinite(qac)
    qref = np.quantile(np.abs(qac[finite]), .95) if finite.any() else 0.0
    mref = np.quantile(np.abs(mdot[finite]), .95) if finite.any() else 0.0
    qtol = max(1.0, 0.01 * float(qref))
    mtol = max(1e-9, 0.01 * float(mref))
    mode = np.full(len(frame), "ambiguous", dtype=object)
    active_flow = finite & (mdot > mtol)
    mode[finite & (np.abs(qac) <= qtol)] = "off"
    mode[active_flow & (qac < -qtol) & (ts < tz)] = "cooling"
    mode[active_flow & (qac > qtol) & (ts > tz)] = "heating"
    return mode, {"qac_activity_threshold_W": qtol, "mdot_activity_threshold_kg_s": mtol}


def _transitions(
    mode: np.ndarray,
    tz: np.ndarray,
    timestamps: Sequence[object] | None = None,
    *,
    dt_seconds: float = 300.0,
) -> tuple[dict[str, list[float]], dict[str, object]]:
    """Detect thermostat transitions only across one valid simulation step.

    Production Phase-B calibration is aligned to twelve disjoint monthly
    Phase-D TRAIN segments.  Adjacent rows after filtering are therefore not
    necessarily adjacent in physical time.  A transition is eligible only when
    the canonical timestamps differ by exactly ``dt_seconds`` (300 s for this
    paper).  Synthetic legacy callers without timestamps retain the historical
    adjacency behavior, which is explicitly reported as unverified provenance.
    """
    names = {
        ("off", "cooling"): "cooling_on",
        ("heating", "cooling"): "cooling_mode_enter",
        ("cooling", "off"): "cooling_off",
        ("off", "heating"): "heating_on",
        ("cooling", "heating"): "heating_mode_enter",
        ("heating", "off"): "heating_off",
    }
    out = {name: [] for name in names.values()}
    ts = None if timestamps is None else pd.to_datetime(pd.Series(timestamps), errors="raise")
    eligible_pairs = 0; skipped_noncontiguous_pairs = 0
    for i in range(1, len(mode)):
        if ts is not None:
            delta = (ts.iloc[i] - ts.iloc[i-1]).total_seconds()
            if not np.isclose(delta, float(dt_seconds), rtol=0.0, atol=1e-6):
                skipped_noncontiguous_pairs += 1
                continue
        eligible_pairs += 1
        key = names.get((str(mode[i-1]), str(mode[i])))
        if key and np.isfinite(tz[i]):
            out[key].append(float(tz[i]))
    return out, {
        "dt_seconds": float(dt_seconds),
        "timestamps_available": ts is not None,
        "eligible_transition_pairs": int(eligible_pairs),
        "skipped_noncontiguous_pairs": int(skipped_noncontiguous_pairs),
        "transition_continuity_rule": "exact_dt" if ts is not None else "legacy_row_adjacency_unverified",
    }


def _median_or_nan(values: Sequence[float]) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def calibrate_thermostat(
    frame: pd.DataFrame,
    *,
    zone_id: str,
    deadband_override_C: float | None = None,
    heating_mode_deadband_override_C: float | None = None,
    setpoint_quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
    setpoint_min_separation_C: float = 0.50,
) -> ThermostatCalibration:
    required = {TZ_COL, TS_COL, MDOT_COL, COOL_COL, HEAT_COL}
    missing = sorted(required.difference(frame.columns))
    if missing: raise KeyError(f"Thermostat calibration missing columns: {missing}")
    if not (0 <= setpoint_quantiles[0] < setpoint_quantiles[1] < setpoint_quantiles[2] <= 1):
        raise ValueError("setpoint_quantiles must be ordered in [0,1]")
    if setpoint_min_separation_C < 0:
        raise ValueError("setpoint_min_separation_C must be nonnegative")

    mode, thresholds = classify_hvac_modes(frame)
    tz = pd.to_numeric(frame[TZ_COL], errors="coerce").to_numpy(float)
    ts = pd.to_numeric(frame[TS_COL], errors="coerce").to_numpy(float)
    mdot = pd.to_numeric(frame[MDOT_COL], errors="coerce").to_numpy(float)
    cool = mode == "cooling"; heat = mode == "heating"; off = mode == "off"

    canonical_ts_col = "_phase_d_timestamp" if "_phase_d_timestamp" in frame.columns else None
    transition_timestamps = frame[canonical_ts_col] if canonical_ts_col else None
    trans, transition_diag = _transitions(mode, tz, transition_timestamps, dt_seconds=300.0)
    # Empirical cycling half-width candidates. If direct on/off transitions are
    # sparse, fall back to a robust Tz spread derived from the same data.
    candidates = []
    co, cf = _median_or_nan(trans["cooling_on"]), _median_or_nan(trans["cooling_off"])
    ho, hf = _median_or_nan(trans["heating_on"]), _median_or_nan(trans["heating_off"])
    if np.isfinite(co) and np.isfinite(cf): candidates.append(0.5 * abs(co - cf))
    if np.isfinite(ho) and np.isfinite(hf): candidates.append(0.5 * abs(ho - hf))
    tzq = _quantiles(tz)
    dead_data = float(np.median(candidates)) if candidates else max(0.1, 0.25 * (tzq["p75"] - tzq["p25"]))

    # Heating/cooling mode-switch separation beyond the ordinary HVAC deadband.
    h_enter = _median_or_nan(trans["heating_mode_enter"] + trans["heating_on"])
    c_enter = _median_or_nan(trans["cooling_mode_enter"] + trans["cooling_on"])
    if np.isfinite(h_enter) and np.isfinite(c_enter):
        mode_db_data = max(0.0, 0.5 * abs(c_enter - h_enter) - dead_data)
    else:
        mode_db_data = max(0.0, 0.5 * (tzq["p90"] - tzq["p10"]) - dead_data)

    dead_used = dead_data if deadband_override_C is None else float(deadband_override_C)
    mode_db_used = mode_db_data if heating_mode_deadband_override_C is None else float(heating_mode_deadband_override_C)
    if dead_used < 0 or mode_db_used < 0: raise ValueError("deadband overrides must be nonnegative")

    qlo, qmid, qhi = setpoint_quantiles
    finite_tz = tz[np.isfinite(tz)]
    sp_low, sp_mid_data, sp_high = [
        float(np.quantile(finite_tz, q)) for q in (qlo, qmid, qhi)
    ]
    # Some commercial-zone schedules create highly concentrated temperature
    # plateaus. Dining in the controlled RestaurantFastFood data has P10≈P50,
    # which would make "low" and "medium" numerically indistinguishable.
    # Preserve data-derived low/high quantiles, but use their midpoint when the
    # empirical median is not meaningfully separated from both.
    if (
        sp_mid_data - sp_low < setpoint_min_separation_C
        or sp_high - sp_mid_data < setpoint_min_separation_C
    ):
        sp_mid = 0.5 * (sp_low + sp_high)
        sp_mid_source = "midpoint_low_high_due_quantile_collapse"
    else:
        sp_mid = sp_mid_data
        sp_mid_source = f"empirical_quantile_{qmid:g}"

    return ThermostatCalibration(
        zone_id=zone_id, row_count=int(len(frame)),
        mode_counts={k: int(np.sum(mode == k)) for k in ("cooling","heating","off","ambiguous")},
        tz_quantiles_C=tzq,
        ts_cooling_C=_weighted_median(ts[cool], mdot[cool]) if cool.any() else float("nan"),
        ts_heating_C=_weighted_median(ts[heat], mdot[heat]) if heat.any() else float("nan"),
        ts_off_C=_weighted_median(ts[off], np.maximum(mdot[off], 0.0)) if off.any() else float("nan"),
        mdot_off_kg_s=float(np.median(mdot[off])) if off.any() else 0.0,
        mdot_cooling_nominal_kg_s=float(np.median(mdot[cool])) if cool.any() else float("nan"),
        mdot_cooling_max_kg_s=float(np.max(mdot[cool])) if cool.any() else float("nan"),
        mdot_heating_nominal_kg_s=float(np.median(mdot[heat])) if heat.any() else float("nan"),
        mdot_heating_max_kg_s=float(np.max(mdot[heat])) if heat.any() else float("nan"),
        mdot_cooling_quantiles_kg_s=_quantiles(mdot[cool]),
        mdot_heating_quantiles_kg_s=_quantiles(mdot[heat]),
        deadband_data_C=float(dead_data), heating_mode_deadband_data_C=float(mode_db_data),
        deadband_used_C=float(dead_used), heating_mode_deadband_used_C=float(mode_db_used),
        setpoint_low_C=sp_low, setpoint_medium_C=sp_mid, setpoint_high_C=sp_high,
        transition_temperatures_C=trans,
        qac_activity_threshold_W=float(thresholds["qac_activity_threshold_W"]),
        mdot_activity_threshold_kg_s=float(thresholds["mdot_activity_threshold_kg_s"]),
        provenance={
            "source": "Phase-B aggregated Tz/Ts/mdot/heating/cooling; training rows only required by caller",
            "qac_sign": "Q_AC = Q_heating - Q_cooling",
            "mode_rule": "signed net QAC plus sign(Ts-Tz)",
            "ts_constant": "mass-flow-weighted median by effective mode",
            "mdot_nominal": "median active mass flow by effective mode",
            "mdot_max": "maximum finite active mass flow by effective mode",
            "deadband_source": "override" if deadband_override_C is not None else "data_300s_transition_calibration",
            "deadband_source_class": "user_override" if deadband_override_C is not None else "data_300s_transition_calibration",
            "deadband_transition_diagnostics": transition_diag,
            "heating_mode_deadband_source": "override" if heating_mode_deadband_override_C is not None else "data",
            "heating_mode_deadband_source_class": "user_override" if heating_mode_deadband_override_C is not None else "data_300s_transition_calibration",
            "deadband_semantics": "half_width_about_setpoint",
            "deadband_total_width_C": float(2.0 * dead_used),
            "setpoint_quantiles": list(setpoint_quantiles),
            "setpoint_min_separation_C": float(setpoint_min_separation_C),
            "setpoint_medium_data_C": float(sp_mid_data),
            "setpoint_medium_source": sp_mid_source,
            "observed_modes": {
                "cooling": bool(cool.any()),
                "heating": bool(heat.any()),
            },
            # Positive mode-specific supply-air temperature offsets used only
            # to construct transparent fallback actuation for an unobserved mode.
            "cooling_supply_deltaT_C": (
                _weighted_median((tz - ts)[cool], mdot[cool]) if cool.any() else float("nan")
            ),
            "heating_supply_deltaT_C": (
                _weighted_median((ts - tz)[heat], mdot[heat]) if heat.any() else float("nan")
            ),
        },
    )


@dataclass(frozen=True)
class ThermostatModeActuation:
    mode: Literal["cooling", "heating"]
    observed_in_train: bool
    parameter_source: str
    T_supply_C: float
    mdot_nominal_kg_s: float
    mdot_max_kg_s: float
    reference_deltaT_C: float
    qac_extrapolation_expected: bool

    def active_mdot(self, choice: FlowChoice) -> float:
        value = self.mdot_nominal_kg_s if choice == "nominal" else self.mdot_max_kg_s
        if not np.isfinite(value) or value < 0:
            raise RuntimeError(f"{self.mode}: invalid resolved mass flow {value}")
        return float(value)


@dataclass(frozen=True)
class ThermostatActuationProfile:
    zone_id: str
    cooling: ThermostatModeActuation
    heating: ThermostatModeActuation
    provenance: dict[str, object]

    def for_mode(self, mode: Literal["cooling", "heating"]) -> ThermostatModeActuation:
        return self.cooling if mode == "cooling" else self.heating

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mode_override(
    overrides: Mapping[str, Mapping[str, float]] | None,
    mode: Literal["cooling", "heating"],
) -> Mapping[str, float]:
    if overrides is None:
        return {}
    raw = overrides.get(mode, {})
    return dict(raw)


def resolve_actuation_profile(
    calibration: ThermostatCalibration,
    *,
    overrides: Mapping[str, Mapping[str, float]] | None = None,
    fallback_strategy: str = "same_zone_opposite_mode_deltaT_mdot",
) -> ThermostatActuationProfile:
    """Resolve executable heating and cooling parameters independently of TRAIN support.

    TRAIN mode support is provenance/confidence, not an actuator constraint.

    Observed modes use data-derived Ts and nominal/max mdot.  If a mode is not
    observed, the default fallback uses only that same zone's observed opposite
    mode: its mass-flow-weighted active |Ts-Tz| and its nominal/max active airflow.
    The temperature difference is reflected around the same zone's medium setpoint,
    with a physical sign guard on the synthesized fallback only, ensuring fallback
    heating Ts is above the high setpoint and fallback cooling Ts is below the low
    setpoint. Observed same-mode TRAIN values are preserved exactly. No
    equipment-definition lookup and no
    other-run lookup are used. Explicit per-mode overrides can replace any resolved
    Ts/mdot value.

    This makes the controller capable of both heating and cooling while making
    unobserved-mode actuation explicitly extrapolative.
    """
    if fallback_strategy != "same_zone_opposite_mode_deltaT_mdot":
        raise ValueError(f"Unsupported fallback_strategy {fallback_strategy!r}")

    c = calibration
    observed = {
        "cooling": c.supports_mode("cooling"),
        "heating": c.supports_mode("heating"),
    }
    resolution: dict[str, dict[str, object]] = {}

    def observed_values(mode: Literal["cooling", "heating"]):
        if mode == "cooling":
            return (
                float(c.ts_cooling_C),
                float(c.mdot_cooling_nominal_kg_s),
                float(c.mdot_cooling_max_kg_s),
                float(c.provenance.get("cooling_supply_deltaT_C", float("nan"))),
            )
        return (
            float(c.ts_heating_C),
            float(c.mdot_heating_nominal_kg_s),
            float(c.mdot_heating_max_kg_s),
            float(c.provenance.get("heating_supply_deltaT_C", float("nan"))),
        )

    def resolve_one(mode: Literal["cooling", "heating"]) -> ThermostatModeActuation:
        override = _mode_override(overrides, mode)
        if observed[mode]:
            ts, md_nom, md_max, delta = observed_values(mode)
            if not np.isfinite(delta) or delta <= 0:
                delta = (
                    max(0.1, c.setpoint_medium_C - ts)
                    if mode == "cooling"
                    else max(0.1, ts - c.setpoint_medium_C)
                )
            base_source = "observed_train"
        else:
            opposite: Literal["cooling", "heating"] = "heating" if mode == "cooling" else "cooling"
            if not observed[opposite]:
                required = {"T_supply_C", "mdot_nominal_kg_s", "mdot_max_kg_s"}
                if not required.issubset(override):
                    raise RuntimeError(
                        f"{c.zone_id}: neither {mode} nor {opposite} is observed in TRAIN; "
                        f"explicit {mode} overrides are required"
                    )
                ts = float(override["T_supply_C"])
                md_nom = float(override["mdot_nominal_kg_s"])
                md_max = float(override["mdot_max_kg_s"])
                delta = abs(ts - c.setpoint_medium_C)
                base_source = "unavailable_in_train"
            else:
                opp_ts, opp_nom, opp_max, opp_delta = observed_values(opposite)
                if not np.isfinite(opp_delta) or opp_delta <= 0:
                    opp_delta = abs(opp_ts - c.setpoint_medium_C)
                delta = max(float(opp_delta), 0.1)
                sign_guard = max(float(c.deadband_used_C), 0.5)
                if mode == "heating":
                    ts = max(
                        c.setpoint_medium_C + delta,
                        c.setpoint_high_C + sign_guard,
                    )
                else:
                    ts = min(
                        c.setpoint_medium_C - delta,
                        c.setpoint_low_C - sign_guard,
                    )
                md_nom = float(opp_nom)
                md_max = float(opp_max)
                base_source = f"fallback_same_zone_{opposite}_deltaT_and_mdot"

        # Per-mode explicit overrides always win.  Keep a structured record of
        # the underlying data/fallback source and the exact user-replaced fields.
        overridden_parameters: list[str] = []
        if "T_supply_C" in override:
            ts = float(override["T_supply_C"])
            overridden_parameters.append("T_supply_C")
        if "mdot_nominal_kg_s" in override:
            md_nom = float(override["mdot_nominal_kg_s"])
            overridden_parameters.append("mdot_nominal_kg_s")
        if "mdot_max_kg_s" in override:
            md_max = float(override["mdot_max_kg_s"])
            overridden_parameters.append("mdot_max_kg_s")
        final_source = "user_override" if overridden_parameters else base_source
        if overridden_parameters:
            source_class = "user_override"
        elif observed[mode]:
            source_class = "data_train_observed"
        elif base_source.startswith("fallback_same_zone_"):
            source_class = "fallback_same_zone_opposite_mode_deltaT_mdot"
        else:
            source_class = base_source

        if not (np.isfinite(ts) and np.isfinite(md_nom) and np.isfinite(md_max)):
            raise RuntimeError(f"{c.zone_id}: non-finite resolved {mode} actuation parameters")
        if md_nom < 0 or md_max < 0 or md_max + 1e-12 < md_nom:
            raise RuntimeError(f"{c.zone_id}: invalid resolved {mode} mass-flow parameters")
        # The synthesized fallback branch above applies its own directional
        # setpoint guard. Do not apply that guard to observed TRAIN actuator
        # values: paper experiment setpoints are evaluation conditions, not
        # validity bounds on historical supply-air temperatures. Explicit
        # overrides likewise remain explicit user choices.
        resolution[mode] = {
            "final_source": final_source,
            "base_source": base_source,
            "source_class": source_class,
            "overridden_parameters": list(overridden_parameters),
            "observed_in_train": bool(observed[mode]),
            "qac_extrapolation_expected": bool(not observed[mode]),
        }
        return ThermostatModeActuation(
            mode=mode,
            observed_in_train=bool(observed[mode]),
            parameter_source=final_source,
            T_supply_C=float(ts),
            mdot_nominal_kg_s=float(md_nom),
            mdot_max_kg_s=float(md_max),
            reference_deltaT_C=float(delta),
            qac_extrapolation_expected=bool(not observed[mode]),
        )

    cooling = resolve_one("cooling")
    heating = resolve_one("heating")
    return ThermostatActuationProfile(
        zone_id=c.zone_id,
        cooling=cooling,
        heating=heating,
        provenance={
            "training_mode_support": dict(observed),
            "parameter_priority": (
                "user_override > observed_train > "
                "fallback_same_zone_opposite_mode_deltaT_mdot"
            ),
            "fallback_strategy": fallback_strategy,
            "fallback_external_sources_used": False,
            "fallback_uses_equipment_definition": False,
            "fallback_uses_other_runs": False,
            "unobserved_mode_semantics": (
                "actuation permitted; Phase-C QAC is extrapolative/OOD for that mode"
            ),
            "overrides": {k: dict(v) for k, v in (overrides or {}).items()},
            "mode_resolution": resolution,
        },
    )


@dataclass
class ThermostatState:
    hvac_on: int = 0
    heating_mode: int = 0


@dataclass(frozen=True)
class ThermostatAction:
    requested_hvac_on: int
    heating_mode: int
    hvac_on: int
    delivered_mode: str
    mode_available: bool
    mode_observed_in_train: bool
    actuation_parameter_source: str
    qac_extrapolation: bool
    action_suppressed: bool
    setpoint_C: float
    m_dot_kg_s: float
    T_supply_C: float
    Q_HVAC_X_W: float
    Q_AC_W: float
    P_HVAC_model_raw_W: float
    P_HVAC_W: float

    @property
    def Q_HVAC_phaseC_W(self) -> float:
        """Corrected/effective HVAC heat returned by the Phase-C QAC model."""
        return self.Q_AC_W

    @property
    def P_HVAC_physical_W(self) -> float:
        """Nonnegative electric HVAC power used for energy accounting."""
        return self.P_HVAC_W


class LegacyHeatingCoolingThermostat:
    """Exact state-transition structure of HEMS_AC_Controller_WithHeatingMode.

    The state machine can command both heating and cooling in every zone.
    TRAIN mode availability controls parameter provenance/confidence only.
    """
    def __init__(
        self,
        calibration: ThermostatCalibration,
        phase_c: PhaseCModelBundle,
        *,
        cooling_mdot_choice: FlowChoice = "nominal",
        heating_mdot_choice: FlowChoice = "nominal",
        unobserved_mode_policy: UnobservedModePolicy = "fallback",
        actuation_profile: ThermostatActuationProfile | None = None,
        actuation_overrides: Mapping[str, Mapping[str, float]] | None = None,
        initial_hvac_on: int = 0,
        initial_heating_mode: int = 0,
    ) -> None:
        if unobserved_mode_policy not in ("fallback", "error"):
            raise ValueError(f"Unsupported unobserved_mode_policy {unobserved_mode_policy!r}")
        self.calibration = calibration
        self.phase_c = phase_c
        self.cooling_mdot_choice = cooling_mdot_choice
        self.heating_mdot_choice = heating_mdot_choice
        self.unobserved_mode_policy = unobserved_mode_policy
        self.actuation_profile = actuation_profile or resolve_actuation_profile(
            calibration,
            overrides=actuation_overrides,
        )
        self.state = ThermostatState(int(initial_hvac_on), int(initial_heating_mode))

    def command(self, T_z_C: float, setpoint_C: float) -> ThermostatAction:
        c = self.calibration
        db = c.deadband_used_C
        hmdb = c.heating_mode_deadband_used_C
        prev_h, prev_m = self.state.hvac_on, self.state.heating_mode

        if prev_m == 0 and T_z_C <= setpoint_C - db - hmdb:
            mode = 1
        elif prev_m == 1 and T_z_C >= setpoint_C + db + hmdb:
            mode = 0
        else:
            mode = prev_m

        if mode == 0:
            if prev_h == 0 and T_z_C >= setpoint_C + db:
                hvac = 1
            elif prev_h == 1 and T_z_C <= setpoint_C - db:
                hvac = 0
            else:
                hvac = prev_h
        else:
            if prev_h == 0 and T_z_C <= setpoint_C - db:
                hvac = 1
            elif prev_h == 1 and T_z_C >= setpoint_C + db:
                hvac = 0
            else:
                hvac = prev_h

        self.state = ThermostatState(hvac, mode)
        requested_hvac = int(hvac)
        requested_mode: Literal["cooling", "heating"] = "heating" if mode == 1 else "cooling"
        params = self.actuation_profile.for_mode(requested_mode)

        if requested_hvac and (not params.observed_in_train) and self.unobserved_mode_policy == "error":
            raise RuntimeError(
                f"{c.zone_id}: thermostat requested {requested_mode}, which was not observed "
                "in TRAIN; fallback is disabled by unobserved_mode_policy='error'"
            )

        # Both heating and cooling are executable.  Unobserved modes use the
        # resolved fallback/override profile and are tagged as extrapolative.
        delivered_hvac = requested_hvac
        if delivered_hvac:
            choice = self.cooling_mdot_choice if requested_mode == "cooling" else self.heating_mdot_choice
            mdot = params.active_mdot(choice)
            ts = params.T_supply_C
            delivered_mode = requested_mode
            qac_extrapolation = bool(params.qac_extrapolation_expected)
        else:
            delivered_mode = "off"
            mdot = c.mdot_off_kg_s
            ts = (
                c.ts_off_C
                if np.isfinite(c.ts_off_C) and mdot > c.mdot_activity_threshold_kg_s
                else float(T_z_C)
            )
            qac_extrapolation = False

        proxy = float(Q_HVAC_X(float(mdot), float(ts), float(T_z_C)))
        qac = float(np.asarray(self.phase_c.predict_corrected_qhvac_from_physics(proxy)).reshape(-1)[0])
        phvac_raw = float(np.asarray(self.phase_c.predict_phvac_from_corrected_qhvac(qac)).reshape(-1)[0])
        phvac = max(0.0, phvac_raw)

        return ThermostatAction(
            requested_hvac_on=requested_hvac,
            heating_mode=mode,
            hvac_on=delivered_hvac,
            delivered_mode=delivered_mode,
            mode_available=True,
            mode_observed_in_train=bool(params.observed_in_train),
            actuation_parameter_source=str(params.parameter_source),
            qac_extrapolation=bool(qac_extrapolation),
            action_suppressed=False,
            setpoint_C=float(setpoint_C),
            m_dot_kg_s=float(mdot),
            T_supply_C=float(ts),
            Q_HVAC_X_W=proxy,
            Q_AC_W=qac,
            P_HVAC_model_raw_W=phvac_raw,
            P_HVAC_W=phvac,
        )


def medium_low_high_medium_schedule(calibration: ThermostatCalibration, n_steps: int) -> np.ndarray:
    if n_steps < 4: raise ValueError("Sim3 schedule requires at least 4 steps")
    values = (calibration.setpoint_medium_C, calibration.setpoint_low_C,
              calibration.setpoint_high_C, calibration.setpoint_medium_C)
    lengths = [n_steps // 4] * 4
    for i in range(n_steps % 4): lengths[i] += 1
    return np.concatenate([np.full(lengths[i], values[i], dtype=float) for i in range(4)])
