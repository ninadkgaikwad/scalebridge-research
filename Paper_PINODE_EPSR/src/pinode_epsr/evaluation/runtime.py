from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..core.common import contiguous_segments
from ..data.phase_d import PhaseDTrajectory
from ..methods.inverse_pinn import InversePINNRC
from ..data.method_data import MethodArrays, inverse_pinn_forcing, node_method_arrays
from ..methods.neural_ode import NeuralODEModel
from ..data.phase_c import PhaseCModelBundle
from .thermostat import LegacyHeatingCoolingThermostat, ThermostatCalibration, ThermostatActuationProfile, medium_low_high_medium_schedule, resolve_actuation_profile
from .metrics import full_prediction_metrics


@dataclass(frozen=True)
class EvaluationResult:
    simulation: str
    case_name: str
    rc_order: int
    trajectory: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    provenance: dict[str, Any]


def _zone_suffixes(case_name: str, zone_ids: Sequence[str]) -> list[tuple[str, str]]:
    return [(zone_ids[0], "A")] if case_name == "all_to_one" else [(zone_ids[0], "D"), (zone_ids[1], "K")]


class PaperModelRuntime:
    """Thin common runtime for the four already-built paper methods.

    It deliberately does not generalize the model architecture; that work belongs
    to ScaleBridge Phase E.0. This adapter only makes Sim1/2/3 semantics identical
    across the current paper methods.
    """
    def __init__(self, model: torch.nn.Module, trajectory: PhaseDTrajectory) -> None:
        self.model = model; self.trajectory = trajectory
        self.arrays = node_method_arrays(trajectory)
        self.is_inverse = isinstance(model, InversePINNRC)
        self.is_node_family = isinstance(model, NeuralODEModel)
        if not (self.is_inverse or self.is_node_family):
            raise TypeError(f"Unsupported paper runtime model: {type(model).__name__}")
        self.rc_order = int(model.config.rc_order)
        self.physical = inverse_pinn_forcing(trajectory) if self.is_inverse else None

    @property
    def zone_ids(self) -> tuple[str, ...]: return self.trajectory.zone_ids

    def _context(self, k: int):
        if self.rc_order == 1: return None, None
        L = int(self.model.config.L_e)
        if k - L + 1 < 0: raise ValueError("insufficient causal history for 2C initialization")
        cy = torch.as_tensor(self.arrays.y[k-L+1:k+1], dtype=torch.float64)
        if isinstance(self.arrays.v, dict):
            cv = {key: torch.as_tensor(v[k-L+1:k+1], dtype=torch.float64) for key, v in self.arrays.v.items()}
        else: cv = torch.as_tensor(self.arrays.v[k-L+1:k+1], dtype=torch.float64)
        return cy, cv

    def initialize(self, k: int) -> torch.Tensor:
        y0 = torch.as_tensor(self.arrays.y[k], dtype=torch.float64)
        if self.is_inverse:
            # Locked Day-6 fairness rule: methods without a causal encoder use Ta==Tm.
            if self.rc_order == 1: return y0.clone()
            if self.trajectory.case_name == "all_to_one": return torch.stack((y0[0], y0[0]))
            return torch.stack((y0[0], y0[0], y0[1], y0[1]))
        cy, cv = self._context(k)
        return self.model.initial_state(y0, context_y=cy, context_v=cv).detach()

    def observe(self, state: torch.Tensor) -> np.ndarray:
        if self.is_inverse:
            x = state.reshape(1, -1)
            return self.model.observed_air(x).detach().cpu().numpy().reshape(-1)
        return self.model.observe(state).detach().cpu().numpy().reshape(-1)

    def _node_forcing_row(self, k: int, qac_override: Mapping[str, float] | None = None):
        qac_override = dict(qac_override or {})
        if isinstance(self.arrays.v, dict):
            out = {}
            for zone, values in self.arrays.v.items():
                row = values[k].copy(); names = self.arrays.v_names[zone]
                suffix = "D" if zone == self.zone_ids[0] else "K"
                for i, name in enumerate(names):
                    if name == f"Q_AC,{suffix}" and zone in qac_override: row[i] = qac_override[zone]
                out[zone] = torch.as_tensor(row, dtype=torch.float64)
            return out
        row = self.arrays.v[k].copy(); names = self.arrays.v_names
        for zone, suffix in _zone_suffixes(self.trajectory.case_name, self.zone_ids):
            if zone not in qac_override: continue
            token = f"Q_AC,{suffix}"
            for i, name in enumerate(names):
                if name == token: row[i] = qac_override[zone]
        return torch.as_tensor(row, dtype=torch.float64)

    def _inverse_forcing_row(self, k: int, qac_override: Mapping[str, float] | None = None) -> dict[str, float]:
        assert self.physical is not None
        row = {name: float(values[k]) for name, values in self.physical.items()}
        for zone, suffix in _zone_suffixes(self.trajectory.case_name, self.zone_ids):
            if qac_override and zone in qac_override: row[f"Q_AC,{suffix}"] = float(qac_override[zone])
        return row

    def qac_at(self, k: int) -> dict[str, float]:
        if self.is_inverse:
            row = self._inverse_forcing_row(k)
            return {zone: float(row[f"Q_AC,{suffix}"]) for zone, suffix in _zone_suffixes(self.trajectory.case_name, self.zone_ids)}
        v = self._node_forcing_row(k)
        out = {}
        if isinstance(v, dict):
            for zone, suffix in _zone_suffixes(self.trajectory.case_name, self.zone_ids):
                names = self.arrays.v_names[zone]; pos = list(names).index(f"Q_AC,{suffix}")
                out[zone] = float(v[zone][pos])
        else:
            for zone, suffix in _zone_suffixes(self.trajectory.case_name, self.zone_ids):
                pos = list(self.arrays.v_names).index(f"Q_AC,{suffix}"); out[zone] = float(v[pos])
        return out

    def step(self, state: torch.Tensor, k: int, *, qac_override: Mapping[str, float] | None = None) -> torch.Tensor:
        if self.is_inverse:
            row = self._inverse_forcing_row(k, qac_override)
            hist = self.model.physical_rollout(state, [row], dt_seconds=300.0, n_substeps=1)
            return hist[-1].detach()
        vrow = self._node_forcing_row(k, qac_override)
        if isinstance(vrow, dict):
            vrow = {key: value.reshape(1, -1) for key, value in vrow.items()}
        else:
            vrow = vrow.reshape(1, -1)
        return self.model.step(state.reshape(1, -1), vrow).reshape(-1).detach()


def _test_segments(runtime: PaperModelRuntime) -> list[np.ndarray]:
    return contiguous_segments(runtime.trajectory.timestamp, runtime.trajectory.partition, runtime.trajectory.included,
                               partition_name="test", dt_seconds=float(runtime.model.config.dt_seconds if hasattr(runtime.model.config,"dt_seconds") else 300.0))


def _metric_dict(zone_ids: Sequence[str], truth: np.ndarray, pred: np.ndarray) -> dict[str, dict[str, float]]:
    return {zone: full_prediction_metrics(truth[:, i], pred[:, i]) for i, zone in enumerate(zone_ids)}



def _phase_c_power_values(bundle: PhaseCModelBundle, q_phase_c_W: float) -> tuple[float, float]:
    raw = float(np.asarray(bundle.predict_phvac_from_qac(q_phase_c_W)).reshape(-1)[0])
    return raw, max(0.0, raw)

def sim1(runtime: PaperModelRuntime, phase_c: Mapping[str, PhaseCModelBundle], *, max_points: int | None = None) -> EvaluationResult:
    rows=[]; truth=[]; pred=[]; count=0
    for seg in _test_segments(runtime):
        context_skip = int(getattr(runtime.model.config, "L_e", 1)) - 1 if (runtime.rc_order == 2 and not runtime.is_inverse) else 0
        for k in seg[context_skip:-1]:
            state=runtime.initialize(int(k)); nxt=runtime.step(state,int(k)); yp=runtime.observe(nxt); yt=runtime.arrays.y[int(k)+1]
            qacs=runtime.qac_at(int(k))
            rec={"timestamp":runtime.trajectory.timestamp.iloc[int(k)+1], "step_index":int(k), "simulation":"sim1"}
            for i,z in enumerate(runtime.zone_ids):
                raw_p, physical_p = _phase_c_power_values(phase_c[z], qacs[z])
                rec[f"Tz_true__{z}"]=float(yt[i]); rec[f"Tz_pred__{z}"]=float(yp[i])
                rec[f"QHVAC_phaseC_W__{z}"]=qacs[z]; rec[f"QAC_W__{z}"]=qacs[z]
                rec[f"abs_QHVAC_phaseC_W__{z}"]=abs(qacs[z])
                rec[f"PHVAC_model_raw_W__{z}"]=raw_p; rec[f"PHVAC_W__{z}"]=physical_p
            rows.append(rec); truth.append(yt); pred.append(yp); count += 1
            if max_points is not None and count >= max_points: break
        if max_points is not None and count >= max_points: break
    yt=np.asarray(truth); yp=np.asarray(pred)
    return EvaluationResult("sim1",runtime.trajectory.case_name,runtime.rc_order,pd.DataFrame(rows),_metric_dict(runtime.zone_ids,yt,yp),
                            {"state_feedback":"reset_true_each_step","qac_source":"recorded_test","phvac_source":"Phase-C from corresponding QAC"})


def sim2(runtime: PaperModelRuntime, phase_c: Mapping[str, PhaseCModelBundle], *, horizon: int | None = 12,
         all_test_segments: bool = False) -> EvaluationResult:
    segs = _test_segments(runtime)
    if not segs:
        raise ValueError("no contiguous test segment")
    selected = segs if all_test_segments else [max(segs, key=len)]
    rows=[]; truth=[]; pred=[]
    for episode_id, seg in enumerate(selected):
        start = int(seg[0])
        if runtime.rc_order == 2 and not runtime.is_inverse:
            start += int(runtime.model.config.L_e) - 1
        usable = [int(k) for k in seg[:-1] if int(k) >= start]
        if horizon is not None:
            usable = usable[: int(horizon)]
        if not usable:
            continue
        state = runtime.initialize(usable[0])
        for k in usable:
            state=runtime.step(state,k); yp=runtime.observe(state); yt=runtime.arrays.y[k+1]; qacs=runtime.qac_at(k)
            rec={"timestamp":runtime.trajectory.timestamp.iloc[k+1],"step_index":k,"simulation":"sim2","test_episode_id":episode_id}
            for i,z in enumerate(runtime.zone_ids):
                raw_p, physical_p = _phase_c_power_values(phase_c[z], qacs[z])
                rec[f"Tz_true__{z}"]=float(yt[i]); rec[f"Tz_pred__{z}"]=float(yp[i])
                rec[f"QHVAC_phaseC_W__{z}"]=qacs[z]; rec[f"QAC_W__{z}"]=qacs[z]
                rec[f"abs_QHVAC_phaseC_W__{z}"]=abs(qacs[z])
                rec[f"PHVAC_model_raw_W__{z}"]=raw_p; rec[f"PHVAC_W__{z}"]=physical_p
            rows.append(rec); truth.append(yt); pred.append(yp)
    if not rows:
        raise ValueError("test segment too short")
    yt=np.asarray(truth); yp=np.asarray(pred)
    return EvaluationResult("sim2",runtime.trajectory.case_name,runtime.rc_order,pd.DataFrame(rows),_metric_dict(runtime.zone_ids,yt,yp),
                            {"state_feedback":"recursive_predicted","qac_source":"recorded_test_phaseC_corrected_QHVAC",
                             "disturbance_source":"recorded_test","test_episode_count":len(selected),
                             "all_test_segments":bool(all_test_segments),
                             "phvac_chain":"abs(QHVAC_phaseC)->Phase-C PHVAC; physical PHVAC=max(raw,0)"})


def sim3(
    runtime: PaperModelRuntime,
    phase_c: Mapping[str, PhaseCModelBundle],
    calibrations: Mapping[str, ThermostatCalibration],
    *,
    horizon: int | None = 24,
    all_test_segments: bool = False,
    cooling_mdot_choice: str = "nominal",
    heating_mdot_choice: str = "nominal",
    unobserved_mode_policy: str = "fallback",
    actuation_profiles: Mapping[str, ThermostatActuationProfile] | None = None,
) -> EvaluationResult:
    """Closed-loop local thermostat simulation.

    Historical test Tz is saved only as a reference trace. Because Sim3 replaces
    historical QAC with controller-generated QAC, historical Tz is not used as a
    model-accuracy target. Primary metrics are setpoint/comfort, HVAC energy/control,
    and explicit observed-vs-extrapolative actuation diagnostics.
    """
    segs = _test_segments(runtime)
    if not segs:
        raise ValueError("no contiguous test segment")
    selected_segments = segs if all_test_segments else [max(segs, key=len)]
    resolved_profiles = {
        z: ((actuation_profiles or {}).get(z) or resolve_actuation_profile(calibrations[z]))
        for z in runtime.zone_ids
    }
    rows = []
    for episode_id, seg in enumerate(selected_segments):
        start = int(seg[0])
        if runtime.rc_order == 2 and not runtime.is_inverse:
            start = int(seg[0]) + int(runtime.model.config.L_e) - 1
        usable = [int(k) for k in seg[:-1] if int(k) >= start]
        if horizon is not None:
            usable = usable[: int(horizon)]
        if not usable:
            continue

        state = runtime.initialize(usable[0])
        controllers = {
            z: LegacyHeatingCoolingThermostat(
                calibrations[z], phase_c[z],
                cooling_mdot_choice=cooling_mdot_choice,
                heating_mdot_choice=heating_mdot_choice,
                unobserved_mode_policy=unobserved_mode_policy,
                actuation_profile=resolved_profiles[z],
            ) for z in runtime.zone_ids
        }
        schedules = {z: medium_low_high_medium_schedule(calibrations[z], len(usable)) for z in runtime.zone_ids}

        for j, k in enumerate(usable):
            current = runtime.observe(state)
            actions = {
                z: controllers[z].command(float(current[i]), float(schedules[z][j]))
                for i, z in enumerate(runtime.zone_ids)
            }
            qac = {z: a.Q_AC_W for z, a in actions.items()}
            state = runtime.step(state, k, qac_override=qac)
            yp = runtime.observe(state)
            yref = runtime.arrays.y[k + 1]
            rec = {
                "timestamp": runtime.trajectory.timestamp.iloc[k + 1],
                "step_index": k,
                "simulation": "sim3",
                "test_episode_id": episode_id,
            }
            for i, z in enumerate(runtime.zone_ids):
                a = actions[z]
                rec.update(
                    {
                        f"Tz_historical_test_reference__{z}": float(yref[i]),
                        f"Tz_sim__{z}": float(yp[i]),
                        f"setpoint_C__{z}": a.setpoint_C,
                        f"requested_hvac_on__{z}": a.requested_hvac_on,
                        f"hvac_on__{z}": a.hvac_on,
                        f"heating_mode__{z}": a.heating_mode,
                        f"delivered_mode__{z}": a.delivered_mode,
                        f"mode_available__{z}": bool(a.mode_available),
                        f"mode_observed_in_train__{z}": bool(a.mode_observed_in_train),
                        f"actuation_parameter_source__{z}": a.actuation_parameter_source,
                        f"qac_extrapolation__{z}": bool(a.qac_extrapolation),
                        f"action_suppressed__{z}": bool(a.action_suppressed),
                        f"m_dot_kg_s__{z}": a.m_dot_kg_s,
                        f"T_supply_C__{z}": a.T_supply_C,
                        f"QHVAC_physics_W__{z}": a.Q_HVAC_X_W,
                        f"QHVAC_X_W__{z}": a.Q_HVAC_X_W,
                        f"QHVAC_phaseC_W__{z}": a.Q_AC_W,
                        f"QAC_W__{z}": a.Q_AC_W,
                        f"abs_QHVAC_phaseC_W__{z}": abs(a.Q_AC_W),
                        f"PHVAC_model_raw_W__{z}": a.P_HVAC_model_raw_W,
                        f"PHVAC_W__{z}": a.P_HVAC_W,
                    }
                )
            rows.append(rec)
    if not rows:
        raise ValueError("test segment too short")
    frame = pd.DataFrame(rows)
    metrics: dict[str, dict[str, float]] = {}
    dt_hours = 300.0 / 3600.0

    def episode_switch_count(column: str) -> float:
        # Never count a discontinuous month-to-month TEST boundary as a 300-s
        # controller transition.  Each Sim3 TEST episode resets independently.
        total = 0
        for _, group in frame.groupby("test_episode_id", sort=True):
            values = group[column].to_numpy()
            if len(values) > 1:
                total += int(np.sum(np.diff(values) != 0))
        return float(total)

    for z in runtime.zone_ids:
        tz = frame[f"Tz_sim__{z}"].to_numpy(float)
        sp = frame[f"setpoint_C__{z}"].to_numpy(float)
        err = tz - sp
        db = float(calibrations[z].deadband_used_C)
        outside = np.maximum(np.abs(err) - db, 0.0)
        ph = np.maximum(frame[f"PHVAC_W__{z}"].to_numpy(float), 0.0)
        md = frame[f"m_dot_kg_s__{z}"].to_numpy(float)
        delivered_hvac = frame[f"hvac_on__{z}"].to_numpy(int)
        requested_hvac = frame[f"requested_hvac_on__{z}"].to_numpy(int)
        requested_heating_mode = frame[f"heating_mode__{z}"].to_numpy(int)
        suppressed = frame[f"action_suppressed__{z}"].astype(bool).to_numpy()
        extrap = frame[f"qac_extrapolation__{z}"].astype(bool).to_numpy()
        observed = frame[f"mode_observed_in_train__{z}"].astype(bool).to_numpy()
        delivered_mode = frame[f"delivered_mode__{z}"].astype(str).to_numpy()

        metrics[z] = {
            "setpoint_rmse_C": float(np.sqrt(np.mean(err**2))),
            "setpoint_mae_C": float(np.mean(np.abs(err))),
            "within_deadband_fraction": float(np.mean(np.abs(err) <= db)),
            "degree_hours_outside_deadband": float(np.sum(outside) * dt_hours),
            "hvac_energy_kWh": float(np.sum(ph) * dt_hours / 1000.0),
            "peak_phvac_W": float(np.max(ph)),
            "mean_mdot_kg_s": float(np.mean(md)),
            "max_mdot_kg_s": float(np.max(md)),
            "hvac_on_fraction": float(np.mean(delivered_hvac)),
            "requested_hvac_on_fraction": float(np.mean(requested_hvac)),
            "heating_mode_fraction": float(np.mean(requested_heating_mode)),
            "delivered_heating_fraction": float(np.mean(delivered_mode == "heating")),
            "delivered_cooling_fraction": float(np.mean(delivered_mode == "cooling")),
            "suppressed_action_count": float(np.sum(suppressed)),
            "suppressed_action_fraction": float(np.mean(suppressed)),
            "unobserved_mode_action_count": float(np.sum((delivered_hvac == 1) & (~observed))),
            "unobserved_mode_action_fraction": float(np.mean((delivered_hvac == 1) & (~observed))),
            "qac_extrapolation_action_count": float(np.sum(extrap)),
            "qac_extrapolation_action_fraction": float(np.mean(extrap)),
            "hvac_switch_count": episode_switch_count(f"hvac_on__{z}"),
            "requested_hvac_switch_count": episode_switch_count(f"requested_hvac_on__{z}"),
            "heating_mode_switch_count": episode_switch_count(f"heating_mode__{z}"),
            "setpoint_change_count": episode_switch_count(f"setpoint_C__{z}"),
        }

    return EvaluationResult(
        "sim3",
        runtime.trajectory.case_name,
        runtime.rc_order,
        frame,
        metrics,
        {
            "state_feedback": "recursive_predicted",
            "qac_source": "thermostat->mdot/Ts->QHVAC_physics->Phase-C QAC model->QHVAC_phaseC",
            "phvac_chain": "abs(QHVAC_phaseC)->Phase-C PHVAC; physical PHVAC=max(raw,0)",
            "test_episode_count": len(selected_segments),
            "all_test_segments": bool(all_test_segments),
            "historical_test_qac_used": False,
            "historical_test_tz_role": "reference_trace_only_not_accuracy_target",
            "setpoint_schedule": "medium->low->high->medium",
            "cooling_mdot_choice": cooling_mdot_choice,
            "heating_mdot_choice": heating_mdot_choice,
            "unobserved_mode_policy": unobserved_mode_policy,
            "training_mode_support": {
                z: {
                    "cooling": calibrations[z].supports_mode("cooling"),
                    "heating": calibrations[z].supports_mode("heating"),
                }
                for z in runtime.zone_ids
            },
            "actuation_support": {
                z: {"cooling": True, "heating": True}
                for z in runtime.zone_ids
            },
            "actuation_profiles": {
                z: resolved_profiles[z].to_dict()
                for z in runtime.zone_ids
            },
        },
    )

