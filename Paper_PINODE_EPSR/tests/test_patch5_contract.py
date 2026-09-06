from __future__ import annotations

from types import SimpleNamespace
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Paper_PINODE_EPSR.metrics import cvrmse, full_prediction_metrics, nmbe, r2
from Paper_PINODE_EPSR.paper_paths import resolve_paper_data_root
from Paper_PINODE_EPSR.phase_c import reference_phase_c_bundle
from Paper_PINODE_EPSR.thermostat import (
    COOL_COL, HEAT_COL, MDOT_COL, TS_COL, TZ_COL,
    LegacyHeatingCoolingThermostat, ThermostatCalibration, calibrate_thermostat,
    classify_hvac_modes, medium_low_high_medium_schedule, resolve_actuation_profile,
)
from Paper_PINODE_EPSR.experiment import suggest_method_hyperparameters


def _training_frame() -> pd.DataFrame:
    # Deliberately includes simultaneous raw heating+cooling in effective cooling/heating rows.
    modes = ["off"]*5 + ["cooling"]*8 + ["off"]*4 + ["heating"]*8 + ["off"]*5
    tz=[]; ts=[]; md=[]; heat=[]; cool=[]
    for i,m in enumerate(modes):
        if m=="cooling":
            z=22.0 + .1*np.sin(i); tz.append(z); ts.append(12.0); md.append(1.0 + .02*i); heat.append(1000.0); cool.append(6000.0)
        elif m=="heating":
            z=18.0 + .1*np.sin(i); tz.append(z); ts.append(32.0); md.append(.8 + .01*i); heat.append(5000.0); cool.append(500.0)
        else:
            z=20.0 + .05*np.sin(i); tz.append(z); ts.append(z); md.append(.05); heat.append(0.0); cool.append(0.0)
    return pd.DataFrame({TZ_COL:tz,TS_COL:ts,MDOT_COL:md,HEAT_COL:heat,COOL_COL:cool})


def _manual_calibration(**overrides) -> ThermostatCalibration:
    base=dict(
        zone_id="Z",row_count=100,mode_counts={"cooling":30,"heating":30,"off":40,"ambiguous":0},
        tz_quantiles_C={"p10":18.0,"p50":20.0,"p90":22.0},ts_cooling_C=12.0,ts_heating_C=32.0,ts_off_C=20.0,
        mdot_off_kg_s=0.05,mdot_cooling_nominal_kg_s=1.0,mdot_cooling_max_kg_s=2.0,
        mdot_heating_nominal_kg_s=.8,mdot_heating_max_kg_s=1.6,
        mdot_cooling_quantiles_kg_s={},mdot_heating_quantiles_kg_s={},deadband_data_C=1.0,
        heating_mode_deadband_data_C=.5,deadband_used_C=1.0,heating_mode_deadband_used_C=.5,
        setpoint_low_C=18.0,setpoint_medium_C=20.0,setpoint_high_C=22.0,transition_temperatures_C={},
        qac_activity_threshold_W=10.0,mdot_activity_threshold_kg_s=.01,provenance={}
    ); base.update(overrides); return ThermostatCalibration(**base)


def test_mode_classifier_uses_signed_net_qac_for_simultaneous_components():
    f=_training_frame(); mode,_=classify_hvac_modes(f)
    assert "cooling" in set(mode) and "heating" in set(mode)
    # Cooling rows contain positive heating too; classification must still be net cooling.
    i=6; assert f.iloc[i][HEAT_COL] > 0 and f.iloc[i][COOL_COL] > 0 and mode[i]=="cooling"


def test_calibration_has_nominal_and_max_flow_choices():
    c=calibrate_thermostat(_training_frame(),zone_id="Z")
    assert c.mdot_cooling_max_kg_s >= c.mdot_cooling_nominal_kg_s > 0
    assert c.mdot_heating_max_kg_s >= c.mdot_heating_nominal_kg_s > 0
    assert c.active_mdot("cooling","nominal") == c.mdot_cooling_nominal_kg_s
    assert c.active_mdot("cooling","max") == c.mdot_cooling_max_kg_s


def test_deadband_data_is_preserved_when_override_is_used():
    c0=calibrate_thermostat(_training_frame(),zone_id="Z")
    c1=calibrate_thermostat(_training_frame(),zone_id="Z",deadband_override_C=.75,heating_mode_deadband_override_C=.25)
    assert c1.deadband_used_C == .75 and c1.heating_mode_deadband_used_C == .25
    assert c1.deadband_data_C == c0.deadband_data_C
    assert c1.heating_mode_deadband_data_C == c0.heating_mode_deadband_data_C
    assert c1.provenance["deadband_source"] == "override"


def test_setpoints_are_three_ordered_data_quantiles():
    c=calibrate_thermostat(_training_frame(),zone_id="Z")
    assert c.setpoint_low_C < c.setpoint_medium_C < c.setpoint_high_C


def test_supply_temperatures_have_correct_mode_order():
    c=calibrate_thermostat(_training_frame(),zone_id="Z")
    assert c.ts_cooling_C < c.setpoint_medium_C < c.ts_heating_C

    # Regression for the real all-to-one case: observed heating Ts may be below
    # a paper experiment's high setpoint and must still pass through unchanged.
    observed = _manual_calibration(
        mode_counts={"cooling": 30, "heating": 20, "off": 50, "ambiguous": 0},
        ts_heating_C=20.678,
        mdot_heating_nominal_kg_s=0.9066,
        mdot_heating_max_kg_s=2.7232,
        setpoint_low_C=19.994,
        setpoint_medium_C=21.953,
        setpoint_high_C=24.997,
    )
    profile = resolve_actuation_profile(observed)
    assert profile.heating.observed_in_train is True
    assert profile.heating.parameter_source == "observed_train"
    assert np.isclose(profile.heating.T_supply_C, 20.678)
    assert np.isclose(profile.heating.mdot_nominal_kg_s, 0.9066)
    assert np.isclose(profile.heating.mdot_max_kg_s, 2.7232)


def test_legacy_controller_cooling_logic_and_nominal_flow():
    ctrl=LegacyHeatingCoolingThermostat(_manual_calibration(),reference_phase_c_bundle("Dining"))
    a=ctrl.command(22.0,20.0)
    assert a.heating_mode==0 and a.hvac_on==1 and a.m_dot_kg_s==1.0 and a.T_supply_C==12.0
    a=ctrl.command(19.0,20.0)
    assert a.hvac_on==0


def test_legacy_controller_heating_mode_logic():
    ctrl=LegacyHeatingCoolingThermostat(_manual_calibration(),reference_phase_c_bundle("Dining"))
    a=ctrl.command(17.0,20.0)
    assert a.heating_mode==1 and a.hvac_on==1 and a.T_supply_C==32.0
    a=ctrl.command(22.0,20.0)
    assert a.heating_mode==0


def test_sim3_can_choose_max_flow_independently_by_mode():
    ctrl=LegacyHeatingCoolingThermostat(_manual_calibration(),reference_phase_c_bundle("Dining"),cooling_mdot_choice="max",heating_mdot_choice="nominal")
    a=ctrl.command(22.0,20.0); assert a.m_dot_kg_s==2.0
    ctrl.state.hvac_on=0; ctrl.state.heating_mode=0
    a=ctrl.command(17.0,20.0); assert a.m_dot_kg_s==.8


def test_off_flow_is_data_calibrated_not_forced_to_zero():
    ctrl=LegacyHeatingCoolingThermostat(_manual_calibration(),reference_phase_c_bundle("Dining"))
    a=ctrl.command(20.0,20.0)
    assert a.hvac_on==0 and a.m_dot_kg_s==.05


def test_phase_c_chain_returns_finite_qac_and_phvac():
    ctrl=LegacyHeatingCoolingThermostat(_manual_calibration(),reference_phase_c_bundle("Dining"))
    a=ctrl.command(22.0,20.0)
    assert np.isfinite([a.Q_HVAC_X_W,a.Q_AC_W,a.P_HVAC_W]).all()


def test_four_segment_schedule_uses_three_unique_conditions():
    c=_manual_calibration(); s=medium_low_high_medium_schedule(c,12)
    assert s[0]==20 and 18 in s and 22 in s and s[-1]==20
    assert set(s)=={18.0,20.0,22.0}


def test_schedule_requires_enough_steps():
    with pytest.raises(ValueError): medium_low_high_medium_schedule(_manual_calibration(),3)


def test_generated_root_environment_is_machine_portable(monkeypatch,tmp_path):
    generated=tmp_path/"Data"/"ScaleBridge"; generated.mkdir(parents=True)
    monkeypatch.delenv("SCALEBRIDGE_PINODE_EPSR_DATA_ROOT",raising=False)
    monkeypatch.setenv("SCALEBRIDGE_GENERATED_DATA_ROOT",str(generated))
    assert resolve_paper_data_root()==(generated/"Paper_PINODE_EPSR").resolve()


def test_explicit_paper_root_override_has_priority(monkeypatch,tmp_path):
    monkeypatch.setenv("SCALEBRIDGE_GENERATED_DATA_ROOT",str(tmp_path/"wrong"))
    target=tmp_path/"paper"
    assert resolve_paper_data_root(target)==target.resolve()


def test_paper_root_env_override_has_priority(monkeypatch,tmp_path):
    target=tmp_path/"paper_env"; monkeypatch.setenv("SCALEBRIDGE_PINODE_EPSR_DATA_ROOT",str(target))
    assert resolve_paper_data_root()==target.resolve()


def test_metrics_include_cvrmse_nmbe_r2():
    y=np.array([20.,21.,22.]); p=np.array([20.1,20.9,22.2]); m=full_prediction_metrics(y,p)
    assert set(m)=={"rmse","mae","bias","cvrmse_percent","nmbe_percent","r2"}
    assert all(np.isfinite(v) for v in m.values())


class DummyTrial:
    def suggest_int(self,name,a,b): return a
    def suggest_categorical(self,name,choices): return choices[0]
    def suggest_float(self,name,a,b,log=False): return a


@pytest.mark.parametrize("method",["inverse_pinn","neural_ode","base_pinode","ebp_pinode"])
def test_common_hpo_dispatch_covers_all_four_methods(method):
    values=suggest_method_hyperparameters(method,DummyTrial(),rc_order=2)
    assert "hidden_layers" in values and "hidden_width" in values and "learning_rate" in values


def test_calibration_rejects_missing_columns():
    with pytest.raises(KeyError): calibrate_thermostat(pd.DataFrame({TZ_COL:[20.]}),zone_id="Z")


def test_negative_deadband_override_rejected():
    with pytest.raises(ValueError): calibrate_thermostat(_training_frame(),zone_id="Z",deadband_override_C=-1)


def test_phase_b_calibration_filter_uses_only_phase_d_training_timestamps():
    from Paper_PINODE_EPSR.data import PhaseDTrajectory
    from Paper_PINODE_EPSR.thermostat_data import filter_phase_b_to_phase_d_training
    ts=pd.date_range("2026-01-01 00:05:00",periods=4,freq="5min")
    traj=PhaseDTrajectory("all_to_one",("RestaurantFastFood_All",),"independent",pd.Series(ts),np.ones(4,bool),
        np.array(["train","train","validation","test"]),np.zeros((4,1)),np.zeros((4,1)),np.zeros((4,1)),np.zeros((4,1)),(),(),(),(),tuple(),pd.DataFrame())
    raw=pd.DataFrame({"timestamp_raw":["01/01  00:05:00","01/01  00:10:00","01/01  00:15:00","01/01  00:20:00"],TZ_COL:[20]*4,TS_COL:[20]*4,MDOT_COL:[0]*4,HEAT_COL:[0]*4,COOL_COL:[0]*4})
    out=filter_phase_b_to_phase_d_training(raw,traj)
    assert out["timestamp_raw"].tolist()==["01/01  00:05:00","01/01  00:10:00"]


def test_inverse_2c_runtime_initialization_contract_is_ta_equals_tm():
    from Paper_PINODE_EPSR.data import PhaseDTrajectory
    from Paper_PINODE_EPSR.experiment import build_paper_model
    from Paper_PINODE_EPSR.evaluation import PaperModelRuntime
    n=12; ts=pd.date_range("2026-01-01",periods=n,freq="5min"); y=20+np.arange(n)*.01
    f=pd.DataFrame({"outdoor_temperature__lag_0":np.full(n,5.),"RestaurantFastFood_All__zone_temperature__lag_0":y,
                    "RestaurantFastFood_All__qac__lag_0":np.zeros(n),"RestaurantFastFood_All__zic__lag_0":np.ones(n),
                    "RestaurantFastFood_All__zir__lag_0":np.ones(n),"RestaurantFastFood_All__qsol1__lag_0":np.zeros(n),
                    "RestaurantFastFood_All__qsol2__lag_0":np.zeros(n)})
    traj=PhaseDTrajectory("all_to_one",("RestaurantFastFood_All",),"independent",pd.Series(ts),np.ones(n,bool),np.array(["train"]*n),
        y[:,None],np.zeros((n,1)),np.zeros((n,1)),y[:,None],(),(),(),(),tuple(),f)
    model,_=build_paper_model("inverse_pinn",traj,rc_order=2,train_indices=np.arange(n))
    x0=PaperModelRuntime(model,traj).initialize(3).detach().numpy()
    assert np.isclose(x0[0],x0[1]) and np.isclose(x0[0],y[3])


def test_setpoint_policy_repairs_collapsed_empirical_median():
    # Mimic Dining: a dominant low plateau makes P10 and P50 nearly identical.
    low = np.full(60, 21.1)
    high = np.full(40, 23.9)
    n = len(low) + len(high)
    f = pd.DataFrame({
        TZ_COL: np.concatenate([low, high]),
        TS_COL: np.concatenate([np.full(60, 24.0), np.full(40, 19.0)]),
        MDOT_COL: np.ones(n),
        HEAT_COL: np.concatenate([np.full(60, 5000.0), np.zeros(40)]),
        COOL_COL: np.concatenate([np.zeros(60), np.full(40, 5000.0)]),
    })
    c = calibrate_thermostat(f, zone_id="DiningLike", setpoint_min_separation_C=0.5)
    assert c.setpoint_low_C < c.setpoint_medium_C < c.setpoint_high_C
    assert np.isclose(c.setpoint_medium_C, 0.5 * (c.setpoint_low_C + c.setpoint_high_C))
    assert c.provenance["setpoint_medium_source"] == "midpoint_low_high_due_quantile_collapse"


def test_calibration_exposes_missing_heating_without_inventing_values():
    f = _training_frame().copy()
    f[HEAT_COL] = 0.0
    c = calibrate_thermostat(f, zone_id="CoolingOnly")
    assert c.supports_mode("cooling")
    assert not c.supports_mode("heating")
    assert np.isnan(c.ts_heating_C)
    assert np.isnan(c.mdot_heating_nominal_kg_s)
    with pytest.raises(RuntimeError):
        c.active_mdot("heating", "nominal")


def test_unobserved_heating_is_applied_with_same_zone_fallback_and_flagged_extrapolative():
    c = _manual_calibration(
        mode_counts={"cooling": 30, "heating": 0, "off": 70, "ambiguous": 0},
        ts_heating_C=float("nan"),
        mdot_heating_nominal_kg_s=float("nan"),
        mdot_heating_max_kg_s=float("nan"),
        provenance={"cooling_supply_deltaT_C": 8.0},
    )
    p = resolve_actuation_profile(c)
    assert p.heating.observed_in_train is False
    assert p.heating.parameter_source.startswith("fallback_same_zone_cooling_deltaT_and_mdot")
    expected_delta = 8.0
    expected_ts = max(
        c.setpoint_medium_C + expected_delta,
        c.setpoint_high_C + max(c.deadband_used_C, 0.5),
    )
    assert np.isclose(p.heating.reference_deltaT_C, expected_delta)
    assert np.isclose(p.heating.T_supply_C, expected_ts)
    assert p.heating.T_supply_C > c.setpoint_high_C
    assert p.heating.mdot_nominal_kg_s == c.mdot_cooling_nominal_kg_s
    assert p.heating.mdot_max_kg_s == c.mdot_cooling_max_kg_s
    assert p.provenance["fallback_uses_equipment_definition"] is False
    assert p.provenance["fallback_uses_other_runs"] is False

    ctrl = LegacyHeatingCoolingThermostat(
        c,
        reference_phase_c_bundle("Kitchen"),
        unobserved_mode_policy="fallback",
        actuation_profile=p,
    )
    a = ctrl.command(17.0, 20.0)
    assert a.heating_mode == 1
    assert a.requested_hvac_on == 1
    assert a.hvac_on == 1
    assert a.delivered_mode == "heating"
    assert a.mode_available is True
    assert a.mode_observed_in_train is False
    assert a.action_suppressed is False
    assert a.qac_extrapolation is True
    assert a.T_supply_C > c.setpoint_high_C
    assert np.isfinite([a.Q_HVAC_X_W, a.Q_AC_W, a.P_HVAC_W]).all()


def test_unobserved_mode_can_be_made_strict_for_diagnostics():
    c = _manual_calibration(
        mode_counts={"cooling": 30, "heating": 0, "off": 70, "ambiguous": 0},
        ts_heating_C=float("nan"),
        mdot_heating_nominal_kg_s=float("nan"),
        mdot_heating_max_kg_s=float("nan"),
        provenance={"cooling_supply_deltaT_C": 8.0},
    )
    ctrl = LegacyHeatingCoolingThermostat(
        c,
        reference_phase_c_bundle("Kitchen"),
        unobserved_mode_policy="error",
    )
    with pytest.raises(RuntimeError, match="not observed in TRAIN"):
        ctrl.command(17.0, 20.0)


def test_unobserved_mode_actuation_can_be_explicitly_overridden():
    c = _manual_calibration(
        mode_counts={"cooling": 30, "heating": 0, "off": 70, "ambiguous": 0},
        ts_heating_C=float("nan"),
        mdot_heating_nominal_kg_s=float("nan"),
        mdot_heating_max_kg_s=float("nan"),
        provenance={"cooling_supply_deltaT_C": 8.0},
    )
    p = resolve_actuation_profile(
        c,
        overrides={
            "heating": {
                "T_supply_C": 35.0,
                "mdot_nominal_kg_s": 0.7,
                "mdot_max_kg_s": 1.4,
            }
        },
    )
    assert p.heating.T_supply_C == 35.0
    assert p.heating.mdot_nominal_kg_s == 0.7
    assert p.heating.mdot_max_kg_s == 1.4
    assert "override" in p.heating.parameter_source
    assert p.heating.observed_in_train is False
    assert p.heating.qac_extrapolation_expected is True


def test_exact_phase_d_train_alignment_coalesces_interval_end_duplicates():
    from Paper_PINODE_EPSR.data import PhaseDTrajectory
    from Paper_PINODE_EPSR.thermostat_data import filter_phase_b_to_phase_d_training

    ts = pd.Series(pd.to_datetime([
        "2026-02-01 00:00:00",
        "2026-02-01 00:05:00",
        "2026-02-01 00:10:00",
    ]))
    traj = PhaseDTrajectory(
        "all_to_one", ("RestaurantFastFood_All",), "independent",
        ts, np.ones(3, bool), np.array(["train", "train", "validation"]),
        np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)),
        (), (), (), (), tuple(), pd.DataFrame(),
    )
    raw = pd.DataFrame({
        "timestamp_raw": [
            "01/31  24:00:00",  # same physical instant as next row
            "02/01  00:00:00",
            "02/01  00:05:00",
            "02/01  00:10:00",
        ],
        TZ_COL: [20.0, np.nan, 20.1, 20.2],
        TS_COL: [12.0, np.nan, 12.0, 12.0],
        MDOT_COL: [1.0, np.nan, 1.0, 1.0],
        HEAT_COL: [0.0, np.nan, 0.0, 0.0],
        COOL_COL: [5000.0, np.nan, 5000.0, 5000.0],
    })
    out = filter_phase_b_to_phase_d_training(raw, traj)
    assert len(out) == 2
    assert out["_phase_d_timestamp"].tolist() == [
        pd.Timestamp("2026-02-01 00:00:00"),
        pd.Timestamp("2026-02-01 00:05:00"),
    ]
    assert out.attrs["training_alignment"]["exact_train_count_match"] is True
    assert out.attrs["training_alignment"]["normalized_duplicate_group_count"] == 1


def test_exact_alignment_allows_dec31_24_to_roll_into_next_year():
    from Paper_PINODE_EPSR.data import PhaseDTrajectory
    from Paper_PINODE_EPSR.thermostat_data import filter_phase_b_to_phase_d_training

    ts = pd.Series(pd.to_datetime([
        "2026-12-31 23:55:00",
        "2027-01-01 00:00:00",
    ]))
    traj = PhaseDTrajectory(
        "all_to_one", ("RestaurantFastFood_All",), "independent",
        ts, np.ones(2, bool), np.array(["train", "train"]),
        np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1)),
        (), (), (), (), tuple(), pd.DataFrame(),
    )
    raw = pd.DataFrame({
        "timestamp_raw": ["12/31  23:55:00", "12/31  24:00:00"],
        TZ_COL: [20.0, 20.0],
        TS_COL: [12.0, 12.0],
        MDOT_COL: [1.0, 1.0],
        HEAT_COL: [0.0, 0.0],
        COOL_COL: [5000.0, 5000.0],
    })
    out = filter_phase_b_to_phase_d_training(raw, traj)
    assert out["_phase_d_timestamp"].tolist() == [
        pd.Timestamp("2026-12-31 23:55:00"),
        pd.Timestamp("2027-01-01 00:00:00"),
    ]
    assert out.attrs["training_alignment"]["exact_train_count_match"] is True
