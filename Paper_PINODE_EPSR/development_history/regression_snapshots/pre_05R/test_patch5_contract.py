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
    classify_hvac_modes, medium_low_high_medium_schedule,
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
