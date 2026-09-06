from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinode_epsr.core.config import PaperConfig
from pinode_epsr.data.phase_d import PhaseDTrajectory
from pinode_epsr.evaluation.runtime import sim2, sim3
from pinode_epsr.evaluation.thermostat import ThermostatCalibration, _transitions, resolve_actuation_profile
from pinode_epsr.production.contracts import ControllerOverrideConfig, HPOConfig
from pinode_epsr.production.matrix import production_matrix
from pinode_epsr.production.hpo import _GeometryRestrictedTrial, hpo_protocol_id
from pinode_epsr.production.paths import resolve_production_layout
from pinode_epsr.production.sampling import select_month_balanced_hpo_sample


def _synthetic_monthly_trajectory(rows_per_month: int = 600):
    ts=[]; part=[]; included=[]
    for month in range(1, 13):
        cursor=pd.Timestamp(f"2026-{month:02d}-01 00:05:00")
        for i in range(rows_per_month):
            ts.append(cursor + pd.Timedelta(minutes=5*i)); part.append("train"); included.append(True)
        ts.append(ts[-1] + pd.Timedelta(minutes=5)); part.append("excluded"); included.append(False)
    n=len(ts)
    return PhaseDTrajectory(
        case_name="all_to_one", zone_ids=("RestaurantFastFood_All",), dependency_mode="independent",
        timestamp=pd.Series(ts), included=np.asarray(included,bool), partition=np.asarray(part,str),
        state=np.zeros((n,1)), control=np.zeros((n,1)), disturbance=np.zeros((n,1)), target=np.zeros((n,1)),
        state_columns=(), control_columns=(), disturbance_columns=(), target_columns=(), manifests=tuple(),
        frame=pd.DataFrame({"timestamp":ts,"included":included,"partition":part}),
    )


def test_production_matrix_has_32_scientific_configurations():
    matrix=production_matrix()
    assert len(matrix)==32
    assert len({m.configuration_id for m in matrix})==32
    assert sum(m.priority=="A" for m in matrix)==8
    assert sum(m.priority=="B" for m in matrix)==8
    assert sum(m.priority=="C" for m in matrix)==16


def test_hpo_percentage_is_configurable_and_strictly_bounded():
    assert HPOConfig(train_percentage=.5).train_percentage==.5
    assert HPOConfig(train_percentage=100).train_percentage==100
    with pytest.raises(ValueError): HPOConfig(train_percentage=0)
    with pytest.raises(ValueError): HPOConfig(train_percentage=101)


def test_month_balanced_hpo_sampling_uses_every_month_and_train_only():
    # Use a realistic monthly TRAIN scale so strict 20% holdout supports N_r=12.
    traj=_synthetic_monthly_trajectory(rows_per_month=6000)
    sample=select_month_balanced_hpo_sample(traj,train_percentage=5.0,holdout_percentage=20.0)
    assert len(sample.monthly_counts)==12
    assert all(v["fit"]>0 and v["holdout"]>0 for v in sample.monthly_counts.values())
    assert np.intersect1d(sample.fit_indices,sample.holdout_indices).size==0
    train=traj.mask("train",included_only=True)
    assert np.all(train[sample.fit_indices]) and np.all(train[sample.holdout_indices])
    assert abs(sample.actual_train_percentage-5.0)<0.25
    assert sample.rollout_windows("fit",N_r=12,L_e=12,rc_order=2)
    assert sample.rollout_windows("holdout",N_r=12,L_e=12,rc_order=2)


def test_tiny_percentage_never_silently_oversamples_when_below_legal_minimum():
    traj=_synthetic_monthly_trajectory(rows_per_month=600)
    with pytest.raises(ValueError, match="never rounds up"):
        select_month_balanced_hpo_sample(traj,train_percentage=.1,holdout_percentage=20.0)


def test_small_percentage_micro_geometry_preserves_floored_target_and_holdout_budgets():
    traj=_synthetic_monthly_trajectory(rows_per_month=6000)
    sample=select_month_balanced_hpo_sample(
        traj,train_percentage=.5,holdout_percentage=20.0,
        conservative_N_r=3,conservative_L_e=12,
    )
    assert sample.actual_train_percentage<=.5
    assert len(sample.monthly_counts)==12
    for values in sample.monthly_counts.values():
        assert values["requested_targets"]==math.floor(values["train_available"]*.005)
        assert values["requested_holdout_targets"]==math.floor(values["requested_targets"]*.20)
        assert values["fit"]+values["holdout"]==values["requested_targets"]
        assert values["holdout"]==values["requested_holdout_targets"]
    assert sample.actual_holdout_percentage<=sample.requested_holdout_percentage
    assert sample.rollout_windows("fit",N_r=3,L_e=12,rc_order=2)
    assert sample.rollout_windows("holdout",N_r=3,L_e=12,rc_order=2)
    # Full production N_r=12 geometry is intentionally infeasible at 0.5%
    # with a strict 20% floored holdout; it must fail rather than inflate.
    with pytest.raises(ValueError, match="will not.*inflate the holdout"):
        select_month_balanced_hpo_sample(
            traj,train_percentage=.5,holdout_percentage=20.0,
            conservative_N_r=12,conservative_L_e=12,
        )


def test_thermostat_transitions_require_exact_300_second_adjacency_when_timestamps_exist():
    mode=np.asarray(["off","cooling","off","heating"],dtype=object)
    tz=np.asarray([22.,23.,21.,20.])
    ts=pd.to_datetime(["2026-01-01 00:05","2026-01-01 00:10","2026-02-01 00:05","2026-02-01 00:10"])
    trans,diag=_transitions(mode,tz,ts,dt_seconds=300.0)
    assert trans["cooling_on"]==[23.0]
    assert trans["cooling_off"]==[]  # month gap is not a 300-s transition
    assert trans["heating_on"]==[20.0]
    assert diag["skipped_noncontiguous_pairs"]==1
    assert diag["transition_continuity_rule"]=="exact_dt"


def test_production_layout_writes_to_epsr_sibling_not_campaign(tmp_path):
    generated=tmp_path/"Data"/"ScaleBridge"; campaign=generated/"campaigns"/"c"; campaign.mkdir(parents=True)
    cfg=PaperConfig(generated_data_root=generated,campaign_id="c")
    paper=generated/"Paper_PINODE_EPSR"
    layout=resolve_production_layout(cfg,paper_data_root=paper,create=True)
    assert layout.campaign_root==campaign.resolve()
    assert layout.paper_data_root==paper.resolve()
    assert layout.hpo_root.name=="01_hpo" and layout.checkpoint_root.name=="03_checkpoints"
    assert not str(layout.hpo_root).startswith(str(campaign))


def test_sim2_and_sim3_expose_all_test_episode_mode():
    assert "all_test_segments" in inspect.signature(sim2).parameters
    assert "all_test_segments" in inspect.signature(sim3).parameters


def test_production_source_has_no_git_dependency():
    root=Path(__file__).parents[1]/"src"/"pinode_epsr"/"production"
    text="\n".join(p.read_text(encoding="utf-8") for p in root.glob("*.py"))
    assert "git rev-parse" not in text
    assert "subprocess.*git" not in text


def test_hpo_protocol_identity_changes_with_scientific_policy_not_trial_budget():
    spec=production_matrix()[0]
    a=HPOConfig(train_percentage=2.0,n_trials=4)
    b=HPOConfig(train_percentage=2.0,n_trials=20)
    c=HPOConfig(train_percentage=5.0,n_trials=20)
    d=HPOConfig(train_percentage=2.0,n_trials=20,objective="recursive_temperature_mae_C")
    assert hpo_protocol_id(spec,a)==hpo_protocol_id(spec,b)
    assert hpo_protocol_id(spec,a)!=hpo_protocol_id(spec,c)
    assert hpo_protocol_id(spec,a)!=hpo_protocol_id(spec,d)


def test_monthly_percentage_and_holdout_are_floored_not_rounded_up():
    traj=_synthetic_monthly_trajectory(rows_per_month=619)
    sample=select_month_balanced_hpo_sample(
        traj,train_percentage=5.0,holdout_percentage=20.0,
        conservative_N_r=3,conservative_L_e=12,
    )
    assert all(v["requested_targets"]==30 for v in sample.monthly_counts.values())
    assert all(v["requested_holdout_targets"]==6 for v in sample.monthly_counts.values())
    assert sample.actual_train_percentage < 5.0
    assert sample.actual_holdout_percentage <= 20.0


def _synthetic_kitchen_calibration() -> ThermostatCalibration:
    q={f"p{p:02d}":1.0 for p in (1,5,10,25,50,75,90,95,99)}
    return ThermostatCalibration(
        zone_id="Kitchen", row_count=100,
        mode_counts={"cooling":90,"heating":0,"off":10,"ambiguous":0},
        tz_quantiles_C=q,
        ts_cooling_C=12.0, ts_heating_C=float("nan"), ts_off_C=23.0,
        mdot_off_kg_s=0.0,
        mdot_cooling_nominal_kg_s=1.0, mdot_cooling_max_kg_s=1.5,
        mdot_heating_nominal_kg_s=float("nan"), mdot_heating_max_kg_s=float("nan"),
        mdot_cooling_quantiles_kg_s=q, mdot_heating_quantiles_kg_s=q,
        deadband_data_C=.7, heating_mode_deadband_data_C=2.9,
        deadband_used_C=1.0, heating_mode_deadband_used_C=2.9,
        setpoint_low_C=19.0, setpoint_medium_C=23.0, setpoint_high_C=26.0,
        transition_temperatures_C={},
        qac_activity_threshold_W=100.0, mdot_activity_threshold_kg_s=.01,
        provenance={"cooling_supply_deltaT_C":9.0,"heating_supply_deltaT_C":float("nan")},
    )


def test_controller_override_contract_uses_half_width_and_user_override_precedence():
    cfg=ControllerOverrideConfig.from_mapping({
        "deadband_half_width_C":1.0,
        "zones":{
            "Kitchen":{
                "heating":{
                    "T_supply_C":34.0,
                    "mdot_nominal_kg_s":0.8,
                    "mdot_max_kg_s":1.1,
                }
            }
        },
    })
    assert cfg.deadband_overrides_C()["Kitchen"]==1.0
    assert cfg.to_dict()["deadband_semantics"]=="half_width_about_setpoint"
    profile=resolve_actuation_profile(
        _synthetic_kitchen_calibration(),
        overrides=cfg.actuation_overrides("Kitchen"),
    )
    assert profile.heating.parameter_source=="user_override"
    assert profile.heating.T_supply_C==34.0
    assert profile.heating.mdot_nominal_kg_s==0.8
    assert profile.heating.mdot_max_kg_s==1.1
    assert profile.heating.qac_extrapolation_expected is True
    assert profile.provenance["mode_resolution"]["heating"]["base_source"]=="fallback_same_zone_cooling_deltaT_and_mdot"
    assert profile.provenance["mode_resolution"]["heating"]["source_class"]=="user_override"
    assert set(profile.provenance["mode_resolution"]["heating"]["overridden_parameters"])=={
        "T_supply_C","mdot_nominal_kg_s","mdot_max_kg_s"
    }


def test_hpo_protocol_identity_includes_rollout_geometry():
    spec=production_matrix()[0]
    full=HPOConfig(train_percentage=2.0,max_rollout_steps=12,max_encoder_history_steps=12)
    micro=HPOConfig(train_percentage=2.0,max_rollout_steps=3,max_encoder_history_steps=12)
    assert hpo_protocol_id(spec,full)!=hpo_protocol_id(spec,micro)


def test_micro_hpo_trial_proxy_restricts_rollout_but_preserves_encoder_history():
    class Trial:
        def __init__(self): self.seen={}
        def suggest_categorical(self,name,choices):
            self.seen[name]=list(choices)
            return list(choices)[-1]
    raw=Trial()
    proxy=_GeometryRestrictedTrial(raw,max_rollout_steps=3,max_encoder_history_steps=12)
    assert proxy.suggest_categorical("N_r",[1,3,6,12])==3
    assert raw.seen["N_r"]==[1,3]
    assert proxy.suggest_categorical("L_e",[3,6,12])==12
    assert raw.seen["L_e"]==[3,6,12]
