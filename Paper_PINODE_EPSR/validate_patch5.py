from __future__ import annotations

"""Day-6 Patch-05 validator: unified tune/freeze + Sim1/2/3 + thermostat runtime.

The real-data stage is intentionally tiny. It proves data/control/runtime paths on
actual Phase-B/Phase-C/Phase-D artifacts; it is not a convergence or accuracy run.
All generated Patch-05 artifacts are written outside the repository under the
portable paper data root.
"""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np

from .common import write_json
from .config import PaperConfig, canonical_case_specs
from .data import load_case
from .evaluation import PaperModelRuntime, sim1, sim2, sim3
from .experiment import build_paper_model, longest_training_prefix, save_evaluation_result, suggest_method_hyperparameters, tiny_fit
from .inverse_pinn import InversePINNRC
from .neuromancer_backend import runtime_info
from .paper_paths import resolve_paper_data_root
from .phase_c import discover_and_load_phase_c_bundle
from .thermostat_data import calibrate_controlled_thermostats
from .thermostat import resolve_actuation_profile
from .training import TuningConfig, run_optuna_tuning

METHODS = ("inverse_pinn", "neural_ode", "base_pinode", "ebp_pinode")


def _load_actuation_overrides(path: str | None) -> dict[str, dict[str, dict[str, float]]]:
    if path is None:
        return {}
    p=Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Actuation override JSON not found: {p}")
    data=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data,dict):
        raise TypeError("Actuation override JSON must be an object keyed by zone")
    return data


def run_pytest_contracts() -> int:
    repo_root=Path(__file__).resolve().parent.parent
    command=[sys.executable,"-m","pytest","-q",
             "Paper_PINODE_EPSR/tests/test_patch1_contract.py",
             "Paper_PINODE_EPSR/tests/test_patch2_contract.py",
             "Paper_PINODE_EPSR/tests/test_patch3_contract.py",
             "Paper_PINODE_EPSR/tests/test_patch4_contract.py",
             "Paper_PINODE_EPSR/tests/test_patch5_contract.py",
             "Paper_PINODE_EPSR/tests/test_reorganization_contract.py"]
    print("[unit-contracts] "+" ".join(command),flush=True)
    return int(subprocess.run(command,cwd=repo_root,check=False).returncode)


def _tiny_hpo(root: Path, *, trials: int) -> dict[str,Any]:
    out={}
    for method in METHODS:
        def objective(trial, method=method):
            vals=suggest_method_hyperparameters(method,trial,rc_order=2)
            # deterministic finite plumbing objective; no test data enter this smoke.
            return float(vals.get("hidden_width",1))*1e-6 + float(vals.get("learning_rate",1e-3))
        _,frozen=run_optuna_tuning(objective,method=method,tuning_scope="representative_TRAIN_only_day06_plumbing_smoke",
                                   config=TuningConfig(n_trials=max(1,trials),representative_max_windows=8,seed=42))
        path=root/"frozen_hyperparameters"/f"{method}_2C_smoke.json"; frozen.save(path)
        out[method]={"path":str(path),"values":frozen.values,"scope":frozen.tuning_scope}
    return out


def _calibration_report(
    config: PaperConfig,
    root: Path,
    deadband: float | None,
    mode_deadband: float | None,
    setpoint_min_separation_C: float,
    actuation_overrides: dict[str, dict[str, dict[str, float]]] | None = None,
):
    db={z:deadband for z in ("RestaurantFastFood_All","Dining","Kitchen") if deadband is not None}
    mdb={z:mode_deadband for z in ("RestaurantFastFood_All","Dining","Kitchen") if mode_deadband is not None}
    calibrations=calibrate_controlled_thermostats(
        config,
        deadband_overrides_C=db,
        heating_mode_deadband_overrides_C=mdb,
        setpoint_min_separation_C=setpoint_min_separation_C,
    )
    serial={z:c.to_dict() for z,c in calibrations.items()}
    cal_path=root/"thermostat_calibration"/"controlled_training_calibration.json"
    write_json(cal_path,serial)

    profiles={z:resolve_actuation_profile(c,overrides=(actuation_overrides or {}).get(z)) for z,c in calibrations.items()}
    profile_serial={z:p.to_dict() for z,p in profiles.items()}
    profile_path=root/"thermostat_calibration"/"controlled_actuation_profiles.json"
    write_json(profile_path,profile_serial)

    for z,c in calibrations.items():
        observed={"cooling":c.supports_mode("cooling"),"heating":c.supports_mode("heating")}
        profile=profiles[z]
        alignment=dict(c.provenance.get("training_alignment",{}))
        print(f"[thermostat] {z}: modes={c.mode_counts} observed={observed} "
              f"Tsp=({c.setpoint_low_C:.3f},{c.setpoint_medium_C:.3f},{c.setpoint_high_C:.3f})",flush=True)
        print(f"             actuation cool Ts/mdot_nom/max="
              f"({profile.cooling.T_supply_C:.3f},{profile.cooling.mdot_nominal_kg_s:.4f},{profile.cooling.mdot_max_kg_s:.4f}) "
              f"source={profile.cooling.parameter_source}",flush=True)
        print(f"             actuation heat Ts/mdot_nom/max="
              f"({profile.heating.T_supply_C:.3f},{profile.heating.mdot_nominal_kg_s:.4f},{profile.heating.mdot_max_kg_s:.4f}) "
              f"source={profile.heating.parameter_source}",flush=True)
        print(f"             train alignment={alignment.get('strict_exact_train_row_count')}/"
              f"{alignment.get('phase_d_train_included_count')} exact={alignment.get('exact_train_count_match')}",flush=True)
        if not alignment.get("exact_train_count_match",False):
            raise RuntimeError(f"{z}: thermostat calibration is not exactly aligned to Phase-D TRAIN+included timestamps")

        # Observed actuator modes are authoritative. Experimental thermostat
        # setpoints must not rewrite or reject their measured Ts/mdot values.
        zone_overrides=(actuation_overrides or {}).get(z,{})
        for mode in ("cooling","heating"):
            if not c.supports_mode(mode) or zone_overrides.get(mode):
                continue
            pm=profile.for_mode(mode)
            if mode=="cooling":
                expected_ts=c.ts_cooling_C
                expected_nom=c.mdot_cooling_nominal_kg_s
                expected_max=c.mdot_cooling_max_kg_s
            else:
                expected_ts=c.ts_heating_C
                expected_nom=c.mdot_heating_nominal_kg_s
                expected_max=c.mdot_heating_max_kg_s
            if pm.parameter_source!="observed_train":
                raise RuntimeError(f"{z}: observed {mode} actuator provenance changed: {pm.parameter_source}")
            if not np.isclose(pm.T_supply_C,expected_ts,rtol=0.0,atol=1e-12):
                raise RuntimeError(f"{z}: observed {mode} Ts was altered")
            if not np.isclose(pm.mdot_nominal_kg_s,expected_nom,rtol=0.0,atol=1e-12):
                raise RuntimeError(f"{z}: observed {mode} nominal mdot was altered")
            if not np.isclose(pm.mdot_max_kg_s,expected_max,rtol=0.0,atol=1e-12):
                raise RuntimeError(f"{z}: observed {mode} max mdot was altered")

    if not calibrations["RestaurantFastFood_All"].supports_mode("cooling") or not calibrations["RestaurantFastFood_All"].supports_mode("heating"):
        raise RuntimeError("Controlled all-to-one calibration must observe both effective cooling and heating")
    if not calibrations["Dining"].supports_mode("cooling") or not calibrations["Dining"].supports_mode("heating"):
        raise RuntimeError("Controlled Dining calibration must observe both effective cooling and heating")
    if not calibrations["Kitchen"].supports_mode("cooling") or calibrations["Kitchen"].supports_mode("heating"):
        raise RuntimeError("Controlled Kitchen TRAIN evidence must be cooling-observed and heating-unobserved")

    # Controller capability is intentionally independent of TRAIN mode support.
    kh=profiles["Kitchen"].heating
    if kh.observed_in_train or not kh.qac_extrapolation_expected:
        raise RuntimeError("Kitchen fallback heating must be marked unobserved/extrapolative")
    kc=calibrations["Kitchen"]
    kdelta=float(kc.provenance["cooling_supply_deltaT_C"])
    kguard=max(float(kc.deadband_used_C),0.5)
    expected_kitchen_heating_Ts=max(
        float(kc.setpoint_medium_C)+kdelta,
        float(kc.setpoint_high_C)+kguard,
    )
    if not np.isclose(kh.reference_deltaT_C,kdelta,rtol=0.0,atol=1e-10):
        raise RuntimeError("Kitchen fallback heating deltaT is not derived from Kitchen observed cooling")
    if not np.isclose(kh.T_supply_C,expected_kitchen_heating_Ts,rtol=0.0,atol=1e-10):
        raise RuntimeError("Kitchen fallback heating Ts does not follow same-zone cooling deltaT rule")
    if not np.isclose(kh.mdot_nominal_kg_s,kc.mdot_cooling_nominal_kg_s,rtol=0.0,atol=1e-12):
        raise RuntimeError("Kitchen fallback heating nominal mdot is not inherited from Kitchen cooling")
    if not np.isclose(kh.mdot_max_kg_s,kc.mdot_cooling_max_kg_s,rtol=0.0,atol=1e-12):
        raise RuntimeError("Kitchen fallback heating max mdot is not inherited from Kitchen cooling")
    if not kh.parameter_source.startswith("fallback_same_zone_cooling_deltaT_and_mdot"):
        raise RuntimeError("Kitchen fallback heating provenance is not same-zone cooling based")

    for z,c in calibrations.items():
        if not (c.setpoint_low_C < c.setpoint_medium_C < c.setpoint_high_C):
            raise RuntimeError(f"{z}: low/medium/high setpoints must be strictly ordered")
        if min(c.setpoint_medium_C-c.setpoint_low_C,c.setpoint_high_C-c.setpoint_medium_C) < setpoint_min_separation_C - 1e-9:
            raise RuntimeError(f"{z}: setpoint conditions are not separated by the requested minimum")

    return calibrations,profiles,{
        "calibration_path":str(cal_path),
        "actuation_profile_path":str(profile_path),
        "zones":serial,
        "actuation_profiles":profile_serial,
    }


def real_data_smoke(config: PaperConfig, root: Path, *, max_rows:int, train_steps:int, sim_points:int,
                    cooling_mdot_choice:str, heating_mdot_choice:str, deadband:float|None, mode_deadband:float|None,
                    unobserved_mode_policy:str, setpoint_min_separation_C:float,
                    optuna_trials:int, actuation_overrides:dict[str,dict[str,dict[str,float]]]|None=None) -> dict[str,Any]:
    report={"status":"running","validation_intent":"tiny path/control-contract coverage; not convergence or accuracy",
            "paper_data_root":str(root),"cases":{},"phase_c_models":{},"neuromancer_runtime":runtime_info().__dict__}
    calibrations,actuation_profiles,cal_report=_calibration_report(config,root,deadband,mode_deadband,setpoint_min_separation_C,actuation_overrides=actuation_overrides); report["thermostat_calibration"]=cal_report
    phase_c={}
    for zone in ("RestaurantFastFood_All","Dining","Kitchen"):
        bundle=discover_and_load_phase_c_bundle(config,zone,phase_c_run_id=config.controlled_phase_c_run_id); phase_c[zone]=bundle
        report["phase_c_models"][zone]=bundle.provenance
    report["hpo_freeze_smoke"]=_tiny_hpo(root,trials=optuna_trials)

    for case_name in canonical_case_specs():
        print(f"[real-data] case={case_name}",flush=True)
        traj=load_case(config,case_name); idx=longest_training_prefix(traj,max_rows=max_rows)
        case_out={"selected_train_rows":int(len(idx)),"methods":{}}
        bundles={z:phase_c[z] for z in traj.zone_ids}; cals={z:calibrations[z] for z in traj.zone_ids}
        for method in METHODS:
            method_out={}
            for rc_order in (1,2):
                print(f"  {method} {rc_order}C: tiny fit + Sim1/2/3",flush=True)
                model,aux=build_paper_model(method,traj,rc_order=rc_order,train_indices=idx)
                history=tiny_fit(model,aux,train_steps=train_steps)
                runtime=PaperModelRuntime(model,traj)
                # Explicitly validate locked 2C initialization distinction.
                if rc_order==2:
                    segs=[s for s in __import__('Paper_PINODE_EPSR.common',fromlist=['contiguous_segments']).contiguous_segments(
                        traj.timestamp,traj.partition,traj.included,partition_name='test',dt_seconds=config.dt_seconds) if len(s)>=4]
                    k=int(max(segs,key=len)[0])+int(getattr(model.config,'L_e',1))-1 if not isinstance(model,InversePINNRC) else int(max(segs,key=len)[0])
                    x0=runtime.initialize(k).detach().cpu().numpy()
                    if isinstance(model,InversePINNRC):
                        if case_name=='all_to_one': ok=np.isclose(x0[0],x0[1])
                        else: ok=np.isclose(x0[0],x0[1]) and np.isclose(x0[2],x0[3])
                        if not ok: raise RuntimeError("Inverse-PINN-RC 2C must initialize Tm=Ta")
                        init_mode="Tm_equals_Ta"
                    else: init_mode="learned_causal_encoder"
                else: init_mode="observed_1C"
                r1=sim1(runtime,bundles,max_points=sim_points)
                r2=sim2(runtime,bundles,horizon=sim_points)
                r3=sim3(runtime,bundles,cals,horizon=max(4,sim_points),cooling_mdot_choice=cooling_mdot_choice,heating_mdot_choice=heating_mdot_choice,unobserved_mode_policy=unobserved_mode_policy,actuation_profiles={z:actuation_profiles[z] for z in traj.zone_ids})
                paths={}
                for r in (r1,r2,r3):
                    if r.trajectory.empty or not np.isfinite(r.trajectory.select_dtypes(include=[np.number]).to_numpy()).all():
                        raise FloatingPointError(f"{method}/{case_name}/{rc_order}C/{r.simulation}: non-finite evaluation output")
                    if r.simulation=='sim3' and r.provenance.get('historical_test_qac_used') is not False:
                        raise RuntimeError("Sim3 must not replay historical test QAC")
                    paths[r.simulation]=save_evaluation_result(r,run_id="patch05_validation",method=method,root=root)
                method_out[str(rc_order)]={"tiny_fit_loss":history,"initialization":init_mode,
                                           "sim1_metrics":r1.metrics,"sim2_metrics":r2.metrics,"sim3_metrics":r3.metrics,
                                           "sim3_mdot_choice":{"cooling":cooling_mdot_choice,"heating":heating_mdot_choice},
                                           "sim3_unobserved_mode_policy":unobserved_mode_policy,"artifacts":paths}
            case_out["methods"][method]=method_out
        report["cases"][case_name]=case_out
    report["status"]="passed"; return report


def parse_args():
    p=argparse.ArgumentParser(description="Validate PINODE/EPSR Day-6 Patch 05")
    p.add_argument("--real-data",action="store_true")
    p.add_argument("--skip-unit-contracts",action="store_true")
    p.add_argument("--max-rows",type=int,default=48); p.add_argument("--train-steps",type=int,default=1)
    p.add_argument("--sim-points",type=int,default=8); p.add_argument("--optuna-trials",type=int,default=2)
    p.add_argument("--cooling-mdot-choice",choices=("nominal","max"),default="nominal")
    p.add_argument("--heating-mdot-choice",choices=("nominal","max"),default="nominal")
    p.add_argument("--deadband-C",type=float,default=None); p.add_argument("--heating-mode-deadband-C",type=float,default=None)
    p.add_argument("--unobserved-mode-policy",choices=("fallback","error"),default="fallback")
    p.add_argument("--setpoint-min-separation-C",type=float,default=0.50)
    p.add_argument("--actuation-overrides-json",default=None)
    p.add_argument("--paper-data-root",default=None)
    return p.parse_args()


def main()->int:
    args=parse_args(); nm=None; payload={}
    run_id="patch05_validation_"+datetime.now().strftime("%Y%m%d_%H%M%S")
    root=resolve_paper_data_root(args.paper_data_root,create=True)/"validation"/run_id; root.mkdir(parents=True,exist_ok=True)
    output=root/"patch05_validation_real.json"
    try:
        print("="*100); print("PINODE / EPSR PATCH 05 DAY-6 VALIDATION"); print("="*100)
        print(f"External validation root: {root}")
        nm=runtime_info(); print(f"Neuromancer version: {nm.version}")
        if not args.skip_unit_contracts:
            code=run_pytest_contracts()
            if code: raise RuntimeError(f"Patch05 cumulative pytest contracts failed with exit code {code}")
        payload={"status":"running","unit_contracts":"passed" if not args.skip_unit_contracts else "skipped",
                 "framework_runtime":nm.__dict__,"paper_data_root":str(root)}
        if args.real_data:
            config=PaperConfig.from_environment()
            actuation_overrides=_load_actuation_overrides(args.actuation_overrides_json)
            payload["real_data"]=real_data_smoke(config,root,max_rows=args.max_rows,train_steps=args.train_steps,
                sim_points=args.sim_points,cooling_mdot_choice=args.cooling_mdot_choice,heating_mdot_choice=args.heating_mdot_choice,
                deadband=args.deadband_C,mode_deadband=args.heating_mode_deadband_C,unobserved_mode_policy=args.unobserved_mode_policy,
                setpoint_min_separation_C=args.setpoint_min_separation_C,optuna_trials=args.optuna_trials,actuation_overrides=actuation_overrides)
        else: payload["real_data"]="not_requested"
        payload["status"]="passed"; write_json(output,payload); print(f"Validation report: {output}"); print("PATCH 05 STATUS: PASSED"); return 0
    except BaseException as exc:
        failure={"status":"failed","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),
                 "partial_report":payload,"paper_data_root":str(root)}
        try: write_json(output,failure); print(f"Failure report: {output}")
        except Exception: pass
        traceback.print_exc(); print("PATCH 05 STATUS: FAILED"); return 1

if __name__=="__main__": raise SystemExit(main())
