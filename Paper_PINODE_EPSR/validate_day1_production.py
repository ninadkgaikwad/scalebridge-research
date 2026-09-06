from __future__ import annotations

"""Validate the Day-1 production/HPO infrastructure on the real controlled campaign.

The validator is deliberately bounded: it validates source contracts, real
Phase-B/C/D resolution, month-balanced TRAIN-only sampling, 300-s thermostat
calibration, Phase-C QHVAC/PHVAC semantics, and model-factory hyperparameter
propagation.  It does not automatically launch the 32-configuration micro
campaign; that is the explicit next qualification command after this validator
passes.
"""

import argparse
import math
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from .common import write_json
from .config import PaperConfig, canonical_case_specs
from .data import load_case
from .experiment import build_paper_model
from .paper_paths import resolve_paper_data_root
from .phase_c import discover_and_load_phase_c_bundle, Q_HVAC_X
from .production import ControllerOverrideConfig, HPOConfig, production_matrix, resolve_production_config, resolve_production_layout, select_month_balanced_hpo_sample
from .thermostat_data import calibrate_controlled_thermostats
from .thermostat import resolve_actuation_profile


def _pytest(repo_root: Path) -> int:
    files = [
        "Paper_PINODE_EPSR/tests/test_patch2_contract.py",
        "Paper_PINODE_EPSR/tests/test_patch3_contract.py",
        "Paper_PINODE_EPSR/tests/test_patch4_contract.py",
        "Paper_PINODE_EPSR/tests/test_patch5_contract.py",
        "Paper_PINODE_EPSR/tests/test_reorganization_contract.py",
        "Paper_PINODE_EPSR/tests/test_day1_production_contract.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", *files]
    print("[pytest] " + " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=repo_root, check=False).returncode)


def _real_checks(config: PaperConfig) -> dict[str, object]:
    layout = resolve_production_layout(config, create=True)
    result: dict[str, object] = {"paths": layout.to_dict(), "matrix_count": len(production_matrix())}
    if result["matrix_count"] != 32:
        raise RuntimeError("Production matrix is not 32 configurations")

    sampling = {}
    for case_name in canonical_case_specs():
        trajectory = load_case(config, case_name)
        case_report = {}
        train_mask = trajectory.mask("train", included_only=True)

        # micro32: 0.5% + strict 20% holdout, with explicitly reduced HPO
        # rollout geometry N_r<=3. L_e<=12 remains legal via causal context.
        micro = select_month_balanced_hpo_sample(
            trajectory, train_percentage=0.5, holdout_percentage=20.0,
            conservative_N_r=3, conservative_L_e=12,
        )
        if len(micro.monthly_counts) != 12:
            raise RuntimeError(f"{case_name}: micro HPO sampling did not cover all 12 TRAIN months")
        for month, values in micro.monthly_counts.items():
            expected = math.floor(values["train_available"] * 0.005)
            expected_hold = math.floor(expected * 0.20)
            if values["requested_targets"] != expected:
                raise RuntimeError(f"{case_name}/{month}: micro target budget is not floored")
            if values["requested_holdout_targets"] != expected_hold or values["holdout"] != expected_hold:
                raise RuntimeError(f"{case_name}/{month}: micro holdout budget is not floored")
        if micro.actual_train_percentage > 0.5 + 1e-12:
            raise RuntimeError(f"{case_name}: micro sampler exceeded requested 0.5%")
        if micro.actual_holdout_percentage > 20.0 + 1e-12:
            raise RuntimeError(f"{case_name}: micro sampler exceeded requested 20% holdout")
        if not micro.rollout_windows("fit", N_r=3, L_e=12, rc_order=2):
            raise RuntimeError(f"{case_name}: no micro 2C fit HPO windows")
        if not micro.rollout_windows("holdout", N_r=3, L_e=12, rc_order=2):
            raise RuntimeError(f"{case_name}: no micro 2C holdout HPO windows")
        try:
            select_month_balanced_hpo_sample(
                trajectory, train_percentage=0.5, holdout_percentage=20.0,
                conservative_N_r=12, conservative_L_e=12,
            )
        except ValueError:
            pass
        else:
            raise RuntimeError(f"{case_name}: 0.5% full N_r=12 geometry should fail instead of inflating")
        case_report["0.5%_micro_Nr3"] = {
            "actual_percentage": micro.actual_train_percentage,
            "requested_holdout_percentage": micro.requested_holdout_percentage,
            "actual_holdout_percentage": micro.actual_holdout_percentage,
            "fit_rows": len(micro.fit_indices),
            "holdout_rows": len(micro.holdout_indices),
            "months": micro.monthly_counts,
        }

        # Production geometry retains N_r<=12/L_e<=12. Validate percentages that
        # are actually feasible with a strict floored 20% inner holdout.
        for pct in (2.0, 5.0, 100.0):
            sample = select_month_balanced_hpo_sample(
                trajectory, train_percentage=pct, holdout_percentage=20.0,
                conservative_N_r=12, conservative_L_e=12,
            )
            if len(sample.monthly_counts) != 12:
                raise RuntimeError(f"{case_name}: HPO sampling did not cover all 12 TRAIN months")
            if not np.all(train_mask[sample.fit_indices]) or not np.all(train_mask[sample.holdout_indices]):
                raise RuntimeError(f"{case_name}: HPO sample escaped authoritative TRAIN")
            if np.intersect1d(sample.fit_indices, sample.holdout_indices).size:
                raise RuntimeError(f"{case_name}: HPO fit/holdout overlap")
            for month, values in sample.monthly_counts.items():
                expected = math.floor(values["train_available"] * pct / 100.0)
                expected_hold = math.floor(expected * 0.20)
                if values["requested_targets"] != expected:
                    raise RuntimeError(f"{case_name}/{month}: target budget is not floored")
                if values["requested_holdout_targets"] != expected_hold or values["holdout"] != expected_hold:
                    raise RuntimeError(f"{case_name}/{month}: holdout budget is not floored")
            if sample.actual_train_percentage > pct + 1e-12:
                raise RuntimeError(f"{case_name}: requested {pct:g}% HPO but sampler exceeded it")
            if sample.actual_holdout_percentage > 20.0 + 1e-12:
                raise RuntimeError(f"{case_name}: sampler exceeded requested 20% holdout")
            if not sample.rollout_windows("fit", N_r=12, L_e=12, rc_order=2):
                raise RuntimeError(f"{case_name}: no conservative 2C fit HPO windows at {pct:g}%")
            if not sample.rollout_windows("holdout", N_r=12, L_e=12, rc_order=2):
                raise RuntimeError(f"{case_name}: no conservative 2C holdout HPO windows at {pct:g}%")
            case_report[f"{pct:g}%"] = {
                "actual_percentage": sample.actual_train_percentage,
                "requested_holdout_percentage": sample.requested_holdout_percentage,
                "actual_holdout_percentage": sample.actual_holdout_percentage,
                "fit_rows": len(sample.fit_indices),
                "holdout_rows": len(sample.holdout_indices),
                "months": sample.monthly_counts,
            }
        sampling[case_name] = case_report
    result["hpo_sampling"] = sampling

    calibrations = calibrate_controlled_thermostats(config)
    cal_report = {}
    for zone, cal in calibrations.items():
        diag = dict(cal.provenance.get("deadband_transition_diagnostics", {}))
        if diag.get("timestamps_available") is not True:
            raise RuntimeError(f"{zone}: 300-s deadband calibration lacks canonical timestamps")
        if not np.isclose(float(diag.get("dt_seconds", -1)), 300.0):
            raise RuntimeError(f"{zone}: thermostat transition dt is not 300 s")
        if int(diag.get("skipped_noncontiguous_pairs", 0)) < 1:
            raise RuntimeError(f"{zone}: expected monthly TRAIN gaps were not excluded from transition calibration")
        cal_report[zone] = cal.to_dict()
    result["thermostat_calibration_300s"] = cal_report

    # Validate explicit +/-1 C half-width override and user-overridden Kitchen
    # heating fallback without changing the default data-derived calibration.
    override_cfg = ControllerOverrideConfig.from_mapping({
        "deadband_half_width_C": 1.0,
        "zones": {
            "Kitchen": {
                "heating": {
                    "T_supply_C": 34.0,
                    "mdot_nominal_kg_s": 0.80,
                    "mdot_max_kg_s": 1.10,
                }
            }
        },
    })
    override_cal = calibrate_controlled_thermostats(
        config,
        deadband_overrides_C=override_cfg.deadband_overrides_C(),
        heating_mode_deadband_overrides_C=override_cfg.heating_mode_deadband_overrides_C(),
    )
    for zone, cal in override_cal.items():
        if not np.isclose(cal.deadband_used_C, 1.0):
            raise RuntimeError(f"{zone}: +/-1 C user deadband override was not applied")
        if cal.provenance.get("deadband_source") != "override":
            raise RuntimeError(f"{zone}: legacy deadband provenance changed unexpectedly")
        if cal.provenance.get("deadband_source_class") != "user_override":
            raise RuntimeError(f"{zone}: rich deadband provenance does not identify user override")
    kitchen_profile = resolve_actuation_profile(
        override_cal["Kitchen"],
        overrides=override_cfg.actuation_overrides("Kitchen"),
    )
    if kitchen_profile.heating.parameter_source != "user_override":
        raise RuntimeError("Kitchen heating override was not classified as user_override")
    if not kitchen_profile.heating.qac_extrapolation_expected:
        raise RuntimeError("Kitchen heating must remain explicitly extrapolative/OOD")
    result["controller_override_contract"] = {
        "config": override_cfg.to_dict(),
        "deadband_used_C": {z: c.deadband_used_C for z, c in override_cal.items()},
        "kitchen_heating_profile": kitchen_profile.to_dict(),
    }

    phase_c_report = {}
    for zone in ("RestaurantFastFood_All", "Dining", "Kitchen"):
        bundle = discover_and_load_phase_c_bundle(config, zone, phase_c_run_id=config.controlled_phase_c_run_id)
        qx = float(Q_HVAC_X(1.0, 30.0, 20.0))
        qy = float(np.asarray(bundle.predict_qac_from_hvac_proxy(qx)).reshape(-1)[0])
        pplus = float(np.asarray(bundle.predict_phvac_from_qac(qy)).reshape(-1)[0])
        pminus = float(np.asarray(bundle.predict_phvac_from_qac(-qy)).reshape(-1)[0])
        if not np.isfinite([qx, qy, pplus, pminus]).all():
            raise FloatingPointError(f"{zone}: non-finite Phase-C chain")
        if not np.isclose(pplus, pminus, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"{zone}: PHVAC model is not driven by abs(corrected QHVAC)")
        phase_c_report[zone] = {
            "QHVAC_physics_probe_W": qx,
            "QHVAC_phaseC_probe_W": qy,
            "PHVAC_raw_probe_W": pplus,
            "provenance": bundle.provenance,
            "chain": "QHVAC_physics -> Phase-C QAC model -> QHVAC_phaseC -> abs -> Phase-C PHVAC",
        }
    result["phase_c_chain"] = phase_c_report

    # Prove frozen HPO values reach the actual model configuration.
    traj = load_case(config, "all_to_one")
    train_idx = np.flatnonzero(traj.mask("train", included_only=True))[:64]
    hp = {
        "hidden_layers": 2, "hidden_width": 16, "activation": "silu",
        "N_r": 3, "N_s": 2, "lambda_wd": 1e-6,
        "learning_rate": 1e-3, "optimizer": "adam",
    }
    model, _ = build_paper_model(
        "neural_ode", traj, rc_order=1, train_indices=train_idx,
        hyperparameters=hp, seed=7,
    )
    applied = {
        "hidden_layers": model.config.hidden_layers,
        "hidden_width": model.config.hidden_width,
        "activation": model.config.activation,
        "N_r": model.config.N_r,
        "N_s": model.config.N_s,
        "lambda_wd": model.config.lambda_wd,
    }
    expected = {k: hp[k] for k in applied}
    if applied != expected:
        raise RuntimeError(f"Frozen hyperparameters did not reach real model config: {applied} != {expected}")
    result["frozen_hyperparameter_application"] = applied
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-real-data", action="store_true")
    args = parser.parse_args()
    root = resolve_paper_data_root(create=True)
    outdir = root / "08_manifests" / "validation" / ("day1_production_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "validation.json"
    payload: dict[str, object] = {"status": "running", "output": str(output)}
    try:
        repo_root = Path(__file__).resolve().parent.parent
        if not args.skip_pytest:
            code = _pytest(repo_root)
            if code:
                raise RuntimeError(f"Targeted pytest contracts failed with exit code {code}")
            payload["pytest"] = "passed"
        else:
            payload["pytest"] = "skipped"
        if not args.skip_real_data:
            payload["real_data"] = _real_checks(resolve_production_config())
        else:
            payload["real_data"] = "skipped"
        payload["status"] = "passed"
        write_json(output, payload)
        print(f"Validation report: {output}")
        print("DAY-1 PRODUCTION VALIDATION: PASSED")
        return 0
    except BaseException as exc:
        payload.update({
            "status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        try:
            write_json(output, payload)
            print(f"Failure report: {output}")
        except Exception:
            pass
        traceback.print_exc()
        print("DAY-1 PRODUCTION VALIDATION: FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
