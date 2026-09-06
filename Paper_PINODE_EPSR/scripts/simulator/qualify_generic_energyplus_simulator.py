from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


EXPECTED_SHA = (
    "da86d8c3c782424d2d4f30e29a97220c9b9042f721e83591f43762f34df77101"
)


def add_src(repo_root: Path):
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def find_candidate(data_root: Path) -> Path:
    from pinode_epsr.simulator.runtime_idf import sha256_file

    root = (
        data_root
        / "02_version_transition_validation"
        / "transition_logs"
    )
    candidates = sorted(
        root.rglob(
            "RestaurantFastFood_Buffalo_EPlus24.1_FINAL_CANDIDATE.idf"
        )
    )
    verified = [
        p for p in candidates
        if sha256_file(p).lower() == EXPECTED_SHA
    ]
    if not verified:
        raise RuntimeError("Accepted V24.1 candidate not found.")
    return max(verified, key=lambda p: p.stat().st_mtime)


def find_epw(data_root: Path) -> Path:
    files = sorted(
        (
            data_root
            / "01_energyplus24_plant"
            / "weather"
        ).glob("*.epw")
    )
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one EPW; found {len(files)}."
        )
    return files[0]


def _float_or_none(value):
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _check_time_aligned_override_history(
    *,
    run_dir: Path,
    control_step_index: int,
    zone: str,
):
    """
    Validate the command/readback identity at the same recorded EnergyPlus
    system timesteps.

    The sensible-load request is intentionally recomputed at
    AfterPredictorAfterHVACManagers as T_zone evolves. Therefore comparing one
    instantaneous TransformedZoneCommand value to the 300-s mean actuator
    readback is invalid. The correct gate compares each recorded transformed
    command with its actuator readback at the same system timestep.
    """
    import csv

    path = (
        run_dir
        / "history"
        / "system_timestep_zone_history.csv"
    )

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (
                int(row["control_step_index"]) == int(control_step_index)
                and row["zone_token"].upper() == zone.upper()
            ):
                rows.append(row)

    if not rows:
        return {
            "rows": 0,
            "fan_max_abs_error": None,
            "load_max_abs_error": None,
            "fan_pass": False,
            "load_pass": False,
            "pass": False,
        }

    fan_errors = []
    load_errors = []

    for row in rows:
        fan_cmd = _float_or_none(
            row.get("transformed_fan_command_kg_s")
        )
        fan_rb = _float_or_none(
            row.get("fan_actuator_readback_kg_s")
        )
        load_cmd = _float_or_none(
            row.get("transformed_sensible_load_request_w")
        )
        load_rb = _float_or_none(
            row.get("load_actuator_readback_w")
        )

        if fan_cmd is None or fan_rb is None:
            fan_errors.append(float("inf"))
        else:
            fan_errors.append(abs(fan_cmd - fan_rb))

        if load_cmd is None or load_rb is None:
            load_errors.append(float("inf"))
        else:
            load_errors.append(abs(load_cmd - load_rb))

    fan_max = max(fan_errors)
    load_max = max(load_errors)

    return {
        "rows": len(rows),
        "fan_max_abs_error": fan_max,
        "load_max_abs_error": load_max,
        "fan_pass": fan_max <= 1e-9,
        "load_pass": load_max <= 1e-6,
        "pass": (
            fan_max <= 1e-9
            and load_max <= 1e-6
        ),
    }


def check_override_step(result, label, run_dir):
    checks = {}
    for zone in ["DINING", "KITCHEN"]:
        x = result.transformed_commands[zone]
        s = result.zone_history_summary[zone]

        seconds = s.get("accumulated_system_seconds")
        duration_ok = (
            seconds is not None
            and abs(float(seconds) - 300.0) <= 1e-6
        )

        aligned = _check_time_aligned_override_history(
            run_dir=run_dir,
            control_step_index=result.control_step_index,
            zone=zone,
        )

        # Interval means are retained as useful diagnostics, but they are NOT
        # compared to the final instantaneous transformed load command.
        mean_transformed_fan = s.get(
            "mean__transformed_fan_command_kg_s"
        )
        mean_fan_readback = s.get(
            "mean__fan_actuator_readback_kg_s"
        )
        mean_transformed_load = s.get(
            "mean__transformed_sensible_load_request_w"
        )
        mean_load_readback = s.get(
            "mean__load_actuator_readback_w"
        )

        checks[zone] = {
            "label": label,
            "control_step_index": result.control_step_index,
            "effective_control_mode": x.effective_control_mode,
            "feasible": x.feasible,
            "fallback_applied": x.fallback_applied,
            "received_mdot_kg_s": x.received_mass_flow_kg_s,
            "received_Tsa_c": x.received_supply_air_temperature_c,
            "final_callback_delta_t_star_c": x.delta_t_star_c,
            "final_callback_transformed_fan_kg_s": (
                x.fan_actuator_command_kg_s
            ),
            "final_callback_transformed_load_w": (
                x.sensible_load_request_w
            ),
            "mean_transformed_fan_kg_s": mean_transformed_fan,
            "mean_fan_readback_kg_s": mean_fan_readback,
            "mean_transformed_load_w": mean_transformed_load,
            "mean_load_readback_w": mean_load_readback,
            "time_aligned_history_check": aligned,
            "accumulated_system_seconds": seconds,
            "full_300s_pass": duration_ok,
            "pass": (
                x.feasible
                and not x.fallback_applied
                and aligned["pass"]
                and duration_ok
            ),
        }

    return checks

def check_fallback_step(result):
    checks = {}
    for zone in ["DINING", "KITCHEN"]:
        x = result.transformed_commands[zone]
        s = result.zone_history_summary[zone]

        seconds = s.get("accumulated_system_seconds")
        effective_fan_readback = s.get(
            "mean__fan_actuator_readback_kg_s"
        )
        effective_load_readback = s.get(
            "mean__load_actuator_readback_w"
        )
        raw_fan_api = s.get(
            "mean__fan_actuator_api_value_raw"
        )
        raw_load_api = s.get(
            "mean__load_actuator_api_value_raw"
        )

        duration_ok = (
            seconds is not None
            and abs(float(seconds) - 300.0) <= 1e-6
        )

        checks[zone] = {
            "effective_control_mode": x.effective_control_mode,
            "feasible": x.feasible,
            "fallback_applied": x.fallback_applied,
            "reason": x.feasibility_reason,
            "fan_command_is_none": x.fan_actuator_command_kg_s is None,
            "load_command_is_none": x.sensible_load_request_w is None,
            "effective_fan_readback_is_none": (
                effective_fan_readback is None
                or (
                    isinstance(effective_fan_readback, float)
                    and math.isnan(effective_fan_readback)
                )
            ),
            "effective_load_readback_is_none": (
                effective_load_readback is None
                or (
                    isinstance(effective_load_readback, float)
                    and math.isnan(effective_load_readback)
                )
            ),
            "raw_fan_api_value_retained_for_forensics": raw_fan_api,
            "raw_load_api_value_retained_for_forensics": raw_load_api,
            "accumulated_system_seconds": seconds,
            "full_300s_pass": duration_ok,
            "pass": (
                x.fallback_applied
                and not x.feasible
                and x.fan_actuator_command_kg_s is None
                and x.sensible_load_request_w is None
                and (
                    effective_fan_readback is None
                    or (
                        isinstance(effective_fan_readback, float)
                        and math.isnan(effective_fan_readback)
                    )
                )
                and (
                    effective_load_readback is None
                    or (
                        isinstance(effective_load_readback, float)
                        and math.isnan(effective_load_readback)
                    )
                )
                and duration_ok
            ),
        }

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--energyplus-root",
        type=Path,
        default=Path(r"C:\EnergyPlusV24-1-0"),
    )
    parser.add_argument("--start-month", type=int, default=8)
    parser.add_argument("--start-day", type=int, default=3)
    parser.add_argument("--start-hour", type=float, default=17.0)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    add_src(repo)

    from pinode_epsr.simulator import (
        ActuatorTransform,
        ControlWindow,
        EnergyPlusSimulator,
        EPSRProjectLayout,
        FeasibilitySupervisor,
        RestaurantFastFoodCommand,
        restaurant_fastfood_signal_specs,
        restaurant_fastfood_zone_specs,
    )

    layout = EPSRProjectLayout.from_repo_root(repo)
    candidate = find_candidate(layout.data_root)
    epw = find_epw(layout.data_root)

    # 45 min provides room for synchronization + three complete 300-s steps.
    simulator = EnergyPlusSimulator(
        layout=layout,
        energyplus_root=args.energyplus_root,
        authoritative_idf=candidate,
        weather_epw=epw,
        authoritative_idf_sha256=EXPECTED_SHA,
        zone_specs=restaurant_fastfood_zone_specs(),
        signal_specs=restaurant_fastfood_signal_specs(),
        control_window=ControlWindow(
            start_month=args.start_month,
            start_day=args.start_day,
            start_hour=args.start_hour,
            end_month=args.start_month,
            end_day=args.start_day,
            end_hour=min(24.0, args.start_hour + 0.75),
        ),
        actuator_transform=ActuatorTransform(),
        feasibility_supervisor=FeasibilitySupervisor(),
        run_label="generic_simulator_qualification_v6_1_1",
        capture_api_registry=True,
    )

    try:
        obs0 = simulator.reset()

        # STEP 1: inside-envelope cooling for both zones.
        cooling_cmd = RestaurantFastFoodCommand.four_physical_commands(
            dining_mass_flow_kg_s=(
                0.65
                * obs0.zones["DINING"].design_max_mass_flow_kg_s
            ),
            dining_supply_air_temperature_c=(
                obs0.zones["DINING"].zone_temperature_c - 6.0
            ),
            kitchen_mass_flow_kg_s=(
                0.60
                * obs0.zones["KITCHEN"].design_max_mass_flow_kg_s
            ),
            kitchen_supply_air_temperature_c=(
                obs0.zones["KITCHEN"].zone_temperature_c - 5.0
            ),
        )
        cooling_result = simulator.step(cooling_cmd)
        cooling_checks = check_override_step(
            cooling_result,
            "inside_envelope_cooling",
            simulator.run_dir,
        )

        # STEP 2: inside-envelope heating for both zones, explicitly proving
        # Kitchen heating is commandable through the same transform.
        obs1 = cooling_result.observation
        heating_cmd = RestaurantFastFoodCommand.four_physical_commands(
            dining_mass_flow_kg_s=(
                0.65
                * obs1.zones["DINING"].design_max_mass_flow_kg_s
            ),
            dining_supply_air_temperature_c=(
                obs1.zones["DINING"].zone_temperature_c + 6.0
            ),
            kitchen_mass_flow_kg_s=(
                0.60
                * obs1.zones["KITCHEN"].design_max_mass_flow_kg_s
            ),
            kitchen_supply_air_temperature_c=(
                obs1.zones["KITCHEN"].zone_temperature_c + 5.0
            ),
        )
        heating_result = simulator.step(heating_cmd)
        heating_checks = check_override_step(
            heating_result,
            "inside_envelope_heating",
            simulator.run_dir,
        )

        # STEP 3: deliberately outside the envelope. Dining is cooling,
        # Kitchen is heating, proving symmetric fallback across signs/zones.
        obs2 = heating_result.observation
        fallback_cmd = RestaurantFastFoodCommand.four_physical_commands(
            dining_mass_flow_kg_s=(
                0.90
                * obs2.zones["DINING"].design_max_mass_flow_kg_s
            ),
            dining_supply_air_temperature_c=(
                obs2.zones["DINING"].zone_temperature_c - 10.0
            ),
            kitchen_mass_flow_kg_s=(
                0.90
                * obs2.zones["KITCHEN"].design_max_mass_flow_kg_s
            ),
            kitchen_supply_air_temperature_c=(
                obs2.zones["KITCHEN"].zone_temperature_c + 10.0
            ),
        )
        fallback_result = simulator.step(fallback_cmd)
        fallback_checks = check_fallback_step(fallback_result)

        required_files = [
            simulator.run_dir
            / "history"
            / "system_timestep_zone_history.csv",
            simulator.run_dir
            / "history"
            / "control_step_zone_history.csv",
            simulator.run_dir
            / "history"
            / "received_command_history.csv",
            simulator.run_dir
            / "history"
            / "control_steps.jsonl",
            simulator.run_dir
            / "history"
            / "signal_catalog.json",
            simulator.run_dir
            / "history"
            / "api_exchange_registry.csv",
        ]
        files_pass = all(p.exists() for p in required_files)

        overall = (
            files_pass
            and all(c["pass"] for c in cooling_checks.values())
            and all(c["pass"] for c in heating_checks.values())
            and all(c["pass"] for c in fallback_checks.values())
        )

        payload = {
            "status": "PASS" if overall else "FAIL",
            "contract_version": "20260830_v6_1_1",
            "run_dir": str(simulator.run_dir),
            "inside_envelope_cooling_checks": cooling_checks,
            "inside_envelope_heating_checks": heating_checks,
            "outside_envelope_fallback_checks": fallback_checks,
            "required_history_files_present": files_pass,
        }

        print(json.dumps(payload, indent=2))

        (
            simulator.run_dir
            / "GENERIC_SIMULATOR_QUALIFICATION_RESULT.json"
        ).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

        return 0 if overall else 2

    finally:
        simulator.close()


if __name__ == "__main__":
    raise SystemExit(main())
