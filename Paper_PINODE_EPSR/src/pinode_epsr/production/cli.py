from __future__ import annotations

import argparse
import json

from ..data.thermostat_data import calibrate_controlled_thermostats
from ..evaluation.thermostat import resolve_actuation_profile
from .campaign import run_micro_campaign, run_production_campaign
from .contracts import (
    HPOConfig,
    ProductionTrainingConfig,
    load_controller_override_config,
)
from .matrix import production_matrix
from .paths import resolve_production_config, resolve_production_layout


def _seeds(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _priorities(scope: str) -> tuple[str, ...]:
    return {"priority-a": ("A",), "priority-ab": ("A", "B"), "full": ("A", "B", "C")}[scope]


def _add_controller_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--controller-config",
        default=None,
        help=(
            "YAML/JSON containing a controller section. Default: "
            "Paper_PINODE_EPSR/configs/production.yaml. "
            "deadband_half_width_C=1.0 means +/-1 C around setpoint."
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PINODE/EPSR production experiment orchestration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("paths")
    sub.add_parser("matrix")

    calibrate = sub.add_parser("calibrate-controller")
    _add_controller_config_argument(calibrate)

    micro = sub.add_parser("micro32", help="tiny all-32 end-to-end qualification")
    micro.add_argument("--hpo-percentage", type=float, default=0.5)
    micro.add_argument("--seed", type=int, default=0)
    micro.add_argument("--fail-fast", action="store_true")
    _add_controller_config_argument(micro)

    prod = sub.add_parser("campaign", help="persistent HPO + training + checkpoint + Sim1/2/3")
    prod.add_argument("--scope", choices=("priority-a", "priority-ab", "full"), default="priority-a")
    prod.add_argument("--seeds", default="0", help="comma-separated multi-start restart seeds")
    prod.add_argument("--hpo-percentage", type=float, default=2.0)
    prod.add_argument("--hpo-holdout-percentage", type=float, default=20.0)
    prod.add_argument("--hpo-objective", choices=(
        "recursive_temperature_normalized",
        "recursive_temperature_rmse_C",
        "recursive_temperature_mae_C",
        "recursive_temperature_cvrmse",
    ), default="recursive_temperature_normalized")
    prod.add_argument("--hpo-trials", type=int, default=12)
    prod.add_argument("--hpo-epochs", type=int, default=25)
    prod.add_argument("--hpo-patience", type=int, default=5)
    prod.add_argument("--training-epochs", type=int, default=500)
    prod.add_argument("--training-patience", type=int, default=50)
    prod.add_argument("--validation-max-windows", type=int, default=256)
    prod.add_argument("--fail-fast", action="store_true")
    _add_controller_config_argument(prod)

    args = parser.parse_args(argv)
    config = resolve_production_config()

    if args.command == "paths":
        print(json.dumps(resolve_production_layout(config, create=False).to_dict(), indent=2)); return 0
    if args.command == "matrix":
        specs = production_matrix(); print(json.dumps([s.to_dict() for s in specs], indent=2)); return 0

    controller = load_controller_override_config(getattr(args, "controller_config", None))

    if args.command == "calibrate-controller":
        calibrations = calibrate_controlled_thermostats(
            config,
            deadband_overrides_C=controller.deadband_overrides_C(),
            heating_mode_deadband_overrides_C=controller.heating_mode_deadband_overrides_C(),
        )
        payload = {
            "controller_override_config": controller.to_dict(),
            "calibrations": {k: v.to_dict() for k, v in calibrations.items()},
            "actuation_profiles": {
                k: resolve_actuation_profile(v, overrides=controller.actuation_overrides(k)).to_dict()
                for k, v in calibrations.items()
            },
        }
        print(json.dumps(payload, indent=2, default=str)); return 0
    if args.command == "micro32":
        summary = run_micro_campaign(
            config, hpo_percentage=args.hpo_percentage, seed=args.seed,
            continue_on_error=not args.fail_fast,
            controller_overrides=controller,
        )
        print(json.dumps({k: summary[k] for k in ("campaign_id","expected_configurations","completed","accepted","rejected","failed")}, indent=2))
        return 0 if summary["failed"] == 0 and summary["rejected"] == 0 else 2
    if args.command == "campaign":
        hpo = HPOConfig(
            train_percentage=args.hpo_percentage,
            holdout_percentage=args.hpo_holdout_percentage,
            objective=args.hpo_objective,
            n_trials=args.hpo_trials,
            max_epochs_per_trial=args.hpo_epochs,
            patience=args.hpo_patience,
            max_rollout_steps=12,
            max_encoder_history_steps=12,
        )
        training = ProductionTrainingConfig(
            max_epochs=args.training_epochs,
            patience=args.training_patience,
            validation_max_windows=args.validation_max_windows,
            continue_on_error=not args.fail_fast,
        )
        summary = run_production_campaign(
            config, priorities=_priorities(args.scope), seeds=_seeds(args.seeds),
            hpo_config=hpo, training_config=training,
            continue_on_error=not args.fail_fast,
            controller_overrides=controller,
        )
        print(json.dumps({k: summary[k] for k in ("production_id","expected_configurations","completed_configurations","failed_configurations")}, indent=2))
        return 0 if summary["failed_configurations"] == 0 else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
