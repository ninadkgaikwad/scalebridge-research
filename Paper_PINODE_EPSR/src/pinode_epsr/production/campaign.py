from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import numpy as np

from ..core.common import build_rollout_windows, contiguous_segments, load_checkpoint, representative_window_subset, write_json
from ..core.config import PaperConfig
from ..data.phase_c import discover_and_load_phase_c_bundle
from ..data.phase_d import load_case
from ..data.thermostat_data import calibrate_controlled_thermostats
from ..evaluation.runtime import PaperModelRuntime
from ..training.experiment import build_paper_model
from ..training.trainer import FrozenHyperparameters, OptimizationConfig
from .checkpoints import CheckpointRecord, acceptance_gates, append_registry, save_candidate
from .contracts import HPOConfig, ProductionTrainingConfig, ControllerOverrideConfig, load_controller_override_config
from .evaluation import run_offline_evaluations
from .hpo import hpo_protocol_id, run_persistent_hpo
from .matrix import ExperimentSpec, production_matrix
from .paths import resolve_production_layout
from .provenance import canonical_source_hash, environment_manifest, stable_id
from .training import fit_model


def _phase_c_for_case(config: PaperConfig, trajectory):
    return {
        z: discover_and_load_phase_c_bundle(config, z, phase_c_run_id=config.controlled_phase_c_run_id)
        for z in trajectory.zone_ids
    }


def run_micro_campaign(
    config: PaperConfig,
    *,
    hpo_percentage: float = 0.5,
    seed: int = 0,
    continue_on_error: bool = True,
    controller_overrides: ControllerOverrideConfig | None = None,
) -> dict[str, object]:
    """Run all 32 configurations through the real lifecycle with tiny budgets."""
    layout = resolve_production_layout(config, create=True)
    controller_overrides = controller_overrides or load_controller_override_config()
    campaign_id = stable_id(
        "micro32",
        {
            "campaign": config.campaign_id,
            "hpo_percentage": hpo_percentage,
            "seed": seed,
            "controller_overrides": controller_overrides.to_dict(),
            "micro_hpo_max_rollout_steps": 3,
        },
    )
    root = layout.manifest_root / "micro_qualification" / campaign_id
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "controller_override_config.json", controller_overrides.to_dict())
    thermostat = calibrate_controlled_thermostats(
        config,
        deadband_overrides_C=controller_overrides.deadband_overrides_C(),
        heating_mode_deadband_overrides_C=controller_overrides.heating_mode_deadband_overrides_C(),
    )
    write_json(root / "thermostat_calibration_300s.json", {k: v.to_dict() for k, v in thermostat.items()})

    rows = []
    for spec in production_matrix():
        started = time.perf_counter()
        record = {"spec": spec.to_dict(), "status": "running"}
        try:
            trajectory = load_case(config, spec.case_name)
            hpo_dir = layout.hpo_root / "micro_qualification" / campaign_id / spec.configuration_id
            hpo_cfg = HPOConfig(
                train_percentage=hpo_percentage, holdout_percentage=20.0,
                objective="recursive_temperature_normalized", n_trials=1,
                max_epochs_per_trial=1, patience=1, seed=seed,
                max_batch_windows=2,
                # 0.5% micro qualification is a plumbing test.  Cap only the
                # rollout search geometry so a floored 20% holdout is legal.
                max_rollout_steps=3,
                max_encoder_history_steps=12,
            )
            _, frozen, sample = run_persistent_hpo(trajectory, spec, output_dir=hpo_dir, config=hpo_cfg)
            full_train = np.flatnonzero(trajectory.mask("train", included_only=True))
            model, _ = build_paper_model(spec.method, trajectory, rc_order=spec.rc_order, train_indices=full_train, hyperparameters=frozen, seed=seed)
            train_segments = contiguous_segments(trajectory.timestamp, trajectory.partition, trajectory.included, partition_name="train", dt_seconds=300.0)
            val_segments = contiguous_segments(trajectory.timestamp, trajectory.partition, trajectory.included, partition_name="validation", dt_seconds=300.0)
            N_r = int(getattr(model.config, "N_r", 1)); L_e = int(getattr(model.config, "L_e", 1)) if spec.rc_order == 2 and spec.method != "inverse_pinn" else 1
            fit_windows = build_rollout_windows(train_segments, partition="train", N_r=N_r, L_e=L_e, is_2c=spec.rc_order == 2)[:4]
            val_windows = build_rollout_windows(val_segments, partition="validation", N_r=N_r, L_e=L_e, is_2c=spec.rc_order == 2)[:4]
            if not val_windows:
                raise RuntimeError("No validation windows")
            outcome = fit_model(model, trajectory, fit_indices=full_train[: min(256, len(full_train))], fit_windows=fit_windows, validation_windows=val_windows,
                                objective="recursive_temperature_normalized", optimization=OptimizationConfig(learning_rate=float(frozen.values.get("learning_rate",1e-3)), optimizer=str(frozen.values.get("optimizer","adam")), max_epochs=1, patience=1, seed=seed), batch_size=2)
            run_id = stable_id("train", {"spec": spec.to_dict(), "seed": seed, "hp": frozen.values})
            ckpt_dir = layout.checkpoint_root / "micro_qualification" / campaign_id / run_id
            ckpt = ckpt_dir / "model.pt"
            save_candidate(ckpt, model, {"run_id": run_id, "spec": spec.to_dict(), "frozen": frozen.values, "training": outcome.to_dict()})
            phase_c = _phase_c_for_case(config, trajectory)
            cal = {z: thermostat[z] for z in trajectory.zone_ids}
            evals, outputs = run_offline_evaluations(
                model, trajectory, phase_c, cal,
                offline_root=layout.offline_root / "micro_qualification" / campaign_id,
                run_id=run_id, micro=True,
                controller_overrides=controller_overrides,
            )
            gates = acceptance_gates(model, trajectory, ckpt, sim1_smoke=evals[0], sim2_smoke=evals[1])
            status = "accepted" if gates.get("accepted") else "rejected"
            append_registry(layout.checkpoint_root / "registry.jsonl", CheckpointRecord(run_id, run_id, spec.method, spec.case_name, spec.rc_order, seed, str(ckpt), status, gates, {"micro_campaign": campaign_id}))
            record.update({"status": status, "run_id": run_id, "gates": gates, "outputs": outputs})
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            if not continue_on_error:
                rows.append(record); break
        record["wall_seconds"] = float(time.perf_counter() - started)
        rows.append(record)
        write_json(root / "progress.json", {"campaign_id": campaign_id, "runs": rows})
    summary = {
        "campaign_id": campaign_id,
        "expected_configurations": 32,
        "completed": len(rows),
        "accepted": sum(r["status"] == "accepted" for r in rows),
        "rejected": sum(r["status"] == "rejected" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "environment": environment_manifest(),
        "runs": rows,
    }
    write_json(root / "summary.json", summary)
    return summary



def run_production_campaign(
    config: PaperConfig,
    *,
    priorities=("A", "B", "C"),
    seeds=(0,),
    hpo_config: HPOConfig | None = None,
    training_config: ProductionTrainingConfig | None = None,
    continue_on_error: bool = True,
    controller_overrides: ControllerOverrideConfig | None = None,
) -> dict[str, object]:
    """Run HPO -> multi-start training -> checkpoint selection -> Sim1/2/3.

    HPO is performed once per scientific configuration and is persistent.  Each
    configured seed is then a stochastic training restart of the same frozen
    hyperparameter configuration.  Phase-D VALIDATION selects the best accepted
    restart; final Sim1/2/3 use TEST only.  Individual failures are recorded and
    do not abort the matrix unless ``continue_on_error`` is false.
    """
    layout = resolve_production_layout(config, create=True)
    hpo_config = hpo_config or HPOConfig()
    training_config = training_config or ProductionTrainingConfig()
    controller_overrides = controller_overrides or load_controller_override_config()
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("At least one training restart seed is required")

    paper_root = Path(__file__).resolve().parents[3]
    source_hash = canonical_source_hash(paper_root)
    campaign_payload = {
        "campaign_id": config.campaign_id,
        "priorities": list(priorities),
        "seeds": list(seeds),
        "hpo": hpo_config.to_dict(),
        "training": training_config.to_dict(),
        "controller_overrides": controller_overrides.to_dict(),
        "source_manifest_sha256": source_hash,
    }
    production_id = stable_id("production", campaign_payload)
    manifest_dir = layout.manifest_root / "production_campaigns" / production_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / "campaign_manifest.json", campaign_payload | {
        "production_id": production_id,
        "paths": layout.to_dict(),
        "environment": environment_manifest(),
    })

    # One physical calibration for aggregate and one each for Dining/Kitchen.
    # Data-derived 300-s calibration is default; user half-width overrides are
    # applied only when explicitly configured.
    write_json(manifest_dir / "controller_override_config.json", controller_overrides.to_dict())
    thermostat = calibrate_controlled_thermostats(
        config,
        deadband_overrides_C=controller_overrides.deadband_overrides_C(),
        heating_mode_deadband_overrides_C=controller_overrides.heating_mode_deadband_overrides_C(),
    )
    write_json(
        manifest_dir / "thermostat_calibration_300s.json",
        {k: v.to_dict() for k, v in thermostat.items()},
    )

    config_records: list[dict[str, object]] = []
    for spec in production_matrix(priorities=priorities):
        spec_started = time.perf_counter()
        spec_record: dict[str, object] = {"spec": spec.to_dict(), "status": "running", "restarts": []}
        try:
            trajectory = load_case(config, spec.case_name)
            phase_c = _phase_c_for_case(config, trajectory)
            cal = {z: thermostat[z] for z in trajectory.zone_ids}

            hpo_dir = layout.hpo_root / "studies" / spec.configuration_id / hpo_protocol_id(spec, hpo_config)
            _, frozen, _ = run_persistent_hpo(
                trajectory, spec, output_dir=hpo_dir, config=hpo_config,
            )
            spec_record["frozen_hyperparameters"] = str(hpo_dir / "frozen_hyperparameters.json")

            full_train = np.flatnonzero(trajectory.mask("train", included_only=True))
            train_segments = contiguous_segments(
                trajectory.timestamp, trajectory.partition, trajectory.included,
                partition_name="train", dt_seconds=300.0,
            )
            val_segments = contiguous_segments(
                trajectory.timestamp, trajectory.partition, trajectory.included,
                partition_name="validation", dt_seconds=300.0,
            )

            accepted_restarts: list[tuple[float, int, object, str, Path]] = []
            for seed in seeds:
                restart_started = time.perf_counter()
                run_id = stable_id("train", {
                    "spec": spec.to_dict(), "seed": seed,
                    "frozen": frozen.values,
                    "source_manifest_sha256": source_hash,
                })
                train_dir = layout.training_root / spec.configuration_id / run_id
                run_manifest_path = train_dir / "run_manifest.json"
                rr: dict[str, object] = {"seed": seed, "run_id": run_id, "status": "running"}
                try:
                    model, _ = build_paper_model(
                        spec.method, trajectory, rc_order=spec.rc_order,
                        train_indices=full_train, hyperparameters=frozen, seed=seed,
                    )
                    if run_manifest_path.is_file():
                        try:
                            existing = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                        except Exception:
                            existing = {}
                        if existing.get("status") == "accepted" and existing.get("checkpoint"):
                            existing_ckpt = Path(existing["checkpoint"])
                            if existing_ckpt.is_file():
                                load_checkpoint(existing_ckpt, model=model)
                                score = float(existing["validation_score"])
                                accepted_restarts.append((score, seed, model, run_id, existing_ckpt))
                                rr.update({
                                    "status": "resumed_accepted",
                                    "checkpoint_id": existing.get("checkpoint_id"),
                                    "checkpoint": str(existing_ckpt),
                                    "validation_score": score,
                                    "resumed": True,
                                })
                                rr["wall_seconds"] = float(time.perf_counter() - restart_started)
                                spec_record["restarts"].append(rr)
                                write_json(
                                    manifest_dir / "progress.json",
                                    {"production_id": production_id, "configurations": config_records + [spec_record]},
                                )
                                continue
                    N_r = int(getattr(model.config, "N_r", 1))
                    L_e = int(getattr(model.config, "L_e", 1)) if spec.rc_order == 2 and spec.method != "inverse_pinn" else 1
                    train_windows = build_rollout_windows(
                        train_segments, partition="train", N_r=N_r, L_e=L_e,
                        is_2c=spec.rc_order == 2,
                    )
                    validation_windows = build_rollout_windows(
                        val_segments, partition="validation", N_r=N_r, L_e=L_e,
                        is_2c=spec.rc_order == 2,
                    )
                    validation_windows = representative_window_subset(
                        validation_windows,
                        max_windows=max(1, int(training_config.validation_max_windows)),
                        seed=seed,
                    )
                    if not train_windows or not validation_windows:
                        raise RuntimeError("No legal production TRAIN/VALIDATION windows")

                    opt = OptimizationConfig(
                        learning_rate=float(frozen.values.get("learning_rate", 1e-3)),
                        optimizer=str(frozen.values.get("optimizer", "adam")),
                        max_epochs=int(training_config.max_epochs),
                        patience=int(training_config.patience),
                        gradient_clip_norm=training_config.gradient_clip_norm,
                        seed=seed,
                    )
                    outcome = fit_model(
                        model, trajectory,
                        fit_indices=full_train,
                        fit_windows=train_windows,
                        validation_windows=validation_windows,
                        objective=hpo_config.objective,
                        optimization=opt,
                        batch_size=min(
                            int(frozen.values.get("batch_size", training_config.max_batch_windows)),
                            int(training_config.max_batch_windows),
                        ),
                    )
                    train_dir.mkdir(parents=True, exist_ok=True)
                    write_json(train_dir / "training_summary.json", outcome.to_dict())
                    write_json(train_dir / "run_manifest.json", {
                        "run_id": run_id, "spec": spec.to_dict(), "seed": seed,
                        "phase_d_source_manifests": list(trajectory.manifests),
                        "frozen_hyperparameters": frozen.values,
                        "hpo_study": str(hpo_dir / "study.db"),
                        "source_manifest_sha256": source_hash,
                        "environment": environment_manifest(),
                        "status": "candidate",
                    })

                    checkpoint_id = stable_id("ckpt", {"run_id": run_id, "validation": outcome.best_validation_score})
                    ckpt_dir = layout.checkpoint_root / "by_id" / checkpoint_id
                    ckpt = ckpt_dir / "model.pt"
                    save_candidate(ckpt, model, {
                        "checkpoint_id": checkpoint_id,
                        "run_id": run_id,
                        "spec": spec.to_dict(),
                        "seed": seed,
                        "frozen_hyperparameters": frozen.values,
                        "validation_objective": hpo_config.objective,
                        "validation_score": outcome.best_validation_score,
                        "source_manifest_sha256": source_hash,
                    })

                    # Acceptance is a functionality/integrity gate only; short
                    # TEST smokes never choose among restarts.
                    runtime = PaperModelRuntime(model, trajectory)
                    from ..evaluation.runtime import sim1, sim2
                    sim1_smoke = sim1(runtime, phase_c, max_points=12)
                    sim2_smoke = sim2(runtime, phase_c, horizon=12, all_test_segments=False)
                    gates = acceptance_gates(
                        model, trajectory, ckpt,
                        sim1_smoke=sim1_smoke, sim2_smoke=sim2_smoke,
                    )
                    status = "accepted" if gates.get("accepted") else "rejected"
                    record = CheckpointRecord(
                        checkpoint_id=checkpoint_id, run_id=run_id,
                        method=spec.method, case_name=spec.case_name,
                        rc_order=spec.rc_order, seed=seed,
                        checkpoint_path=str(ckpt), status=status,
                        acceptance=gates,
                        provenance={
                            "production_id": production_id,
                            "source_manifest_sha256": source_hash,
                            "validation_score": outcome.best_validation_score,
                        },
                    )
                    append_registry(layout.checkpoint_root / "registry.jsonl", record)
                    write_json(ckpt_dir / "acceptance.json", gates)
                    write_json(run_manifest_path, {
                        "run_id": run_id, "spec": spec.to_dict(), "seed": seed,
                        "phase_d_source_manifests": list(trajectory.manifests),
                        "frozen_hyperparameters": frozen.values,
                        "hpo_study": str(hpo_dir / "study.db"),
                        "source_manifest_sha256": source_hash,
                        "environment": environment_manifest(),
                        "status": status,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint": str(ckpt),
                        "validation_score": outcome.best_validation_score,
                        "acceptance": gates,
                    })
                    rr.update({
                        "status": status, "run_id": run_id,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint": str(ckpt),
                        "validation_score": outcome.best_validation_score,
                        "acceptance": gates,
                    })
                    if status == "accepted":
                        accepted_restarts.append((float(outcome.best_validation_score), seed, model, run_id, ckpt))
                except Exception as exc:
                    rr.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
                    if not continue_on_error:
                        raise
                rr["wall_seconds"] = float(time.perf_counter() - restart_started)
                spec_record["restarts"].append(rr)
                write_json(manifest_dir / "progress.json", {"production_id": production_id, "configurations": config_records + [spec_record]})

            if not accepted_restarts:
                spec_record["status"] = "failed_no_accepted_restart"
            else:
                accepted_restarts.sort(key=lambda x: (x[0], x[1]))
                best_score, best_seed, best_model, best_run_id, best_ckpt = accepted_restarts[0]
                evals, outputs = run_offline_evaluations(
                    best_model, trajectory, phase_c, cal,
                    offline_root=layout.offline_root,
                    run_id=best_run_id,
                    micro=False,
                    controller_overrides=controller_overrides,
                )
                spec_record.update({
                    "status": "completed",
                    "selected_seed": best_seed,
                    "selected_validation_score": best_score,
                    "selected_checkpoint": str(best_ckpt),
                    "selected_run_id": best_run_id,
                    "offline_outputs": outputs,
                })
        except Exception as exc:
            spec_record.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            if not continue_on_error:
                config_records.append(spec_record)
                break
        spec_record["wall_seconds"] = float(time.perf_counter() - spec_started)
        config_records.append(spec_record)
        write_json(manifest_dir / "progress.json", {"production_id": production_id, "configurations": config_records})

    summary = {
        "production_id": production_id,
        "expected_configurations": len(production_matrix(priorities=priorities)),
        "completed_configurations": sum(r.get("status") == "completed" for r in config_records),
        "failed_configurations": sum(str(r.get("status", "")).startswith("failed") for r in config_records),
        "configurations": config_records,
        "source_manifest_sha256": source_hash,
        "environment": environment_manifest(),
    }
    write_json(manifest_dir / "summary.json", summary)
    return summary
