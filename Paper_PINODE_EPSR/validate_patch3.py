from __future__ import annotations

"""Day-4 Patch-03 validator for Part-4 Base PINODE.

The real-data stage is deliberately tiny.  Its purpose is contract/path coverage,
not convergence or paper-result accuracy.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch

from .base_pinode import BasePINODEConfig, BasePINODEModel
from .common import contiguous_segments, write_json
from .config import PaperConfig, canonical_case_specs
from .data import load_case
from .method_data import node_method_arrays
from .neuromancer_backend import runtime_info
from .phase_c import discover_and_load_phase_c_bundle
from .training import OptimizationConfig, optimize_steps


def _one_base_pinode_loss(model: BasePINODEModel, y: np.ndarray, v, rc_order: int):
    N_r = model.config.N_r
    if rc_order == 1:
        k = 0
        context_y = context_v = None
    else:
        k = model.config.L_e - 1
        context_y = torch.tensor(
            y[k - model.config.L_e + 1 : k + 1], dtype=torch.float64
        )
        if isinstance(v, Mapping):
            context_v = {
                key: torch.tensor(
                    value[k - model.config.L_e + 1 : k + 1], dtype=torch.float64
                )
                for key, value in v.items()
            }
        else:
            context_v = torch.tensor(
                v[k - model.config.L_e + 1 : k + 1], dtype=torch.float64
            )

    if k + N_r >= len(y):
        raise ValueError(
            f"tiny Base-PINODE window requires at least {k + N_r + 1} rows; got {len(y)}"
        )
    y_true = torch.tensor(y[k : k + N_r + 1], dtype=torch.float64)
    if isinstance(v, Mapping):
        vseq = {
            key: torch.tensor(value[k : k + N_r], dtype=torch.float64)
            for key, value in v.items()
        }
    else:
        vseq = torch.tensor(v[k : k + N_r], dtype=torch.float64)
    return model.rollout_loss(
        y_true=y_true,
        v_sequence=vseq,
        context_y=context_y,
        context_v=context_v,
    )["total"]


def _physical_summary(model: BasePINODEModel) -> dict[str, Any]:
    params = model.physical_parameters()
    serial = {name: float(value.detach().cpu()) for name, value in params.items()}
    positive_rc = all(
        value > 0.0 for name, value in serial.items() if name.startswith(("R_", "C_"))
    )
    finite = all(np.isfinite(value) for value in serial.values())
    out: dict[str, Any] = {
        "positive_rc": bool(positive_rc),
        "finite_physical_parameters": bool(finite),
        "physical_parameters": serial,
        "q_star_zone_W": model.q_star_zone.detach().cpu().tolist(),
    }
    if model.config.case_name == "identity_dep2":
        out["dep2_lambda_c_sum"] = serial["lambda_c_D"] + serial["lambda_c_K"]
        out["dep2_lambda_r_sum"] = serial["lambda_r_D"] + serial["lambda_r_K"]
        if model.config.rc_order == 1:
            out["dep2_1c_eta_locked_to_one"] = bool(
                np.isclose(serial["eta_r_D"], 1.0)
                and np.isclose(serial["eta_r_K"], 1.0)
            )
    return out


def real_data_smoke(
    config: PaperConfig, *, max_rows: int, train_steps: int
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cases": {},
        "phase_c_models": {},
        "neuromancer_runtime": runtime_info().__dict__,
        "validation_intent": "tiny path/contract coverage; not convergence or accuracy",
        "status": "running",
    }

    for case_name in canonical_case_specs():
        print(
            f"[real-data] case={case_name}: loading Phase-D ML_SciML trajectory...",
            flush=True,
        )
        trajectory = load_case(config, case_name)
        segments = contiguous_segments(
            trajectory.timestamp,
            trajectory.partition,
            trajectory.included,
            partition_name="train",
            dt_seconds=config.dt_seconds,
        )
        if not segments:
            raise RuntimeError(f"{case_name}: no contiguous training segments")
        segment = max(segments, key=len)
        if len(segment) < 8:
            raise RuntimeError(
                f"{case_name}: longest train segment is too short ({len(segment)})"
            )
        idx = segment[: min(max_rows, len(segment))]
        arrays = node_method_arrays(trajectory, row_indices=idx)

        case_report: dict[str, Any] = {
            "selected_train_rows": int(len(idx)),
            "rc_orders": {},
        }
        for rc_order in (1, 2):
            print(
                f"[real-data] case={case_name} rc_order={rc_order}: "
                "Base PINODE tiny optimization...",
                flush=True,
            )
            model = BasePINODEModel(
                BasePINODEConfig(
                    case_name=case_name,
                    rc_order=rc_order,
                    hidden_layers=1,
                    hidden_width=8,
                    activation="tanh",
                    N_r=2,
                    L_e=3,
                    N_s=1,
                    delta_T_m_max=8.0,
                    lambda_y=1.0,
                    lambda_f=0.1,
                    lambda_wd=0.0,
                    dt_seconds=config.dt_seconds,
                ),
                y_training=arrays.y,
                v_training=arrays.v,
                y_names=arrays.y_names,
                v_names=arrays.v_names,
            )
            history = optimize_steps(
                model,
                lambda: _one_base_pinode_loss(
                    model, arrays.y, arrays.v, rc_order
                ),
                config=OptimizationConfig(
                    learning_rate=1e-3,
                    max_epochs=max(1, train_steps),
                    patience=max(2, train_steps + 1),
                ),
                steps=train_steps,
            )
            expected_stage_count = model.config.N_r * model.config.N_s * 4
            if model.stage_residual_count != expected_stage_count:
                raise RuntimeError(
                    f"{case_name}/{rc_order}C: expected {expected_stage_count} "
                    f"Neuromancer RK4 stage residuals, got {model.stage_residual_count}"
                )
            if not all(np.isfinite(value) for value in history):
                raise FloatingPointError(
                    f"{case_name}/{rc_order}C: non-finite Base-PINODE loss history"
                )
            summary = _physical_summary(model)
            if not summary["positive_rc"] or not summary["finite_physical_parameters"]:
                raise FloatingPointError(
                    f"{case_name}/{rc_order}C: invalid physical parameters"
                )
            provenance = model.provenance()
            if provenance["physics"]["hard_projection"]:
                raise RuntimeError("Base PINODE must not hard-project its derivative")
            case_report["rc_orders"][str(rc_order)] = {
                "base_pinode_loss": history,
                "expected_rk4_stage_count": expected_stage_count,
                "observed_rk4_stage_count": model.stage_residual_count,
                "forcing_dimensions": provenance["forcing_dimensions"],
                "integrated_derivative": provenance["physics"]["integrated_derivative"],
                "constraint_type": provenance["physics"]["constraint_type"],
                **summary,
            }
        report["cases"][case_name] = case_report

    # Retain the Day-3 downstream integration guard.  These models are not thermal
    # RHS inputs in Patch 03; they are loaded now so the paper-specific pipeline
    # continues to prove compatibility with the later Sim3/MPC evaluation stack.
    for zone in ("RestaurantFastFood_All", "Dining", "Kitchen"):
        print(
            f"[real-data] Phase-C zone={zone}: loading persisted QAC + PHVAC models...",
            flush=True,
        )
        bundle = discover_and_load_phase_c_bundle(
            config,
            zone,
            phase_c_run_id=config.controlled_phase_c_run_id,
        )
        proxy = np.array([-5000.0, 0.0, 5000.0], dtype=float)
        qac = np.asarray(bundle.predict_qac_from_hvac_proxy(proxy), dtype=float)
        phvac = np.asarray(bundle.predict_phvac_from_qac(qac), dtype=float)
        if not np.isfinite(qac).all() or not np.isfinite(phvac).all():
            raise FloatingPointError(f"{zone}: non-finite Phase-C model predictions")
        report["phase_c_models"][zone] = {
            "provenance": bundle.provenance,
            "qac_predictions_W": qac.tolist(),
            "phvac_predictions_W": phvac.tolist(),
        }

    report["status"] = "passed"
    return report


def run_pytest_contracts() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "Paper_PINODE_EPSR/tests/test_patch1_contract.py",
        "Paper_PINODE_EPSR/tests/test_patch2_contract.py",
        "Paper_PINODE_EPSR/tests/test_patch3_contract.py",
    ]
    print("[unit-contracts] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=repo_root, check=False)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PINODE/EPSR Day-4 Patch 03 Base PINODE"
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Also run the tiny controlled RestaurantFastFood/Buffalo Base-PINODE matrix",
    )
    parser.add_argument(
        "--skip-unit-contracts",
        action="store_true",
        help="Skip pytest only when the complete Patch-03 contract suite already passed",
    )
    parser.add_argument("--max-rows", type=int, default=48)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument(
        "--output",
        default="Paper_PINODE_EPSR/results/patch03_validation_real.json",
    )
    return parser.parse_args()


def _failure_payload(
    exc: BaseException, *, nm: Any | None, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "real_data_requested": bool(args.real_data),
        "framework_runtime": None if nm is None else nm.__dict__,
    }


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    nm = None
    payload: dict[str, Any] = {}
    try:
        print("=" * 100, flush=True)
        print("PINODE / EPSR PATCH 03 BASE-PINODE VALIDATION", flush=True)
        print("=" * 100, flush=True)
        nm = runtime_info()
        print(f"Neuromancer version: {nm.version}", flush=True)
        print(f"Neuromancer RK4: {nm.rk4_class}", flush=True)
        print(f"Neuromancer Node: {nm.node_class}", flush=True)
        print(f"Neuromancer System: {nm.system_class}", flush=True)
        print(f"Neuromancer MLP: {nm.mlp_class}", flush=True)
        print(f"Neuromancer Problem: {nm.problem_class}", flush=True)
        print(f"Neuromancer PenaltyLoss: {nm.penalty_loss_class}", flush=True)
        if not nm.rk4_class.startswith("neuromancer."):
            raise RuntimeError(f"Unexpected RK4 provider: {nm.rk4_class}")

        if args.skip_unit_contracts:
            print("[unit-contracts] SKIPPED by explicit CLI request.", flush=True)
            unit_status = "skipped_by_request"
        else:
            print(
                "[unit-contracts] Running Patch-1 + Patch-2A + Patch-3 suite "
                "in isolated interpreter...",
                flush=True,
            )
            rc = run_pytest_contracts()
            if rc != 0:
                raise RuntimeError(
                    f"contract pytest subprocess failed with exit code {rc}"
                )
            unit_status = "passed"

        payload = {
            "status": "running",
            "unit_contracts": unit_status,
            "framework_contract": {
                "sciml": "neuromancer",
                "tensor_autograd_optimizer": "pytorch",
                "hyperparameter_search": "optuna",
                "integration": "neuromancer.dynamics.integrators.RK4",
                "recursive_graph": "neuromancer.system.Node+System",
                "named_data_contract": "neuromancer.dataset.DictDataset + dataset.collate_fn",
                "training_loss_key": "train_loss",
                "custom_rk4_in_paper_code": False,
                "direct_torchdiffeq_calls_in_paper_code": False,
                "runtime": nm.__dict__,
            },
            "base_pinode_contract": {
                "integrated_derivative": "raw_f_tilde_omega",
                "physics": "soft_full_RC_residual",
                "hard_projection": False,
                "physics_evaluation": "every_Neuromancer_RK4_RHS_call",
            },
            "real_data_requested": bool(args.real_data),
            "math_contracts": [
                "PINODE_EPSR_Part1_RC_Representations_v1.tex",
                "PINODE_EPSR_Part3_NeuralODE_Detailed.tex",
                "PINODE_EPSR_Part4_Base_PINODE_Detailed.tex",
            ],
        }

        if args.real_data:
            print("[real-data] Starting tiny Base-PINODE matrix...", flush=True)
            config = PaperConfig.from_environment()
            payload["real_data"] = real_data_smoke(
                config,
                max_rows=args.max_rows,
                train_steps=args.train_steps,
            )

        payload["status"] = "passed"
        write_json(output, payload)
        print(f"\nValidation report: {output}", flush=True)
        print("PATCH 03 STATUS: PASSED", flush=True)
        return 0
    except BaseException as exc:
        payload = _failure_payload(exc, nm=nm, args=args)
        try:
            write_json(output, payload)
            print(f"\nFailure report written: {output}", flush=True)
        except Exception as write_exc:
            print(f"Failed to write validation JSON: {write_exc}", flush=True)
        traceback.print_exc()
        print("PATCH 03 STATUS: FAILED", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
