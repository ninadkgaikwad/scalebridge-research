from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch

from .common import contiguous_segments, write_json
from .config import PaperConfig, canonical_case_specs
from .data import load_case
from .inverse_pinn import InversePINNConfig, InversePINNRC
from .method_data import inverse_pinn_forcing, node_method_arrays
from .neural_ode import NeuralODEConfig, NeuralODEModel
from .neuromancer_backend import runtime_info
from .phase_c import discover_and_load_phase_c_bundle
from .training import OptimizationConfig, optimize_steps


def _to_tensor_v(v):
    if isinstance(v, Mapping):
        return {k: torch.tensor(value, dtype=torch.float64) for k, value in v.items()}
    return torch.tensor(v, dtype=torch.float64)


def _one_node_loss(model: NeuralODEModel, y: np.ndarray, v, rc_order: int):
    N_r = model.config.N_r
    if rc_order == 1:
        k = 0
        context_y = context_v = None
    else:
        k = model.config.L_e - 1
        context_y = torch.tensor(y[k - model.config.L_e + 1 : k + 1], dtype=torch.float64)
        if isinstance(v, Mapping):
            context_v = {key: torch.tensor(value[k - model.config.L_e + 1 : k + 1], dtype=torch.float64) for key, value in v.items()}
        else:
            context_v = torch.tensor(v[k - model.config.L_e + 1 : k + 1], dtype=torch.float64)
    y_true = torch.tensor(y[k : k + N_r + 1], dtype=torch.float64)
    if isinstance(v, Mapping):
        vseq = {key: torch.tensor(value[k : k + N_r], dtype=torch.float64) for key, value in v.items()}
    else:
        vseq = torch.tensor(v[k : k + N_r], dtype=torch.float64)
    return model.rollout_loss(y_true=y_true, v_sequence=vseq, context_y=context_y, context_v=context_v)["total"]


def real_data_smoke(config: PaperConfig, *, max_rows: int, train_steps: int) -> dict[str, Any]:
    """Small controlled RestaurantFastFood/Buffalo validation across all Patch-2 options."""

    report: dict[str, Any] = {
        "cases": {},
        "phase_c_models": {},
        "neuromancer_runtime": runtime_info().__dict__,
        "status": "running",
    }
    for case_name in canonical_case_specs():
        print(f"[real-data] case={case_name}: loading Phase-D ML_SciML trajectory...", flush=True)
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
            raise RuntimeError(f"{case_name}: longest train segment is too short ({len(segment)})")
        idx = segment[: min(max_rows, len(segment))]
        arrays = node_method_arrays(trajectory, row_indices=idx)
        forcing = inverse_pinn_forcing(trajectory, row_indices=idx)
        t0 = trajectory.timestamp.iloc[int(idx[0])]
        t_seconds = np.asarray([(trajectory.timestamp.iloc[int(i)] - t0).total_seconds() for i in idx], dtype=float)

        case_report: dict[str, Any] = {"selected_train_rows": int(len(idx)), "rc_orders": {}}
        for rc_order in (1, 2):
            print(
                f"[real-data] case={case_name} rc_order={rc_order}: "
                "Inverse PINN-RC tiny optimization...",
                flush=True,
            )
            inv = InversePINNRC(
                InversePINNConfig(
                    case_name=case_name,
                    rc_order=rc_order,
                    hidden_layers=1,
                    hidden_width=8,
                    lambda_y=1.0,
                    lambda_f=0.1,
                ),
                y_training=arrays.y,
                t_training_seconds=t_seconds,
            )
            y_t = torch.tensor(arrays.y, dtype=torch.float64)
            t_t = torch.tensor(t_seconds, dtype=torch.float64)
            inv_history = optimize_steps(
                inv,
                lambda: inv.loss(t_seconds=t_t, y_measured=y_t, forcing=forcing)["total"],
                config=OptimizationConfig(learning_rate=1e-3, max_epochs=max(1, train_steps), patience=max(2, train_steps + 1)),
                steps=train_steps,
            )

            print(
                f"[real-data] case={case_name} rc_order={rc_order}: "
                "Neural ODE tiny optimization...",
                flush=True,
            )
            node_cfg = NeuralODEConfig(
                case_name=case_name,
                rc_order=rc_order,
                hidden_layers=1,
                hidden_width=8,
                N_r=2,
                L_e=3,
                N_s=1,
                delta_T_m_max=8.0,
            )
            node = NeuralODEModel(
                node_cfg,
                y_training=arrays.y,
                v_training=arrays.v,
                y_names=arrays.y_names,
                v_names=arrays.v_names,
            )
            node_history = optimize_steps(
                node,
                lambda: _one_node_loss(node, arrays.y, arrays.v, rc_order),
                config=OptimizationConfig(learning_rate=1e-3, max_epochs=max(1, train_steps), patience=max(2, train_steps + 1)),
                steps=train_steps,
            )
            case_report["rc_orders"][str(rc_order)] = {
                "inverse_pinn_loss": inv_history,
                "inverse_pinn_positive_rc": all(
                    float(value.detach()) > 0.0
                    for name, value in inv.physical_parameters().items()
                    if name.startswith(("R_", "C_"))
                ),
                "node_loss": node_history,
                "node_forcing_dimensions": node.provenance()["forcing_dimensions"],
            }
        report["cases"][case_name] = case_report

    # Required real Phase-C model loading test. No hard-coded fallback is allowed here.
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
    """Run contract tests in an isolated Python process.

    In-process pytest invocation can retain plugin/framework state inside the
    validation process.  An isolated interpreter matches normal command-line
    pytest use and prevents unit-test teardown from contaminating the following
    real-data Neuromancer smoke stage.
    """

    repo_root = Path(__file__).resolve().parent.parent
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "Paper_PINODE_EPSR/tests/test_patch1_contract.py",
        "Paper_PINODE_EPSR/tests/test_patch2_contract.py",
    ]
    print("[unit-contracts] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=repo_root, check=False)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PINODE/EPSR Day-3 Patch 02")
    parser.add_argument("--real-data", action="store_true", help="Also run the controlled RestaurantFastFood/Buffalo real-data smoke matrix")
    parser.add_argument(
        "--skip-unit-contracts",
        action="store_true",
        help="Skip pytest only when the same Patch-02A contract suite has already passed in this environment",
    )
    parser.add_argument("--max-rows", type=int, default=48, help="Maximum contiguous training rows per real-data case")
    parser.add_argument("--train-steps", type=int, default=2, help="Tiny optimizer steps per method/case/order in real smoke")
    parser.add_argument("--output", default="Paper_PINODE_EPSR/results/patch02_validation.json")
    return parser.parse_args()


def _failure_payload(exc: BaseException, *, nm: Any | None, args: argparse.Namespace) -> dict[str, Any]:
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
        print("PINODE / EPSR PATCH 02A VALIDATION", flush=True)
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
            print("[unit-contracts] Running Patch-1 + Patch-2A suite in isolated interpreter...", flush=True)
            rc = run_pytest_contracts()
            if rc != 0:
                raise RuntimeError(f"contract pytest subprocess failed with exit code {rc}")
            unit_status = "passed"

        payload = {
            "status": "running",
            "unit_contracts": unit_status,
            "framework_contract": {
                "sciml": "neuromancer",
                "tensor_autograd_optimizer": "pytorch",
                "hyperparameter_search": "optuna",
                "integration": "neuromancer.dynamics.integrators.RK4",
                "named_data_contract": "neuromancer.dataset.DictDataset + dataset.collate_fn",
                "training_loss_key": "train_loss",
                "custom_rk4_in_paper_code": False,
                "direct_torchdiffeq_calls_in_paper_code": False,
                "runtime": nm.__dict__,
            },
            "real_data_requested": bool(args.real_data),
            "math_contracts": [
                "PINODE_EPSR_Part1_RC_Representations_v1.tex",
                "PINODE_EPSR_Part2_Inverse_PINN_RC.tex",
                "PINODE_EPSR_Part3_NeuralODE_Detailed.tex",
            ],
        }

        if args.real_data:
            print("[real-data] Starting controlled matrix...", flush=True)
            config = PaperConfig.from_environment()
            payload["real_data"] = real_data_smoke(
                config, max_rows=args.max_rows, train_steps=args.train_steps
            )
        else:
            payload["real_data"] = "not_requested"

        payload["status"] = "passed"
        write_json(output, payload)
        print(f"\nValidation report: {output}", flush=True)
        print("PATCH 02A STATUS: PASSED", flush=True)
        return 0

    except BaseException as exc:
        failure = _failure_payload(exc, nm=nm, args=args)
        if payload:
            failure["partial_report"] = payload
        try:
            write_json(output, failure)
            print(f"\nFailure report: {output}", flush=True)
        except Exception as write_exc:
            print(f"Failed to write validation JSON: {write_exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("PATCH 02A STATUS: FAILED", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
