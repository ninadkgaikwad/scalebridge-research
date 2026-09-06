from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
import torch

from ..methods.base_pinode import BasePINODEConfig, BasePINODEModel
from ..core.common import contiguous_segments, write_json
from ..data.phase_d import PhaseDTrajectory
from ..methods.ebp_pinode import EBPPINODEConfig, EBPPINODEModel
from ..evaluation.runtime import EvaluationResult
from ..methods.inverse_pinn import InversePINNConfig, InversePINNRC
from ..data.method_data import inverse_pinn_forcing, node_method_arrays
from ..methods.neural_ode import NeuralODEConfig, NeuralODEModel
from ..core.paths import resolve_paper_data_root
from .trainer import (
    FrozenHyperparameters, OptimizationConfig, optimize_steps,
    suggest_base_pinode_hyperparameters, suggest_ebp_pinode_hyperparameters,
    suggest_inverse_pinn_hyperparameters, suggest_node_hyperparameters,
)

MethodName = Literal["inverse_pinn", "neural_ode", "base_pinode", "ebp_pinode"]


def suggest_method_hyperparameters(method: MethodName, trial, *, rc_order: int) -> dict[str, Any]:
    if method == "inverse_pinn": return suggest_inverse_pinn_hyperparameters(trial)
    if method == "neural_ode": return suggest_node_hyperparameters(trial, rc_order=rc_order)
    if method == "base_pinode": return suggest_base_pinode_hyperparameters(trial, rc_order=rc_order)
    if method == "ebp_pinode": return suggest_ebp_pinode_hyperparameters(trial, rc_order=rc_order)
    raise ValueError(method)


def longest_training_prefix(trajectory: PhaseDTrajectory, *, max_rows: int) -> np.ndarray:
    segs = contiguous_segments(trajectory.timestamp, trajectory.partition, trajectory.included,
                               partition_name="train", dt_seconds=300.0)
    if not segs: raise ValueError("no contiguous training segment")
    seg = max(segs, key=len)
    return np.asarray(seg[:min(max_rows, len(seg))], dtype=int)


def _hyperparameter_values(hyperparameters: Mapping[str, Any] | FrozenHyperparameters | None) -> dict[str, Any]:
    if hyperparameters is None:
        return {}
    if isinstance(hyperparameters, FrozenHyperparameters):
        return dict(hyperparameters.values)
    return dict(hyperparameters)


def build_paper_model(method: MethodName, trajectory: PhaseDTrajectory, *, rc_order: int,
                      train_indices: np.ndarray, hidden_layers: int=1, hidden_width: int=8,
                      N_r: int=2, L_e: int=3, N_s: int=1, seed: int=42,
                      hyperparameters: Mapping[str, Any] | FrozenHyperparameters | None = None):
    """Construct one validated paper method from explicit/frozen hyperparameters.

    Legacy keyword defaults remain for historical smoke tests.  Production code
    passes a mapping or ``FrozenHyperparameters``; every searched mathematical
    parameter is then propagated into the actual method config rather than
    merely being serialized by Optuna.
    """
    hp = _hyperparameter_values(hyperparameters)
    hidden_layers = int(hp.get("hidden_layers", hidden_layers))
    hidden_width = int(hp.get("hidden_width", hidden_width))
    activation = str(hp.get("activation", "tanh"))
    N_r = int(hp.get("N_r", N_r))
    L_e = int(hp.get("L_e", L_e))
    N_s = int(hp.get("N_s", N_s))
    delta_T_m_max = float(hp.get("delta_T_m_max", 8.0))
    lambda_wd = float(hp.get("lambda_wd", 0.0))

    arrays = node_method_arrays(trajectory, row_indices=train_indices)
    if method == "inverse_pinn":
        t0 = trajectory.timestamp.iloc[int(train_indices[0])]
        t_seconds = np.asarray([(trajectory.timestamp.iloc[int(i)]-t0).total_seconds() for i in train_indices], float)
        model = InversePINNRC(
            InversePINNConfig(
                case_name=trajectory.case_name, rc_order=rc_order,
                hidden_layers=hidden_layers, hidden_width=hidden_width,
                activation=activation, lambda_y=1.0,
                lambda_f=float(hp.get("lambda_f", .1)), seed=seed,
            ),
            y_training=arrays.y, t_training_seconds=t_seconds,
        )
        aux = {"arrays":arrays, "t_seconds":t_seconds,
               "forcing":inverse_pinn_forcing(trajectory,row_indices=train_indices)}
        return model, aux

    common = dict(
        case_name=trajectory.case_name, rc_order=rc_order,
        hidden_layers=hidden_layers, hidden_width=hidden_width,
        activation=activation, N_r=N_r, L_e=L_e, N_s=N_s,
        delta_T_m_max=delta_T_m_max, lambda_wd=lambda_wd, seed=seed,
    )
    if method == "neural_ode":
        cfg = NeuralODEConfig(**common); cls = NeuralODEModel
    elif method == "base_pinode":
        cfg = BasePINODEConfig(
            **common, lambda_y=1.0,
            lambda_f=float(hp.get("lambda_f", .1)), dt_seconds=300.0,
        ); cls = BasePINODEModel
    elif method == "ebp_pinode":
        cfg = EBPPINODEConfig(
            **common, lambda_y=1.0,
            lambda_f=0.0,
            lambda_int=float(hp.get("lambda_int", .1 if rc_order == 2 else 0.0)),
            lambda_corr=float(hp.get("lambda_corr", 1e-8)),
            dt_seconds=300.0,
        ); cls = EBPPINODEModel
    else:
        raise ValueError(method)
    model = cls(cfg, y_training=arrays.y, v_training=arrays.v,
                y_names=arrays.y_names, v_names=arrays.v_names)
    return model, {"arrays": arrays, "hyperparameters_applied": hp}


def _node_loss(model, arrays, rc_order: int):
    N_r=model.config.N_r
    if rc_order==1: k=0; cy=cv=None
    else:
        k=model.config.L_e-1; cy=torch.tensor(arrays.y[k-model.config.L_e+1:k+1],dtype=torch.float64)
        if isinstance(arrays.v,Mapping): cv={key:torch.tensor(v[k-model.config.L_e+1:k+1],dtype=torch.float64) for key,v in arrays.v.items()}
        else: cv=torch.tensor(arrays.v[k-model.config.L_e+1:k+1],dtype=torch.float64)
    yt=torch.tensor(arrays.y[k:k+N_r+1],dtype=torch.float64)
    if isinstance(arrays.v,Mapping): vs={key:torch.tensor(v[k:k+N_r],dtype=torch.float64) for key,v in arrays.v.items()}
    else: vs=torch.tensor(arrays.v[k:k+N_r],dtype=torch.float64)
    return model.rollout_loss(y_true=yt,v_sequence=vs,context_y=cy,context_v=cv)["total"]


def tiny_fit(model, aux: dict[str,Any], *, train_steps: int=1) -> list[float]:
    if isinstance(model,InversePINNRC):
        arrays=aux["arrays"]
        closure=lambda: model.loss(t_seconds=torch.tensor(aux["t_seconds"],dtype=torch.float64),
                                   y_measured=torch.tensor(arrays.y,dtype=torch.float64),forcing=aux["forcing"])["total"]
    else:
        closure=lambda: _node_loss(model,aux["arrays"],int(model.config.rc_order))
    return optimize_steps(model,closure,config=OptimizationConfig(learning_rate=1e-3,max_epochs=max(1,train_steps),patience=max(2,train_steps+1)),steps=train_steps)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    method: str
    case_name: str
    rc_order: int
    seed: int
    data_root: str
    frozen_hyperparameters: str | None = None
    checkpoint: str | None = None
    qac_artifacts: dict[str,str] | None = None
    phvac_artifacts: dict[str,str] | None = None
    thermostat_calibration: dict[str,str] | None = None
    status: str = "created"

    def save(self,path:Path)->None: write_json(path,asdict(self))


def save_evaluation_result(result: EvaluationResult, *, run_id: str, method: str,
                           root: str | Path | None=None) -> dict[str,str]:
    base=resolve_paper_data_root(root,create=True)/"evaluations"/run_id/result.simulation/method/result.case_name/f"{result.rc_order}C"
    base.mkdir(parents=True,exist_ok=True)
    trajectory_path=base/"trajectory.parquet"; metrics_path=base/"metrics.json"; provenance_path=base/"provenance.json"
    result.trajectory.to_parquet(trajectory_path,index=False)
    write_json(metrics_path,{"metrics":result.metrics})
    write_json(provenance_path,result.provenance)
    return {"trajectory":str(trajectory_path),"metrics":str(metrics_path),"provenance":str(provenance_path)}
