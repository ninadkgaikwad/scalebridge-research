from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from Paper_PINODE_EPSR.base_pinode import BasePINODEConfig, BasePINODEModel
from Paper_PINODE_EPSR.neural_ode import NeuralODEConfig, NeuralODEModel
from Paper_PINODE_EPSR.training import (
    OptimizationConfig,
    TuningConfig,
    optimize_steps,
    run_optuna_tuning,
    suggest_base_pinode_hyperparameters,
)

CASES = ("all_to_one", "identity_ind", "identity_dep1", "identity_dep2")


def _y(case: str, n: int = 36) -> np.ndarray:
    k = np.arange(n, dtype=float)
    if case == "all_to_one":
        return (22.0 + 0.5 * np.sin(k / 5.0))[:, None]
    return np.column_stack(
        (22.0 + 0.5 * np.sin(k / 5.0), 23.0 + 0.35 * np.cos(k / 6.0))
    )


def _raw_v(case: str, n: int = 36):
    k = np.arange(n, dtype=float)
    To = 12.0 + 2.0 * np.sin(k / 9.0)
    if case == "all_to_one":
        names = ("T_o", "Q_AC,A", "Q_ZIC,A", "Q_ZIR,A", "Q_Sol1,A", "Q_Sol2,A")
        v = np.column_stack(
            (
                To,
                -1500.0 + 100.0 * np.sin(k / 3.0),
                500.0 + 5.0 * k,
                220.0 + 3.0 * k,
                180.0 * np.maximum(np.sin(k / 8.0), 0.0),
                90.0 * np.maximum(np.sin(k / 8.0), 0.0),
            )
        )
        return v, names
    if case == "identity_ind":
        names = {
            "Dining": ("T_o", "Q_AC,D", "Q_ZIC,D", "Q_ZIR,D", "Q_Sol1,D", "Q_Sol2,D"),
            "Kitchen": ("T_o", "Q_AC,K", "Q_ZIC,K", "Q_ZIR,K"),
        }
        v = {
            "Dining": np.column_stack(
                (To, -800.0 + 40.0 * np.sin(k), 320.0 + 3.0 * k, 170.0 + 2.0 * k,
                 100.0 * np.maximum(np.sin(k / 8.0), 0.0), 50.0 * np.maximum(np.sin(k / 8.0), 0.0))
            ),
            "Kitchen": np.column_stack(
                (To, -600.0 + 30.0 * np.cos(k), 430.0 + 4.0 * k, 240.0 + 2.0 * k)
            ),
        }
        return v, names
    if case == "identity_dep1":
        names = (
            "T_o", "Q_AC,D", "Q_AC,K", "Q_ZIC,D", "Q_ZIR,D", "Q_Sol1,D",
            "Q_Sol2,D", "Q_ZIC,K", "Q_ZIR,K",
        )
        v = np.column_stack(
            (To, -800.0 + 40.0 * np.sin(k), -600.0 + 30.0 * np.cos(k),
             320.0 + 3.0 * k, 170.0 + 2.0 * k,
             100.0 * np.maximum(np.sin(k / 8.0), 0.0),
             50.0 * np.maximum(np.sin(k / 8.0), 0.0),
             430.0 + 4.0 * k, 240.0 + 2.0 * k)
        )
        return v, names
    names = ("T_o", "Q_AC,D", "Q_AC,K", "Q_ZIC,A", "Q_ZIR,A", "Q_Sol1,A", "Q_Sol2,A")
    v = np.column_stack(
        (To, -800.0 + 40.0 * np.sin(k), -600.0 + 30.0 * np.cos(k),
         380.0 + 3.0 * k, 205.0 + 2.0 * k,
         90.0 * np.maximum(np.sin(k / 8.0), 0.0),
         45.0 * np.maximum(np.sin(k / 8.0), 0.0))
    )
    return v, names


def _model(case: str, rc_order: int, *, N_r: int = 2, N_s: int = 1, lambda_f: float = 0.1):
    y = _y(case)
    v, names = _raw_v(case)
    m = BasePINODEModel(
        BasePINODEConfig(
            case_name=case,
            rc_order=rc_order,
            hidden_layers=1,
            hidden_width=8,
            N_r=N_r,
            L_e=3,
            N_s=N_s,
            lambda_y=1.0,
            lambda_f=lambda_f,
            lambda_wd=0.0,
        ),
        y_training=y,
        v_training=v,
        v_names=names,
    )
    return m, y, v, names


def _window(model: BasePINODEModel, y: np.ndarray, v):
    if model.config.rc_order == 1:
        k = 0
        cy = cv = None
    else:
        k = model.config.L_e - 1
        cy = torch.tensor(y[: model.config.L_e], dtype=torch.float64)
        if isinstance(v, dict):
            cv = {key: torch.tensor(val[: model.config.L_e], dtype=torch.float64) for key, val in v.items()}
        else:
            cv = torch.tensor(v[: model.config.L_e], dtype=torch.float64)
    yt = torch.tensor(y[k : k + model.config.N_r + 1], dtype=torch.float64)
    if isinstance(v, dict):
        vs = {key: torch.tensor(val[k : k + model.config.N_r], dtype=torch.float64) for key, val in v.items()}
    else:
        vs = torch.tensor(v[k : k + model.config.N_r], dtype=torch.float64)
    return yt, vs, cy, cv


def test_base_pinode_source_uses_neuromancer_rk4_and_has_no_projection_or_direct_torchdiffeq():
    source = (Path(__file__).parents[1] / "base_pinode.py").read_text(encoding="utf-8")
    assert "rk4_interval(" in source
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("torchdiffeq") for name in imported)
    assert "torch.linalg.solve" not in source
    assert "k1 =" not in source and "k2 =" not in source and "k3 =" not in source and "k4 =" not in source
    assert "return g" in source


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rc_order", (1, 2))
def test_base_pinode_all_architectures_orders_finite_gradients_positive_rc_and_stage_count(case, rc_order):
    model, y, v, _ = _model(case, rc_order, N_r=2, N_s=1)
    yt, vs, cy, cv = _window(model, y, v)
    out = model.rollout_loss(y_true=yt, v_sequence=vs, context_y=cy, context_v=cv)
    assert torch.isfinite(out["total"])
    assert torch.isfinite(out["physics"])
    assert model.stage_residual_count == 2 * 1 * 4
    assert out["stage_residual"].shape[-1] == model.state_dim
    out["total"].backward()
    neural_grads = [p.grad for p in model.vector_fields.parameters()]
    rc_grads = [p.grad for p in list(model.rho_R.parameters()) + list(model.rho_C.parameters())]
    assert any(g is not None and torch.isfinite(g).all() for g in neural_grads)
    assert any(g is not None and torch.isfinite(g).all() for g in rc_grads)
    for name, value in model.physical_parameters().items():
        if name.startswith(("R_", "C_")):
            assert float(value.detach()) > 0.0


def test_stage_count_tracks_true_four_rk4_rhs_calls_per_substep():
    model, y, v, _ = _model("all_to_one", 1, N_r=3, N_s=2)
    yt, vs, cy, cv = _window(model, y, v)
    model.rollout_loss(y_true=yt, v_sequence=vs, context_y=cy, context_v=cv)
    assert model.stage_residual_count == 3 * 2 * 4
    assert model.stage_raw_derivative_tensor().shape[0] == 24


def test_lambda_f_zero_has_identical_dynamics_to_warm_started_node():
    base, y, v, names = _model("identity_dep1", 2, N_r=2, N_s=2, lambda_f=0.0)
    node_model = NeuralODEModel(
        NeuralODEConfig(case_name="identity_dep1", rc_order=2, hidden_layers=1, hidden_width=8, N_r=2, L_e=3, N_s=2),
        y_training=y, v_training=v, v_names=names,
    )
    base.warm_start_from_node(node_model)
    yt, vs, cy, cv = _window(base, y, v)
    yb, _ = base.rollout(y0=yt[0], v_sequence=vs, context_y=cy, context_v=cv)
    yn, _ = node_model.rollout(y0=yt[0], v_sequence=vs, context_y=cy, context_v=cv)
    assert torch.allclose(yb, yn, atol=1e-12, rtol=1e-12)


def test_dep2_allocation_sums_and_1c_identifiability_lock_and_2c_eta_learning():
    m1, *_ = _model("identity_dep2", 1)
    p1 = m1.physical_parameters()
    assert torch.allclose(p1["lambda_c_D"] + p1["lambda_c_K"], torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(p1["lambda_r_D"] + p1["lambda_r_K"], torch.tensor(2.0, dtype=torch.float64))
    assert float(p1["eta_r_D"]) == 1.0 and float(p1["eta_r_K"]) == 1.0
    assert len(m1.rho_eta) == 0
    m2, *_ = _model("identity_dep2", 2)
    assert set(m2.rho_eta) == {"eta_r_D", "eta_r_K"}


def test_interzone_resistance_only_on_dep1_and_dep2():
    for case in CASES:
        model, *_ = _model(case, 1)
        has = "R_DK" in model.rho_R
        assert has is (case in {"identity_dep1", "identity_dep2"})


def test_2c_hidden_mass_initialization_is_bounded_and_encoder_is_neuromancer_network():
    model, y, v, _ = _model("identity_dep2", 2)
    y0 = torch.tensor(y[2], dtype=torch.float64)
    cy = torch.tensor(y[:3], dtype=torch.float64)
    cv = torch.tensor(v[:3], dtype=torch.float64)
    z = model.initial_state(y0, context_y=cy, context_v=cv)
    x = model.mu_x + model.S_x * z
    assert torch.all(torch.abs(x[[1, 3]] - x[[0, 2]]) < model.config.delta_T_m_max + 1e-12)
    net = model.encoders["joint"].network
    assert "neuromancer" in net.__class__.__module__.lower()


def test_q_star_is_positive_training_only_and_2c_repeats_zone_scale_for_air_mass():
    model, *_ = _model("identity_dep1", 2)
    assert torch.all(model.q_star_zone > 0)
    assert torch.allclose(model.q_star_residual[[0, 1]], model.q_star_zone[0].repeat(2))
    assert torch.allclose(model.q_star_residual[[2, 3]], model.q_star_zone[1].repeat(2))
    assert not model.q_star_zone.requires_grad


def test_neural_regularization_excludes_physical_rc_parameters():
    model, *_ = _model("all_to_one", 2)
    expected = sum((p**2).sum() for p in model.vector_fields.parameters()) + sum((p**2).sum() for p in model.encoders.parameters())
    assert torch.allclose(model.neural_regularization(), expected)


def test_node_warm_start_copies_only_omega_psi_not_rc():
    base, y, v, names = _model("all_to_one", 2)
    node_model = NeuralODEModel(
        NeuralODEConfig(case_name="all_to_one", rc_order=2, hidden_layers=1, hidden_width=8, N_r=2, L_e=3, N_s=1),
        y_training=y, v_training=v, v_names=names,
    )
    rc_before = {k: p.detach().clone() for k, p in base.rho_R.items()} | {k: p.detach().clone() for k, p in base.rho_C.items()}
    with torch.no_grad():
        for p in node_model.parameters():
            p.add_(0.123)
    base.warm_start_from_node(node_model)
    rc_after = {k: p.detach().clone() for k, p in base.rho_R.items()} | {k: p.detach().clone() for k, p in base.rho_C.items()}
    assert all(torch.allclose(rc_before[k], rc_after[k]) for k in rc_before)
    assert all(torch.allclose(a, b) for a, b in zip(base.vector_fields.parameters(), node_model.vector_fields.parameters()))


def test_base_pinode_provenance_distinguishes_soft_penalty_from_ebp_projection():
    model, *_ = _model("all_to_one", 1)
    p = model.provenance()
    assert p["method"] == "base_pinode"
    assert p["physics"]["constraint_type"] == "soft_penalty"
    assert p["physics"]["hard_projection"] is False
    assert p["physics"]["integrated_derivative"] == "raw_f_tilde_omega"
    assert p["framework"]["integration"] == "neuromancer.dynamics.integrators.RK4"


def test_base_pinode_optuna_search_space_and_two_trial_freeze_smoke():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = _y("all_to_one", 36)
    v, names = _raw_v("all_to_one", 36)
    seen = []

    def objective(trial):
        hp = suggest_base_pinode_hyperparameters(trial, rc_order=1)
        seen.append(hp)
        model = BasePINODEModel(
            BasePINODEConfig(
                case_name="all_to_one", rc_order=1,
                hidden_layers=hp["hidden_layers"], hidden_width=hp["hidden_width"],
                activation=hp["activation"], N_r=hp["N_r"], N_s=hp["N_s"],
                lambda_f=hp["lambda_f"], lambda_wd=hp["lambda_wd"],
            ),
            y_training=y, v_training=v, v_names=names,
        )
        N_r = model.config.N_r
        yt = torch.tensor(y[: N_r + 1], dtype=torch.float64)
        vs = torch.tensor(v[:N_r], dtype=torch.float64)
        history = optimize_steps(
            model,
            lambda: model.rollout_loss(y_true=yt, v_sequence=vs)["total"],
            config=OptimizationConfig(
                learning_rate=hp["learning_rate"], optimizer=hp["optimizer"],
                max_epochs=1, patience=2, seed=11,
            ),
            steps=1,
        )
        return float(history[-1])

    study, frozen = run_optuna_tuning(
        objective,
        method="base_pinode",
        tuning_scope="synthetic_training_only_patch03",
        config=TuningConfig(n_trials=2, seed=11),
    )
    assert len(study.trials) == 2 and len(seen) == 2
    assert frozen.method == "base_pinode"
    assert "lambda_f" in frozen.values and "N_s" in frozen.values and "N_r" in frozen.values


def test_patch3_validator_uses_isolated_pytest_and_always_has_failure_writer_path():
    source = (Path(__file__).parents[1] / "validate_patch3.py").read_text(encoding="utf-8")
    assert "subprocess.run" in source
    assert '"-m",\n        "pytest"' in source
    assert "write_json(output, payload)" in source
    assert "PATCH 03 STATUS: FAILED" in source
