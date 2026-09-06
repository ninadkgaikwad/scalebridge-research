from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from Paper_PINODE_EPSR.ebp_pinode import (
    EBPPINODEConfig,
    EBPPINODEModel,
    weighted_energy_projection,
)
from Paper_PINODE_EPSR.neural_ode import NeuralODEConfig, NeuralODEModel
from Paper_PINODE_EPSR.training import (
    OptimizationConfig,
    TuningConfig,
    optimize_steps,
    run_optuna_tuning,
    suggest_ebp_pinode_hyperparameters,
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


def _model(
    case: str,
    rc_order: int,
    *,
    N_r: int = 2,
    N_s: int = 1,
    lambda_int: float = 0.1,
    lambda_corr: float = 1e-8,
):
    y = _y(case)
    v, names = _raw_v(case)
    model = EBPPINODEModel(
        EBPPINODEConfig(
            case_name=case,
            rc_order=rc_order,
            hidden_layers=1,
            hidden_width=8,
            activation="tanh",
            N_r=N_r,
            L_e=3,
            N_s=N_s,
            lambda_y=1.0,
            lambda_int=lambda_int,
            lambda_corr=lambda_corr,
            lambda_wd=0.0,
        ),
        y_training=y,
        v_training=v,
        v_names=names,
    )
    return model, y, v, names


def _window(model: EBPPINODEModel, y: np.ndarray, v):
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


def test_ebp_source_uses_neuromancer_rk4_torch_solve_and_no_explicit_inverse_or_torchdiffeq():
    source = (Path(__file__).parents[1] / "ebp_pinode.py").read_text(encoding="utf-8")
    assert "rk4_interval(" in source
    assert "torch.linalg.solve" in source
    assert "return g_P" in source
    assert ".inverse(" not in source
    assert "torch.inverse(" not in source
    assert "torch.linalg.inv(" not in source
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("torchdiffeq") for name in imported)
    assert "k1 =" not in source and "k2 =" not in source and "k3 =" not in source and "k4 =" not in source


def test_weighted_projector_exact_feasibility_kkt_and_correction_energy_identity():
    f = torch.tensor([[1.0, -0.5], [0.3, 0.8]], dtype=torch.float64)
    A = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    b = torch.tensor([[5.0], [2.0]], dtype=torch.float64)
    W = torch.tensor([4.0, 16.0], dtype=torch.float64)
    out = weighted_energy_projection(f, A, b, W)
    assert torch.max(torch.abs(out["rho_P"])) < 1e-12
    assert torch.max(torch.abs(out["stationarity"])) < 1e-12
    assert torch.allclose(out["correction_energy"], out["rho_solve_energy"], atol=1e-12, rtol=1e-12)
    assert torch.any(torch.abs(out["f_P"] - f) > 1e-12)


def test_weighted_projector_leaves_already_feasible_derivative_unchanged():
    A = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    f = torch.tensor([[1.0, 0.75]], dtype=torch.float64)
    b = (A @ f.T).T
    out = weighted_energy_projection(f, A, b, torch.tensor([4.0, 16.0], dtype=torch.float64))
    assert torch.allclose(out["f_P"], f, atol=1e-12, rtol=1e-12)
    assert torch.max(torch.abs(out["correction"])) < 1e-12


def test_weighted_projector_pythagorean_identity_against_feasible_reference():
    f_tilde = torch.tensor([[2.2, -0.7]], dtype=torch.float64)
    A = torch.tensor([[3.0, 5.0]], dtype=torch.float64)
    b = torch.tensor([[7.0]], dtype=torch.float64)
    W = torch.tensor([9.0, 25.0], dtype=torch.float64)
    out = weighted_energy_projection(f_tilde, A, b, W)
    # One feasible reference distinct from the projection.
    f_star = torch.tensor([[4.0, -1.0]], dtype=torch.float64)  # 3*4 + 5*(-1) = 7
    lhs = torch.sum((f_tilde - f_star) ** 2 * W, dim=-1)
    rhs = torch.sum((f_tilde - out["f_P"]) ** 2 * W, dim=-1) + torch.sum((out["f_P"] - f_star) ** 2 * W, dim=-1)
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_weighted_projector_is_differentiable_through_solve_and_metric():
    f = torch.tensor([[1.0, -0.5]], dtype=torch.float64, requires_grad=True)
    A = torch.tensor([[[2.0, 4.0]]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[5.0]], dtype=torch.float64, requires_grad=True)
    W = torch.tensor([[4.0, 16.0]], dtype=torch.float64, requires_grad=True)
    out = weighted_energy_projection(f, A, b, W)
    loss = out["f_P"].square().sum() + 1e-3 * out["correction_energy"].sum()
    loss.backward()
    for tensor in (f, A, b, W):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_rank_nullity_contract_for_paper_orders():
    m1, y1, v1, _ = _model("all_to_one", 1)
    x1 = torch.tensor([[float(y1[0, 0])]], dtype=torch.float64)
    vv1 = torch.tensor(v1[:1], dtype=torch.float64)
    A1, _, _ = m1.energy_constraint(x1, vv1)
    assert int(torch.linalg.matrix_rank(A1[0])) == 1
    assert A1.shape[-1] - int(torch.linalg.matrix_rank(A1[0])) == 0

    m2, y2, v2, _ = _model("all_to_one", 2)
    yt, vs, cy, cv = _window(m2, y2, v2)
    z0 = m2.initial_state(yt[0], context_y=cy, context_v=cv).reshape(1, -1)
    x2 = m2.mu_x + m2.S_x * z0
    A2, _, _ = m2.energy_constraint(x2, vs[:1])
    assert A2.shape[-1] - int(torch.linalg.matrix_rank(A2[0])) == 1

    md, yd, vd, _ = _model("identity_dep1", 2)
    ytd, vsd, cyd, cvd = _window(md, yd, vd)
    z0d = md.initial_state(ytd[0], context_y=cyd, context_v=cvd).reshape(1, -1)
    xd = md.mu_x + md.S_x * z0d
    Ad, _, _ = md.energy_constraint(xd, vsd[:1])
    assert Ad.shape[-1] - int(torch.linalg.matrix_rank(Ad[0])) == 2


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rc_order", (1, 2))
def test_ebp_all_architectures_orders_finite_gradients_positive_rc_exact_balance_and_stage_count(case, rc_order):
    model, y, v, _ = _model(case, rc_order, N_r=2, N_s=1, lambda_corr=1e-8)
    yt, vs, cy, cv = _window(model, y, v)
    out = model.rollout_loss(y_true=yt, v_sequence=vs, context_y=cy, context_v=cv)
    assert torch.isfinite(out["total"])
    assert torch.isfinite(out["correction"])
    assert model.projection_stage_count == 2 * 1 * 4
    assert torch.max(torch.abs(out["stage_rho_P"])) < 1e-6
    assert torch.max(out["stage_stationarity_relative"]) < 1e-12
    assert torch.allclose(out["stage_correction_energy"], out["stage_rho_solve_energy"], atol=1e-5, rtol=1e-10)
    out["total"].backward()
    neural_grads = [p.grad for p in model.vector_fields.parameters()]
    rc_grads = [p.grad for p in list(model.rho_R.parameters()) + list(model.rho_C.parameters())]
    assert any(g is not None and torch.isfinite(g).all() for g in neural_grads)
    assert any(g is not None and torch.isfinite(g).all() for g in rc_grads)
    for name, value in model.physical_parameters().items():
        if name.startswith(("R_", "C_")):
            assert float(value.detach()) > 0.0


def test_projection_stage_count_tracks_true_four_rk4_rhs_calls_per_substep():
    model, y, v, _ = _model("all_to_one", 2, N_r=3, N_s=2)
    yt, vs, cy, cv = _window(model, y, v)
    model.rollout_loss(y_true=yt, v_sequence=vs, context_y=cy, context_v=cv)
    assert model.projection_stage_count == 3 * 2 * 4
    assert model.stage_projected_derivative_tensor().shape[0] == 24


def test_1c_projection_has_zero_neural_derivative_freedom():
    model, y, v, _ = _model("all_to_one", 1)
    state = torch.tensor([[float(y[0, 0])]], dtype=torch.float64)
    vv = torch.tensor(v[:1], dtype=torch.float64)
    p1 = model.project_physical_derivative(state, torch.tensor([[0.123]], dtype=torch.float64), vv)
    p2 = model.project_physical_derivative(state, torch.tensor([[-9.0]], dtype=torch.float64), vv)
    A, b, _ = model.energy_constraint(state, vv)
    physical = b[:, 0] / A[:, 0, 0]
    assert torch.allclose(p1["f_P"][:, 0], physical, atol=1e-12, rtol=1e-12)
    assert torch.allclose(p2["f_P"][:, 0], physical, atol=1e-12, rtol=1e-12)
    assert torch.allclose(p1["f_P"], p2["f_P"], atol=1e-12, rtol=1e-12)


def test_2c_projection_preserves_one_nullspace_direction_and_corrects_energy_normal_direction():
    model, y, v, _ = _model("all_to_one", 2)
    yt, vs, cy, cv = _window(model, y, v)
    z0 = model.initial_state(yt[0], context_y=cy, context_v=cv).reshape(1, -1)
    state = model.mu_x + model.S_x * z0
    vv = vs[:1]
    A, b, _ = model.energy_constraint(state, vv)
    # Two raw derivatives separated by a null-space vector [Cm, -Ca].
    p = model.physical_parameters()
    null = torch.stack((p["C_m_A"], -p["C_a_A"])).reshape(1, -1)
    null = null / torch.linalg.vector_norm(null)
    raw1 = torch.tensor([[0.02, -0.01]], dtype=torch.float64)
    raw2 = raw1 + 0.05 * null
    assert torch.max(torch.abs((A @ null.unsqueeze(-1)).squeeze(-1))) < 1e-12
    proj1 = model.project_physical_derivative(state, raw1, vv)["f_P"]
    proj2 = model.project_physical_derivative(state, raw2, vv)["f_P"]
    assert torch.allclose(proj2 - proj1, raw2 - raw1, atol=1e-12, rtol=1e-12)
    assert torch.max(torch.abs((A @ proj1.unsqueeze(-1)).squeeze(-1) - b)) < 1e-6


def test_1c_has_no_internal_loss_and_2c_internal_pairs_sum_to_hard_balance_zero():
    m1, y1, v1, _ = _model("identity_dep1", 1)
    yt1, vs1, cy1, cv1 = _window(m1, y1, v1)
    out1 = m1.rollout_loss(y_true=yt1, v_sequence=vs1, context_y=cy1, context_v=cv1)
    assert float(out1["internal_physics"]) == 0.0
    assert out1["stage_internal_residual"].shape[-1] == 0

    m2, y2, v2, _ = _model("identity_dep1", 2)
    yt2, vs2, cy2, cv2 = _window(m2, y2, v2)
    out2 = m2.rollout_loss(y_true=yt2, v_sequence=vs2, context_y=cy2, context_v=cv2)
    r = out2["stage_internal_residual"]
    assert torch.max(torch.abs(r[..., 0] + r[..., 1])) < 1e-6
    assert torch.max(torch.abs(r[..., 2] + r[..., 3])) < 1e-6
    assert torch.isfinite(out2["internal_physics"])


def test_dep2_allocation_sums_1c_eta_lock_and_2c_eta_internal_split():
    m1, *_ = _model("identity_dep2", 1)
    p1 = m1.physical_parameters()
    assert torch.allclose(p1["lambda_c_D"] + p1["lambda_c_K"], torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(p1["lambda_r_D"] + p1["lambda_r_K"], torch.tensor(2.0, dtype=torch.float64))
    assert float(p1["eta_r_D"]) == 1.0 and float(p1["eta_r_K"]) == 1.0
    assert len(m1.rho_eta) == 0
    m2, *_ = _model("identity_dep2", 2)
    assert set(m2.rho_eta) == {"eta_r_D", "eta_r_K"}


def test_2c_causal_hidden_state_initializer_is_inherited_bounded_and_neuromancer_mlp():
    model, y, v, _ = _model("identity_dep2", 2)
    y0 = torch.tensor(y[2], dtype=torch.float64)
    cy = torch.tensor(y[:3], dtype=torch.float64)
    cv = torch.tensor(v[:3], dtype=torch.float64)
    z = model.initial_state(y0, context_y=cy, context_v=cv)
    x = model.mu_x + model.S_x * z
    assert torch.all(torch.abs(x[[1, 3]] - x[[0, 2]]) < model.config.delta_T_m_max + 1e-12)
    assert "neuromancer" in model.encoders["joint"].network.__class__.__module__.lower()


def test_ebp_warm_start_from_node_copies_neural_encoder_only_not_rc_parameters():
    ebp, y, v, names = _model("all_to_one", 2)
    node = NeuralODEModel(
        NeuralODEConfig(case_name="all_to_one", rc_order=2, hidden_layers=1, hidden_width=8, N_r=2, L_e=3, N_s=1),
        y_training=y, v_training=v, v_names=names,
    )
    rc_before = {k: p.detach().clone() for k, p in ebp.rho_R.items()} | {k: p.detach().clone() for k, p in ebp.rho_C.items()}
    with torch.no_grad():
        for p in node.parameters():
            p.add_(0.123)
    ebp.warm_start_from_node(node)
    rc_after = {k: p.detach().clone() for k, p in ebp.rho_R.items()} | {k: p.detach().clone() for k, p in ebp.rho_C.items()}
    assert all(torch.allclose(rc_before[k], rc_after[k]) for k in rc_before)
    assert all(torch.allclose(a, b) for a, b in zip(ebp.vector_fields.parameters(), node.vector_fields.parameters()))


def test_ebp_checkpoint_round_trip_preserves_projection_parameters_and_prediction():
    model, y, v, _ = _model("identity_dep2", 2)
    yt, vs, cy, cv = _window(model, y, v)
    pred1, _ = model.rollout(y0=yt[0], v_sequence=vs, context_y=cy, context_v=cv)
    state = {k: val.detach().clone() for k, val in model.state_dict().items()}
    clone, *_ = _model("identity_dep2", 2)
    clone.load_state_dict(state)
    pred2, _ = clone.rollout(y0=yt[0], v_sequence=vs, context_y=cy, context_v=cv)
    assert torch.allclose(pred1, pred2, atol=1e-12, rtol=1e-12)
    assert torch.allclose(model.alpha_c, clone.alpha_c)
    assert torch.allclose(model.alpha_r, clone.alpha_r)


def test_ebp_provenance_locks_hard_projection_projected_deployment_and_solve():
    model, *_ = _model("all_to_one", 2)
    p = model.provenance()
    assert p["method"] == "ebp_pinode"
    assert p["physics"]["constraint_type"] == "hard_projection"
    assert p["physics"]["hard_projection"] is True
    assert p["physics"]["integrated_derivative"] == "projected_f_P"
    assert p["physics"]["projection_solve"] == "torch.linalg.solve(M,rho)"
    assert p["physics"]["explicit_matrix_inverse"] is False
    assert p["framework"]["integration"] == "neuromancer.dynamics.integrators.RK4"


def test_ebp_optuna_search_space_and_two_trial_freeze_smoke():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y = _y("all_to_one", 36)
    v, names = _raw_v("all_to_one", 36)
    seen = []

    def objective(trial):
        hp = suggest_ebp_pinode_hyperparameters(trial, rc_order=1)
        seen.append(hp)
        model = EBPPINODEModel(
            EBPPINODEConfig(
                case_name="all_to_one", rc_order=1,
                hidden_layers=hp["hidden_layers"], hidden_width=hp["hidden_width"],
                activation=hp["activation"], N_r=hp["N_r"], N_s=hp["N_s"],
                lambda_corr=1e-10, lambda_wd=hp["lambda_wd"],
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
                max_epochs=1, patience=2, seed=13,
            ),
            steps=1,
        )
        return float(history[-1])

    study, frozen = run_optuna_tuning(
        objective,
        method="ebp_pinode",
        tuning_scope="synthetic_training_only_patch04",
        config=TuningConfig(n_trials=2, seed=13),
    )
    assert len(study.trials) == 2 and len(seen) == 2
    assert frozen.method == "ebp_pinode"
    assert "lambda_corr" in frozen.values and "N_s" in frozen.values and "N_r" in frozen.values
    assert "lambda_int" not in frozen.values  # 1C has no internal residual term.


def test_ebp_2c_optuna_search_includes_internal_weight_and_fixed_metric_is_not_tuned():
    optuna = pytest.importorskip("optuna")
    trial = optuna.trial.FixedTrial(
        {
            "hidden_layers": 1, "hidden_width": 16, "activation": "tanh",
            "learning_rate": 1e-3, "optimizer": "adam", "N_r": 3, "N_s": 1,
            "lambda_corr": 1e-4, "lambda_wd": 1e-6, "batch_size": 8,
            "lambda_int": 0.1, "L_e": 3, "delta_T_m_max": 8.0,
        }
    )
    hp = suggest_ebp_pinode_hyperparameters(trial, rc_order=2)
    assert hp["lambda_int"] == 0.1
    assert "W" not in hp and "projection_metric" not in hp


def test_patch4_validator_uses_isolated_cumulative_pytest_and_always_has_failure_writer_path():
    source = (Path(__file__).parents[1] / "validate_patch4.py").read_text(encoding="utf-8")
    assert "subprocess.run" in source
    assert "test_patch1_contract.py" in source
    assert "test_patch2_contract.py" in source
    assert "test_patch3_contract.py" in source
    assert "test_patch4_contract.py" in source
    assert "write_json(output, payload)" in source
    assert "PATCH 04 STATUS: FAILED" in source
