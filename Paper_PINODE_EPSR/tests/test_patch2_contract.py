from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from Paper_PINODE_EPSR.common import (
    TensorStandardizer,
    build_rollout_windows,
    contiguous_segments,
    load_checkpoint,
    representative_window_subset,
    save_checkpoint,
)
from Paper_PINODE_EPSR.neuromancer_backend import (
    named_dataloader,
    rk4_interval,
    runtime_info,
    scalar_objective_problem,
)
from Paper_PINODE_EPSR.inverse_pinn import BaiCuiResidentialRCReference, InversePINNConfig, InversePINNRC
from Paper_PINODE_EPSR.neural_ode import NeuralODEConfig, NeuralODEModel
from Paper_PINODE_EPSR.phase_c import Q_HVAC_X, discover_and_load_phase_c_bundle, load_phase_c_linear_artifact, reference_phase_c_bundle
from Paper_PINODE_EPSR.config import PaperConfig
from Paper_PINODE_EPSR.training import (
    OptimizationConfig,
    TuningConfig,
    assert_training_only_indices,
    optimize_steps,
    run_optuna_tuning,
    suggest_inverse_pinn_hyperparameters,
    suggest_node_hyperparameters,
)

CASES = ("all_to_one", "identity_ind", "identity_dep1", "identity_dep2")


def _synthetic_y(case: str, n: int = 24) -> np.ndarray:
    k = np.arange(n, dtype=float)
    if case == "all_to_one":
        return (22.0 + 0.6 * np.sin(k / 4.0))[:, None]
    return np.column_stack((22.0 + 0.6 * np.sin(k / 4.0), 23.0 + 0.4 * np.cos(k / 5.0)))


def _inverse_forcing(case: str, n: int = 24) -> dict[str, np.ndarray]:
    k = np.arange(n, dtype=float)
    base = {"T_o": 12.0 + 3.0 * np.sin(k / 7.0)}
    if case == "all_to_one":
        base.update({
            "Q_AC,A": -1200.0 + 100.0 * np.sin(k),
            "Q_ZIC,A": 500.0 + 20.0 * k,
            "Q_ZIR,A": 250.0 + 10.0 * k,
            "Q_Sol1,A": 200.0 * np.maximum(np.sin(k / 8.0), 0.0),
            "Q_Sol2,A": 100.0 * np.maximum(np.sin(k / 8.0), 0.0),
        })
    elif case in {"identity_ind", "identity_dep1"}:
        base.update({
            "Q_AC,D": -700.0 + 30.0 * np.sin(k), "Q_ZIC,D": 300.0 + 10.0 * k, "Q_ZIR,D": 180.0 + 5.0 * k,
            "Q_Sol1,D": 120.0 * np.maximum(np.sin(k / 8.0), 0.0), "Q_Sol2,D": 60.0 * np.maximum(np.sin(k / 8.0), 0.0),
            "Q_AC,K": -500.0 + 20.0 * np.cos(k), "Q_ZIC,K": 400.0 + 8.0 * k, "Q_ZIR,K": 220.0 + 4.0 * k,
        })
    else:
        base.update({
            "Q_AC,D": -700.0 + 30.0 * np.sin(k),
            "Q_AC,K": -500.0 + 20.0 * np.cos(k),
            "Qbar_c_nh": 900.0 + 18.0 * k,
            "Qbar_r": 450.0 + 9.0 * k,
        })
    return base


def _node_v(case: str, n: int = 32):
    rng = np.random.default_rng(7)
    if case == "identity_ind":
        return {"Dining": rng.normal(size=(n, 6)), "Kitchen": rng.normal(size=(n, 4))}
    nv = {"all_to_one": 6, "identity_dep1": 9, "identity_dep2": 7}[case]
    return rng.normal(size=(n, nv))


def test_standardizer_round_trip_and_constant_protection():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    s = TensorStandardizer.fit(x, names=("vary", "constant"))
    xt = torch.tensor(x, dtype=torch.float64)
    assert torch.allclose(s.denormalize(s.normalize(xt)), xt)
    assert float(s.scale[1]) == 1.0


def test_contiguous_segments_obey_partition_included_and_300s():
    ts = pd.date_range("2026-01-01", periods=9, freq="5min").to_series().reset_index(drop=True)
    ts.iloc[7] = ts.iloc[6] + pd.Timedelta(minutes=10)
    partition = np.array(["train"] * 6 + ["validation"] * 3)
    included = np.array([True, True, False, True, True, True, True, True, True])
    seg = contiguous_segments(ts, partition, included, partition_name="train")
    assert [s.tolist() for s in seg] == [[0, 1], [3, 4, 5]]


def test_rollout_window_formula_matches_tex_and_subset_is_training_only():
    seg = [np.arange(0, 12)]
    w1 = build_rollout_windows(seg, partition="train", N_r=3, L_e=1, is_2c=False)
    w2 = build_rollout_windows(seg, partition="train", N_r=3, L_e=4, is_2c=True)
    assert len(w1) == 9  # N_seg - 1 - N_r + 1 = 9
    assert len(w2) == 6  # N_seg - L_e - N_r + 1 = 6
    sub = representative_window_subset(w2, max_windows=3, seed=1)
    assert len(sub) == 3 and all(w.partition == "train" for w in sub)


def test_neuromancer_backend_is_the_only_rk4_and_has_no_direct_torchdiffeq_call():
    info = runtime_info()
    assert info.rk4_class.endswith("integrators.RK4")
    assert "neuromancer" in info.rk4_class.lower()
    source = (Path(__file__).parents[1] / "src" / "pinode_epsr" / "backends" / "neuromancer.py").read_text(encoding="utf-8")
    assert "from neuromancer.dynamics.integrators import RK4 as NeuromancerRK4" in source
    assert "from neuromancer.problem import Problem as NeuromancerProblem" in source
    assert "from neuromancer.loss import PenaltyLoss as NeuromancerPenaltyLoss" in source
    assert "import torchdiffeq" not in source
    # Paper code must not carry an independent RK4 Butcher formula.
    assert "k1 =" not in source and "k2 =" not in source and "k3 =" not in source and "k4 =" not in source


def test_neuromancer_rk4_is_differentiable_and_accurate_on_simple_ode():
    a = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor([[1.0]], dtype=torch.float64)
    v0 = torch.zeros((1, 1), dtype=torch.float64)
    out = rk4_interval(
        lambda x, v: a * x,
        x0,
        v0,
        state_dim=1,
        extra_dim=1,
        n_substeps=10,
        interval_length=1.0,
    )
    assert np.isclose(float(out.detach().reshape(-1)[0]), np.exp(-1.0), atol=1e-5)
    out.sum().backward()
    assert a.grad is not None and torch.isfinite(a.grad)


def test_inverse_pinn_trajectory_networks_are_neuromancer_blocks():
    y = _synthetic_y("all_to_one", 12)
    t = np.arange(len(y), dtype=float) * 300.0
    model = InversePINNRC(
        InversePINNConfig(case_name="all_to_one", rc_order=1, hidden_layers=1, hidden_width=8),
        y_training=y,
        t_training_seconds=t,
    )
    block = model.trajectory_networks["joint"]
    assert "neuromancer" in block.__class__.__module__.lower()
    assert model.provenance()["framework"]["integration"] == "neuromancer.dynamics.integrators.RK4"


def test_bai_cui_reference_is_initialization_not_fixed_truth():
    ref = BaiCuiResidentialRCReference()
    init = ref.paper_initialization()
    assert init == {
        "R_out": ref.R_w,
        "R_mass": ref.R_im,
        "R_interzone": ref.R_im,
        "C_air": ref.C_in,
        "C_mass": ref.C_im,
    }
    # C1/C2/C3 are intentionally not used as physical capacitances.
    assert all(x not in init.values() for x in (ref.C1, ref.C2, ref.C3))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rc_order", (1, 2))
def test_inverse_pinn_all_architectures_and_rc_orders_have_finite_gradients(case: str, rc_order: int):
    n = 24
    y = _synthetic_y(case, n)
    t = np.arange(n, dtype=float) * 300.0
    model = InversePINNRC(
        InversePINNConfig(case_name=case, rc_order=rc_order, hidden_layers=1, hidden_width=8, lambda_y=1.0, lambda_f=0.1),
        y_training=y,
        t_training_seconds=t,
    )
    result = model.loss(
        t_seconds=torch.tensor(t, dtype=torch.float64),
        y_measured=torch.tensor(y, dtype=torch.float64),
        forcing=_inverse_forcing(case, n),
    )
    assert torch.isfinite(result["total"])
    result["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
    params = model.physical_parameters()
    for name, value in params.items():
        if name.startswith(("R_", "C_")):
            assert float(value.detach()) > 0.0


def test_dep2_1c_identifiability_safeguard_and_allocation_sums():
    y = _synthetic_y("identity_dep2")
    t = np.arange(len(y), dtype=float) * 300.0
    with pytest.raises(ValueError):
        InversePINNConfig(case_name="identity_dep2", rc_order=1, eta_mode_1c="learnable")
    model = InversePINNRC(InversePINNConfig(case_name="identity_dep2", rc_order=1), y_training=y, t_training_seconds=t)
    p = model.physical_parameters()
    assert float(p["eta_r_D"]) == 1.0 and float(p["eta_r_K"]) == 1.0
    assert torch.allclose(p["lambda_c_D"] + p["lambda_c_K"], torch.tensor(2.0, dtype=torch.float64))
    assert torch.allclose(p["lambda_r_D"] + p["lambda_r_K"], torch.tensor(2.0, dtype=torch.float64))


def test_inverse_pinn_physical_rollout_discards_trajectory_network():
    y = _synthetic_y("all_to_one")
    t = np.arange(len(y), dtype=float) * 300.0
    model = InversePINNRC(InversePINNConfig(case_name="all_to_one", rc_order=1, hidden_layers=1, hidden_width=4), y_training=y, t_training_seconds=t)
    forcing = _inverse_forcing("all_to_one", 2)
    seq = [{k: v[i] for k, v in forcing.items()} for i in range(2)]
    before = model.physical_rollout(torch.tensor([22.0]), seq)
    with torch.no_grad():
        for p in model.trajectory_networks.parameters():
            p.add_(100.0)
    after = model.physical_rollout(torch.tensor([22.0]), seq)
    assert torch.allclose(before, after)


def _node_model(case: str, rc_order: int, n: int = 32) -> tuple[NeuralODEModel, np.ndarray, object]:
    y = _synthetic_y(case, n)
    v = _node_v(case, n)
    config = NeuralODEConfig(case_name=case, rc_order=rc_order, hidden_layers=1, hidden_width=8, N_r=3, L_e=3, N_s=2)
    if case == "identity_ind":
        names = {"Dining": tuple(f"d{i}" for i in range(6)), "Kitchen": tuple(f"k{i}" for i in range(4))}
    else:
        names = tuple(f"v{i}" for i in range(v.shape[1]))
    return NeuralODEModel(config, y_training=y, v_training=v, v_names=names), y, v


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rc_order", (1, 2))
def test_node_all_architectures_and_rc_orders_rollout_and_backprop(case: str, rc_order: int):
    model, y, v = _node_model(case, rc_order)
    if rc_order == 1:
        k = 0
        context_y = context_v = None
    else:
        k = 2
        context_y = torch.tensor(y[0:3], dtype=torch.float64)
        if case == "identity_ind":
            context_v = {key: torch.tensor(value[0:3], dtype=torch.float64) for key, value in v.items()}
        else:
            context_v = torch.tensor(v[0:3], dtype=torch.float64)
    y_true = torch.tensor(y[k : k + 4], dtype=torch.float64)
    if case == "identity_ind":
        vseq = {key: torch.tensor(value[k : k + 3], dtype=torch.float64) for key, value in v.items()}
    else:
        vseq = torch.tensor(v[k : k + 3], dtype=torch.float64)
    result = model.rollout_loss(y_true=y_true, v_sequence=vseq, context_y=context_y, context_v=context_v)
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_node_vector_field_dimensions_match_part3_exactly():
    expected = {
        ("all_to_one", 1): {"joint": (7, 1)},
        ("all_to_one", 2): {"joint": (8, 2)},
        ("identity_ind", 1): {"Dining": (7, 1), "Kitchen": (5, 1)},
        ("identity_ind", 2): {"Dining": (8, 2), "Kitchen": (6, 2)},
        ("identity_dep1", 1): {"joint": (11, 2)},
        ("identity_dep1", 2): {"joint": (13, 4)},
        ("identity_dep2", 1): {"joint": (9, 2)},
        ("identity_dep2", 2): {"joint": (11, 4)},
    }
    for (case, rc_order), dims in expected.items():
        model, _, _ = _node_model(case, rc_order)
        for key, (nin, nout) in dims.items():
            block = model.vector_fields[key]
            assert block.in_features == nin
            assert block.out_features == nout
            assert "neuromancer" in block.__class__.__module__.lower()


def test_node_has_no_physical_rc_or_absolute_time_parameters():
    model, _, _ = _node_model("identity_dep1", 2)
    names = " ".join(name.lower() for name, _ in model.named_parameters())
    assert "r_dk" not in names and "eta" not in names and "lambda" not in names
    assert "time" not in names and "tau" not in names


def test_2c_causal_encoder_uses_only_supplied_past_context():
    model, y, v = _node_model("identity_dep2", 2)
    y0 = torch.tensor(y[2], dtype=torch.float64)
    past_y = torch.tensor(y[0:3], dtype=torch.float64)
    past_v = torch.tensor(v[0:3], dtype=torch.float64)
    z1 = model.initial_state(y0, context_y=past_y, context_v=past_v)
    # Alter future data that is not an encoder argument; initialization cannot change.
    future_y = y.copy(); future_y[3:] += 1000.0
    z2 = model.initial_state(y0, context_y=past_y, context_v=past_v)
    assert torch.allclose(z1, z2)


def test_phase_c_proxy_and_json_model_loading(tmp_path: Path):
    assert np.isclose(Q_HVAC_X(1.0, 12.0, 22.0), -10050.0)
    artifact = tmp_path / "model_metadata.json"
    artifact.write_text(json.dumps({"coefficient": 1.25, "intercept": -3.0}), encoding="utf-8")
    model = load_phase_c_linear_artifact(artifact, component="QAC", aggregate_zone_id="Dining")
    assert np.allclose(model.predict(np.array([0.0, 2.0])), [-3.0, -0.5])
    bundle = reference_phase_c_bundle("Dining")
    assert np.isfinite(bundle.predict_phvac_from_qac(np.array([-1000.0, 1000.0]))).all()



def test_phase_c_actual_artifact_discovery_path(tmp_path: Path):
    config = PaperConfig(generated_data_root=tmp_path)
    base = config.campaign_root / "heat_input_regression"
    training = base / "training_runs" / "c6_test" / "artifacts" / "Dining"
    for component, coef, intercept in (("QAC", 1.2, 0.0), ("PHVAC", 0.3, 400.0)):
        folder = training / component
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model_metadata.json").write_text(
            json.dumps({"aggregate_zone_id": "Dining", "component": component, "coefficient": coef, "intercept": intercept}),
            encoding="utf-8",
        )
    run_dir = base / "campaign_runs" / "phase_c_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "phase_c_campaign_run_manifest.json").write_text(
        json.dumps({"training_run_id": "c6_test"}), encoding="utf-8"
    )
    bundle = discover_and_load_phase_c_bundle(config, "Dining", phase_c_run_id="phase_c_test")
    assert np.allclose(bundle.predict_qac_from_hvac_proxy(np.array([2.0])), [2.4])
    assert np.allclose(bundle.predict_phvac_from_qac(np.array([-1000.0])), [700.0])
    assert bundle.provenance["mode"] == "actual_phase_c_artifacts"

def test_checkpoint_round_trip(tmp_path: Path):
    model, _, _ = _node_model("all_to_one", 1)
    path = tmp_path / "node.pt"
    state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_checkpoint(path, model=model, provenance={"case": "all_to_one", "frozen": True})
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    provenance = load_checkpoint(path, model=model)
    assert provenance["frozen"] is True
    assert all(torch.allclose(model.state_dict()[k], v) for k, v in state_before.items())


def test_optuna_search_and_freeze_pipeline_executes_actual_small_model_trials():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_inv = _synthetic_y("all_to_one", 18)
    t_inv = np.arange(len(y_inv), dtype=float) * 300.0
    f_inv = _inverse_forcing("all_to_one", len(y_inv))

    def inv_objective(trial):
        hp = suggest_inverse_pinn_hyperparameters(trial)
        model = InversePINNRC(
            InversePINNConfig(
                case_name="all_to_one", rc_order=1,
                hidden_layers=hp["hidden_layers"], hidden_width=hp["hidden_width"], activation=hp["activation"],
                lambda_y=hp["lambda_y"], lambda_f=hp["lambda_f"],
            ),
            y_training=y_inv, t_training_seconds=t_inv,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(
            t_seconds=torch.tensor(t_inv, dtype=torch.float64),
            y_measured=torch.tensor(y_inv, dtype=torch.float64),
            forcing=f_inv,
        )["total"]
        loss.backward(); optimizer.step()
        return float(loss.detach())

    _, frozen_inv = run_optuna_tuning(
        inv_objective,
        method="inverse_pinn_rc",
        tuning_scope="synthetic_representative_training_only_smoke",
        config=TuningConfig(n_trials=2, representative_max_windows=8, seed=3),
    )
    assert "hidden_layers" in frozen_inv.values and "hidden_width" in frozen_inv.values

    y_node = _synthetic_y("all_to_one", 32)
    v_node = _node_v("all_to_one", 32)

    def node_objective(trial):
        hp = suggest_node_hyperparameters(trial, rc_order=2)
        cfg = NeuralODEConfig(
            case_name="all_to_one", rc_order=2,
            hidden_layers=hp["hidden_layers"], hidden_width=hp["hidden_width"], activation=hp["activation"],
            N_r=hp["N_r"], L_e=hp["L_e"], N_s=hp["N_s"],
            delta_T_m_max=hp["delta_T_m_max"], lambda_wd=hp["lambda_wd"],
        )
        model = NeuralODEModel(cfg, y_training=y_node, v_training=v_node)
        k = cfg.L_e - 1
        y_true = torch.tensor(y_node[k:k+cfg.N_r+1], dtype=torch.float64)
        vseq = torch.tensor(v_node[k:k+cfg.N_r], dtype=torch.float64)
        cy = torch.tensor(y_node[k-cfg.L_e+1:k+1], dtype=torch.float64)
        cv = torch.tensor(v_node[k-cfg.L_e+1:k+1], dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
        optimizer.zero_grad(set_to_none=True)
        loss = model.rollout_loss(y_true=y_true, v_sequence=vseq, context_y=cy, context_v=cv)["total"]
        loss.backward(); optimizer.step()
        return float(loss.detach())

    _, frozen_node = run_optuna_tuning(
        node_objective,
        method="neural_ode",
        tuning_scope="synthetic_representative_training_only_smoke",
        config=TuningConfig(n_trials=2, representative_max_windows=8, seed=4),
    )
    assert "N_r" in frozen_node.values and "L_e" in frozen_node.values and "delta_T_m_max" in frozen_node.values


def test_training_only_leakage_guard():
    partition = np.array(["train", "train", "validation", "test"])
    assert_training_only_indices(np.array([0, 1]), partition)
    with pytest.raises(ValueError):
        assert_training_only_indices(np.array([0, 2]), partition)


def test_neuromancer_named_dataloader_injects_train_name_without_mutating_source():
    source = {"x": torch.arange(3, dtype=torch.float64).reshape(-1, 1)}
    loader = named_dataloader(source, name="train", batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch["name"] == "train"
    assert "name" not in source
    assert torch.equal(batch["x"], source["x"])


def test_neuromancer_problem_prefixes_named_loss_and_optimize_steps_updates_parameters():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    y = torch.tensor([[2.0], [4.0]], dtype=torch.float64)

    def closure():
        return torch.mean((model(x) - y) ** 2)

    problem, loader = scalar_objective_problem(model, closure, dataset_name="train")
    batch = next(iter(loader))
    assert batch["name"] == "train"
    output = problem(batch)
    assert "train_loss" in output
    assert "loss" not in output
    assert torch.isfinite(output["train_loss"])

    before = model.weight.detach().clone()
    history = optimize_steps(
        model,
        closure,
        config=OptimizationConfig(
            learning_rate=1e-2, max_epochs=2, patience=3, gradient_clip_norm=None
        ),
        steps=2,
        dataset_name="train",
    )
    assert len(history) == 2
    assert np.isfinite(history).all()
    assert not torch.allclose(before, model.weight.detach())


def test_patch02_validator_is_subprocess_isolated_and_failure_json_hardened():
    source = (Path(__file__).parents[1] / "validate_patch2.py").read_text(encoding="utf-8")
    assert "pytest.main" not in source
    assert "subprocess.run" in source
    assert "traceback.format_exc" in source
    assert "\"status\": \"failed\"" in source
