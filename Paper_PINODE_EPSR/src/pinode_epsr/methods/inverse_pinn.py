from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..core.common import TensorStandardizer
from ..backends.neuromancer import build_mlp, node, rk4_interval, runtime_info, system

RCOrder = Literal[1, 2]
EtaMode1C = Literal["full", "zero", "fixed", "learnable"]
EtaMode2C = Literal["mass_only", "air_only", "fixed", "learnable"]


@dataclass(frozen=True)
class BaiCuiResidentialRCReference:
    """Residential RC values supplied by the user as initialization scales only.

    They originate from a Bai Cui / ORNL Texas detached-family Elsevier paper
    snippet supplied in this chat.  They are NOT asserted to be physical truth
    for RestaurantFastFood and are never frozen during identification.

    C1/C2/C3 are retained as auxiliary source values because their physical
    meaning/units are not established by the supplied snippet; they are not
    silently treated as J/K capacitances.
    """

    R_w: float = 0.0134
    R_attic: float = 0.0235
    R_roof: float = 0.00156
    R_im: float = 0.00171
    R_win: float = 0.021
    C_w: float = 10_383_364.0
    C_attic: float = 704_168.0
    C_im: float = 23_396_403.0
    C_in: float = 8_665_588.0
    C1: float = 0.691
    C2: float = 0.784
    C3: float = 0.1
    source_note: str = "Bai Cui / ORNL Texas detached-family reference supplied by user; exact citation pending"

    def paper_initialization(self) -> dict[str, float]:
        """Transparent heuristic map into this paper's reduced RC parameters."""

        return {
            "R_out": self.R_w,
            "R_mass": self.R_im,
            "R_interzone": self.R_im,
            "C_air": self.C_in,
            "C_mass": self.C_im,
        }


@dataclass(frozen=True)
class InversePINNConfig:
    case_name: str
    rc_order: RCOrder
    hidden_layers: int = 2
    hidden_width: int = 32
    activation: str = "tanh"
    lambda_y: float = 1.0
    lambda_f: float = 1.0
    eta_mode_1c: EtaMode1C = "full"
    eta_mode_2c: EtaMode2C = "mass_only"
    eta_fixed: float = 1.0
    R_min: float = 1e-6
    C_min: float = 1e3
    seed: int = 42

    def __post_init__(self) -> None:
        if self.case_name not in {"all_to_one", "identity_ind", "identity_dep1", "identity_dep2"}:
            raise ValueError(f"Unknown case_name: {self.case_name}")
        if self.rc_order not in (1, 2):
            raise ValueError("rc_order must be 1 or 2")
        if self.lambda_y < 0 or self.lambda_f < 0:
            raise ValueError("loss weights must be nonnegative")
        if not 0.0 <= self.eta_fixed <= 1.0:
            raise ValueError("eta_fixed must be in [0,1]")
        # Part-2 identifiability safeguard: DEP2 1C learns alpha_r and therefore eta=1.
        if self.case_name == "identity_dep2" and self.rc_order == 1 and self.eta_mode_1c == "learnable":
            raise ValueError("Part-2 DEP2 1C forbids jointly learning eta_r and lambda_r; eta_r must be fixed to 1")


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    if value > 30:
        return value
    return float(np.log(np.expm1(value)))


def _parameter_from_physical(initial: float, minimum: float) -> nn.Parameter:
    if initial <= minimum:
        initial = minimum * 1.01
    rho = _inverse_softplus(initial - minimum)
    return nn.Parameter(torch.tensor(rho, dtype=torch.float64))


class InversePINNRC(nn.Module):
    """Part-2 inverse PINN trajectory surrogate + trainable physical RC parameters.

    The neural trajectory is identification-only. `physical_rollout` uses only
    the retained RC parameterization, matching the deployment contract in the
    authoritative Part-2 TeX.
    """

    def __init__(
        self,
        config: InversePINNConfig,
        *,
        y_training: np.ndarray | torch.Tensor,
        t_training_seconds: np.ndarray | torch.Tensor,
        reference: BaiCuiResidentialRCReference | None = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.reference = reference or BaiCuiResidentialRCReference()
        y = torch.as_tensor(y_training, dtype=torch.float64)
        if y.ndim == 1:
            y = y[:, None]
        expected_y = 1 if config.case_name == "all_to_one" else 2
        if y.shape[1] != expected_y:
            raise ValueError(f"{config.case_name} expects {expected_y} measured air temperatures, got {y.shape[1]}")
        t = torch.as_tensor(t_training_seconds, dtype=torch.float64).reshape(-1)
        if len(t) != len(y) or len(t) < 2:
            raise ValueError("t_training_seconds and y_training must have the same length >=2")

        y_scaler = TensorStandardizer.fit(y, names=("T_A",) if expected_y == 1 else ("T_D", "T_K"))
        state_mean, state_scale = self._expand_state_scaling(y_scaler.mean, y_scaler.scale)
        self.register_buffer("mu_x", state_mean.clone().detach())
        self.register_buffer("S_x", state_scale.clone().detach())
        self.register_buffer("mu_y", y_scaler.mean.clone().detach())
        self.register_buffer("S_y", y_scaler.scale.clone().detach())
        t_min, t_max = t.min(), t.max()
        t_center = 0.5 * (t_min + t_max)
        t_scale = 0.5 * (t_max - t_min)
        if float(t_scale) <= 0:
            raise ValueError("training time span must be positive")
        self.register_buffer("t_center", t_center.detach())
        self.register_buffer("s_t", t_scale.detach())

        self.trajectory_networks = self._build_trajectory_networks()
        self.rho_R = nn.ParameterDict()
        self.rho_C = nn.ParameterDict()
        self.rho_eta = nn.ParameterDict()
        self._initialize_physical_parameters()
        if config.case_name == "identity_dep2":
            self.alpha_c = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.alpha_r = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        else:
            self.register_parameter("alpha_c", None)
            self.register_parameter("alpha_r", None)

    @property
    def state_dim(self) -> int:
        if self.config.case_name == "all_to_one":
            return self.config.rc_order
        return 2 if self.config.rc_order == 1 else 4

    @property
    def output_dim(self) -> int:
        return 1 if self.config.case_name == "all_to_one" else 2

    @property
    def state_order(self) -> tuple[str, ...]:
        if self.config.case_name == "all_to_one":
            return ("T_A",) if self.config.rc_order == 1 else ("T_a,A", "T_m,A")
        return ("T_D", "T_K") if self.config.rc_order == 1 else ("T_a,D", "T_m,D", "T_a,K", "T_m,K")

    def _expand_state_scaling(self, mean_y: torch.Tensor, scale_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.rc_order == 1:
            return mean_y, scale_y
        if self.config.case_name == "all_to_one":
            return mean_y.repeat(2), scale_y.repeat(2)
        return (
            torch.stack((mean_y[0], mean_y[0], mean_y[1], mean_y[1])),
            torch.stack((scale_y[0], scale_y[0], scale_y[1], scale_y[1])),
        )

    def _build_trajectory_networks(self) -> nn.ModuleDict:
        c = self.config
        if c.case_name == "identity_ind":
            return nn.ModuleDict(
                {
                    "D": build_mlp(1, c.rc_order, hidden_layers=c.hidden_layers, hidden_width=c.hidden_width, activation=c.activation).double(),
                    "K": build_mlp(1, c.rc_order, hidden_layers=c.hidden_layers, hidden_width=c.hidden_width, activation=c.activation).double(),
                }
            )
        return nn.ModuleDict(
            {
                "joint": build_mlp(1, self.state_dim, hidden_layers=c.hidden_layers, hidden_width=c.hidden_width, activation=c.activation).double()
            }
        )

    def _initialize_physical_parameters(self) -> None:
        init = self.reference.paper_initialization()
        c = self.config
        zones = ("A",) if c.case_name == "all_to_one" else ("D", "K")
        for z in zones:
            self.rho_R[f"R_{z}o"] = _parameter_from_physical(init["R_out"], c.R_min)
            if c.rc_order == 1:
                self.rho_C[f"C_{z}"] = _parameter_from_physical(init["C_air"], c.C_min)
            else:
                self.rho_R[f"R_{z}m"] = _parameter_from_physical(init["R_mass"], c.R_min)
                self.rho_C[f"C_a_{z}"] = _parameter_from_physical(init["C_air"], c.C_min)
                self.rho_C[f"C_m_{z}"] = _parameter_from_physical(init["C_mass"], c.C_min)
        if c.case_name in {"identity_dep1", "identity_dep2"}:
            self.rho_R["R_DK"] = _parameter_from_physical(init["R_interzone"], c.R_min)

        eta_mode = c.eta_mode_1c if c.rc_order == 1 else c.eta_mode_2c
        if eta_mode == "learnable":
            eps = 1e-6
            eta0 = min(max(c.eta_fixed, eps), 1.0 - eps)
            rho0 = float(np.log(eta0 / (1.0 - eta0)))
            for z in zones:
                # DEP2-1C cannot get here due to __post_init__.
                self.rho_eta[f"eta_r_{z}"] = nn.Parameter(torch.tensor(rho0, dtype=torch.float64))

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        c = self.config
        out: dict[str, torch.Tensor] = {}
        for name, rho in self.rho_R.items():
            out[name] = c.R_min + F.softplus(rho)
        for name, rho in self.rho_C.items():
            out[name] = c.C_min + F.softplus(rho)
        zones = ("A",) if c.case_name == "all_to_one" else ("D", "K")
        for z in zones:
            out[f"eta_r_{z}"] = self.eta_r(z)
        if c.case_name == "identity_dep2":
            lam = self.dep2_allocations()
            out.update(lam)
        return out

    def eta_r(self, zone: str) -> torch.Tensor:
        c = self.config
        if c.case_name == "identity_dep2" and c.rc_order == 1:
            return self.mu_x.new_tensor(1.0)  # authoritative Part-2 identifiability safeguard
        mode = c.eta_mode_1c if c.rc_order == 1 else c.eta_mode_2c
        if mode in {"full", "mass_only"}:
            return self.mu_x.new_tensor(1.0)
        if mode in {"zero", "air_only"}:
            return self.mu_x.new_tensor(0.0)
        if mode == "fixed":
            return self.mu_x.new_tensor(c.eta_fixed)
        if mode == "learnable":
            return torch.sigmoid(self.rho_eta[f"eta_r_{zone}"])
        raise ValueError(f"Unknown eta mode {mode!r}")

    def dep2_allocations(self) -> dict[str, torch.Tensor]:
        if self.config.case_name != "identity_dep2" or self.alpha_c is None or self.alpha_r is None:
            raise RuntimeError("DEP2 allocations requested for a non-DEP2 model")
        sc = torch.sigmoid(self.alpha_c)
        sr = torch.sigmoid(self.alpha_r)
        return {
            "lambda_c_D": 2.0 * sc,
            "lambda_c_K": 2.0 * (1.0 - sc),
            "lambda_r_D": 2.0 * sr,
            "lambda_r_K": 2.0 * (1.0 - sr),
        }

    def _z_from_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """Evaluate N_phi through NeuroMANCER Node objects (Part-2 P2.6)."""

        inp = tau.reshape(-1, 1)
        if self.config.case_name == "identity_ind":
            node_D = node(self.trajectory_networks["D"], ["tau"], ["z_D"], name="InvPINN_N_phi_D")
            node_K = node(self.trajectory_networks["K"], ["tau"], ["z_K"], name="InvPINN_N_phi_K")
            zD = node_D({"tau": inp})["z_D"]
            zK = node_K({"tau": inp})["z_K"]
            if self.config.rc_order == 1:
                return torch.cat((zD, zK), dim=1)
            return torch.cat((zD[:, 0:1], zD[:, 1:2], zK[:, 0:1], zK[:, 1:2]), dim=1)
        trajectory_node = node(
            self.trajectory_networks["joint"], ["tau"], ["z_phi"], name="InvPINN_N_phi"
        )
        return trajectory_node({"tau": inp})["z_phi"]

    def state_and_dxdt(self, t_seconds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.as_tensor(t_seconds, dtype=self.mu_x.dtype, device=self.mu_x.device).reshape(-1)
        tau = ((t - self.t_center) / self.s_t).clone().detach().requires_grad_(True)
        z = self._z_from_tau(tau)
        x = self.mu_x + self.S_x * z
        dz_dtau_cols = []
        for j in range(z.shape[1]):
            grad = torch.autograd.grad(z[:, j].sum(), tau, create_graph=True, retain_graph=True)[0]
            dz_dtau_cols.append(grad)
        dz_dtau = torch.stack(dz_dtau_cols, dim=1)
        dxdt = (self.S_x / self.s_t) * dz_dtau
        return x, dxdt, tau

    def observed_air(self, state: torch.Tensor) -> torch.Tensor:
        if self.config.rc_order == 1:
            return state
        if self.config.case_name == "all_to_one":
            return state[:, 0:1]
        return state[:, [0, 2]]

    @staticmethod
    def _tensor_forcing(forcing: Mapping[str, Any], like: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in forcing.items():
            tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
            if tensor.ndim == 0:
                tensor = tensor.repeat(like.shape[0])
            out[key] = tensor.reshape(-1)
        return out

    @staticmethod
    def _get(f: Mapping[str, torch.Tensor], key: str, like: torch.Tensor) -> torch.Tensor:
        if key in f:
            return f[key]
        return torch.zeros(like.shape[0], dtype=like.dtype, device=like.device)

    def _zone_heat(self, f: Mapping[str, torch.Tensor], suffix: str, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Qc = self._get(f, f"Q_AC,{suffix}", like) + self._get(f, f"Q_ZIC,{suffix}", like) + self._get(f, f"Q_Sol1,{suffix}", like)
        Qr = self._get(f, f"Q_ZIR,{suffix}", like) + self._get(f, f"Q_Sol2,{suffix}", like)
        return Qc, Qr

    def rc_residuals(self, state: torch.Tensor, dxdt: torch.Tensor, forcing: Mapping[str, Any]) -> torch.Tensor:
        p = self.physical_parameters()
        f = self._tensor_forcing(forcing, state)
        To = f["T_o"]
        c = self.config

        if c.case_name == "all_to_one":
            Qc, Qr = self._zone_heat(f, "A", state)
            if c.rc_order == 1:
                T = state[:, 0]; dT = dxdt[:, 0]
                r = p["C_A"] * dT - (To - T) / p["R_Ao"] - Qc - p["eta_r_A"] * Qr
                return r[:, None]
            Ta, Tm = state[:, 0], state[:, 1]
            dTa, dTm = dxdt[:, 0], dxdt[:, 1]
            eta = p["eta_r_A"]
            ra = p["C_a_A"] * dTa - (To - Ta) / p["R_Ao"] - (Tm - Ta) / p["R_Am"] - Qc - (1.0 - eta) * Qr
            rm = p["C_m_A"] * dTm - (Ta - Tm) / p["R_Am"] - eta * Qr
            return torch.stack((ra, rm), dim=1)

        TD_idx, TK_idx = (0, 1) if c.rc_order == 1 else (0, 2)
        TD, TK = state[:, TD_idx], state[:, TK_idx]
        dTD, dTK = dxdt[:, TD_idx], dxdt[:, TK_idx]
        coupling_D = coupling_K = torch.zeros_like(TD)
        if c.case_name in {"identity_dep1", "identity_dep2"}:
            coupling_D = (TK - TD) / p["R_DK"]
            coupling_K = (TD - TK) / p["R_DK"]

        if c.case_name == "identity_dep2":
            lam = self.dep2_allocations()
            QcD = self._get(f, "Q_AC,D", state) + lam["lambda_c_D"] * f["Qbar_c_nh"]
            QcK = self._get(f, "Q_AC,K", state) + lam["lambda_c_K"] * f["Qbar_c_nh"]
            QrD = lam["lambda_r_D"] * f["Qbar_r"]
            QrK = lam["lambda_r_K"] * f["Qbar_r"]
        else:
            QcD, QrD = self._zone_heat(f, "D", state)
            QcK, QrK = self._zone_heat(f, "K", state)

        if c.rc_order == 1:
            rD = p["C_D"] * dTD - (To - TD) / p["R_Do"] - coupling_D - QcD - p["eta_r_D"] * QrD
            rK = p["C_K"] * dTK - (To - TK) / p["R_Ko"] - coupling_K - QcK - p["eta_r_K"] * QrK
            return torch.stack((rD, rK), dim=1)

        TmD, TmK = state[:, 1], state[:, 3]
        dTmD, dTmK = dxdt[:, 1], dxdt[:, 3]
        etaD, etaK = p["eta_r_D"], p["eta_r_K"]
        raD = p["C_a_D"] * dTD - (To - TD) / p["R_Do"] - (TmD - TD) / p["R_Dm"] - coupling_D - QcD - (1.0 - etaD) * QrD
        rmD = p["C_m_D"] * dTmD - (TD - TmD) / p["R_Dm"] - etaD * QrD
        raK = p["C_a_K"] * dTK - (To - TK) / p["R_Ko"] - (TmK - TK) / p["R_Km"] - coupling_K - QcK - (1.0 - etaK) * QrK
        rmK = p["C_m_K"] * dTmK - (TK - TmK) / p["R_Km"] - etaK * QrK
        return torch.stack((raD, rmD, raK, rmK), dim=1)

    def loss(
        self,
        *,
        t_seconds: torch.Tensor,
        y_measured: torch.Tensor,
        forcing: Mapping[str, Any],
        q_star: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state, dxdt, _ = self.state_and_dxdt(t_seconds)
        y = torch.as_tensor(y_measured, dtype=state.dtype, device=state.device)
        if y.ndim == 1:
            y = y[:, None]
        yhat = self.observed_air(state)
        Ly = torch.mean(((yhat - y) / self.S_y) ** 2)
        residual = self.rc_residuals(state, dxdt, forcing)
        if q_star is None:
            q_star = torch.sqrt(torch.mean(residual.detach() ** 2, dim=0)).clamp_min(1.0)
        qscale = torch.as_tensor(q_star, dtype=residual.dtype, device=residual.device).reshape(1, -1).clamp_min(1.0)
        Lf = torch.mean((residual / qscale) ** 2)
        total = self.config.lambda_y * Ly + self.config.lambda_f * Lf
        return {"total": total, "data": Ly, "physics": Lf, "residual": residual, "state": state, "yhat": yhat}

    @property
    def physical_forcing_keys(self) -> tuple[str, ...]:
        if self.config.case_name == "all_to_one":
            return ("T_o", "Q_AC,A", "Q_ZIC,A", "Q_ZIR,A", "Q_Sol1,A", "Q_Sol2,A")
        if self.config.case_name in {"identity_ind", "identity_dep1"}:
            return (
                "T_o",
                "Q_AC,D", "Q_ZIC,D", "Q_ZIR,D", "Q_Sol1,D", "Q_Sol2,D",
                "Q_AC,K", "Q_ZIC,K", "Q_ZIR,K", "Q_Sol1,K", "Q_Sol2,K",
            )
        return ("T_o", "Q_AC,D", "Q_AC,K", "Qbar_c_nh", "Qbar_r")

    def _physical_rhs_batch(
        self, state: torch.Tensor, forcing: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Retained physical RC ODE in seconds; N_phi is not used."""

        if state.ndim != 2:
            raise ValueError("NeuroMANCER RC RHS expects state [batch, n_x]")
        zero = torch.zeros_like(state)
        # residual = C*dxdt - balance; residual(dxdt=0) = -balance.
        r0 = self.rc_residuals(state, zero, forcing)
        p = self.physical_parameters()
        if self.config.case_name == "all_to_one":
            C = (
                torch.stack((p["C_A"],))
                if self.config.rc_order == 1
                else torch.stack((p["C_a_A"], p["C_m_A"]))
            )
        else:
            C = (
                torch.stack((p["C_D"], p["C_K"]))
                if self.config.rc_order == 1
                else torch.stack((p["C_a_D"], p["C_m_D"], p["C_a_K"], p["C_m_K"]))
            )
        return -r0 / C.reshape(1, -1)

    def _physical_rhs_tensor(self, state: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        forcing = {key: v[:, i] for i, key in enumerate(self.physical_forcing_keys)}
        return self._physical_rhs_batch(state, forcing)

    def physical_rollout(
        self,
        x0: torch.Tensor,
        forcing_sequence: Sequence[Mapping[str, Any]],
        *,
        dt_seconds: float = 300.0,
        n_substeps: int = 1,
    ) -> torch.Tensor:
        """Deploy identified RC ODE with NeuroMANCER Node/System + RK4 only."""

        x0_t = torch.as_tensor(x0, dtype=self.mu_x.dtype, device=self.mu_x.device).reshape(1, 1, -1)
        keys = self.physical_forcing_keys
        rows = [
            [float(torch.as_tensor(row.get(key, 0.0)).detach().cpu()) for key in keys]
            for row in forcing_sequence
        ]
        if not rows:
            return x0_t.squeeze(0)
        v = torch.as_tensor(rows, dtype=x0_t.dtype, device=x0_t.device).unsqueeze(0)

        def interval_map(x: torch.Tensor, v_k: torch.Tensor) -> torch.Tensor:
            return rk4_interval(
                self._physical_rhs_tensor,
                x,
                v_k,
                state_dim=self.state_dim,
                extra_dim=len(keys),
                n_substeps=n_substeps,
                interval_length=float(dt_seconds),
            )

        rc_node = node(interval_map, ["x", "v"], ["x"], name="InvPINN_retained_RC_RK4")
        rc_system = system([rc_node], nstep_key="v", name="InvPINN_retained_RC_System")
        result = rc_system({"x": x0_t, "v": v})
        return result["x"].squeeze(0)

    def provenance(self) -> dict[str, Any]:
        return {
            "method": "inverse_pinn_rc",
            "math_contract": "PINODE_EPSR_Part2_Inverse_PINN_RC.tex",
            "config": asdict(self.config),
            "state_order": list(self.state_order),
            "rc_initialization": self.reference.paper_initialization(),
            "rc_initialization_source": self.reference.source_note,
            "auxiliary_reference_values_not_mapped_to_capacitance": {"C1": self.reference.C1, "C2": self.reference.C2, "C3": self.reference.C3},
            "deployment": "physical_rc_ode_only",
            "framework": {
                "sciml": "neuromancer",
                "tensor_autograd_optimizer": "pytorch",
                "integration": "neuromancer.dynamics.integrators.RK4",
                "direct_torchdiffeq_calls": False,
                "runtime": runtime_info().__dict__,
            },
        }
