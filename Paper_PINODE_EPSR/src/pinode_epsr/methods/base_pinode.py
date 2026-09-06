from __future__ import annotations

"""Part-4 Base PINODE: neural ODE dynamics with a soft full-RC residual.

Scientific contract
-------------------
Authoritative source: ``PINODE_EPSR_Part4_Base_PINODE_Detailed.tex``.

The model evolves the normalized state with the unrestricted neural derivative

    dz/dtau = g_omega(z, v_tilde)

and converts every neural derivative evaluation back to physical units

    f_tilde_omega = (S_x / Delta t) g_omega.

The complete selected RC model is evaluated as a *soft* residual at every RHS
call made by NeuroMANCER RK4.  The raw neural derivative is returned unchanged
to the integrator: there is no projection in Base PINODE.

Framework division
------------------
* NeuroMANCER: MLP blocks, Node/System rollout graph, RK4 integration,
  Problem/PenaltyLoss through the shared training helper.
* PyTorch: tensors, autograd, parameter transforms, optimizer.
* Optuna: hyperparameter search via ``training.suggest_base_pinode_hyperparameters``.
* No direct torchdiffeq call and no home-written RK4 formula in paper code.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .inverse_pinn import BaiCuiResidentialRCReference
from .neural_ode import NeuralODEConfig, NeuralODEModel
from ..backends.neuromancer import rk4_interval, runtime_info

EtaMode = Literal["auto", "full", "zero", "fixed", "learnable", "mass_only", "air_only"]


@dataclass(frozen=True)
class BasePINODEConfig(NeuralODEConfig):
    """Part-4 Base-PINODE configuration.

    ``eta_mode_*='auto'`` means learn eta_r wherever Part 4 permits it, while
    the primary DEP2-1C model is automatically locked to eta_r=1 to preserve
    the Part-4 identifiability safeguard.
    """

    lambda_y: float = 1.0
    lambda_f: float = 1.0
    eta_mode_1c: EtaMode = "auto"
    eta_mode_2c: EtaMode = "auto"
    eta_fixed: float = 0.5
    R_min: float = 1e-6
    C_min: float = 1e3
    dt_seconds: float = 300.0
    epsilon_q: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lambda_y < 0.0 or self.lambda_f < 0.0 or self.lambda_wd < 0.0:
            raise ValueError("lambda_y, lambda_f and lambda_wd must be nonnegative")
        if self.dt_seconds <= 0.0:
            raise ValueError("dt_seconds must be positive")
        if self.R_min <= 0.0 or self.C_min <= 0.0:
            raise ValueError("R_min and C_min must be positive")
        if self.epsilon_q <= 0.0:
            raise ValueError("epsilon_q must be positive")
        if not 0.0 < self.eta_fixed < 1.0:
            raise ValueError("eta_fixed must be strictly between 0 and 1")
        valid = {"auto", "full", "zero", "fixed", "learnable", "mass_only", "air_only"}
        if self.eta_mode_1c not in valid or self.eta_mode_2c not in valid:
            raise ValueError(f"eta mode must be one of {sorted(valid)}")


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    if value > 30.0:
        return value
    return float(np.log(np.expm1(value)))


def _parameter_from_physical(initial: float, minimum: float) -> nn.Parameter:
    if initial <= minimum:
        initial = minimum * 1.01
    rho = _inverse_softplus(initial - minimum)
    return nn.Parameter(torch.tensor(rho, dtype=torch.float64))


def _logit(p: float) -> float:
    eps = 1e-8
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


class BasePINODEModel(NeuralODEModel):
    """Part-4 soft-physics PINODE for all four paper spatial architectures.

    The neural vector field and 2C causal encoder are inherited from the
    validated Day-3 NODE implementation.  This class adds an independently
    initialized trainable RC parameterization plus stage-wise residual logging.

    Importantly, the return value of ``_stage_rhs`` is *exactly* ``g_omega``.
    The RC residual is only appended to the training objective; it never alters
    the derivative that NeuroMANCER RK4 integrates.
    """

    def __init__(
        self,
        config: BasePINODEConfig,
        *,
        y_training: np.ndarray | torch.Tensor,
        v_training: np.ndarray
        | Mapping[str, np.ndarray]
        | torch.Tensor
        | Mapping[str, torch.Tensor],
        y_names: Sequence[str] | None = None,
        v_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
        reference: BaiCuiResidentialRCReference | None = None,
    ) -> None:
        super().__init__(
            config,
            y_training=y_training,
            v_training=v_training,
            y_names=y_names,
            v_names=v_names,
        )
        self.config: BasePINODEConfig = config
        self.reference = reference or BaiCuiResidentialRCReference()

        self.rho_R = nn.ParameterDict()
        self.rho_C = nn.ParameterDict()
        self.rho_eta = nn.ParameterDict()
        self._initialize_physical_parameters()

        if config.case_name == "identity_dep2":
            # alpha=0 -> lambda_D=lambda_K=1, the neutral equal-allocation start.
            self.alpha_c = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.alpha_r = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        else:
            self.register_parameter("alpha_c", None)
            self.register_parameter("alpha_r", None)

        self._assert_semantic_forcing_names()
        q_zone = self._compute_training_q_star(v_training).detach()
        self.register_buffer("q_star_zone", q_zone)
        self.register_buffer("q_star_residual", self._expand_q_star(q_zone))

        # Python list intentionally used only as an autograd-preserving collector
        # during one rollout.  It is cleared at every rollout start.
        self._stage_residuals: list[torch.Tensor] = []
        self._stage_raw_derivatives: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Physical parameterization: Part 4 P4.14--P4.25
    # ------------------------------------------------------------------
    def _resolved_eta_mode(self) -> str:
        c = self.config
        if c.case_name == "identity_dep2" and c.rc_order == 1:
            return "full"  # P4.34--P4.35 identifiability lock
        mode = c.eta_mode_1c if c.rc_order == 1 else c.eta_mode_2c
        if mode == "auto":
            return "learnable"
        return mode

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

        if self._resolved_eta_mode() == "learnable":
            rho0 = _logit(c.eta_fixed)
            for z in zones:
                self.rho_eta[f"eta_r_{z}"] = nn.Parameter(
                    torch.tensor(rho0, dtype=torch.float64)
                )

    def eta_r(self, zone: str) -> torch.Tensor:
        mode = self._resolved_eta_mode()
        if mode in {"full", "mass_only"}:
            return self.mu_x.new_tensor(1.0)
        if mode in {"zero", "air_only"}:
            return self.mu_x.new_tensor(0.0)
        if mode == "fixed":
            return self.mu_x.new_tensor(self.config.eta_fixed)
        if mode == "learnable":
            return torch.sigmoid(self.rho_eta[f"eta_r_{zone}"])
        raise ValueError(f"Unknown eta mode {mode!r}")

    def dep2_allocations(self) -> dict[str, torch.Tensor]:
        if self.config.case_name != "identity_dep2" or self.alpha_c is None or self.alpha_r is None:
            raise RuntimeError("DEP2 allocations requested for a non-DEP2 Base PINODE")
        sc = torch.sigmoid(self.alpha_c)
        sr = torch.sigmoid(self.alpha_r)
        return {
            "lambda_c_D": 2.0 * sc,
            "lambda_c_K": 2.0 * (1.0 - sc),
            "lambda_r_D": 2.0 * sr,
            "lambda_r_K": 2.0 * (1.0 - sr),
        }

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        c = self.config
        out: dict[str, torch.Tensor] = {}
        for name, rho in self.rho_R.items():
            out[name] = c.R_min + F.softplus(rho)
        for name, rho in self.rho_C.items():
            out[name] = c.C_min + F.softplus(rho)
        zones = ("A",) if c.case_name == "all_to_one" else ("D", "K")
        for zone in zones:
            out[f"eta_r_{zone}"] = self.eta_r(zone)
        if c.case_name == "identity_dep2":
            out.update(self.dep2_allocations())
        return out

    # ------------------------------------------------------------------
    # Raw forcing -> physical RC forcing dictionary
    # ------------------------------------------------------------------
    def _assert_semantic_forcing_names(self) -> None:
        required: set[str]
        if self.config.case_name == "all_to_one":
            required = {"T_o", "Q_AC,A", "Q_ZIC,A", "Q_ZIR,A", "Q_Sol1,A", "Q_Sol2,A"}
        elif self.config.case_name == "identity_ind":
            required = {
                "T_o", "Q_AC,D", "Q_ZIC,D", "Q_ZIR,D", "Q_Sol1,D", "Q_Sol2,D",
                "Q_AC,K", "Q_ZIC,K", "Q_ZIR,K",
            }
        elif self.config.case_name == "identity_dep1":
            required = {
                "T_o", "Q_AC,D", "Q_AC,K", "Q_ZIC,D", "Q_ZIR,D",
                "Q_Sol1,D", "Q_Sol2,D", "Q_ZIC,K", "Q_ZIR,K",
            }
        else:
            required = {
                "T_o", "Q_AC,D", "Q_AC,K", "Q_ZIC,A", "Q_ZIR,A", "Q_Sol1,A", "Q_Sol2,A",
            }
        available = set()
        for names in self._v_feature_names.values():
            available.update(names)
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                "Base PINODE requires semantic Part-4 forcing names; missing "
                + ", ".join(missing)
            )

    @staticmethod
    def _to_tensor(value: Any, *, like: torch.Tensor | None = None) -> torch.Tensor:
        if like is None:
            return torch.as_tensor(value, dtype=torch.float64)
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def _raw_forcing_dict(
        self,
        v: torch.Tensor | Mapping[str, torch.Tensor] | np.ndarray | Mapping[str, np.ndarray],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.config.case_name == "identity_ind":
            if not isinstance(v, Mapping):
                raise TypeError("identity_ind Base PINODE requires Dining/Kitchen forcing mapping")
            for canonical, alias in (("Dining", "D"), ("Kitchen", "K")):
                value = v.get(canonical, v.get(alias))
                if value is None:
                    raise KeyError(f"identity_ind forcing missing {canonical}/{alias}")
                tensor = self._to_tensor(value)
                names = self._v_feature_names[canonical]
                if tensor.shape[-1] != len(names):
                    raise ValueError(f"{canonical} forcing width does not match semantic names")
                for j, name in enumerate(names):
                    # T_o appears in both independent zone vectors; the physical
                    # environment value is common, so the first copy is sufficient.
                    if name not in out:
                        out[name] = tensor[..., j]
            return out

        if isinstance(v, Mapping):
            raise TypeError(f"{self.config.case_name} Base PINODE requires a joint forcing tensor")
        tensor = self._to_tensor(v)
        names = self._v_feature_names["joint"]
        if tensor.shape[-1] != len(names):
            raise ValueError("joint forcing width does not match semantic names")
        return {name: tensor[..., j] for j, name in enumerate(names)}

    @staticmethod
    def _get(f: Mapping[str, torch.Tensor], key: str, like: torch.Tensor) -> torch.Tensor:
        if key in f:
            return f[key].to(like)
        return torch.zeros(like.shape[:-1], dtype=like.dtype, device=like.device)

    def _zone_heat(
        self, f: Mapping[str, torch.Tensor], suffix: str, like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Qc = (
            self._get(f, f"Q_AC,{suffix}", like)
            + self._get(f, f"Q_ZIC,{suffix}", like)
            + self._get(f, f"Q_Sol1,{suffix}", like)
        )
        Qr = self._get(f, f"Q_ZIR,{suffix}", like) + self._get(
            f, f"Q_Sol2,{suffix}", like
        )
        return Qc, Qr

    # ------------------------------------------------------------------
    # P4.40--P4.46 full RC residual in physical units
    # ------------------------------------------------------------------
    def f_tilde_omega(
        self,
        z: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Raw neural physical-time derivative, Eq. P4.3/P4.67."""

        g = self.g_omega(z, v)
        return (self.S_x.to(g) / g.new_tensor(self.config.dt_seconds)) * g

    def rc_residuals(
        self,
        state: torch.Tensor,
        f_tilde: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Full selected RC residual M_theta f_tilde - b_theta in watts."""

        if state.ndim != 2 or f_tilde.ndim != 2:
            raise ValueError("Base PINODE RC residual expects [batch, n_x] state/derivative")
        if state.shape != f_tilde.shape:
            raise ValueError("state and f_tilde must have identical shape")

        p = self.physical_parameters()
        f = self._raw_forcing_dict(v)
        To = f["T_o"].to(state)
        c = self.config

        if c.case_name == "all_to_one":
            Qc, Qr = self._zone_heat(f, "A", state)
            if c.rc_order == 1:
                T = state[:, 0]
                dT = f_tilde[:, 0]
                r = p["C_A"] * dT - (To - T) / p["R_Ao"] - Qc - p["eta_r_A"] * Qr
                return r[:, None]
            Ta, Tm = state[:, 0], state[:, 1]
            dTa, dTm = f_tilde[:, 0], f_tilde[:, 1]
            eta = p["eta_r_A"]
            ra = (
                p["C_a_A"] * dTa
                - (To - Ta) / p["R_Ao"]
                - (Tm - Ta) / p["R_Am"]
                - Qc
                - (1.0 - eta) * Qr
            )
            rm = p["C_m_A"] * dTm - (Ta - Tm) / p["R_Am"] - eta * Qr
            return torch.stack((ra, rm), dim=1)

        TD_idx, TK_idx = (0, 1) if c.rc_order == 1 else (0, 2)
        TD, TK = state[:, TD_idx], state[:, TK_idx]
        dTD, dTK = f_tilde[:, TD_idx], f_tilde[:, TK_idx]
        coupling_D = coupling_K = torch.zeros_like(TD)
        if c.case_name in {"identity_dep1", "identity_dep2"}:
            coupling_D = (TK - TD) / p["R_DK"]
            coupling_K = (TD - TK) / p["R_DK"]

        if c.case_name == "identity_dep2":
            lam = self.dep2_allocations()
            qbar_c = (f["Q_ZIC,A"] + f["Q_Sol1,A"]).to(state)
            qbar_r = (f["Q_ZIR,A"] + f["Q_Sol2,A"]).to(state)
            QcD = f["Q_AC,D"].to(state) + lam["lambda_c_D"] * qbar_c
            QcK = f["Q_AC,K"].to(state) + lam["lambda_c_K"] * qbar_c
            QrD = lam["lambda_r_D"] * qbar_r
            QrK = lam["lambda_r_K"] * qbar_r
        else:
            QcD, QrD = self._zone_heat(f, "D", state)
            QcK, QrK = self._zone_heat(f, "K", state)

        if c.rc_order == 1:
            rD = (
                p["C_D"] * dTD
                - (To - TD) / p["R_Do"]
                - coupling_D
                - QcD
                - p["eta_r_D"] * QrD
            )
            rK = (
                p["C_K"] * dTK
                - (To - TK) / p["R_Ko"]
                - coupling_K
                - QcK
                - p["eta_r_K"] * QrK
            )
            return torch.stack((rD, rK), dim=1)

        TmD, TmK = state[:, 1], state[:, 3]
        dTmD, dTmK = f_tilde[:, 1], f_tilde[:, 3]
        etaD, etaK = p["eta_r_D"], p["eta_r_K"]
        raD = (
            p["C_a_D"] * dTD
            - (To - TD) / p["R_Do"]
            - (TmD - TD) / p["R_Dm"]
            - coupling_D
            - QcD
            - (1.0 - etaD) * QrD
        )
        rmD = p["C_m_D"] * dTmD - (TD - TmD) / p["R_Dm"] - etaD * QrD
        raK = (
            p["C_a_K"] * dTK
            - (To - TK) / p["R_Ko"]
            - (TmK - TK) / p["R_Km"]
            - coupling_K
            - QcK
            - (1.0 - etaK) * QrK
        )
        rmK = p["C_m_K"] * dTmK - (TK - TmK) / p["R_Km"] - etaK * QrK
        return torch.stack((raD, rmD, raK, rmK), dim=1)

    # ------------------------------------------------------------------
    # P4.141 training-only residual scales
    # ------------------------------------------------------------------
    def _compute_training_q_star(self, v_training: Any) -> torch.Tensor:
        f = self._raw_forcing_dict(v_training)
        like = self._to_tensor(next(iter(f.values()))).reshape(-1, 1)
        eps = like.new_tensor(self.config.epsilon_q)

        if self.config.case_name == "all_to_one":
            Qc, Qr = self._zone_heat(f, "A", like)
            qA = torch.sqrt(torch.mean(Qc**2 + Qr**2)) + eps
            return qA.reshape(1)

        if self.config.case_name == "identity_dep2":
            # The scale is fitted once from training forcing and then frozen.
            # At initialization alpha_c=alpha_r=0, therefore lambda=1 for both
            # zones.  This neutral allocation prevents a trainable parameter
            # from changing the denominator of its own physics penalty.
            qbar_c = self._to_tensor(f["Q_ZIC,A"] + f["Q_Sol1,A"])
            qbar_r = self._to_tensor(f["Q_ZIR,A"] + f["Q_Sol2,A"])
            QcD = self._to_tensor(f["Q_AC,D"]) + qbar_c
            QcK = self._to_tensor(f["Q_AC,K"]) + qbar_c
            QrD = qbar_r
            QrK = qbar_r
        else:
            QcD, QrD = self._zone_heat(f, "D", like)
            QcK, QrK = self._zone_heat(f, "K", like)

        qD = torch.sqrt(torch.mean(QcD**2 + QrD**2)) + eps
        qK = torch.sqrt(torch.mean(QcK**2 + QrK**2)) + eps
        return torch.stack((qD, qK))

    def _expand_q_star(self, q_zone: torch.Tensor) -> torch.Tensor:
        if self.config.rc_order == 1:
            return q_zone.clone()
        if self.config.case_name == "all_to_one":
            return q_zone.repeat(2)
        return torch.stack((q_zone[0], q_zone[0], q_zone[1], q_zone[1]))

    # ------------------------------------------------------------------
    # True NeuroMANCER RK4 stage residual collection: P4.46--P4.60
    # ------------------------------------------------------------------
    def _stage_rhs(
        self,
        z: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        g = self.g_omega(z, v)
        x = self.mu_x.to(z) + self.S_x.to(z) * z
        f_tilde = (self.S_x.to(g) / g.new_tensor(self.config.dt_seconds)) * g
        residual = self.rc_residuals(x, f_tilde, v)
        self._stage_raw_derivatives.append(f_tilde)
        self._stage_residuals.append(residual)
        # P4.4/P4.43/P4.150: integrate the raw derivative, never a correction.
        return g

    def step(
        self, z: torch.Tensor, v: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """One sampled interval with raw g_omega integrated by NeuroMANCER RK4."""

        extra_dim = int(v.shape[-1]) if isinstance(v, torch.Tensor) else 0
        return rk4_interval(
            lambda zz, vv: self._stage_rhs(zz, vv),
            z,
            v,
            state_dim=self.state_dim,
            extra_dim=extra_dim,
            n_substeps=self.config.N_s,
            interval_length=1.0,
        )

    def rollout(
        self,
        *,
        y0: torch.Tensor,
        v_sequence: torch.Tensor | Mapping[str, torch.Tensor],
        context_y: torch.Tensor | None = None,
        context_v: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._stage_residuals = []
        self._stage_raw_derivatives = []
        # Parent rollout is already the validated NeuroMANCER Node/System graph;
        # because it dispatches to self.step(), every RK4 RHS call is captured.
        return super().rollout(
            y0=y0,
            v_sequence=v_sequence,
            context_y=context_y,
            context_v=context_v,
        )

    @property
    def stage_residual_count(self) -> int:
        return len(self._stage_residuals)

    def stage_residual_tensor(self) -> torch.Tensor:
        if not self._stage_residuals:
            raise RuntimeError("No RK4 stage residuals have been collected; run a rollout first")
        return torch.stack(self._stage_residuals, dim=0)

    def stage_raw_derivative_tensor(self) -> torch.Tensor:
        if not self._stage_raw_derivatives:
            raise RuntimeError("No RK4 stage derivatives have been collected; run a rollout first")
        return torch.stack(self._stage_raw_derivatives, dim=0)

    def neural_regularization(self) -> torch.Tensor:
        """P4.148: regularize omega and psi only, never theta_RC."""

        total = torch.zeros((), dtype=self.mu_x.dtype, device=self.mu_x.device)
        for module in (self.vector_fields, self.encoders):
            for p in module.parameters():
                total = total + torch.sum(p**2)
        return total

    def rollout_loss(
        self,
        *,
        y_true: torch.Tensor,
        v_sequence: torch.Tensor | Mapping[str, torch.Tensor],
        context_y: torch.Tensor | None = None,
        context_v: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Literal Part-4 one-window objective, Eqs. P4.47/P4.144/P4.147."""

        y_true = torch.as_tensor(y_true, dtype=self.mu_x.dtype, device=self.mu_x.device)
        if y_true.shape[-2] < 2:
            raise ValueError("y_true must include initial sample plus >=1 future sample")
        y0 = y_true[..., 0, :]
        yhat, z_hist = self.rollout(
            y0=y0,
            v_sequence=v_sequence,
            context_y=context_y,
            context_v=context_v,
        )

        # P4.144: 1/N_r sum_l ||S_y^-1 (yhat-y)||_2^2.
        err = (yhat[..., 1:, :] - y_true[..., 1:, :]) / self.S_y.to(yhat)
        Ldata = torch.sum(err**2, dim=-1).mean()

        stages = self.stage_residual_tensor()
        scale = self.q_star_residual.to(stages).reshape(1, 1, -1)
        normalized = stages / scale
        # P4.47: mean over rollout/substep/stage of squared L2 residual norm.
        Lphys = torch.sum(normalized**2, dim=-1).mean()

        Lreg = self.neural_regularization()
        total = (
            self.config.lambda_y * Ldata
            + self.config.lambda_f * Lphys
            + self.config.lambda_wd * Lreg
        )
        return {
            "total": total,
            "data": Ldata,
            "physics": Lphys,
            "regularization": Lreg,
            "yhat": yhat,
            "z_hist": z_hist,
            "stage_residual": stages,
            "stage_raw_derivative": self.stage_raw_derivative_tensor(),
        }

    # ------------------------------------------------------------------
    # P4.151 NODE warm-start support
    # ------------------------------------------------------------------
    def warm_start_from_node(self, node_model: NeuralODEModel) -> None:
        """Copy only omega/psi from a compatible NODE; never copy RC parameters."""

        if node_model.config.case_name != self.config.case_name or node_model.config.rc_order != self.config.rc_order:
            raise ValueError("NODE warm start must use the same case_name and rc_order")
        for name in ("mu_x", "S_x", "mu_y", "S_y"):
            if not torch.allclose(getattr(self, name), getattr(node_model, name)):
                raise ValueError(f"NODE warm start normalization mismatch for {name}")
        if set(self.v_scalers) != set(node_model.v_scalers):
            raise ValueError("NODE warm start forcing-scaler keys do not match")
        for key in self.v_scalers:
            a_mu, a_s = self._v_stats(key)
            b_mu, b_s = node_model._v_stats(key)
            if not torch.allclose(a_mu, b_mu) or not torch.allclose(a_s, b_s):
                raise ValueError(f"NODE warm start forcing normalization mismatch for {key}")
        self.vector_fields.load_state_dict(node_model.vector_fields.state_dict())
        self.encoders.load_state_dict(node_model.encoders.state_dict())

    def provenance(self) -> dict[str, Any]:
        v_dims = {key: int(self._v_stats(key)[0].numel()) for key in self.v_scalers}
        return {
            "method": "base_pinode",
            "math_contract": "PINODE_EPSR_Part4_Base_PINODE_Detailed.tex",
            "config": asdict(self.config),
            "state_order": list(self.state_order),
            "forcing_dimensions": v_dims,
            "rc_initialization": self.reference.paper_initialization(),
            "rc_initialization_source": self.reference.source_note,
            "rc_initialization_is_truth": False,
            "physics": {
                "residual": "full_selected_RC_model",
                "constraint_type": "soft_penalty",
                "hard_projection": False,
                "integrated_derivative": "raw_f_tilde_omega",
                "stage_evaluation": "every_Neuromancer_RK4_RHS_call",
                "q_star": self.q_star_zone.detach().cpu().tolist(),
                "dep2_q_star_convention": "frozen training-only neutral allocation lambda=1 at initialization",
            },
            "normalization": {
                "state": "training-only mu_x/S_x",
                "forcing": "training-only mu_v/S_v",
                "output_loss": "training-only S_y",
                "physical_derivative": "f_tilde_omega=(S_x/dt_seconds)g_omega",
            },
            "framework": {
                "sciml": "neuromancer",
                "tensor_autograd_optimizer": "pytorch",
                "hyperparameter_search": "optuna",
                "integration": "neuromancer.dynamics.integrators.RK4",
                "recursive_graph": "neuromancer.system.Node+System",
                "named_data": "neuromancer.dataset.DictDataset + collate_fn",
                "direct_torchdiffeq_calls": False,
                "runtime": runtime_info().__dict__,
            },
        }
