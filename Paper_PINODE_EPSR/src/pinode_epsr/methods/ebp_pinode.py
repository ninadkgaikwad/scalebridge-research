from __future__ import annotations

"""Part-5 Energy-Balance-Projected PINODE (EBP-PINODE).

Authoritative mathematical source:
``PINODE_EPSR_Part5_EBP_PINODE_Detailed.tex``.

The raw neural field proposes a normalized derivative

    g_omega(z, v_tilde) = dz/dtau,

which is converted to a physical derivative

    f_tilde_omega = (S_x / Delta t) g_omega.

At every actual NeuroMANCER RK4 right-hand-side evaluation, this raw derivative
is projected onto the exact learned zone-energy manifold A f = b using the
weighted projection

    rho = A f_tilde_omega - b
    M   = A W^{-1} A^T
    solve M nu = rho
    f_P = f_tilde_omega - W^{-1} A^T nu.

The normalized derivative actually returned to NeuroMANCER RK4 is

    g_P = Delta t S_x^{-1} f_P.

Thus projection is part of both training and deployment.  No hard-balance
penalty is needed because the forward model itself satisfies A f_P = b up to
floating-point tolerance.

For 2C, one independent air/mass redistribution residual per zone remains after
the hard total-energy projection and is penalized softly.  For 1C, no internal
neural derivative freedom remains and the internal-physics term is absent.

Framework division
------------------
* NeuroMANCER: MLP blocks inherited from NODE/Base PINODE, Node/System recursive
  graph, and RK4 numerical integration.
* PyTorch: tensors/autograd, positive parameter transforms, differentiable
  ``torch.linalg.solve``, and optimizer.
* Optuna: representative-training-only hyperparameter search via
  ``training.suggest_ebp_pinode_hyperparameters``.
* No explicit matrix inverse, no custom RK4, and no direct torchdiffeq call.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .base_pinode import BasePINODEConfig, BasePINODEModel
from ..backends.neuromancer import rk4_interval, runtime_info


@dataclass(frozen=True)
class EBPPINODEConfig(BasePINODEConfig):
    """Part-5 EBP-PINODE configuration.

    ``lambda_int`` applies only to the remaining 2C internal air/mass RC
    residual.  ``lambda_corr`` penalizes weighted projection correction energy
    c^T W c; it is optional for conservation but useful for encouraging the raw
    neural proposal to remain near the feasible manifold.
    """

    lambda_int: float = 1.0
    lambda_corr: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lambda_int < 0.0 or self.lambda_corr < 0.0:
            raise ValueError("lambda_int and lambda_corr must be nonnegative")


from ..physics.energy_projection import weighted_energy_projection
class EBPPINODEModel(BasePINODEModel):
    """Part-5 EBP-PINODE for all paper spatial architectures and 1C/2C orders."""

    def __init__(self, config: EBPPINODEConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.config: EBPPINODEConfig = config
        self._clear_projection_stage_cache()

    def _clear_projection_stage_cache(self) -> None:
        self._stage_raw_derivatives = []
        self._stage_projected_derivatives: list[torch.Tensor] = []
        self._stage_rho: list[torch.Tensor] = []
        self._stage_rho_P: list[torch.Tensor] = []
        self._stage_M: list[torch.Tensor] = []
        self._stage_nu: list[torch.Tensor] = []
        self._stage_correction: list[torch.Tensor] = []
        self._stage_correction_energy: list[torch.Tensor] = []
        self._stage_rho_solve_energy: list[torch.Tensor] = []
        self._stage_stationarity: list[torch.Tensor] = []
        self._stage_stationarity_relative: list[torch.Tensor] = []
        self._stage_internal_residuals: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Part-5 architecture-specific exact total-energy manifold A f = b
    # ------------------------------------------------------------------
    def energy_constraint(
        self,
        state: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``A``, ``b`` and diagonal ``W`` in physical units.

        The primary Part-5 metric is capacitance-squared:
        W_i = diag(C_i^2) for 1C and
        W_i = diag(C_a,i^2, C_m,i^2) for 2C.
        """

        if state.ndim != 2:
            raise ValueError("EBP energy constraint expects state shape [batch, n_x]")
        p = self.physical_parameters()
        f = self._raw_forcing_dict(v)
        To = f["T_o"].to(state)
        c = self.config
        batch = state.shape[0]

        if c.case_name == "all_to_one":
            Qc, Qr = self._zone_heat(f, "A", state)
            if c.rc_order == 1:
                T = state[:, 0]
                C = p["C_A"].to(state)
                b = (To - T) / p["R_Ao"] + Qc + p["eta_r_A"] * Qr
                A = torch.zeros((batch, 1, 1), dtype=state.dtype, device=state.device)
                A[:, 0, 0] = C
                W_diag = torch.stack((C.square(),))
                return A, b[:, None], W_diag

            Ta = state[:, 0]
            Ca, Cm = p["C_a_A"].to(state), p["C_m_A"].to(state)
            b = (To - Ta) / p["R_Ao"] + Qc + Qr
            A = torch.zeros((batch, 1, 2), dtype=state.dtype, device=state.device)
            A[:, 0, 0] = Ca
            A[:, 0, 1] = Cm
            W_diag = torch.stack((Ca.square(), Cm.square()))
            return A, b[:, None], W_diag

        TD_idx, TK_idx = (0, 1) if c.rc_order == 1 else (0, 2)
        TD, TK = state[:, TD_idx], state[:, TK_idx]
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
            CD, CK = p["C_D"].to(state), p["C_K"].to(state)
            bD = (
                (To - TD) / p["R_Do"]
                + coupling_D
                + QcD
                + p["eta_r_D"] * QrD
            )
            bK = (
                (To - TK) / p["R_Ko"]
                + coupling_K
                + QcK
                + p["eta_r_K"] * QrK
            )
            A = torch.zeros((batch, 2, 2), dtype=state.dtype, device=state.device)
            A[:, 0, 0] = CD
            A[:, 1, 1] = CK
            W_diag = torch.stack((CD.square(), CK.square()))
            return A, torch.stack((bD, bK), dim=-1), W_diag

        CaD, CmD = p["C_a_D"].to(state), p["C_m_D"].to(state)
        CaK, CmK = p["C_a_K"].to(state), p["C_m_K"].to(state)
        bD = (To - TD) / p["R_Do"] + coupling_D + QcD + QrD
        bK = (To - TK) / p["R_Ko"] + coupling_K + QcK + QrK
        A = torch.zeros((batch, 2, 4), dtype=state.dtype, device=state.device)
        A[:, 0, 0] = CaD
        A[:, 0, 1] = CmD
        A[:, 1, 2] = CaK
        A[:, 1, 3] = CmK
        W_diag = torch.stack(
            (CaD.square(), CmD.square(), CaK.square(), CmK.square())
        )
        return A, torch.stack((bD, bK), dim=-1), W_diag

    def project_physical_derivative(
        self,
        state: torch.Tensor,
        f_tilde: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        A, b, W_diag = self.energy_constraint(state, v)
        out = weighted_energy_projection(f_tilde, A, b, W_diag)
        out["A"] = A
        out["b"] = b
        return out

    def internal_projected_residuals(
        self,
        state: torch.Tensor,
        f_P: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Projected full RC residuals retained only for 2C internal physics.

        For each 2C zone, the air and mass residuals sum to the already-enforced
        hard total-energy residual (approximately zero), leaving one independent
        redistribution direction.  We keep both r_a^P and r_m^P exactly as
        Eq. E5.112 specifies.  For 1C this loss is absent.
        """

        if self.config.rc_order == 1:
            return state.new_zeros((state.shape[0], 0))
        return self.rc_residuals(state, f_P, v)

    # ------------------------------------------------------------------
    # Stage-wise projected RHS used by the existing NeuroMANCER RK4 wrapper
    # ------------------------------------------------------------------
    def _stage_rhs(
        self,
        z: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        g_omega = self.g_omega(z, v)
        x = self.mu_x.to(z) + self.S_x.to(z) * z
        f_tilde = (self.S_x.to(g_omega) / g_omega.new_tensor(self.config.dt_seconds)) * g_omega

        proj = self.project_physical_derivative(x, f_tilde, v)
        f_P = proj["f_P"]
        g_P = g_omega.new_tensor(self.config.dt_seconds) * f_P / self.S_x.to(f_P)

        self._stage_raw_derivatives.append(f_tilde)
        self._stage_projected_derivatives.append(f_P)
        self._stage_rho.append(proj["rho"])
        self._stage_rho_P.append(proj["rho_P"])
        self._stage_M.append(proj["M"])
        self._stage_nu.append(proj["nu"])
        self._stage_correction.append(proj["correction"])
        self._stage_correction_energy.append(proj["correction_energy"])
        self._stage_rho_solve_energy.append(proj["rho_solve_energy"])
        self._stage_stationarity.append(proj["stationarity"])
        self._stage_stationarity_relative.append(proj["stationarity_relative"])

        internal = self.internal_projected_residuals(x, f_P, v)
        if self.config.rc_order == 2:
            self._stage_internal_residuals.append(internal)

        # E5.92/E5.212: NeuroMANCER RK4 integrates projected normalized g_P.
        return g_P

    def step(
        self,
        z: torch.Tensor,
        v: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
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

    def rollout(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self._clear_projection_stage_cache()
        # NeuralODEModel.rollout dispatches to self.step(), so the same
        # NeuroMANCER Node/System graph now propagates the projected RHS.
        return super().rollout(**kwargs)

    @property
    def projection_stage_count(self) -> int:
        return len(self._stage_projected_derivatives)

    @staticmethod
    def _stack_stage(values: list[torch.Tensor], *, name: str) -> torch.Tensor:
        if not values:
            raise RuntimeError(f"No RK4-stage {name} values collected; run a rollout first")
        return torch.stack(values, dim=0)

    def stage_projected_derivative_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_projected_derivatives, name="projected derivative")

    def stage_rho_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_rho, name="raw balance residual")

    def stage_rho_P_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_rho_P, name="projected balance residual")

    def stage_M_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_M, name="M")

    def stage_nu_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_nu, name="nu")

    def stage_correction_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_correction, name="correction")

    def stage_correction_energy_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_correction_energy, name="correction energy")

    def stage_rho_solve_energy_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_rho_solve_energy, name="rho solve energy")

    def stage_stationarity_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_stationarity, name="KKT stationarity")

    def stage_stationarity_relative_tensor(self) -> torch.Tensor:
        return self._stack_stage(self._stage_stationarity_relative, name="relative KKT stationarity")

    def stage_internal_residual_tensor(self) -> torch.Tensor:
        if self.config.rc_order == 1:
            # Explicitly represent "term absent" as a zero-width tensor for
            # diagnostics; no 1C internal loss is formed.
            ref = self.stage_rho_P_tensor()
            return ref.new_zeros((*ref.shape[:-1], 0))
        return self._stack_stage(self._stage_internal_residuals, name="internal residual")

    def rollout_loss(
        self,
        *,
        y_true: torch.Tensor,
        v_sequence: torch.Tensor | Mapping[str, torch.Tensor],
        context_y: torch.Tensor | None = None,
        context_v: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Literal Part-5 objective, Eqs. E5.110--E5.117."""

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

        # E5.110/E5.111: 1/N_r sum ||S_y^-1(yhat-y)||_2^2.
        err = (yhat[..., 1:, :] - y_true[..., 1:, :]) / self.S_y.to(yhat)
        Ldata = torch.sum(err**2, dim=-1).mean()

        if self.config.rc_order == 2:
            internal = self.stage_internal_residual_tensor()
            scale = self.q_star_residual.to(internal).reshape(1, 1, -1)
            Lint = torch.sum((internal / scale) ** 2, dim=-1).mean()
        else:
            internal = self.stage_internal_residual_tensor()
            Lint = Ldata.new_zeros(())

        # E5.113/E5.114: c^T W c = rho^T M^-1 rho.
        Lcorr = self.stage_correction_energy_tensor().mean()
        Lreg = self.neural_regularization()

        total = (
            self.config.lambda_y * Ldata
            + self.config.lambda_int * Lint
            + self.config.lambda_corr * Lcorr
            + self.config.lambda_wd * Lreg
        )

        return {
            "total": total,
            "data": Ldata,
            "internal_physics": Lint,
            "correction": Lcorr,
            "regularization": Lreg,
            "yhat": yhat,
            "z_hist": z_hist,
            "stage_raw_derivative": self.stage_raw_derivative_tensor(),
            "stage_projected_derivative": self.stage_projected_derivative_tensor(),
            "stage_rho": self.stage_rho_tensor(),
            "stage_rho_P": self.stage_rho_P_tensor(),
            "stage_M": self.stage_M_tensor(),
            "stage_nu": self.stage_nu_tensor(),
            "stage_correction": self.stage_correction_tensor(),
            "stage_correction_energy": self.stage_correction_energy_tensor(),
            "stage_rho_solve_energy": self.stage_rho_solve_energy_tensor(),
            "stage_stationarity": self.stage_stationarity_tensor(),
            "stage_stationarity_relative": self.stage_stationarity_relative_tensor(),
            "stage_internal_residual": internal,
        }

    def projection_diagnostics(self) -> dict[str, float]:
        """Aggregate small-run projection diagnostics from the most recent rollout."""

        rho = self.stage_rho_tensor()
        rho_P = self.stage_rho_P_tensor()
        correction = self.stage_correction_tensor()
        corr_energy = self.stage_correction_energy_tensor()
        rho_energy = self.stage_rho_solve_energy_tensor()
        stationarity = self.stage_stationarity_tensor()
        stationarity_relative = self.stage_stationarity_relative_tensor()

        return {
            "raw_balance_rms_W": float(torch.sqrt(torch.mean(rho.square())).detach().cpu()),
            "raw_balance_max_abs_W": float(torch.max(torch.abs(rho)).detach().cpu()),
            "projected_balance_rms_W": float(torch.sqrt(torch.mean(rho_P.square())).detach().cpu()),
            "projected_balance_max_abs_W": float(torch.max(torch.abs(rho_P)).detach().cpu()),
            "correction_rms_K_per_s": float(torch.sqrt(torch.mean(correction.square())).detach().cpu()),
            "correction_energy_mean": float(torch.mean(corr_energy).detach().cpu()),
            "rho_solve_energy_mean": float(torch.mean(rho_energy).detach().cpu()),
            "correction_energy_identity_max_abs": float(
                torch.max(torch.abs(corr_energy - rho_energy)).detach().cpu()
            ),
            "kkt_stationarity_max_abs": float(torch.max(torch.abs(stationarity)).detach().cpu()),
            "kkt_stationarity_relative_max": float(torch.max(stationarity_relative).detach().cpu()),
        }

    def provenance(self) -> dict[str, Any]:
        base = super().provenance()
        base["method"] = "ebp_pinode"
        base["math_contract"] = "PINODE_EPSR_Part5_EBP_PINODE_Detailed.tex"
        base["config"] = asdict(self.config)
        base["physics"] = {
            "constraint": "exact_zone_total_energy_A_f_P_equals_b",
            "constraint_type": "hard_projection",
            "hard_projection": True,
            "raw_derivative": "f_tilde_omega",
            "integrated_derivative": "projected_f_P",
            "normalized_integrated_derivative": "g_P=dt_seconds*S_x^-1*f_P",
            "projection_metric": "W=diag(capacitance^2)",
            "projection_solve": "torch.linalg.solve(M,rho)",
            "explicit_matrix_inverse": False,
            "projection_stage_evaluation": "every_Neuromancer_RK4_RHS_call",
            "internal_physics": "2C_only_projected_air_mass_RC_residual",
            "one_c_internal_loss": False,
            "q_star": self.q_star_zone.detach().cpu().tolist(),
            "dep2_q_star_convention": "frozen training-only neutral allocation lambda=1 at initialization",
        }
        base["framework"] = {
            "sciml": "neuromancer",
            "tensor_autograd_optimizer_linear_solve": "pytorch",
            "hyperparameter_search": "optuna",
            "integration": "neuromancer.dynamics.integrators.RK4",
            "recursive_graph": "neuromancer.system.Node+System",
            "named_data": "neuromancer.dataset.DictDataset + collate_fn",
            "direct_torchdiffeq_calls": False,
            "runtime": runtime_info().__dict__,
        }
        return base
