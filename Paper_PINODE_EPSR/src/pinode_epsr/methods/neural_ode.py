from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ..core.common import TensorStandardizer
from ..backends.neuromancer import build_mlp, node, rk4_interval, runtime_info, system

RCOrder = Literal[1, 2]


@dataclass(frozen=True)
class NeuralODEConfig:
    case_name: str
    rc_order: RCOrder
    hidden_layers: int = 2
    hidden_width: int = 32
    activation: str = "tanh"
    N_r: int = 6
    L_e: int = 6
    N_s: int = 2
    delta_T_m_max: float = 8.0
    lambda_wd: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.case_name not in {"all_to_one", "identity_ind", "identity_dep1", "identity_dep2"}:
            raise ValueError(f"Unknown case_name: {self.case_name}")
        if self.rc_order not in (1, 2):
            raise ValueError("rc_order must be 1 or 2")
        if self.N_r < 1 or self.N_s < 1:
            raise ValueError("N_r and N_s must be >=1")
        if self.rc_order == 2 and self.L_e < 1:
            raise ValueError("NODE-2C requires L_e>=1")
        if self.delta_T_m_max <= 0:
            raise ValueError("delta_T_m_max must be positive")


class CausalMassEncoder(nn.Module):
    """Part-3 causal hidden-mass initializer, evaluated once per rollout."""

    def __init__(self, input_dim: int, output_dim: int, *, hidden_layers: int, hidden_width: int, activation: str) -> None:
        super().__init__()
        self.network = build_mlp(
            input_dim,
            output_dim,
            hidden_layers=hidden_layers,
            hidden_width=hidden_width,
            activation=activation,
        ).double()

    def forward(self, context_flat: torch.Tensor) -> torch.Tensor:
        return self.network(context_flat)


class NeuralODEModel(nn.Module):
    """Part-3 data-only NODE; contains no R/C/eta/lambda parameters."""

    def __init__(
        self,
        config: NeuralODEConfig,
        *,
        y_training: np.ndarray | torch.Tensor,
        v_training: np.ndarray | Mapping[str, np.ndarray] | torch.Tensor | Mapping[str, torch.Tensor],
        y_names: Sequence[str] | None = None,
        v_names: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        y = torch.as_tensor(y_training, dtype=torch.float64)
        if y.ndim == 1:
            y = y[:, None]
        expected_y = 1 if config.case_name == "all_to_one" else 2
        if y.shape[1] != expected_y:
            raise ValueError(f"{config.case_name} expects {expected_y} outputs, got {y.shape[1]}")
        ynames = tuple(y_names or (("T_A",) if expected_y == 1 else ("T_D", "T_K")))
        self.y_scaler = TensorStandardizer.fit(y, names=ynames)
        self.register_buffer("mu_y", self.y_scaler.mean.clone())
        self.register_buffer("S_y", self.y_scaler.scale.clone())
        mu_x, S_x = self._state_scaling(self.mu_y, self.S_y)
        self.register_buffer("mu_x", mu_x)
        self.register_buffer("S_x", S_x)

        self.v_scalers: dict[str, TensorStandardizer] = {}
        self._v_mean_names: list[str] = []
        self._v_scale_names: list[str] = []
        self._v_feature_names: dict[str, tuple[str, ...]] = {}
        if config.case_name == "identity_ind":
            if not isinstance(v_training, Mapping):
                raise TypeError("identity_ind NODE requires separate Dining/Kitchen forcing arrays")
            if not isinstance(v_names, Mapping):
                v_names = {key: tuple(f"v{i}" for i in range(np.asarray(value).shape[1])) for key, value in v_training.items()}
            for key in ("Dining", "Kitchen"):
                if key not in v_training:
                    # Allow D/K aliases in synthetic tests.
                    alt = "D" if key == "Dining" else "K"
                    if alt in v_training:
                        value = v_training[alt]
                        names = v_names.get(alt, ()) if isinstance(v_names, Mapping) else ()
                    else:
                        raise KeyError(f"identity_ind forcing missing {key}")
                else:
                    value = v_training[key]
                    names = v_names.get(key, ()) if isinstance(v_names, Mapping) else ()
                scaler = TensorStandardizer.fit(value, names=names)
                self.v_scalers[key] = scaler
                safe = key.lower()
                self.register_buffer(f"mu_v_{safe}", scaler.mean.clone())
                self.register_buffer(f"S_v_{safe}", scaler.scale.clone())
                self._v_feature_names[key] = scaler.names
        else:
            if isinstance(v_training, Mapping):
                raise TypeError(f"{config.case_name} NODE requires one joint forcing array")
            names = tuple(v_names) if isinstance(v_names, Sequence) and not isinstance(v_names, str) else ()
            scaler = TensorStandardizer.fit(v_training, names=names)
            self.v_scalers["joint"] = scaler
            self.register_buffer("mu_v_joint", scaler.mean.clone())
            self.register_buffer("S_v_joint", scaler.scale.clone())
            self._v_feature_names["joint"] = scaler.names

        self.vector_fields = self._build_vector_fields()
        self.encoders = self._build_encoders()

    @property
    def state_dim(self) -> int:
        if self.config.case_name == "all_to_one":
            return self.config.rc_order
        return 2 if self.config.rc_order == 1 else 4

    @property
    def state_order(self) -> tuple[str, ...]:
        if self.config.case_name == "all_to_one":
            return ("T_A",) if self.config.rc_order == 1 else ("T_a,A", "T_m,A")
        return ("T_D", "T_K") if self.config.rc_order == 1 else ("T_a,D", "T_m,D", "T_a,K", "T_m,K")

    def _state_scaling(self, mean_y: torch.Tensor, scale_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.rc_order == 1:
            return mean_y.clone(), scale_y.clone()
        if self.config.case_name == "all_to_one":
            return mean_y.repeat(2), scale_y.repeat(2)
        return (
            torch.stack((mean_y[0], mean_y[0], mean_y[1], mean_y[1])),
            torch.stack((scale_y[0], scale_y[0], scale_y[1], scale_y[1])),
        )

    def _v_stats(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        safe = key.lower()
        if key == "joint":
            return self.mu_v_joint, self.S_v_joint
        return getattr(self, f"mu_v_{safe}"), getattr(self, f"S_v_{safe}")

    def normalize_v(self, v: torch.Tensor, key: str = "joint") -> torch.Tensor:
        mu, scale = self._v_stats(key)
        return (v - mu.to(v)) / scale.to(v)

    def _joint_nv(self) -> int:
        return int(self._v_stats("joint")[0].numel())

    def _build_vector_fields(self) -> nn.ModuleDict:
        c = self.config
        kwargs = dict(hidden_layers=c.hidden_layers, hidden_width=c.hidden_width, activation=c.activation)
        if c.case_name == "identity_ind":
            nvD = int(self._v_stats("Dining")[0].numel())
            nvK = int(self._v_stats("Kitchen")[0].numel())
            return nn.ModuleDict(
                {
                    "Dining": build_mlp(c.rc_order + nvD, c.rc_order, **kwargs).double(),
                    "Kitchen": build_mlp(c.rc_order + nvK, c.rc_order, **kwargs).double(),
                }
            )
        return nn.ModuleDict({"joint": build_mlp(self.state_dim + self._joint_nv(), self.state_dim, **kwargs).double()})

    def _build_encoders(self) -> nn.ModuleDict:
        c = self.config
        if c.rc_order == 1:
            return nn.ModuleDict()
        kwargs = dict(hidden_layers=max(1, c.hidden_layers), hidden_width=c.hidden_width, activation=c.activation)
        if c.case_name == "identity_ind":
            nvD = int(self._v_stats("Dining")[0].numel())
            nvK = int(self._v_stats("Kitchen")[0].numel())
            return nn.ModuleDict(
                {
                    "Dining": CausalMassEncoder(c.L_e * (1 + nvD), 1, **kwargs),
                    "Kitchen": CausalMassEncoder(c.L_e * (1 + nvK), 1, **kwargs),
                }
            )
        ny = 1 if c.case_name == "all_to_one" else 2
        hidden = 1 if c.case_name == "all_to_one" else 2
        return nn.ModuleDict({"joint": CausalMassEncoder(c.L_e * (ny + self._joint_nv()), hidden, **kwargs)})

    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mu_y.to(y)) / self.S_y.to(y)

    def denormalize_y(self, z_y: torch.Tensor) -> torch.Tensor:
        return self.mu_y.to(z_y) + self.S_y.to(z_y) * z_y

    def _g_ind(self, z: torch.Tensor, v: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.config.rc_order == 1:
            zD = z[..., 0:1]
            zK = z[..., 1:2]
        else:
            zD = z[..., 0:2]
            zK = z[..., 2:4]
        vD = self.normalize_v(v.get("Dining", v.get("D")), "Dining")
        vK = self.normalize_v(v.get("Kitchen", v.get("K")), "Kitchen")
        gD = self.vector_fields["Dining"](torch.cat((zD, vD), dim=-1))
        gK = self.vector_fields["Kitchen"](torch.cat((zK, vK), dim=-1))
        return torch.cat((gD, gK), dim=-1)

    def g_omega(self, z: torch.Tensor, v: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Normalized-time derivative dz/dtau; absolute time is intentionally absent."""

        if self.config.case_name == "identity_ind":
            if not isinstance(v, Mapping):
                raise TypeError("identity_ind g_omega requires a forcing mapping")
            return self._g_ind(z, v)
        if isinstance(v, Mapping):
            raise TypeError(f"{self.config.case_name} g_omega requires a joint forcing tensor")
        vt = self.normalize_v(v, "joint")
        return self.vector_fields["joint"](torch.cat((z, vt), dim=-1))

    def _context_flat_joint(self, context_y: torch.Tensor, context_v: torch.Tensor) -> torch.Tensor:
        if context_y.shape[-2] != self.config.L_e or context_v.shape[-2] != self.config.L_e:
            raise ValueError(f"NODE-2C encoder requires exactly L_e={self.config.L_e} causal samples")
        yt = self.normalize_y(context_y)
        vt = self.normalize_v(context_v)
        return torch.cat((yt, vt), dim=-1).reshape(*yt.shape[:-2], -1)

    def initial_state(
        self,
        y0: torch.Tensor,
        *,
        context_y: torch.Tensor | None = None,
        context_v: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        y0 = torch.as_tensor(y0, dtype=self.mu_x.dtype, device=self.mu_x.device)
        z_air = self.normalize_y(y0)
        if self.config.rc_order == 1:
            return z_air
        if context_y is None or context_v is None:
            raise ValueError("NODE-2C requires causal context_y/context_v at rollout initialization")

        dmax = y0.new_tensor(self.config.delta_T_m_max)
        if self.config.case_name == "identity_ind":
            if not isinstance(context_v, Mapping):
                raise TypeError("identity_ind NODE-2C requires separate forcing contexts")
            outputs = []
            for idx, key in enumerate(("Dining", "Kitchen")):
                vctx = context_v.get(key, context_v.get("D" if key == "Dining" else "K"))
                yctx = context_y[..., :, idx : idx + 1]
                if yctx.shape[-2] != self.config.L_e or vctx.shape[-2] != self.config.L_e:
                    raise ValueError("causal context length mismatch")
                flat = torch.cat((self.normalize_y(context_y)[..., :, idx : idx + 1], self.normalize_v(vctx, key)), dim=-1).reshape(*yctx.shape[:-2], -1)
                delta = dmax * torch.tanh(self.encoders[key](flat)).squeeze(-1)
                T_air = y0[..., idx]
                T_mass = T_air + delta
                mu, scale = self.mu_y[idx].to(y0), self.S_y[idx].to(y0)
                z_mass = (T_mass - mu) / scale
                outputs.extend((z_air[..., idx], z_mass))
            return torch.stack(outputs, dim=-1)

        if isinstance(context_v, Mapping):
            raise TypeError("joint NODE-2C requires joint forcing context")
        flat = self._context_flat_joint(context_y, context_v)
        delta = dmax * torch.tanh(self.encoders["joint"](flat))
        if self.config.case_name == "all_to_one":
            T_air = y0[..., 0]
            T_mass = T_air + delta[..., 0]
            z_mass = (T_mass - self.mu_y[0].to(y0)) / self.S_y[0].to(y0)
            return torch.stack((z_air[..., 0], z_mass), dim=-1)
        TmD = y0[..., 0] + delta[..., 0]
        TmK = y0[..., 1] + delta[..., 1]
        zTmD = (TmD - self.mu_y[0].to(y0)) / self.S_y[0].to(y0)
        zTmK = (TmK - self.mu_y[1].to(y0)) / self.S_y[1].to(y0)
        return torch.stack((z_air[..., 0], zTmD, z_air[..., 1], zTmK), dim=-1)

    def observe_normalized(self, z: torch.Tensor) -> torch.Tensor:
        if self.config.rc_order == 1:
            return z
        if self.config.case_name == "all_to_one":
            return z[..., 0:1]
        return z[..., [0, 2]]

    def observe(self, z: torch.Tensor) -> torch.Tensor:
        return self.denormalize_y(self.observe_normalized(z))

    def step(self, z: torch.Tensor, v: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        """One observed interval using NeuroMANCER RK4 and zero-order-held forcing."""

        # Part-3 normalized time: every observed 5-min interval is Delta tau=1.
        extra_dim = 0
        if isinstance(v, torch.Tensor):
            extra_dim = int(v.shape[-1])
        return rk4_interval(
            lambda zz, vv: self.g_omega(zz, vv),
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
        """Recursive Part-3 rollout through NeuroMANCER Node + System."""

        z0 = self.initial_state(y0, context_y=context_y, context_v=context_v)
        unbatched = z0.ndim == 1
        if unbatched:
            z0 = z0.unsqueeze(0)
        z_initial = z0.unsqueeze(1)  # [batch, 1, n_x]

        if self.config.case_name == "identity_ind":
            if not isinstance(v_sequence, Mapping):
                raise TypeError("identity_ind rollout requires forcing mapping")
            vD = v_sequence.get("Dining", v_sequence.get("D"))
            vK = v_sequence.get("Kitchen", v_sequence.get("K"))
            if vD is None or vK is None:
                raise KeyError("identity_ind forcing requires Dining/D and Kitchen/K")
            if vD.ndim == 2:
                vD = vD.unsqueeze(0)
                vK = vK.unsqueeze(0)
            if vD.shape[:-1] != vK.shape[:-1]:
                raise ValueError("identity_ind forcing sequences have different batch/time shapes")

            def interval_map(z: torch.Tensor, v_D: torch.Tensor, v_K: torch.Tensor) -> torch.Tensor:
                return self.step(z, {"Dining": v_D, "Kitchen": v_K})

            step_node = node(
                interval_map, ["z", "v_D", "v_K"], ["z"], name="NODE_RK4_step_IND"
            )
            rollout_system = system(
                [step_node], nstep_key="v_D", name="NODE_System_IND"
            )
            result = rollout_system({"z": z_initial, "v_D": vD, "v_K": vK})
        else:
            if isinstance(v_sequence, Mapping):
                raise TypeError("joint rollout requires forcing tensor")
            v = v_sequence.unsqueeze(0) if v_sequence.ndim == 2 else v_sequence

            def interval_map(z: torch.Tensor, v_k: torch.Tensor) -> torch.Tensor:
                return self.step(z, v_k)

            step_node = node(interval_map, ["z", "v"], ["z"], name="NODE_RK4_step")
            rollout_system = system([step_node], nstep_key="v", name="NODE_System")
            result = rollout_system({"z": z_initial, "v": v})

        z_hist = result["z"]
        yhat = self.observe(z_hist)
        if unbatched:
            return yhat.squeeze(0), z_hist.squeeze(0)
        return yhat, z_hist

    def rollout_loss(
        self,
        *,
        y_true: torch.Tensor,
        v_sequence: torch.Tensor | Mapping[str, torch.Tensor],
        context_y: torch.Tensor | None = None,
        context_v: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Multi-step normalized output error; no physics loss (Part 3)."""

        y_true = torch.as_tensor(y_true, dtype=self.mu_x.dtype, device=self.mu_x.device)
        if y_true.shape[-2] < 2:
            raise ValueError("y_true must include initial sample plus >=1 future sample")
        y0 = y_true[..., 0, :]
        yhat, _ = self.rollout(y0=y0, v_sequence=v_sequence, context_y=context_y, context_v=context_v)
        # Exclude the exactly measured initialization sample from prediction error.
        err = (yhat[..., 1:, :] - y_true[..., 1:, :]) / self.S_y.to(yhat)
        Lroll = torch.mean(err**2)
        Lreg = torch.zeros((), dtype=Lroll.dtype, device=Lroll.device)
        if self.config.lambda_wd > 0:
            Lreg = sum((p**2).sum() for p in self.parameters())
        total = Lroll + self.config.lambda_wd * Lreg
        return {"total": total, "rollout": Lroll, "regularization": Lreg, "yhat": yhat}

    def provenance(self) -> dict[str, Any]:
        v_dims = {key: int(self._v_stats(key)[0].numel()) for key in self.v_scalers}
        return {
            "method": "neural_ode",
            "math_contract": "PINODE_EPSR_Part3_NeuralODE_Detailed.tex",
            "config": asdict(self.config),
            "state_order": list(self.state_order),
            "forcing_dimensions": v_dims,
            "absolute_time_input": False,
            "physics_loss": False,
            "normalized_interval_delta_tau": 1.0,
            "framework": {
                "sciml": "neuromancer",
                "tensor_autograd_optimizer": "pytorch",
                "integration": "neuromancer.dynamics.integrators.RK4",
                "recursive_graph": "neuromancer.system.Node+System",
                "direct_torchdiffeq_calls": False,
                "runtime": runtime_info().__dict__,
            },
        }
