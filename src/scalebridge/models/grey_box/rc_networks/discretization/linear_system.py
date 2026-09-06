from __future__ import annotations

"""Graph-general linear state-space realization of a compiled E0-3 RC model."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from ..compiler import CompiledRCModel
from ..specification import RCCompileError


@dataclass(frozen=True)
class LinearRCStateSpace:
    """Authoritative linear RC matrices materialized from the E0-3 compiler.

    No RC topology is re-derived here. ``CompiledRCModel.matrices`` remains the
    source of truth. E0-5 only converts the resulting matrices into

        Xdot = A X + B_T T_B + B_Q Q.
    """

    C: np.ndarray
    L_CC: np.ndarray
    L_CB: np.ndarray
    Gamma: np.ndarray
    H: np.ndarray
    A: np.ndarray
    B_boundary: np.ndarray
    B_thermal: np.ndarray
    state_keys: tuple[str, ...]
    boundary_labels: tuple[str, ...]
    thermal_port_keys: tuple[str, ...]

    @property
    def state_dimension(self) -> int:
        return int(self.A.shape[0])

    @property
    def boundary_dimension(self) -> int:
        return int(self.B_boundary.shape[1])

    @property
    def thermal_dimension(self) -> int:
        return int(self.B_thermal.shape[1])

    @property
    def input_dimension(self) -> int:
        return self.boundary_dimension + self.thermal_dimension

    @property
    def B(self) -> np.ndarray:
        return np.concatenate((self.B_boundary, self.B_thermal), axis=1)

    def to_torch(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "TorchLinearRCStateSpace":
        device_obj = torch.device(device)
        kwargs = {"dtype": dtype, "device": device_obj}
        return TorchLinearRCStateSpace(
            C=torch.as_tensor(self.C, **kwargs),
            L_CC=torch.as_tensor(self.L_CC, **kwargs),
            L_CB=torch.as_tensor(self.L_CB, **kwargs),
            Gamma=torch.as_tensor(self.Gamma, **kwargs),
            H=torch.as_tensor(self.H, **kwargs),
            A=torch.as_tensor(self.A, **kwargs),
            B_boundary=torch.as_tensor(self.B_boundary, **kwargs),
            B_thermal=torch.as_tensor(self.B_thermal, **kwargs),
            state_keys=self.state_keys,
            boundary_labels=self.boundary_labels,
            thermal_port_keys=self.thermal_port_keys,
        )


@dataclass(frozen=True)
class TorchLinearRCStateSpace:
    C: torch.Tensor
    L_CC: torch.Tensor
    L_CB: torch.Tensor
    Gamma: torch.Tensor
    H: torch.Tensor
    A: torch.Tensor
    B_boundary: torch.Tensor
    B_thermal: torch.Tensor
    state_keys: tuple[str, ...]
    boundary_labels: tuple[str, ...]
    thermal_port_keys: tuple[str, ...]

    @property
    def state_dimension(self) -> int:
        return int(self.A.shape[0])

    @property
    def boundary_dimension(self) -> int:
        return int(self.B_boundary.shape[1])

    @property
    def thermal_dimension(self) -> int:
        return int(self.B_thermal.shape[1])

    @property
    def input_dimension(self) -> int:
        return self.boundary_dimension + self.thermal_dimension

    @property
    def B(self) -> torch.Tensor:
        return torch.cat((self.B_boundary, self.B_thermal), dim=1)

    def rhs(
        self,
        state: torch.Tensor,
        boundary: torch.Tensor,
        thermal: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the frozen linear E0-3 RHS in batch-major tensor form."""

        return (
            state @ self.A.transpose(-1, -2)
            + boundary @ self.B_boundary.transpose(-1, -2)
            + thermal @ self.B_thermal.transpose(-1, -2)
        )


def compile_linear_state_space(
    model: CompiledRCModel,
    parameter_values: Mapping[str, float],
) -> LinearRCStateSpace:
    """Build ``A, B`` dynamically from an arbitrary valid compiled RC graph.

    The implementation intentionally uses row-wise division by the positive
    capacitance vector instead of explicitly constructing ``C^{-1}``.
    """

    matrices = model.matrices(parameter_values)
    c = np.asarray(matrices.C, dtype=float).reshape(-1)
    if c.shape != (model.state_dimension,) or not np.all(np.isfinite(c)):
        raise RCCompileError("Compiled capacitance vector is invalid")
    if np.any(c <= 0.0):
        raise RCCompileError("Compiled capacitance vector must be strictly positive")

    lcc = np.asarray(matrices.L_CC, dtype=float)
    lcb = np.asarray(matrices.L_CB, dtype=float)
    gamma = np.asarray(matrices.Gamma, dtype=float)
    h = np.asarray(matrices.H, dtype=float)

    a = -lcc / c[:, None]
    b_boundary = -lcb / c[:, None]
    b_thermal = gamma / c[:, None]

    arrays = (lcc, lcb, gamma, h, a, b_boundary, b_thermal)
    if not all(np.all(np.isfinite(item)) for item in arrays):
        raise RCCompileError("Compiled linear RC state-space contains non-finite values")

    return LinearRCStateSpace(
        C=c.copy(),
        L_CC=lcc.copy(),
        L_CB=lcb.copy(),
        Gamma=gamma.copy(),
        H=h.copy(),
        A=a,
        B_boundary=b_boundary,
        B_thermal=b_thermal,
        state_keys=tuple(node.key for node in model.state_nodes),
        boundary_labels=tuple(node.boundary_label for node in model.boundary_nodes),
        thermal_port_keys=tuple(port.key for port in model.thermal_ports),
    )
