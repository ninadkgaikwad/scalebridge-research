from __future__ import annotations

"""Differentiable weighted energy-balance projection used by EBP-PINODE.

Scientific contract: Part 5 EBP-PINODE weighted minimum-change projection.
"""

import torch

def weighted_energy_projection(
    f_tilde: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    W_diag: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable weighted projection onto ``A f = b``.

    Parameters
    ----------
    f_tilde:
        Raw physical derivative, shape ``[batch, n]``.
    A:
        Constraint matrix, shape ``[m, n]`` or ``[batch, m, n]``.
    b:
        Constraint right-hand side, shape ``[batch, m]`` or ``[m]``.
    W_diag:
        Positive diagonal of W, shape ``[n]`` or ``[batch, n]``.

    Returns
    -------
    Dictionary containing ``f_P``, ``rho``, ``rho_P``, ``M``, ``nu``,
    ``correction``, ``correction_energy`` and ``stationarity``.

    Notes
    -----
    The implementation intentionally never forms W^{-1} or M^{-1} as dense
    matrix inverses.  W is diagonal, so its reciprocal diagonal is used, and
    ``torch.linalg.solve`` computes ``nu`` from ``M nu = rho``.
    """

    if f_tilde.ndim == 1:
        f_tilde = f_tilde.unsqueeze(0)
    if f_tilde.ndim != 2:
        raise ValueError("f_tilde must have shape [batch, n]")
    batch, n = f_tilde.shape

    if A.ndim == 2:
        A = A.unsqueeze(0).expand(batch, -1, -1)
    if A.ndim != 3 or A.shape[0] != batch or A.shape[2] != n:
        raise ValueError("A must have shape [m,n] or [batch,m,n] compatible with f_tilde")
    m = A.shape[1]

    if b.ndim == 1:
        b = b.unsqueeze(0).expand(batch, -1)
    if b.ndim != 2 or b.shape != (batch, m):
        raise ValueError("b must have shape [m] or [batch,m] compatible with A")

    if W_diag.ndim == 1:
        W_diag = W_diag.unsqueeze(0).expand(batch, -1)
    if W_diag.ndim != 2 or W_diag.shape != (batch, n):
        raise ValueError("W_diag must have shape [n] or [batch,n]")
    if not torch.all(W_diag > 0):
        raise ValueError("W must be positive definite; every W diagonal entry must be > 0")

    A = A.to(f_tilde)
    b = b.to(f_tilde)
    W_diag = W_diag.to(f_tilde)

    W_inv_diag = W_diag.reciprocal()
    rho = torch.matmul(A, f_tilde.unsqueeze(-1)).squeeze(-1) - b

    # M = A W^{-1} A^T, using the reciprocal diagonal directly.
    A_W_inv = A * W_inv_diag.unsqueeze(-2)
    M = torch.matmul(A_W_inv, A.transpose(-1, -2))

    # E5.94: differentiable direct solve. Never form M^{-1}.
    nu = torch.linalg.solve(M, rho.unsqueeze(-1)).squeeze(-1)

    A_T_nu = torch.matmul(A.transpose(-1, -2), nu.unsqueeze(-1)).squeeze(-1)
    correction = -W_inv_diag * A_T_nu
    f_P = f_tilde + correction

    rho_P = torch.matmul(A, f_P.unsqueeze(-1)).squeeze(-1) - b
    stationarity = W_diag * correction + A_T_nu
    stationarity_scale = (W_diag * correction).abs() + A_T_nu.abs()
    stationarity_relative = stationarity.abs() / stationarity_scale.clamp_min(1.0)

    correction_energy = torch.sum(correction * W_diag * correction, dim=-1)
    # Since M nu = rho, rho^T M^{-1} rho = rho^T nu.
    rho_solve_energy = torch.sum(rho * nu, dim=-1)

    return {
        "f_P": f_P,
        "rho": rho,
        "rho_P": rho_P,
        "M": M,
        "nu": nu,
        "correction": correction,
        "correction_energy": correction_energy,
        "rho_solve_energy": rho_solve_energy,
        "stationarity": stationarity,
        "stationarity_relative": stationarity_relative,
        "W_diag": W_diag,
    }

