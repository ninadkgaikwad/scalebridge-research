from __future__ import annotations

import numpy as np


def equal_weight_temperature(t_dining: np.ndarray, t_kitchen: np.ndarray) -> np.ndarray:
    """Paper aggregation temperature for the locked equal-weight two-zone case."""
    return 0.5 * (np.asarray(t_dining, dtype=float) + np.asarray(t_kitchen, dtype=float))


def temperature_dispersion(t_dining: np.ndarray, t_kitchen: np.ndarray) -> np.ndarray:
    """Two-zone dispersion diagnostic used by the aggregation theorem experiments."""
    td = np.asarray(t_dining, dtype=float)
    tk = np.asarray(t_kitchen, dtype=float)
    tbar = equal_weight_temperature(td, tk)
    return np.sqrt(0.5 * ((td - tbar) ** 2 + (tk - tbar) ** 2))


def aggregation_residual(
    aggregate_derivative: np.ndarray,
    identity_derivatives: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute r_agg = dT_agg/dt - sum_i w_i dT_i/dt.

    This is intentionally model-agnostic; Patch 1 provides the diagnostic primitive
    while later patches supply derivatives from learned/physical models.
    """
    deriv = np.asarray(identity_derivatives, dtype=float)
    if deriv.ndim != 2:
        raise ValueError("identity_derivatives must be [time, zone]")
    if weights is None:
        weights = np.full(deriv.shape[1], 1.0 / deriv.shape[1])
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("aggregation weights must sum to one")
    return np.asarray(aggregate_derivative, dtype=float) - deriv @ weights


def airflow_allocation_mismatch(
    mdot_dining: np.ndarray,
    mdot_kitchen: np.ndarray,
    delta_t: np.ndarray,
    cp_air_j_per_kgk: float = 1005.0,
) -> np.ndarray:
    """Theorem diagnostic cp * (mdot_D - mdot_K) * delta_T."""
    return (
        cp_air_j_per_kgk
        * (np.asarray(mdot_dining, dtype=float) - np.asarray(mdot_kitchen, dtype=float))
        * np.asarray(delta_t, dtype=float)
    )
