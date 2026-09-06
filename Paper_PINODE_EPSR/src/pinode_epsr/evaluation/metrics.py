from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(err)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.mean(err))


def per_output_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, float]]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: truth={yt.shape}, prediction={yp.shape}")
    return [
        {"rmse": rmse(yt[:, i], yp[:, i]), "mae": mae(yt[:, i], yp[:, i]), "bias": bias(yt[:, i], yp[:, i])}
        for i in range(yt.shape[1])
    ]


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    denom = float(np.mean(yt))
    return float("nan") if abs(denom) < 1e-12 else 100.0 * rmse(yt, y_pred) / abs(denom)


def nmbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    denom = float(np.mean(yt))
    return float("nan") if abs(denom) < 1e-12 else 100.0 * bias(yt, y_pred) / abs(denom)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yp - yt) ** 2)); ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def full_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "bias": bias(y_true, y_pred),
            "cvrmse_percent": cvrmse(y_true, y_pred), "nmbe_percent": nmbe(y_true, y_pred), "r2": r2(y_true, y_pred)}
