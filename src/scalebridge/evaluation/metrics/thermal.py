from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nmbe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sum(y_true - y_pred) / ((len(y_true) - 1) * np.mean(y_true)))


def cvrmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    return float(rmse(y_true, y_pred) / np.mean(y_true))
