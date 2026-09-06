from __future__ import annotations

"""Layered E0-6 backend parity helpers."""

import numpy as np

from .contracts import ParityComparison, ParityTolerance


def normalized_linf_error(a, b, tolerance: ParityTolerance) -> ParityComparison:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        return ParityComparison(
            label="shape_mismatch",
            passed=False,
            normalized_linf=float("inf"),
            max_abs_error=float("inf"),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
            metadata={"shape_a": aa.shape, "shape_b": bb.shape},
        )
    diff = np.abs(aa - bb)
    denom = tolerance.atol + tolerance.rtol * np.maximum(np.abs(aa), np.abs(bb))
    normalized = diff / denom
    score = float(np.max(normalized)) if normalized.size else 0.0
    return ParityComparison(
        label="parity",
        passed=bool(score <= 1.0),
        normalized_linf=score,
        max_abs_error=float(np.max(diff)) if diff.size else 0.0,
        atol=tolerance.atol,
        rtol=tolerance.rtol,
    )
