from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from ..core.config import PaperConfig


@dataclass(frozen=True)
class PhaseCLinearSurrogate:
    """One-input affine Phase-C model adapter used for QAC/PHVAC evaluation."""

    coefficient: float
    intercept: float
    component: str
    aggregate_zone_id: str
    source: str

    def predict(self, x: np.ndarray | torch.Tensor | list[float] | float):
        if isinstance(x, torch.Tensor):
            return x * x.new_tensor(self.coefficient) + x.new_tensor(self.intercept)
        arr = np.asarray(x, dtype=float)
        return self.coefficient * arr + self.intercept


@dataclass(frozen=True)
class PhaseCModelBundle:
    qac: Any
    phvac: Any
    aggregate_zone_id: str
    provenance: dict[str, Any]

    def predict_qac_from_hvac_proxy(self, Q_HVAC_X):
        """Legacy API: Phase-C QAC model maps physics QHVAC_X to corrected QHVAC_Y."""
        return self.qac.predict(Q_HVAC_X)

    def predict_corrected_qhvac_from_physics(self, Q_HVAC_X):
        """Explicit physical alias for the persisted Phase-C QAC mapping."""
        return self.predict_qac_from_hvac_proxy(Q_HVAC_X)

    def predict_phvac_from_qac(self, Q_AC):
        """Legacy API: PHVAC is fitted against abs(corrected Phase-C QHVAC)."""
        if isinstance(Q_AC, torch.Tensor):
            return self.phvac.predict(torch.abs(Q_AC))
        return self.phvac.predict(np.abs(np.asarray(Q_AC, dtype=float)))

    def predict_phvac_from_corrected_qhvac(self, Q_HVAC_phaseC):
        """Explicit physical alias for abs(QHVAC_phaseC) -> Phase-C PHVAC."""
        return self.predict_phvac_from_qac(Q_HVAC_phaseC)


def Q_HVAC_X(m_dot, T_supply, T_z):
    """Locked Phase-C HVAC proxy: 1000*1.005*m_dot*(T_supply-T_z) [W]."""

    return 1000.0 * 1.005 * m_dot * (T_supply - T_z)


# Approximate validated controlled-run coefficients carried by the handoff. They
# are an explicit FALLBACK for unit/synthetic testing, never a substitute for
# loading the actual Phase-C artifacts when real-data validation is requested.
_PHASE_C_REFERENCE = {
    "RestaurantFastFood_All": {"QAC": (0.47553, 0.0), "PHVAC": (0.648654, -333.194)},
    "Dining": {"QAC": (1.01245, 0.0), "PHVAC": (0.279740, 469.134)},
    "Kitchen": {"QAC": (1.01547, 0.0), "PHVAC": (0.157693, -2.015)},
}


def reference_phase_c_bundle(aggregate_zone_id: str) -> PhaseCModelBundle:
    if aggregate_zone_id not in _PHASE_C_REFERENCE:
        raise KeyError(f"No controlled-run Phase-C reference for {aggregate_zone_id!r}")
    qac_coef, qac_intercept = _PHASE_C_REFERENCE[aggregate_zone_id]["QAC"]
    p_coef, p_intercept = _PHASE_C_REFERENCE[aggregate_zone_id]["PHVAC"]
    return PhaseCModelBundle(
        qac=PhaseCLinearSurrogate(qac_coef, qac_intercept, "QAC", aggregate_zone_id, "handoff_reference"),
        phvac=PhaseCLinearSurrogate(p_coef, p_intercept, "PHVAC", aggregate_zone_id, "handoff_reference"),
        aggregate_zone_id=aggregate_zone_id,
        provenance={"mode": "handoff_reference", "warning": "synthetic/unit-test fallback only"},
    )


def _walk_json_for_linear_parameters(obj: Any) -> tuple[float, float] | None:
    if isinstance(obj, dict):
        lower = {str(k).lower(): v for k, v in obj.items()}
        coef_keys = ("coefficient", "coef", "slope", "weight")
        int_keys = ("intercept", "bias")
        coef = next((lower[k] for k in coef_keys if k in lower and np.isscalar(lower[k])), None)
        intercept = next((lower[k] for k in int_keys if k in lower and np.isscalar(lower[k])), None)
        if coef is not None:
            return float(coef), float(intercept if intercept is not None else 0.0)
        for value in obj.values():
            found = _walk_json_for_linear_parameters(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _walk_json_for_linear_parameters(value)
            if found is not None:
                return found
    return None


def _linear_from_json(path: Path, *, component: str, aggregate_zone_id: str) -> PhaseCLinearSurrogate | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    found = _walk_json_for_linear_parameters(payload)
    if found is None:
        return None
    return PhaseCLinearSurrogate(found[0], found[1], component, aggregate_zone_id, str(path))


def _linear_from_torch(path: Path, *, component: str, aggregate_zone_id: str) -> PhaseCLinearSurrogate | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        candidates.append(payload)
        sd = payload.get("state_dict")
        if isinstance(sd, dict):
            candidates.append(sd)
    for cand in candidates:
        # First try semantic keys.
        found = _walk_json_for_linear_parameters(cand)
        if found is not None:
            return PhaseCLinearSurrogate(found[0], found[1], component, aggregate_zone_id, str(path))
        # Then accept an unambiguous scalar linear layer state_dict.
        weights = [v for k, v in cand.items() if "weight" in str(k).lower() and torch.is_tensor(v) and v.numel() == 1]
        biases = [v for k, v in cand.items() if "bias" in str(k).lower() and torch.is_tensor(v) and v.numel() == 1]
        if len(weights) == 1 and len(biases) <= 1:
            return PhaseCLinearSurrogate(
                float(weights[0].reshape(-1)[0]),
                float(biases[0].reshape(-1)[0]) if biases else 0.0,
                component,
                aggregate_zone_id,
                str(path),
            )
    return None


def load_phase_c_linear_artifact(path: Path, *, component: str, aggregate_zone_id: str):
    """Load the persisted one-input Phase-C QAC/PHVAC mapping.

    The production Phase-C lifecycle explicitly supports save/load. This paper
    adapter first tries the production serialization module if it exposes a load
    callable, then handles transparent JSON/torch scalar-linear artifacts. The
    function deliberately fails instead of fabricating a model.
    """

    path = Path(path)
    try:
        module = importlib.import_module("scalebridge.models.heat_input_regression.serialization")
    except Exception:
        module = None
    if module is not None:
        for name in ("load_heat_input_regression_model", "load_model", "load"):
            loader = getattr(module, name, None)
            if callable(loader):
                for candidate in (path, str(path)):
                    try:
                        model = loader(candidate)
                    except Exception:
                        continue
                    if hasattr(model, "predict"):
                        return model

    files: list[Path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(path)

    for file in files:
        if file.suffix.lower() == ".json":
            model = _linear_from_json(file, component=component, aggregate_zone_id=aggregate_zone_id)
            if model is not None:
                return model
        if file.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            model = _linear_from_torch(file, component=component, aggregate_zone_id=aggregate_zone_id)
            if model is not None:
                return model
    raise RuntimeError(f"Could not load a scalar linear {component} model from {path}")


def _candidate_component_paths(training_root: Path, aggregate_zone_id: str, component: str) -> list[Path]:
    tokens = (aggregate_zone_id.lower(), component.lower())
    candidates: list[tuple[int, Path]] = []

    # Prefer the authoritative C6 index when present. It commonly carries the
    # exact per-model artifact directory and avoids guessing directory names.
    for csv_path in training_root.rglob("training_results.csv"):
        try:
            import pandas as pd
            table = pd.read_csv(csv_path)
        except Exception:
            continue
        for _, row in table.iterrows():
            text = " ".join(str(v) for v in row.values).lower()
            if not all(token in text for token in tokens):
                continue
            for field, value in row.items():
                if pd.isna(value):
                    continue
                name = str(field).lower()
                if any(key in name for key in ("artifact", "model_dir", "model_path", "output_dir")):
                    candidate = Path(str(value))
                    if not candidate.is_absolute():
                        candidate = csv_path.parent / candidate
                    if candidate.exists():
                        candidates.append((10, candidate))

    for path in training_root.rglob("*"):
        if not path.is_file():
            continue
        haystack = str(path).lower()
        score = sum(token in haystack for token in tokens)
        if score < 2 and path.suffix.lower() in {".json", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")[:1_000_000].lower()
            except Exception:
                text = ""
            score = sum(token in text for token in tokens)
        if score >= 2:
            candidates.append((score, path))
    return [p for _, p in sorted(candidates, key=lambda item: (-item[0], len(str(item[1]))))]


def discover_and_load_phase_c_bundle(
    config: PaperConfig,
    aggregate_zone_id: str,
    *,
    phase_c_run_id: str | None = None,
) -> PhaseCModelBundle:
    """Discover actual controlled/production Phase-C QAC + PHVAC artifacts."""

    base = config.campaign_root / "heat_input_regression"
    training_root = base / "training_runs"
    if not training_root.exists():
        raise FileNotFoundError(f"Missing Phase-C training root: {training_root}")

    if phase_c_run_id:
        run_manifest = base / "campaign_runs" / phase_c_run_id / "phase_c_campaign_run_manifest.json"
        if not run_manifest.exists():
            raise FileNotFoundError(f"Missing Phase-C run manifest: {run_manifest}")
        run_text = run_manifest.read_text(encoding="utf-8", errors="ignore")
        # Narrow training roots to paths named in the authoritative run manifest when possible.
        mentioned = [d for d in training_root.iterdir() if d.is_dir() and d.name in run_text]
        roots = mentioned or [training_root]
    else:
        roots = [training_root]

    loaded: dict[str, Any] = {}
    provenance: dict[str, Any] = {"mode": "actual_phase_c_artifacts", "phase_c_run_id": phase_c_run_id, "artifacts": {}}
    for component in ("QAC", "PHVAC"):
        candidates: list[Path] = []
        for root in roots:
            candidates.extend(_candidate_component_paths(root, aggregate_zone_id, component))
        errors: list[str] = []
        for candidate in candidates:
            try:
                model = load_phase_c_linear_artifact(candidate, component=component, aggregate_zone_id=aggregate_zone_id)
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            loaded[component] = model
            provenance["artifacts"][component] = str(candidate)
            break
        if component not in loaded:
            preview = "\n".join(errors[:5])
            raise RuntimeError(
                f"Could not discover/load actual Phase-C {component} model for {aggregate_zone_id}. "
                f"Candidates={len(candidates)}. {preview}"
            )

    return PhaseCModelBundle(
        qac=loaded["QAC"],
        phvac=loaded["PHVAC"],
        aggregate_zone_id=aggregate_zone_id,
        provenance=provenance,
    )
