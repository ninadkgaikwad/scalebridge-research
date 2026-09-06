from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.common import load_checkpoint, save_checkpoint, write_json
from ..evaluation.runtime import PaperModelRuntime
from ..methods.inverse_pinn import InversePINNRC


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    run_id: str
    method: str
    case_name: str
    rc_order: int
    seed: int
    checkpoint_path: str
    status: str
    acceptance: dict[str, object]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def append_registry(path: Path, record: CheckpointRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")


def save_candidate(path: Path, model, provenance: dict[str, object]) -> None:
    save_checkpoint(path, model=model, provenance=provenance)


def acceptance_gates(model, trajectory, checkpoint_path: Path, *, sim1_smoke=None, sim2_smoke=None) -> dict[str, object]:
    gates: dict[str, object] = {}
    params = [p.detach() for p in model.parameters()]
    gates["finite_parameters"] = bool(all(torch.isfinite(p).all().item() for p in params))
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    provenance = load_checkpoint(checkpoint_path, model=model)
    after = model.state_dict()
    gates["reload_state_dict_parity"] = bool(all(torch.equal(before[k].cpu(), after[k].detach().cpu()) for k in before))
    runtime = PaperModelRuntime(model, trajectory)
    # Small deterministic prediction parity probe.
    train = np.flatnonzero(trajectory.mask("train", included_only=True))
    probe = int(train[min(len(train) - 1, max(0, int(getattr(model.config, "L_e", 1)) - 1))])
    try:
        s = runtime.initialize(probe); p = runtime.observe(runtime.step(s, probe))
        gates["reload_prediction_finite"] = bool(np.isfinite(p).all())
    except Exception as exc:
        gates["reload_prediction_finite"] = False
        gates["reload_prediction_error"] = str(exc)
    if isinstance(model, InversePINNRC):
        try:
            phys = model.physical_parameters()
            values = []
            for value in phys.values():
                if torch.is_tensor(value):
                    values.extend(value.detach().cpu().reshape(-1).tolist())
                elif isinstance(value, (int, float)):
                    values.append(float(value))
            gates["inverse_physical_parameters_finite"] = bool(np.isfinite(values).all()) if values else True
            positive = [v for k, value in phys.items() if str(k).startswith(("R", "C")) for v in (value.detach().cpu().reshape(-1).tolist() if torch.is_tensor(value) else [value])]
            gates["inverse_RC_positive"] = bool(all(float(v) > 0 for v in positive)) if positive else True
        except Exception as exc:
            gates["inverse_physical_parameters_finite"] = False
            gates["inverse_physical_parameter_error"] = str(exc)
    if sim1_smoke is not None:
        gates["sim1_smoke_finite"] = bool(np.isfinite(sim1_smoke.trajectory.select_dtypes(include=[np.number]).to_numpy()).all())
    if sim2_smoke is not None:
        gates["sim2_smoke_finite"] = bool(np.isfinite(sim2_smoke.trajectory.select_dtypes(include=[np.number]).to_numpy()).all())
    gates["checkpoint_provenance_present"] = bool(provenance)
    gates["accepted"] = bool(all(v is True for k, v in gates.items() if not k.endswith("error")))
    return gates
