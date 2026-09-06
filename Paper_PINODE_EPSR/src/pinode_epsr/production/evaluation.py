from __future__ import annotations

from pathlib import Path

from ..core.common import write_json
from ..evaluation.runtime import PaperModelRuntime, sim1, sim2, sim3
from ..evaluation.thermostat import resolve_actuation_profile
from .contracts import ControllerOverrideConfig


def _save(result, base: Path) -> dict[str, str]:
    base.mkdir(parents=True, exist_ok=True)
    trajectory = base / "trajectory.parquet"
    metrics = base / "metrics.json"
    provenance = base / "provenance.json"
    result.trajectory.to_parquet(trajectory, index=False)
    write_json(metrics, {"metrics": result.metrics})
    write_json(provenance, result.provenance)
    return {"trajectory": str(trajectory), "metrics": str(metrics), "provenance": str(provenance)}


def run_offline_evaluations(
    model,
    trajectory,
    phase_c,
    calibrations,
    *,
    offline_root: Path,
    run_id: str,
    micro: bool = False,
    controller_overrides: ControllerOverrideConfig | None = None,
):
    runtime = PaperModelRuntime(model, trajectory)
    controller_overrides = controller_overrides or ControllerOverrideConfig()
    profiles = {
        z: resolve_actuation_profile(
            calibrations[z],
            overrides=controller_overrides.actuation_overrides(z),
        )
        for z in trajectory.zone_ids
    }
    r1 = sim1(runtime, phase_c, max_points=(12 if micro else None))
    r2 = sim2(runtime, phase_c, horizon=(12 if micro else None), all_test_segments=not micro)
    r3 = sim3(
        runtime,
        phase_c,
        calibrations,
        horizon=(12 if micro else None),
        all_test_segments=not micro,
        actuation_profiles=profiles,
    )
    outputs = {
        "sim1": _save(r1, offline_root / "sim1" / run_id),
        "sim2": _save(r2, offline_root / "sim2" / run_id),
        "sim3": _save(r3, offline_root / "sim3" / run_id),
    }
    return (r1, r2, r3), outputs
