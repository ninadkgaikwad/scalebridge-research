"""Persistence for BGIRS Phase D campaign definitions."""
from __future__ import annotations

import json
from pathlib import Path

from scalebridge.dashapp.schemas.pipeline.phase_d import PhaseDCampaignDefinition
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


def definition_root() -> Path:
    path = resolve_generated_data_root() / "campaign_definitions" / "phase_d"
    path.mkdir(parents=True, exist_ok=True)
    return path


def definition_path(campaign_id: str) -> Path:
    return definition_root() / f"{str(campaign_id).strip()}.json"


def definition_exists(campaign_id: str) -> bool:
    return definition_path(campaign_id).is_file()


def load_definition(campaign_id: str) -> PhaseDCampaignDefinition:
    path = definition_path(campaign_id)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    # D1.1 backward compatibility: definitions saved before repeatable target
    # horizons used runner_config.ml_target_horizon as one scalar. Upgrade that
    # envelope in memory without rewriting the user's file until it is saved.
    runner = payload.get("runner_config")
    if isinstance(runner, dict) and "ml_target_horizons" not in runner:
        if "ml_target_horizon" in runner:
            legacy = runner.pop("ml_target_horizon")
            runner["ml_target_horizons"] = [legacy]
    return PhaseDCampaignDefinition.model_validate(payload)


def save_definition(definition: PhaseDCampaignDefinition, *, replace: bool = False) -> Path:
    path = definition_path(definition.phase_d_campaign_id)
    if path.exists() and not replace:
        raise FileExistsError(
            f"Phase D definition already exists: {definition.phase_d_campaign_id}. "
            "Enable Replace existing definition to overwrite it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(definition.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def list_definitions() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(definition_root().glob("*.json")):
        try:
            definition = load_definition(path.stem)
        except Exception:
            continue
        config = definition.runner_config
        rows.append(
            {
                "campaign_id": definition.phase_d_campaign_id,
                "phase_d_campaign_id": definition.phase_d_campaign_id,
                "parent_generation_campaign_id": definition.parent_generation_campaign_id,
                "parent_phase_c_run_key": definition.parent_phase_c_run_key,
                "machine_id": definition.machine_id,
                "matrix_run_id": config.matrix_run_id,
                "phase_c_campaign_run_id": config.phase_c_campaign_run_id,
                "heat_representation": config.heat_representation,
                "ml_policies": tuple(config.ml_policies),
                "ob_policies": tuple(config.ob_policies),
                "path": str(path),
            }
        )
    return rows
