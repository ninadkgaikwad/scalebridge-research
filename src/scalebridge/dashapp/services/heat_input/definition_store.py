"""Persistence for BGIRS Phase C Heat-Input campaign definitions."""
from __future__ import annotations

import json
from pathlib import Path

from scalebridge.dashapp.schemas.pipeline.heat_input import HeatInputCampaignDefinition
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


def definition_root() -> Path:
    """Return the persistent Phase C definition directory."""
    path = resolve_generated_data_root() / "campaign_definitions" / "heat_input"
    path.mkdir(parents=True, exist_ok=True)
    return path


def definition_path(campaign_id: str) -> Path:
    """Return the JSON path for one saved Phase C definition."""
    return definition_root() / f"{str(campaign_id).strip()}.json"


def definition_exists(campaign_id: str) -> bool:
    """Return whether a saved definition already exists."""
    return definition_path(campaign_id).is_file()


def load_definition(campaign_id: str) -> HeatInputCampaignDefinition:
    """Load and validate one saved definition."""
    path = definition_path(campaign_id)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return HeatInputCampaignDefinition.model_validate(payload)


def save_definition(
    definition: HeatInputCampaignDefinition,
    *,
    replace: bool = False,
) -> Path:
    """Persist a deterministic definition, refusing silent replacement."""
    path = definition_path(definition.phase_c_campaign_id)
    if path.exists() and not replace:
        raise FileExistsError(
            f"Phase C definition already exists: {definition.phase_c_campaign_id}. "
            "Enable Replace existing definition to overwrite it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(definition.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def list_definitions() -> list[dict[str, object]]:
    """Return compact metadata for readable saved Phase C definitions."""
    rows: list[dict[str, object]] = []
    for path in sorted(definition_root().glob("*.json")):
        try:
            definition = load_definition(path.stem)
        except Exception:
            continue
        config = definition.runner_config
        rows.append(
            {
                "campaign_id": definition.phase_c_campaign_id,
                "phase_c_campaign_id": definition.phase_c_campaign_id,
                "parent_aggregation_campaign_id": (
                    definition.parent_aggregation_campaign_id
                ),
                "parent_generation_campaign_id": (
                    definition.parent_generation_campaign_id
                ),
                "machine_id": definition.machine_id,
                "matrix_run_id": config.matrix_run_id,
                "estimator_types": tuple(config.estimator_types),
                "start_stage": config.start_stage,
                "stop_stage": config.stop_stage,
                "path": str(path),
            }
        )
    return rows
