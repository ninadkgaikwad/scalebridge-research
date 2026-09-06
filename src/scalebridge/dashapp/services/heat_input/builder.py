"""Campaign-Builder services driven by the generalized Phase C runner contract."""
from __future__ import annotations

from typing import Any, Iterable

from scalebridge.data.heat_input_regression.campaign_config import PhaseCCampaignConfig
from scalebridge.dashapp.schemas.pipeline.heat_input import HeatInputCampaignDefinition
from scalebridge.models.heat_input_regression.registry import list_model_specifications

from .upstream_aggregation import resolve_parent_context, validate_matrix_selection


# These are public PhaseCCampaignConfig fields, but the Campaign Builder resolves
# them from upstream lineage rather than presenting duplicate arbitrary text
# inputs.  matrix_run_id has its own dedicated selector.
LINEAGE_MANAGED_FIELDS = frozenset(
    {"campaign_root", "campaign_id", "generated_data_root", "matrix_run_id"}
)


CURATED_CONFIG_FIELDS = frozenset(
    {
        "case_id",
        "aggregation_id",
        "weight_mode",
        "model_ids",
        "internal_gain_predictor_method",
        "hvac_target_method",
        "split_strategy",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "estimator_types",
        "pytorch_devices",
        "validation_profile",
        "mlflow_enabled",
    }
)

GROUP_TITLES = {
    "identity": "Run Identity",
    "scope": "Scientific Scope",
    "features_targets": "Features and Targets",
    "split": "Train / Validation / Test Split",
    "datasets": "Dataset Construction",
    "model_api": "Model API Validation",
    "training": "Model Training",
    "evaluation": "Evaluation Outputs",
    "inference": "Annual Inference",
    "validation": "Validation Thresholds",
    "mlflow": "MLflow Tracking",
    "diagnostics": "Diagnostics",
    "execution": "Execution / Recovery",
}

GROUP_ORDER = tuple(GROUP_TITLES)


def capability_manifest() -> dict[str, Any]:
    """Return the live machine-readable Phase C capability contract."""
    return PhaseCCampaignConfig.capability_manifest()


def runner_fields() -> list[dict[str, Any]]:
    """Return all 81 public runner fields in authoritative order."""
    return list(capability_manifest()["fields"])


def editable_runner_fields(
    *,
    visibility: str | None = None,
) -> list[dict[str, Any]]:
    """Return fields edited directly by generic Tab-1 controls."""
    rows = [
        field
        for field in runner_fields()
        if field["name"] not in LINEAGE_MANAGED_FIELDS
    ]
    if visibility is not None:
        rows = [
            field for field in rows
            if str(field.get("ui_visibility")) == visibility
        ]
    return rows


def grouped_runner_fields(visibility: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group direct controls using metadata emitted by PhaseCCampaignConfig."""
    fields = editable_runner_fields(visibility=visibility)
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    for group in GROUP_ORDER:
        items = [field for field in fields if field.get("ui_group") == group]
        if items:
            grouped.append((group, items))
    return grouped


def model_registry_rows() -> list[dict[str, Any]]:
    """Return the immutable 19-model relationship registry for display."""
    return [spec.to_dict() for spec in list_model_specifications()]


def model_id_options() -> list[dict[str, str]]:
    """Return model choices in the authoritative registry order."""
    return [
        {
            "label": f"{spec.model_id} — {spec.display_name}",
            "value": spec.model_id,
        }
        for spec in list_model_specifications()
    ]


def control_spec(field: dict[str, Any]) -> dict[str, Any]:
    """Translate capability JSON-schema metadata to a generic Dash control spec."""
    schema = dict(field.get("json_schema") or {})
    core, nullable = _unwrap_nullable(schema)
    field_type = core.get("type")
    enum = list(core.get("enum") or [])
    default = field.get("default")
    name = str(field["name"])

    if nullable and field_type == "boolean":
        kind = "tri_bool"
    elif field_type == "boolean":
        kind = "bool"
    elif field_type == "array":
        item_schema = dict(core.get("items") or {})
        if item_schema.get("enum"):
            kind = "multi_choice"
            enum = list(item_schema["enum"])
        elif name == "model_ids":
            kind = "model_ids"
        else:
            kind = "list_text"
    elif enum:
        kind = "choice"
    elif field_type in {"integer", "number"}:
        kind = "number"
    else:
        kind = "text"

    return {
        "name": name,
        "kind": kind,
        "default": default,
        "choices": enum,
        "nullable": nullable,
        "minimum": core.get("minimum"),
        "exclusive_minimum": core.get("exclusiveMinimum"),
        "maximum": core.get("maximum"),
        "description": field.get("description", ""),
        "group": field.get("ui_group", ""),
        "visibility": field.get("ui_visibility", ""),
        "stages": list(field.get("phase_c_stages") or []),
    }


def collect_config_values(
    field_ids: Iterable[dict[str, Any]] | None,
    field_values: Iterable[Any] | None,
) -> dict[str, Any]:
    """Normalize pattern-matching Dash values to PhaseCCampaignConfig primitives."""
    values: dict[str, Any] = {}
    specs = {field["name"]: control_spec(field) for field in editable_runner_fields()}
    for component_id, raw in zip(field_ids or [], field_values or []):
        name = str((component_id or {}).get("field") or "").strip()
        if not name or name not in specs:
            continue
        spec = specs[name]
        values[name] = _normalize_ui_value(raw, spec)
    return values


def build_definition(
    *,
    phase_c_campaign_id: str,
    parent_aggregation_campaign_id: str,
    matrix_run_id: str | None,
    machine_id: str,
    config_values: dict[str, Any] | None = None,
    display_name: str | None = None,
    notes: str | None = None,
) -> HeatInputCampaignDefinition:
    """Build and validate the complete saved Phase C definition."""
    phase_c_campaign_id = str(phase_c_campaign_id or "").strip()
    parent_aggregation_campaign_id = str(
        parent_aggregation_campaign_id or ""
    ).strip()
    machine_id = str(machine_id or "").strip()
    if not parent_aggregation_campaign_id:
        raise ValueError("Select a Parent Aggregation Campaign")
    if not machine_id:
        raise ValueError("Machine ID is required")

    context = resolve_parent_context(parent_aggregation_campaign_id)
    payload = PhaseCCampaignConfig().to_dict()
    for key, value in (config_values or {}).items():
        if key in LINEAGE_MANAGED_FIELDS:
            continue
        if key not in CURATED_CONFIG_FIELDS:
            raise ValueError(f"Unsupported Campaign Builder field: {key}")
        payload[key] = value

    # The simplified Builder always represents a complete Phase C campaign.
    # Recovery/truncation semantics remain available in the authoritative CLI,
    # but they are intentionally not part of the normal Dash campaign surface.
    payload["c1_aggregation_run_root"] = None
    payload["start_stage"] = "C1"
    payload["stop_stage"] = "C9"
    payload["continue_on_error"] = False
    payload["overwrite_existing"] = False

    matrix_run_id = str(matrix_run_id or "").strip()
    if not matrix_run_id:
        raise ValueError("Select an Aggregation Matrix Run")
    validate_matrix_selection(parent_aggregation_campaign_id, matrix_run_id)
    payload["matrix_run_id"] = matrix_run_id

    payload["campaign_root"] = str(context["campaign_root"])
    payload["campaign_id"] = str(context["parent_generation_campaign_id"])
    payload["generated_data_root"] = str(context["generated_data_root"])

    allowed_models = {row["model_id"] for row in model_registry_rows()}
    requested_models = set(payload.get("model_ids") or [])
    unknown_models = sorted(requested_models - allowed_models)
    if unknown_models:
        raise ValueError(
            "Unsupported model_ids: " + ", ".join(unknown_models)
        )

    config = PhaseCCampaignConfig.model_validate(payload)
    return HeatInputCampaignDefinition(
        phase_c_campaign_id=phase_c_campaign_id,
        parent_aggregation_campaign_id=parent_aggregation_campaign_id,
        parent_generation_campaign_id=str(
            context["parent_generation_campaign_id"]
        ),
        machine_id=machine_id,
        display_name=str(display_name or "").strip() or None,
        notes=str(notes or "").strip() or None,
        runner_config=config,
    )


def definition_summary(definition: HeatInputCampaignDefinition) -> dict[str, Any]:
    """Return compact information for preview/status rendering."""
    config = definition.runner_config
    return {
        "phase_c_campaign_id": definition.phase_c_campaign_id,
        "parent_aggregation_campaign_id": definition.parent_aggregation_campaign_id,
        "parent_generation_campaign_id": definition.parent_generation_campaign_id,
        "machine_id": definition.machine_id,
        "matrix_run_id": config.matrix_run_id,
        "case_id": config.case_id,
        "aggregation_id": config.aggregation_id,
        "weight_mode": config.weight_mode,
        "model_ids": list(config.model_ids),
        "internal_gain_predictor_method": config.internal_gain_predictor_method,
        "hvac_target_method": config.hvac_target_method,
        "estimators": list(config.estimator_types),
        "devices": list(config.pytorch_devices),
        "split": (
            f"{config.split_strategy} "
            f"{config.train_fraction:.3g}/{config.validation_fraction:.3g}/"
            f"{config.test_fraction:.3g}"
        ),
        "mlflow_enabled": config.mlflow_enabled,
        "validation_enabled": config.validation_profile != "none",
    }


def _unwrap_nullable(schema: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if "anyOf" not in schema:
        return schema, False
    candidates = [
        dict(item)
        for item in schema.get("anyOf") or []
        if item.get("type") != "null"
    ]
    nullable = len(candidates) != len(schema.get("anyOf") or [])
    if len(candidates) == 1:
        merged = dict(candidates[0])
        for key in ("minimum", "maximum", "exclusiveMinimum", "description"):
            if key in schema and key not in merged:
                merged[key] = schema[key]
        return merged, nullable
    return schema, nullable


def _normalize_ui_value(raw: Any, spec: dict[str, Any]) -> Any:
    kind = spec["kind"]
    default = spec["default"]

    if kind == "bool":
        return bool(raw)
    if kind == "tri_bool":
        if raw in (None, "", "inherit"):
            return None
        if raw in (True, "true", "True", 1):
            return True
        if raw in (False, "false", "False", 0):
            return False
        return raw
    if kind in {"multi_choice", "model_ids"}:
        return list(raw or [])
    if kind == "list_text":
        if isinstance(raw, (list, tuple)):
            return [str(item).strip() for item in raw if str(item).strip()]
        text = str(raw or "").replace("\n", ",")
        return [part.strip() for part in text.split(",") if part.strip()]
    if raw in (None, ""):
        return None if default is None else default
    return raw
