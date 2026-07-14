# -*- coding: utf-8 -*-
"""MLflow tracking helpers for ScaleBridge aggregation runs.

This module keeps MLflow-specific logic out of aggregation engines/runners.
The aggregation code should orchestrate data discovery, rule application, plan
selection, and output writing; this module handles optional experiment/run/
metric/artifact logging.

Design notes
------------
- MLflow is imported lazily only when tracking is enabled.
- Tracking failures for individual params/metrics/artifacts do not fail the
  aggregation workflow.
- Per-case/per-plan matrix logging uses a unique prefix based on
  case_id + aggregation_id + weight_mode + aggregation_run_id. This avoids
  collisions when the same case appears many times in a 4B4C x aggregation-level
  x weight-mode matrix.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def maybe_start_mlflow_parent_run(
    *,
    enabled: bool,
    campaign_id: str,
    experiment_name: str | None,
    run_name: str | None,
    params: dict[str, Any],
):
    """Start optional MLflow parent run.

    Parameters
    ----------
    enabled:
        If False, no MLflow import or tracking call is made.
    campaign_id:
        ScaleBridge campaign ID.
    experiment_name:
        Optional MLflow experiment name. If omitted, a campaign-specific default
        is used.
    run_name:
        Optional MLflow run name. If omitted, a timestamped campaign run name is
        used.
    params:
        Campaign-level parameters to log.

    Returns
    -------
    object | None
        Active MLflow run object if enabled, otherwise None.
    """
    if not enabled:
        return None

    try:
        import mlflow
    except Exception as exc:
        raise RuntimeError(
            "MLflow tracking was requested with --mlflow, but mlflow could not "
            "be imported."
        ) from exc

    effective_experiment_name = (
        experiment_name or f"ScaleBridge_Aggregation_{campaign_id}"
    )
    effective_run_name = (
        run_name
        or f"aggregation_campaign_{campaign_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    mlflow.set_experiment(effective_experiment_name)
    active_run = mlflow.start_run(run_name=effective_run_name)

    for key, value in params.items():
        safe_log_param(key, value)

    return active_run


def mlflow_log_case_result(
    *,
    enabled: bool,
    case_result: dict[str, Any],
    artifact_root: Path | None = None,
) -> None:
    """Log one aggregation case or case-plan result to the active MLflow run.

    For matrix runs, the same case appears multiple times across aggregation
    levels and weight modes. Therefore this function builds a unique key prefix
    using:

        case_id + aggregation_id + weight_mode + aggregation_run_id

    This keeps metrics and params queryable without overwriting each other.
    """
    if not enabled:
        return

    case_id = str(case_result.get("case_id", "unknown_case"))
    aggregation_id = str(
        case_result.get("aggregation_id")
        or case_result.get("plan_aggregation_id")
        or case_result.get("loaded_plan_aggregation_id")
        or "unknown_aggregation"
    )
    weight_mode = str(
        case_result.get("weight_mode")
        or case_result.get("plan_weight_mode")
        or case_result.get("loaded_plan_weight_mode")
        or "unknown_weight"
    )
    aggregation_run_id = str(
        case_result.get("aggregation_run_id")
        or "unknown_run"
    )

    prefix = build_case_result_prefix(
        case_id=case_id,
        aggregation_id=aggregation_id,
        weight_mode=weight_mode,
        aggregation_run_id=aggregation_run_id,
    )

    metric_keys = [
        "aggregation_level_index",
        "aggregate_zone_count",
        "source_zone_count",
        "aggregation_compression_ratio",
        "loaded_variable_count",
        "aggregated_long_rows",
        "static_equipment_rows",
        "equipment_contribution_rows",
        "diagnostic_rows",
        "rule_summary_rows",
        "runtime_seconds",
    ]

    for key in metric_keys:
        safe_log_metric(f"{prefix}.{key}", case_result.get(key, None))

    param_keys = [
        "status",
        "case_id",
        "source_generation_run_id",
        "aggregation_run_id",
        "building_type",
        "climate_zone",
        "weather_location",
        "aggregation_id",
        "aggregation_level",
        "aggregation_family",
        "weight_mode",
        "plan_strategy",
        "rule_set",
        "loaded_plan_aggregation_id",
        "loaded_plan_strategy",
        "loaded_plan_rule_set",
        "loaded_plan_weight_mode",
        "plan_build_id",
        "plan_path",
        "run_root",
        "error_type",
        "error_message",
    ]

    for key in param_keys:
        if key in case_result:
            safe_log_param(f"{prefix}.{key}", case_result.get(key, ""))

    tag_keys = [
        "status",
        "building_type",
        "climate_zone",
        "weather_location",
        "aggregation_id",
        "aggregation_family",
        "weight_mode",
    ]
    for key in tag_keys:
        if key in case_result:
            safe_set_tag(f"{prefix}.{key}", case_result.get(key, ""))

    if artifact_root is not None and artifact_root.exists():
        try:
            import mlflow

            mlflow.log_artifacts(
                str(artifact_root),
                artifact_path=f"aggregation_cases/{prefix}",
            )
        except Exception:
            pass


def mlflow_log_campaign_summary(
    *,
    enabled: bool,
    summary: dict[str, Any],
    summary_dir: Path,
) -> None:
    """Log campaign-level or matrix-level aggregation summary to MLflow."""
    if not enabled:
        return

    metric_keys = [
        "case_count",
        "successful_case_count",
        "failed_case_count",
        "selected_plan_count",
        "successful_plan_count",
        "failed_plan_count",
        "runtime_seconds",
    ]

    for key in metric_keys:
        if key in summary:
            safe_log_metric(key, summary.get(key))

    param_keys = [
        "campaign_id",
        "campaign_aggregation_run_id",
        "matrix_run_id",
        "strategy",
        "aggregation_id",
        "rule_set",
        "weight_mode",
        "aggregation_ids",
        "weight_modes",
        "building_types",
        "climate_zones",
        "weather_locations",
        "summary_dir",
    ]

    for key in param_keys:
        if key in summary:
            safe_log_param(key, summary.get(key, ""))

    for key in [
        "campaign_id",
        "campaign_aggregation_run_id",
        "matrix_run_id",
    ]:
        if key in summary:
            safe_set_tag(key, summary.get(key, ""))

    if summary_dir.exists():
        try:
            import mlflow

            mlflow.log_artifacts(
                str(summary_dir),
                artifact_path="aggregation_campaign_summary",
            )
        except Exception:
            pass


def maybe_end_mlflow_run(enabled: bool) -> None:
    """End the active MLflow run if tracking is enabled."""
    if not enabled:
        return

    import mlflow

    mlflow.end_run()


def build_case_result_prefix(
    *,
    case_id: str,
    aggregation_id: str,
    weight_mode: str,
    aggregation_run_id: str,
) -> str:
    """Build a unique prefix for per-case/per-plan MLflow keys."""
    parts = [
        "case",
        sanitize_metric_name(case_id),
        sanitize_metric_name(aggregation_id),
        sanitize_metric_name(weight_mode),
        sanitize_metric_name(shorten_token(aggregation_run_id, max_length=48)),
    ]
    return ".".join(part for part in parts if part)


def safe_log_metric(key: str, value: Any) -> None:
    """Log one metric if it can be converted to float."""
    if value in {"", None}:
        return

    try:
        import mlflow

        mlflow.log_metric(sanitize_metric_name(key), float(value))
    except Exception:
        pass


def safe_log_param(key: str, value: Any) -> None:
    """Log one MLflow param defensively.

    MLflow params are immutable within a run. If the same key is logged twice
    with a different value, MLflow raises. We intentionally skip that failure.
    """
    if value is None:
        value = ""

    try:
        import mlflow

        mlflow.log_param(sanitize_param_key(key), stringify_param_value(value))
    except Exception:
        pass


def safe_set_tag(key: str, value: Any) -> None:
    """Set one MLflow tag defensively."""
    if value is None:
        value = ""

    try:
        import mlflow

        mlflow.set_tag(sanitize_param_key(key), stringify_param_value(value))
    except Exception:
        pass


def stringify_param_value(value: Any, *, max_length: int = 500) -> str:
    """Convert MLflow param/tag value to a compact string."""
    if isinstance(value, (list, tuple, set)):
        text = ",".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = ",".join(f"{key}={val}" for key, val in sorted(value.items()))
    else:
        text = str(value)

    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def sanitize_metric_name(value: str) -> str:
    """Sanitize text for use inside MLflow metric names."""
    return "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_"
        for char in str(value)
    )


def sanitize_param_key(value: str) -> str:
    """Sanitize text for use as an MLflow param/tag key."""
    return "".join(
        char if char.isalnum() or char in {"_", "-", ".", "/"} else "_"
        for char in str(value)
    )


def shorten_token(value: str, *, max_length: int) -> str:
    """Shorten a token from the left-preserved side."""
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[:max_length]
