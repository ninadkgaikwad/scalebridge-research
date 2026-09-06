"""Read-only, manifest-first access to Phase C Heat-Input Regression results."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Iterable
import zipfile

import pandas as pd

from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root


CAMPAIGN_MANIFEST = "phase_c_campaign_run_manifest.json"
CAMPAIGN_PLAN = "phase_c_campaign_plan.json"

_STAGE_LAYOUT = {
    "C1": ("audit_runs", "heat_input_regression_audit_manifest.json"),
    "C2": ("feature_runs", "heat_input_feature_run_manifest.json"),
    "C3": ("split_runs", "split_run_manifest.json"),
    "C4": ("dataset_runs", "dataset_run_manifest.json"),
    "C5": ("model_api_validation", "c5_model_api_validation_manifest.json"),
    "C6": ("training_runs", "training_run_manifest.json"),
    "C7": ("evaluation_runs", "evaluation_run_manifest.json"),
    "C8": ("inference_runs", "inference_run_manifest.json"),
    "C9": ("mlflow_registration_runs", "phase_c_mlflow_registration_manifest.json"),
}

_STAGE_RUN_FLAGS = {
    "C1": "--audit-run-id",
    "C2": "--feature-run-id",
    "C3": "--split-run-id",
    "C4": "--dataset-run-id",
    "C6": "--training-run-id",
    "C7": "--evaluation-run-id",
    "C8": "--inference-run-id",
}

_CONTEXT_KEYS = [
    "case_id",
    "aggregation_id",
    "weight_mode",
    "aggregate_zone_id",
    "model_id",
]

_FILTER_TO_COLUMN = {
    "building_types": "building_type",
    "weather_locations": "weather_location",
    "climate_zones": "climate_zone",
    "case_ids": "case_id",
    "aggregation_ids": "aggregation_id",
    "weight_modes": "weight_mode",
    "aggregate_zone_ids": "aggregate_zone_id",
    "model_ids": "model_id",
    "estimator_types": "estimator_type",
}

_FILTER_ORDER = [
    "building_type",
    "weather_location",
    "climate_zone",
    "case_id",
    "aggregation_id",
    "weight_mode",
    "aggregate_zone_id",
    "model_id",
    "estimator_type",
]


class ResultSelectionTooBroad(ValueError):
    """Raised before detailed result files are read for an overly broad selection."""


def campaigns_root() -> Path:
    """Return the shared ScaleBridge campaign root without mutating it."""
    return resolve_generated_data_root() / "campaigns"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _as_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    safe = frame.where(pd.notna(frame), None)
    return safe.to_dict(orient="records")


def _command_value(result: dict[str, Any], flag: str) -> str:
    command = result.get("command") or []
    if not isinstance(command, list):
        return ""
    for index, token in enumerate(command[:-1]):
        if str(token) == flag:
            return str(command[index + 1])
    return ""


def _stage_operations(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    operations: dict[str, dict[str, Any]] = {}
    for row in manifest.get("results") or []:
        if not isinstance(row, dict):
            continue
        match = re.match(r"^(C[1-9])\s", str(row.get("name") or ""))
        if match:
            operations[match.group(1)] = row
    return operations


def _validation_stage(name: str) -> str:
    match = re.match(r"^VALIDATE\s+(C[1-9])(?:_|\s|$)", name)
    if match:
        return match.group(1)
    if name == "VALIDATE source":
        return "C1"
    return ""


def discover_phase_c_runs() -> list[dict[str, Any]]:
    """Discover Phase C campaign runs by reading only campaign-level manifests."""
    rows: list[dict[str, Any]] = []
    root = campaigns_root()
    if not root.is_dir():
        return rows

    for campaign_root in sorted(path for path in root.iterdir() if path.is_dir()):
        run_root = campaign_root / "heat_input_regression" / "campaign_runs"
        if not run_root.is_dir():
            continue
        for phase_run_root in sorted(
            (path for path in run_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        ):
            manifest_path = phase_run_root / CAMPAIGN_MANIFEST
            manifest = _read_json(manifest_path)
            if not manifest:
                continue
            availability = manifest.get("availability_summary") or {}
            rows.append(
                {
                    "campaign_id": str(manifest.get("campaign_id") or campaign_root.name),
                    "phase_c_run_id": str(
                        manifest.get("phase_c_run_id") or phase_run_root.name
                    ),
                    "matrix_run_id": str(manifest.get("matrix_run_id") or ""),
                    "status": str(manifest.get("status") or ""),
                    "created_at_utc": str(manifest.get("created_at_utc") or ""),
                    "runtime_seconds": manifest.get("runtime_seconds"),
                    "command_count": int(manifest.get("command_count") or 0),
                    "passed_command_count": int(manifest.get("passed_command_count") or 0),
                    "failed_command_count": int(manifest.get("failed_command_count") or 0),
                    "candidate_model_count": int(
                        availability.get("candidate_model_count") or 0
                    ),
                    "applicable_model_count": int(
                        availability.get("applicable_model_count") or 0
                    ),
                    "trained_model_count": int(
                        availability.get("trained_model_count") or 0
                    ),
                    "evaluated_model_count": int(
                        availability.get("evaluated_model_count") or 0
                    ),
                    "inference_zone_count": int(
                        availability.get("inference_zone_count") or 0
                    ),
                    "inferred_component_count": int(
                        availability.get("inferred_component_count") or 0
                    ),
                    "campaign_root": str(campaign_root),
                    "run_root": str(phase_run_root),
                    "manifest_path": str(manifest_path),
                }
            )
    rows.sort(key=lambda row: (row["created_at_utc"], row["phase_c_run_id"]), reverse=True)
    return rows


def run_options() -> list[dict[str, str]]:
    """Return dropdown options for discovered Phase C runs."""
    return [
        {
            "label": (
                f"{row['phase_c_run_id']} | {row['campaign_id']} | "
                f"{row['status'] or 'unknown'}"
            ),
            "value": f"{row['campaign_id']}::{row['phase_c_run_id']}",
        }
        for row in discover_phase_c_runs()
    ]


def encode_run_key(campaign_id: str, phase_c_run_id: str) -> str:
    return f"{campaign_id}::{phase_c_run_id}"


def decode_run_key(value: str) -> tuple[str, str]:
    text = str(value or "")
    if "::" not in text:
        raise ValueError("Expected Phase C run key '<campaign_id>::<phase_c_run_id>'")
    campaign_id, phase_c_run_id = text.split("::", 1)
    if not campaign_id or not phase_c_run_id:
        raise ValueError(f"Invalid Phase C run key: {value!r}")
    return campaign_id, phase_c_run_id


def load_run_ref(campaign_id: str, phase_c_run_id: str) -> dict[str, Any]:
    campaign_root = campaigns_root() / str(campaign_id)
    run_root = (
        campaign_root
        / "heat_input_regression"
        / "campaign_runs"
        / str(phase_c_run_id)
    )
    manifest_path = run_root / CAMPAIGN_MANIFEST
    manifest = _read_json(manifest_path)
    if not manifest:
        raise FileNotFoundError(f"Phase C run manifest not found or unreadable: {manifest_path}")
    return {
        "campaign_id": str(manifest.get("campaign_id") or campaign_id),
        "phase_c_run_id": str(manifest.get("phase_c_run_id") or phase_c_run_id),
        "campaign_root": campaign_root,
        "run_root": run_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def load_run_ref_from_key(run_key: str) -> dict[str, Any]:
    return load_run_ref(*decode_run_key(run_key))


def stage_summary(run_ref: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize C1-C9 operation and validator status from one campaign manifest."""
    manifest = run_ref["manifest"]
    operations = _stage_operations(manifest)
    validators: dict[str, list[dict[str, Any]]] = {f"C{i}": [] for i in range(1, 10)}
    for row in manifest.get("results") or []:
        if not isinstance(row, dict):
            continue
        stage = _validation_stage(str(row.get("name") or ""))
        if stage:
            validators[stage].append(row)

    rows: list[dict[str, Any]] = []
    for stage in [f"C{i}" for i in range(1, 10)]:
        operation = operations.get(stage) or {}
        validation_rows = validators[stage]
        validation_failures = sum(
            str(row.get("status") or "").casefold() not in {"passed", "completed"}
            for row in validation_rows
        )
        if validation_rows:
            validation_status = "passed" if validation_failures == 0 else "failed"
        else:
            validation_status = "not-recorded"
        rows.append(
            {
                "stage": stage,
                "operation": str(operation.get("name") or stage),
                "status": str(operation.get("status") or "not-run"),
                "runtime_seconds": operation.get("runtime_seconds"),
                "return_code": operation.get("return_code"),
                "validation_status": validation_status,
                "validation_count": len(validation_rows),
                "validation_failure_count": validation_failures,
                "log_path": str(operation.get("log_path") or ""),
            }
        )
    return rows


def _path_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return PureWindowsPath(text.replace("/", "\\")).name


def stage_roots(run_ref: dict[str, Any]) -> dict[str, Path]:
    """Resolve local C1-C9 roots from the campaign command plan, not stale absolute paths."""
    manifest = run_ref["manifest"]
    operations = _stage_operations(manifest)
    heat_root = Path(run_ref["campaign_root"]) / "heat_input_regression"
    roots: dict[str, Path] = {}

    for stage, (folder, _manifest_name) in _STAGE_LAYOUT.items():
        if stage == "C9":
            roots[stage] = heat_root / folder / str(run_ref["phase_c_run_id"])
            continue
        operation = operations.get(stage) or {}
        if stage == "C5":
            run_id = _path_name(_command_value(operation, "--output-root"))
        else:
            run_id = _command_value(operation, _STAGE_RUN_FLAGS.get(stage, ""))
        if run_id:
            roots[stage] = heat_root / folder / run_id
    return roots


def stage_manifest_paths(run_ref: dict[str, Any]) -> dict[str, Path]:
    roots = stage_roots(run_ref)
    return {
        stage: roots[stage] / _STAGE_LAYOUT[stage][1]
        for stage in roots
        if stage in _STAGE_LAYOUT
    }


def _localize_artifact_path(
    stored_path: str | Path,
    *,
    stage_root: Path,
) -> Path:
    """Map an artifact path persisted on another machine into the selected local stage root."""
    text = str(stored_path or "").strip()
    if not text:
        return Path()
    direct = Path(text)
    if direct.is_file():
        return direct

    normalized = text.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    run_name = stage_root.name
    try:
        index = [part.casefold() for part in parts].index(run_name.casefold())
    except ValueError:
        return stage_root / PureWindowsPath(text).name
    relative = parts[index + 1 :]
    return stage_root.joinpath(*relative)


def run_summary(run_ref: dict[str, Any]) -> dict[str, Any]:
    manifest = run_ref["manifest"]
    availability = manifest.get("availability_summary") or {}
    roots = stage_roots(run_ref)
    c9 = _read_json(roots.get("C9", Path()) / _STAGE_LAYOUT["C9"][1])
    tracking_uri = str(c9.get("tracking_uri") or "")
    experiment_id = str(c9.get("experiment_id") or "")
    parent_run_id = str(c9.get("parent_run_id") or "")
    mlflow_url = ""
    if tracking_uri and experiment_id and parent_run_id:
        mlflow_url = (
            tracking_uri.rstrip("/")
            + f"/#/experiments/{experiment_id}/runs/{parent_run_id}"
        )
    return {
        "campaign_id": run_ref["campaign_id"],
        "phase_c_run_id": run_ref["phase_c_run_id"],
        "matrix_run_id": str(manifest.get("matrix_run_id") or ""),
        "status": str(manifest.get("status") or ""),
        "created_at_utc": str(manifest.get("created_at_utc") or ""),
        "runtime_seconds": manifest.get("runtime_seconds"),
        "command_count": int(manifest.get("command_count") or 0),
        "passed_command_count": int(manifest.get("passed_command_count") or 0),
        "failed_command_count": int(manifest.get("failed_command_count") or 0),
        "availability_summary": dict(availability),
        "campaign_root": str(run_ref["campaign_root"]),
        "manifest_path": str(run_ref["manifest_path"]),
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_experiment_id": experiment_id,
        "mlflow_parent_run_id": parent_run_id,
        "mlflow_url": mlflow_url,
        "mlflow_stage_run_count": c9.get("stage_run_count"),
        "mlflow_task_run_count": c9.get("task_run_count"),
    }


def _stage_csv(run_ref: dict[str, Any], stage: str, name: str) -> pd.DataFrame:
    root = stage_roots(run_ref).get(stage)
    return _read_csv(root / name) if root else pd.DataFrame()


def audit_zone_catalog(run_ref: dict[str, Any]) -> pd.DataFrame:
    return _stage_csv(run_ref, "C1", "audit_zone_results.csv")


def dataset_catalog(run_ref: dict[str, Any]) -> pd.DataFrame:
    return _stage_csv(run_ref, "C4", "dataset_model_results.csv")


def training_catalog(run_ref: dict[str, Any]) -> pd.DataFrame:
    frame = _stage_csv(run_ref, "C6", "training_results.csv")
    return _join_context(frame, dataset_catalog(run_ref))


def evaluation_catalog(run_ref: dict[str, Any]) -> pd.DataFrame:
    frame = _stage_csv(run_ref, "C7", "evaluation_results.csv")
    return _join_context(frame, dataset_catalog(run_ref))


def inference_catalog(run_ref: dict[str, Any]) -> pd.DataFrame:
    frame = _stage_csv(run_ref, "C8", "inference_results.csv")
    zones = audit_zone_catalog(run_ref)
    if frame.empty or zones.empty:
        return frame
    zone_keys = ["case_id", "aggregation_id", "weight_mode", "aggregate_zone_id"]
    context_cols = zone_keys + [
        column
        for column in (
            "building_type",
            "weather_location",
            "climate_zone",
            "strategy",
            "rule_set",
            "aggregation_run_id",
        )
        if column in zones.columns
    ]
    context = zones[context_cols].drop_duplicates(subset=zone_keys)
    return frame.merge(context, on=zone_keys, how="left", validate="many_to_one")


def _join_context(frame: pd.DataFrame, datasets: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or datasets.empty:
        return frame
    keys = [key for key in _CONTEXT_KEYS if key in frame.columns and key in datasets.columns]
    if len(keys) < 4:
        return frame
    additions = [
        column
        for column in (
            "building_type",
            "weather_location",
            "climate_zone",
            "strategy",
            "rule_set",
            "aggregation_run_id",
            "aggregation_family",
            "aggregation_level",
        )
        if column in datasets.columns and column not in frame.columns
    ]
    context = datasets[keys + additions].drop_duplicates(subset=keys)
    return frame.merge(context, on=keys, how="left", validate="many_to_one")


def filter_frame(frame: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    selected = frame
    for argument, column in _FILTER_TO_COLUMN.items():
        values = _as_values(filters.get(argument))
        if values and column in selected.columns:
            selected = selected[selected[column].astype(str).isin(values)]
    return selected


def filter_records(rows: Iterable[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(rows or []))
    return _clean_records(filter_frame(frame, **filters))


def faceted_filter_options(run_ref: dict[str, Any], **filters: Any) -> dict[str, list[str]]:
    """Build mutually constrained filter values without reading detailed predictions."""
    base = evaluation_catalog(run_ref)
    if base.empty:
        base = training_catalog(run_ref)
    selected_by_column = {
        column: _as_values(filters.get(argument))
        for argument, column in _FILTER_TO_COLUMN.items()
    }
    options: dict[str, list[str]] = {}
    for column in _FILTER_ORDER:
        if column not in base.columns:
            options[column] = []
            continue
        current = base
        for other_column, values in selected_by_column.items():
            if other_column == column or not values or other_column not in current.columns:
                continue
            current = current[current[other_column].astype(str).isin(values)]
        options[column] = sorted(
            {
                str(value)
                for value in current[column].dropna().astype(str)
                if str(value).strip()
            },
            key=str.casefold,
        )
    return options


def evaluation_modes(model_ids: Any = None) -> list[str]:
    models = set(_as_values(model_ids))
    if not models:
        return ["direct", "oracle", "chained"]
    modes: set[str] = set()
    if models - {"PHVAC"}:
        modes.add("direct")
    if "PHVAC" in models:
        modes.update({"oracle", "chained"})
    return [mode for mode in ("direct", "oracle", "chained") if mode in modes]


def selected_evaluation_rows(
    run_ref: dict[str, Any],
    *,
    max_artifacts: int = 24,
    **filters: Any,
) -> list[dict[str, Any]]:
    selected = filter_frame(evaluation_catalog(run_ref), **filters)
    if "status" in selected.columns:
        selected = selected[selected["status"].astype(str) == "completed"]
    if len(selected) > max_artifacts:
        raise ResultSelectionTooBroad(
            f"Selection resolves to {len(selected)} evaluation artifacts. "
            f"Narrow the filters to at most {max_artifacts} before loading details."
        )
    return _clean_records(selected)


def _evaluation_manifest(
    run_ref: dict[str, Any],
    row: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    stage_root = stage_roots(run_ref).get("C7")
    if stage_root is None:
        raise FileNotFoundError("C7 stage root is unavailable for this Phase C run")
    manifest_path = _localize_artifact_path(
        row.get("manifest_path") or "",
        stage_root=stage_root,
    )
    payload = _read_json(manifest_path)
    if not payload:
        raise FileNotFoundError(f"Evaluation manifest unavailable: {manifest_path}")
    return payload, manifest_path


def load_evaluation_metrics(
    run_ref: dict[str, Any],
    *,
    splits: Any = None,
    evaluation_modes_selected: Any = None,
    **filters: Any,
) -> list[dict[str, Any]]:
    rows = selected_evaluation_rows(run_ref, **filters)
    split_values = set(_as_values(splits))
    mode_values = set(_as_values(evaluation_modes_selected))
    stage_root = stage_roots(run_ref).get("C7")
    if stage_root is None:
        return []

    outputs: list[pd.DataFrame] = []
    for row in rows:
        manifest, _ = _evaluation_manifest(run_ref, row)
        metrics_path = _localize_artifact_path(
            manifest.get("split_metrics_path") or row.get("metrics_path") or "",
            stage_root=stage_root,
        )
        metrics = _read_csv(metrics_path)
        if metrics.empty:
            continue
        if split_values and "split" in metrics.columns:
            metrics = metrics[metrics["split"].astype(str).isin(split_values)]
        if mode_values and "evaluation_mode" in metrics.columns:
            metrics = metrics[metrics["evaluation_mode"].astype(str).isin(mode_values)]
        if metrics.empty:
            continue
        for key in (
            "building_type",
            "weather_location",
            "climate_zone",
            "case_id",
            "aggregation_id",
            "weight_mode",
            "aggregate_zone_id",
            "model_id",
            "estimator_type",
            "requested_device",
            "resolved_device",
        ):
            metrics[key] = row.get(key)
        metrics["metrics_path"] = str(metrics_path)
        outputs.append(metrics)
    if not outputs:
        return []
    return _clean_records(pd.concat(outputs, ignore_index=True))


def load_model_metadata(
    run_ref: dict[str, Any],
    *,
    max_rows: int = 250,
    **filters: Any,
) -> list[dict[str, Any]]:
    selected = filter_frame(training_catalog(run_ref), **filters)
    if len(selected) > max_rows:
        raise ResultSelectionTooBroad(
            f"Selection resolves to {len(selected)} trained artifacts. "
            f"Narrow the filters to at most {max_rows}."
        )
    preferred = [
        "building_type",
        "weather_location",
        "climate_zone",
        "case_id",
        "aggregation_id",
        "weight_mode",
        "aggregate_zone_id",
        "model_id",
        "estimator_type",
        "requested_device",
        "device",
        "fit_intercept",
        "intercept_policy_source",
        "model_role",
        "input_transform",
        "dependency_model_id",
        "target_allocation",
        "coefficient",
        "intercept",
        "training_rmse",
        "converged",
        "epochs_completed",
        "reload_predictions_match",
        "runtime_seconds",
    ]
    columns = [column for column in preferred if column in selected.columns]
    return _clean_records(selected[columns])


def _filter_time(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    elif "timestamp_raw" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp_raw"], errors="coerce")
    else:
        return frame
    frame = frame.copy()
    frame["timestamp"] = timestamp
    frame = frame[frame["timestamp"].notna()]
    if start:
        frame = frame[frame["timestamp"] >= pd.Timestamp(start)]
    if end:
        frame = frame[frame["timestamp"] <= pd.Timestamp(end)]
    return frame


def load_evaluation_series(
    run_ref: dict[str, Any],
    *,
    splits: Any,
    evaluation_modes_selected: Any,
    resolution: str = "preview",
    start: str | None = None,
    end: str | None = None,
    max_artifacts: int = 12,
    **filters: Any,
) -> list[dict[str, Any]]:
    rows = selected_evaluation_rows(run_ref, max_artifacts=max_artifacts, **filters)
    splits_list = _as_values(splits) or ["test"]
    modes_list = _as_values(evaluation_modes_selected)
    stage_root = stage_roots(run_ref).get("C7")
    if stage_root is None:
        return []

    series: list[dict[str, Any]] = []
    for row in rows:
        manifest, _ = _evaluation_manifest(run_ref, row)
        valid_modes = [str(value) for value in manifest.get("evaluation_modes") or []]
        requested_modes = modes_list or valid_modes or ["direct"]
        for split in splits_list:
            if resolution == "full":
                stored = (manifest.get("prediction_paths") or {}).get(split, "")
            else:
                stored = (manifest.get("preview_paths") or {}).get(split, "")
                if not stored:
                    stored = (manifest.get("prediction_paths") or {}).get(split, "")
            path = _localize_artifact_path(stored, stage_root=stage_root)
            if not path.is_file():
                continue
            if path.suffix.casefold() == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path)
            frame = _filter_time(frame, start, end)
            if frame.empty:
                continue
            for mode in requested_modes:
                if mode not in valid_modes:
                    continue
                prediction_column = "prediction_chained" if mode == "chained" else "prediction"
                if prediction_column not in frame.columns or "y" not in frame.columns:
                    continue
                identity = (
                    f"{row.get('building_type') or row.get('case_id')} | "
                    f"{row.get('aggregation_id')} | {row.get('aggregate_zone_id')} | "
                    f"{row.get('model_id')} | {row.get('estimator_type')} | {split} | {mode}"
                )
                timestamps = frame["timestamp"].astype(str).tolist()
                series.append(
                    {
                        "name": f"{identity} | target",
                        "identity": identity,
                        "role": "target",
                        "timestamp": timestamps,
                        "value": pd.to_numeric(frame["y"], errors="coerce").tolist(),
                        "source_path": str(path),
                    }
                )
                series.append(
                    {
                        "name": f"{identity} | prediction",
                        "identity": identity,
                        "role": "prediction",
                        "timestamp": timestamps,
                        "value": pd.to_numeric(
                            frame[prediction_column], errors="coerce"
                        ).tolist(),
                        "source_path": str(path),
                    }
                )
    return series


def building_phvac_metrics(
    run_ref: dict[str, Any],
    *,
    splits: Any = None,
    evaluation_modes_selected: Any = None,
    **filters: Any,
) -> list[dict[str, Any]]:
    root = stage_roots(run_ref).get("C7")
    if root is None:
        return []
    frame = _read_csv(root / "building_phvac_reconstruction" / "building_phvac_metrics.csv")
    if frame.empty:
        return []
    for argument, column in (
        ("case_ids", "case_id"),
        ("aggregation_ids", "aggregation_id"),
        ("weight_modes", "weight_mode"),
        ("estimator_types", "estimator_type"),
    ):
        values = _as_values(filters.get(argument))
        if values and column in frame.columns:
            frame = frame[frame[column].astype(str).isin(values)]
    split_values = _as_values(splits)
    if split_values and "split" in frame.columns:
        frame = frame[frame["split"].astype(str).isin(split_values)]
    mode_values = _as_values(evaluation_modes_selected)
    if mode_values and "evaluation_mode" in frame.columns:
        frame = frame[frame["evaluation_mode"].astype(str).isin(mode_values)]
    return _clean_records(frame)


def zone_key(row: dict[str, Any]) -> str:
    values = [
        row.get("case_id"),
        row.get("aggregation_id"),
        row.get("weight_mode"),
        row.get("aggregate_zone_id"),
    ]
    return json.dumps([str(value or "") for value in values], separators=(",", ":"))


def decode_zone_key(value: str) -> tuple[str, str, str, str]:
    try:
        payload = json.loads(str(value))
    except Exception as exc:
        raise ValueError(f"Invalid annual inference zone key: {value!r}") from exc
    if not isinstance(payload, list) or len(payload) != 4:
        raise ValueError(f"Invalid annual inference zone key: {value!r}")
    return tuple(str(item) for item in payload)  # type: ignore[return-value]


def inference_zone_options(run_ref: dict[str, Any], **filters: Any) -> list[dict[str, str]]:
    frame = filter_frame(inference_catalog(run_ref), **filters)
    options: list[dict[str, str]] = []
    for row in _clean_records(frame):
        options.append(
            {
                "label": (
                    f"{row.get('building_type') or row.get('case_id')} | "
                    f"{row.get('aggregation_id')} | {row.get('weight_mode')} | "
                    f"{row.get('aggregate_zone_id')} | {row.get('component_count')} components"
                ),
                "value": zone_key(row),
            }
        )
    return options


def _selected_inference_row(run_ref: dict[str, Any], value: str) -> dict[str, Any]:
    wanted = decode_zone_key(value)
    frame = inference_catalog(run_ref)
    keys = ["case_id", "aggregation_id", "weight_mode", "aggregate_zone_id"]
    for column, expected in zip(keys, wanted):
        if column not in frame.columns:
            raise KeyError(f"C8 inference index lacks required column {column!r}")
        frame = frame[frame[column].astype(str) == expected]
    if len(frame) != 1:
        raise ValueError(f"Expected one C8 zone result for {wanted}, found {len(frame)}")
    return _clean_records(frame)[0]


def _annual_manifest(
    run_ref: dict[str, Any],
    selected_zone_key: str,
) -> tuple[dict[str, Any], Path]:
    row = _selected_inference_row(run_ref, selected_zone_key)
    root = stage_roots(run_ref).get("C8")
    if root is None:
        raise FileNotFoundError("C8 stage root is unavailable for this Phase C run")
    path = _localize_artifact_path(row.get("manifest_path") or "", stage_root=root)
    payload = _read_json(path)
    if not payload:
        raise FileNotFoundError(f"Annual inference manifest unavailable: {path}")
    return payload, path


def annual_component_catalog(
    run_ref: dict[str, Any],
    selected_zone_key: str,
) -> list[dict[str, Any]]:
    manifest, _ = _annual_manifest(run_ref, selected_zone_key)
    root = stage_roots(run_ref)["C8"]
    registry_path = _localize_artifact_path(
        (manifest.get("outputs") or {}).get("component_prediction_registry") or "",
        stage_root=root,
    )
    registry = _read_csv(registry_path)
    if registry.empty:
        return []
    rows: list[dict[str, Any]] = []
    for record in _clean_records(registry):
        primary = str(record.get("output_prediction_column") or "")
        if primary:
            mode = "chained" if record.get("model_id") == "PHVAC" else "direct"
            rows.append(
                {
                    **record,
                    "prediction_column": primary,
                    "evaluation_mode": mode,
                    "label": (
                        f"{record.get('model_id')} | {primary} | {mode} | "
                        f"{record.get('prediction_units') or 'units not recorded'}"
                    ),
                }
            )
        oracle = str(record.get("oracle_output_prediction_column") or "")
        if oracle:
            rows.append(
                {
                    **record,
                    "prediction_column": oracle,
                    "evaluation_mode": "oracle",
                    "label": (
                        f"{record.get('model_id')} | {oracle} | oracle | "
                        f"{record.get('prediction_units') or 'units not recorded'}"
                    ),
                }
            )
    return rows


def load_annual_series(
    run_ref: dict[str, Any],
    *,
    selected_zone_key: str,
    prediction_columns: Any,
    resolution: str = "full",
    start: str | None = None,
    end: str | None = None,
    max_components: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog = annual_component_catalog(run_ref, selected_zone_key)
    allowed = {str(row["prediction_column"]): row for row in catalog}
    selected = _as_values(prediction_columns)
    if not selected:
        raise ValueError("Select at least one annual component prediction")
    unknown = [column for column in selected if column not in allowed]
    if unknown:
        raise ValueError(f"Unknown annual prediction column(s): {unknown}")
    if len(selected) > max_components:
        raise ResultSelectionTooBroad(
            f"Select at most {max_components} annual component traces at once."
        )

    manifest, _ = _annual_manifest(run_ref, selected_zone_key)
    root = stage_roots(run_ref)["C8"]
    outputs = manifest.get("outputs") or {}
    stored = (
        outputs.get("annual_component_predictions")
        if resolution == "full"
        else outputs.get("annual_component_predictions_preview")
    )
    if not stored:
        stored = outputs.get("annual_component_predictions")
    path = _localize_artifact_path(stored or "", stage_root=root)
    if not path.is_file():
        raise FileNotFoundError(f"Annual prediction table unavailable: {path}")

    requested_columns = ["timestamp_raw", "timestamp", *selected]
    if path.suffix.casefold() == ".parquet":
        frame = pd.read_parquet(path, columns=requested_columns)
    else:
        frame = pd.read_csv(path, usecols=lambda column: column in requested_columns)
    frame = _filter_time(frame, start, end)
    if frame.empty:
        return [], []

    series: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    timestamps = frame["timestamp"].astype(str).tolist()
    for column in selected:
        metadata = allowed[column]
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.dropna()
        series.append(
            {
                "name": str(metadata["label"]),
                "model_id": metadata.get("model_id"),
                "prediction_column": column,
                "evaluation_mode": metadata.get("evaluation_mode"),
                "units": metadata.get("prediction_units"),
                "timestamp": timestamps,
                "value": values.tolist(),
                "source_path": str(path),
            }
        )
        summary_rows.append(
            {
                "model_id": metadata.get("model_id"),
                "prediction_column": column,
                "evaluation_mode": metadata.get("evaluation_mode"),
                "units": metadata.get("prediction_units"),
                "row_count": int(len(values)),
                "available_count": int(values.notna().sum()),
                "unavailable_count": int(values.isna().sum()),
                "minimum": float(finite.min()) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
                "mean": float(finite.mean()) if len(finite) else None,
            }
        )
    return series, summary_rows


def validation_overview(run_ref: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact validator rows; diagnostics remain lazy until a stage is selected."""
    specs = {
        "C2": ("feature_validation_results.csv", "validation_status"),
        "C3": ("split_validation_results.csv", "status"),
        "C4": ("dataset_validation_results.csv", "status"),
        "C6": ("training_validation_results.csv", "status"),
        "C7": ("evaluation_validation_results.csv", "status"),
        "C8": ("inference_validation_results.csv", "status"),
    }
    rows: list[dict[str, Any]] = []
    for stage, (name, status_column) in specs.items():
        frame = _stage_csv(run_ref, stage, name)
        if frame.empty:
            rows.append(
                {
                    "stage": stage,
                    "row_count": 0,
                    "passed_count": 0,
                    "failed_count": 0,
                    "status": "not-recorded",
                }
            )
            continue
        values = frame.get(status_column, pd.Series(dtype=str)).astype(str).str.casefold()
        passed = int(values.isin({"passed", "completed"}).sum())
        failed = int(len(frame) - passed)
        rows.append(
            {
                "stage": stage,
                "row_count": int(len(frame)),
                "passed_count": passed,
                "failed_count": failed,
                "status": "passed" if failed == 0 else "failed",
            }
        )
    return rows


def validation_diagnostics(
    run_ref: dict[str, Any],
    stage: str,
    *,
    limit: int = 500,
    **filters: Any,
) -> list[dict[str, Any]]:
    names = {
        "C3": "split_validation_diagnostics.csv",
        "C4": "dataset_validation_diagnostics.csv",
        "C6": "training_validation_diagnostics.csv",
        "C7": "evaluation_validation_diagnostics.csv",
        "C8": "inference_validation_diagnostics.csv",
    }
    name = names.get(str(stage))
    if not name:
        return []
    frame = _stage_csv(run_ref, str(stage), name)
    if frame.empty:
        return []
    frame = filter_frame(frame, **filters)
    if len(frame) > limit:
        frame = frame.head(limit)
    return _clean_records(frame)


def structural_availability_rows(run_ref: dict[str, Any]) -> list[dict[str, Any]]:
    frame = audit_zone_catalog(run_ref)
    if frame.empty:
        return []
    columns = [
        column
        for column in (
            "building_type",
            "weather_location",
            "climate_zone",
            "case_id",
            "aggregation_id",
            "weight_mode",
            "aggregate_zone_id",
            "candidate_model_count",
            "applicable_model_count",
            "structurally_inapplicable_model_count",
            "invalid_model_count",
            "missing_expected_data_model_count",
            "status",
        )
        if column in frame.columns
    ]
    return _clean_records(frame[columns])


def _selection_manifest(
    *,
    run_ref: dict[str, Any],
    export_kind: str,
    selection: dict[str, Any],
    source_paths: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_kind": export_kind,
        "campaign_id": run_ref["campaign_id"],
        "phase_c_run_id": run_ref["phase_c_run_id"],
        "selection": selection,
        "source_campaign_manifest": str(run_ref["manifest_path"]),
        "source_paths": sorted({str(path) for path in source_paths if str(path)}),
        "contract": "selected_equals_displayed_equals_exported",
    }


def _zip_payload(files: dict[str, bytes | str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def build_campaign_summary_export(run_ref: dict[str, Any]) -> tuple[bytes, str]:
    summary = run_summary(run_ref)
    selection = {"scope": "selected_phase_c_run"}
    provenance = _selection_manifest(
        run_ref=run_ref,
        export_kind="campaign_summary",
        selection=selection,
        source_paths=[str(run_ref["manifest_path"])],
    )
    files = {
        "campaign_summary.json": json.dumps(summary, indent=2, sort_keys=True),
        "stage_summary.csv": pd.DataFrame(stage_summary(run_ref)).to_csv(index=False),
        "structural_availability.csv": pd.DataFrame(
            structural_availability_rows(run_ref)
        ).to_csv(index=False),
        "validation_overview.csv": pd.DataFrame(validation_overview(run_ref)).to_csv(
            index=False
        ),
        "provenance_manifest.json": json.dumps(provenance, indent=2, sort_keys=True),
    }
    filename = f"{run_ref['phase_c_run_id']}__phase_c_summary.zip"
    return _zip_payload(files), filename


def build_evaluation_export(
    run_ref: dict[str, Any],
    *,
    selection: dict[str, Any],
) -> tuple[bytes, str]:
    filters = dict(selection.get("filters") or {})
    splits = selection.get("splits")
    modes = selection.get("evaluation_modes")
    metrics = load_evaluation_metrics(
        run_ref,
        splits=splits,
        evaluation_modes_selected=modes,
        **filters,
    )
    models = load_model_metadata(run_ref, **filters)
    series = load_evaluation_series(
        run_ref,
        splits=splits,
        evaluation_modes_selected=modes,
        resolution=str(selection.get("resolution") or "preview"),
        start=selection.get("start"),
        end=selection.get("end"),
        **filters,
    )
    long_rows: list[dict[str, Any]] = []
    for item in series:
        for timestamp, value in zip(item["timestamp"], item["value"]):
            long_rows.append(
                {
                    "series": item["name"],
                    "identity": item["identity"],
                    "role": item["role"],
                    "timestamp": timestamp,
                    "value": value,
                }
            )
    source_paths = [item.get("source_path", "") for item in series]
    provenance = _selection_manifest(
        run_ref=run_ref,
        export_kind="evaluation_selection",
        selection=selection,
        source_paths=source_paths,
    )
    files = {
        "evaluation_metrics.csv": pd.DataFrame(metrics).to_csv(index=False),
        "model_metadata.csv": pd.DataFrame(models).to_csv(index=False),
        "evaluation_series.csv": pd.DataFrame(long_rows).to_csv(index=False),
        "provenance_manifest.json": json.dumps(provenance, indent=2, sort_keys=True),
    }
    filename = f"{run_ref['phase_c_run_id']}__phase_c_evaluation_selection.zip"
    return _zip_payload(files), filename


def build_annual_export(
    run_ref: dict[str, Any],
    *,
    selection: dict[str, Any],
) -> tuple[bytes, str]:
    series, summary = load_annual_series(
        run_ref,
        selected_zone_key=str(selection.get("zone_key") or ""),
        prediction_columns=selection.get("prediction_columns"),
        resolution=str(selection.get("resolution") or "full"),
        start=selection.get("start"),
        end=selection.get("end"),
    )
    long_rows: list[dict[str, Any]] = []
    for item in series:
        for timestamp, value in zip(item["timestamp"], item["value"]):
            long_rows.append(
                {
                    "series": item["name"],
                    "model_id": item.get("model_id"),
                    "prediction_column": item.get("prediction_column"),
                    "evaluation_mode": item.get("evaluation_mode"),
                    "units": item.get("units"),
                    "timestamp": timestamp,
                    "value": value,
                }
            )
    source_paths = [item.get("source_path", "") for item in series]
    provenance = _selection_manifest(
        run_ref=run_ref,
        export_kind="annual_inference_selection",
        selection=selection,
        source_paths=source_paths,
    )
    files = {
        "annual_inference_series.csv": pd.DataFrame(long_rows).to_csv(index=False),
        "annual_inference_summary.csv": pd.DataFrame(summary).to_csv(index=False),
        "provenance_manifest.json": json.dumps(provenance, indent=2, sort_keys=True),
    }
    filename = f"{run_ref['phase_c_run_id']}__phase_c_annual_selection.zip"
    return _zip_payload(files), filename

# ---------------------------------------------------------------------------
# Extended Tab-3 inventories, diagnostics and provenance exports.
# ---------------------------------------------------------------------------


def lineage_summary(run_ref: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the saved Phase C definition and upstream lineage when available."""
    summary = run_summary(run_ref)
    generated_root = resolve_generated_data_root()
    definition_root = generated_root / "campaign_definitions" / "heat_input"
    matches: list[dict[str, Any]] = []
    if definition_root.is_dir():
        for path in sorted(definition_root.glob("*.json")):
            payload = _read_json(path)
            config = payload.get("runner_config") or {}
            if str(config.get("campaign_id") or "") != run_ref["campaign_id"]:
                continue
            if str(config.get("matrix_run_id") or "") != summary.get("matrix_run_id"):
                continue
            matches.append(
                {
                    "phase_c_campaign_id": payload.get("phase_c_campaign_id"),
                    "parent_aggregation_campaign_id": payload.get(
                        "parent_aggregation_campaign_id"
                    ),
                    "parent_generation_campaign_id": payload.get(
                        "parent_generation_campaign_id"
                    ),
                    "machine_id": payload.get("machine_id"),
                    "definition_path": str(path),
                }
            )
    if not matches:
        matches = [
            {
                "phase_c_campaign_id": "",
                "parent_aggregation_campaign_id": "",
                "parent_generation_campaign_id": run_ref["campaign_id"],
                "machine_id": "",
                "definition_path": "",
            }
        ]
    datasets = dataset_catalog(run_ref)
    generation_run_ids = (
        sorted(set(datasets["source_generation_run_id"].dropna().astype(str)))
        if "source_generation_run_id" in datasets.columns
        else []
    )
    aggregation_run_ids = (
        sorted(set(datasets["aggregation_run_id"].dropna().astype(str)))
        if "aggregation_run_id" in datasets.columns
        else []
    )
    rows = []
    for match in matches:
        rows.append(
            {
                "phase_c_run_id": run_ref["phase_c_run_id"],
                "phase_c_campaign_id": match.get("phase_c_campaign_id") or "",
                "parent_aggregation_campaign_id": match.get(
                    "parent_aggregation_campaign_id"
                )
                or "",
                "matrix_run_id": summary.get("matrix_run_id") or "",
                "parent_generation_campaign_id": match.get(
                    "parent_generation_campaign_id"
                )
                or run_ref["campaign_id"],
                "machine_id": match.get("machine_id") or "",
                "source_generation_run_ids": ", ".join(generation_run_ids),
                "aggregation_run_ids": ", ".join(aggregation_run_ids),
                "definition_path": match.get("definition_path") or "",
            }
        )
    return rows


def dataset_inventory(run_ref: dict[str, Any], **filters: Any) -> list[dict[str, Any]]:
    """Return compact C4 dataset rows without opening model Parquets."""
    frame = filter_frame(dataset_catalog(run_ref), **filters)
    columns = [
        column
        for column in (
            "building_type",
            "weather_location",
            "climate_zone",
            "case_id",
            "aggregation_id",
            "strategy",
            "rule_set",
            "weight_mode",
            "aggregate_zone_id",
            "model_id",
            "status",
            "predictor_column",
            "target_column",
            "source_row_count",
            "valid_pair_count",
            "invalid_pair_count",
            "train_row_count",
            "validation_row_count",
            "test_row_count",
            "output_root",
        )
        if column in frame.columns
    ]
    return _clean_records(frame[columns]) if columns else []


def target_model_inventory(run_ref: dict[str, Any], **filters: Any) -> list[dict[str, Any]]:
    """Join the authoritative 19-model registry to C1/C4 availability."""
    from scalebridge.models.heat_input_regression.registry import list_model_specifications

    datasets = filter_frame(dataset_catalog(run_ref), **filters)
    audits = filter_frame(audit_zone_catalog(run_ref), **filters)
    selected_models = set(_as_values(filters.get("model_ids")))
    rows: list[dict[str, Any]] = []
    for spec in list_model_specifications():
        if selected_models and spec.model_id not in selected_models:
            continue
        model_rows = datasets[datasets["model_id"].astype(str) == spec.model_id]
        status = "applicable" if not model_rows.empty else "not-materialized"
        reason = ""
        if model_rows.empty and not audits.empty:
            status = "structural-or-unavailable"
            reason = "No C4 dataset row for the currently filtered zone scope."
        rows.append(
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "source_family": spec.source_family,
                "component": spec.component,
                "predictor_kind": spec.predictor_kind,
                "target_semantic": spec.target_semantic_name,
                "prediction_column": spec.output_prediction_column,
                "target_units": spec.expected_target_units,
                "availability_status": status,
                "reason": reason,
                "materialized_dataset_count": int(len(model_rows)),
            }
        )
    return rows


def split_summary_rows(
    run_ref: dict[str, Any],
    *,
    max_detail_zones: int = 24,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Return C3 split coverage, lazily resolving detailed split summaries."""
    frame = _stage_csv(run_ref, "C3", "split_zone_results.csv")
    if frame.empty:
        return []
    frame = filter_frame(frame, **filters)
    rows = _clean_records(frame)
    stage_root = stage_roots(run_ref).get("C3")
    load_detail = stage_root is not None and len(rows) <= max_detail_zones
    outputs: list[dict[str, Any]] = []

    for row in rows:
        detailed: dict[str, dict[str, Any]] = {}
        if load_detail:
            zone_root = _localize_artifact_path(
                row.get("output_root") or "",
                stage_root=stage_root,
            )
            detail_frame = _read_csv(zone_root / "split_summary.csv")
            if not detail_frame.empty and "split" in detail_frame.columns:
                detailed = {
                    str(item.get("split")): item
                    for item in _clean_records(detail_frame)
                }

        included_total = sum(
            int(row.get(field) or 0)
            for field in (
                "train_row_count",
                "validation_row_count",
                "test_row_count",
            )
        )
        assignment_total = int(row.get("assignment_row_count") or 0)
        for split, count_field in (
            ("train", "train_row_count"),
            ("validation", "validation_row_count"),
            ("test", "test_row_count"),
            ("excluded", "excluded_row_count"),
        ):
            count = int(row.get(count_field) or 0)
            detail = detailed.get(split) or {}
            fraction = detail.get("fraction_of_included")
            if fraction is None:
                denominator = included_total if split != "excluded" else assignment_total
                fraction = count / denominator if denominator else None
            outputs.append(
                {
                    "building_type": row.get("building_type"),
                    "weather_location": row.get("weather_location"),
                    "case_id": row.get("case_id"),
                    "aggregation_id": row.get("aggregation_id"),
                    "weight_mode": row.get("weight_mode"),
                    "aggregate_zone_id": row.get("aggregate_zone_id"),
                    "split": split,
                    "row_count": int(detail.get("row_count") or count),
                    "fraction_of_included": fraction,
                    "first_timestamp": detail.get("first_timestamp") or "",
                    "last_timestamp": detail.get("last_timestamp") or "",
                    "month_count": detail.get("month_count"),
                    "day_count": detail.get("day_count"),
                    "output_root": row.get("output_root"),
                    "detail_loaded": bool(detail),
                }
            )
    return outputs


def generalization_metrics(
    run_ref: dict[str, Any],
    *,
    evaluation_modes_selected: Any = None,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Pivot comparable C7 split metrics into train/validation/test deltas."""
    metrics = pd.DataFrame(
        load_evaluation_metrics(
            run_ref,
            splits=["train", "validation", "test"],
            evaluation_modes_selected=evaluation_modes_selected,
            **filters,
        )
    )
    if metrics.empty:
        return []
    metric_names = [
        name
        for name in ("rmse", "mae", "r2", "mean_bias_error", "nrmse_by_range")
        if name in metrics.columns
    ]
    identity = [
        name
        for name in (
            "building_type",
            "weather_location",
            "case_id",
            "aggregation_id",
            "weight_mode",
            "aggregate_zone_id",
            "model_id",
            "estimator_type",
            "evaluation_mode",
        )
        if name in metrics.columns
    ]
    outputs: list[dict[str, Any]] = []
    for _, group in metrics.groupby(identity, dropna=False):
        base = {name: group.iloc[0].get(name) for name in identity}
        for metric in metric_names:
            values = {
                str(row.get("split")): row.get(metric)
                for row in _clean_records(group[["split", metric]])
            }
            train = values.get("train")
            validation = values.get("validation")
            test = values.get("test")
            outputs.append(
                {
                    **base,
                    "metric": metric,
                    "train": train,
                    "validation": validation,
                    "test": test,
                    "validation_minus_train": (
                        float(validation) - float(train)
                        if validation is not None and train is not None
                        else None
                    ),
                    "test_minus_validation": (
                        float(test) - float(validation)
                        if test is not None and validation is not None
                        else None
                    ),
                }
            )
    return outputs


def artifact_inventory(run_ref: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only stage-level compact artifacts; never recursively scan model trees."""
    rows: list[dict[str, Any]] = []
    known_files = {
        "C1": ["heat_input_regression_audit_manifest.json", "audit_zone_results.csv"],
        "C2": ["heat_input_feature_run_manifest.json", "feature_zone_results.csv"],
        "C3": ["split_run_manifest.json", "split_zone_results.csv"],
        "C4": ["dataset_run_manifest.json", "dataset_model_results.csv"],
        "C5": ["c5_model_api_validation_manifest.json"],
        "C6": ["training_run_manifest.json", "training_results.csv"],
        "C7": ["evaluation_run_manifest.json", "evaluation_results.csv"],
        "C8": ["inference_run_manifest.json", "inference_results.csv"],
        "C9": ["phase_c_mlflow_registration_manifest.json"],
    }
    roots = stage_roots(run_ref)
    for stage, names in known_files.items():
        root = roots.get(stage)
        if root is None:
            continue
        for name in names:
            path = root / name
            rows.append(
                {
                    "stage": stage,
                    "type": "manifest" if path.suffix == ".json" else "index",
                    "path": str(path),
                    "exists": path.is_file(),
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                    "producer": _STAGE_LAYOUT[stage][1],
                    "downstream_consumer": "Dash Results / downstream Phase D as applicable",
                }
            )
    return rows



def _dataset_signal_columns(
    row: dict[str, Any],
    frame_columns: Iterable[str],
) -> tuple[list[str], str]:
    """Resolve C4 predictor/target columns without assuming generic x/y names."""
    columns = [str(column) for column in frame_columns]
    column_set = set(columns)
    target = str(row.get("target_column") or "").strip()
    if target not in column_set and "y" in column_set:
        target = "y"

    predictor_raw = row.get("predictor_column")
    predictor_candidates: list[str] = []
    if isinstance(predictor_raw, (list, tuple)):
        predictor_candidates = [str(value).strip() for value in predictor_raw]
    else:
        text = str(predictor_raw or "").strip()
        if text:
            if text.startswith("["):
                try:
                    payload = json.loads(text)
                except Exception:
                    payload = None
                if isinstance(payload, list):
                    predictor_candidates = [str(value).strip() for value in payload]
            if not predictor_candidates:
                predictor_candidates = [
                    part.strip()
                    for part in re.split(r"\s*[;,|]\s*", text)
                    if part.strip()
                ]
    predictors = [name for name in predictor_candidates if name in column_set]
    if not predictors and "x" in column_set:
        predictors = ["x"]
    if not predictors:
        meta = {
            "timestamp",
            "timestamp_raw",
            "datetime",
            "date_time",
            "split",
            "row_index",
            "step_index",
            target,
        }
        predictors = [column for column in columns if column not in meta]
    return predictors, target


def load_dataset_series(
    run_ref: dict[str, Any],
    *,
    resolution: str = "preview",
    start: str | None = None,
    end: str | None = None,
    max_datasets: int = 1,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Load X/Y trajectories for exactly one selected C4 model dataset.

    The compact preview CSV is preferred for interactive inspection. Full resolution
    reads only timestamp, selected predictor, and target columns from the C4 parquet.
    """
    selected = filter_frame(dataset_catalog(run_ref), **filters)
    if "status" in selected.columns:
        selected = selected[selected["status"].astype(str) == "completed"]
    if len(selected) != max_datasets:
        raise ResultSelectionTooBroad(
            f"Dataset trajectory requires exactly {max_datasets} selected C4 dataset; "
            f"current filters resolve to {len(selected)}."
        )
    row = _clean_records(selected)[0]
    root = stage_roots(run_ref).get("C4")
    if root is None:
        return []
    model_root = _localize_artifact_path(row.get("output_root") or "", stage_root=root)
    path = model_root / (
        "regression_pairs_full.parquet"
        if str(resolution).casefold() == "full"
        else "regression_pairs_preview.csv"
    )
    if not path.is_file() and str(resolution).casefold() == "preview":
        fallback = model_root / "regression_pairs_full.parquet"
        if fallback.is_file():
            path = fallback
    if not path.is_file():
        raise FileNotFoundError(f"C4 regression-pair table unavailable: {path}")

    if path.suffix.casefold() == ".parquet":
        try:
            import pyarrow.parquet as pq

            available = pq.ParquetFile(path).schema.names
        except Exception:
            available = []
        predictors, target = _dataset_signal_columns(row, available)
        timestamp_columns = [
            column
            for column in ("timestamp", "timestamp_raw", "datetime", "date_time")
            if column in available
        ]
        requested = list(dict.fromkeys([*timestamp_columns, *predictors, target]))
        requested = [column for column in requested if column]
        frame = pd.read_parquet(path, columns=requested or None)
    else:
        frame = pd.read_csv(path)
        predictors, target = _dataset_signal_columns(row, frame.columns)

    if "timestamp" not in frame.columns:
        for column in ("timestamp_raw", "datetime", "date_time"):
            if column in frame.columns:
                frame = frame.copy()
                frame["timestamp"] = pd.to_datetime(frame[column], errors="coerce")
                break
    frame = _filter_time(frame, start, end)
    if frame.empty:
        return []
    if "timestamp" not in frame.columns:
        raise KeyError(f"C4 regression-pair table has no timestamp column: {path}")

    identity = (
        f"{row.get('building_type') or row.get('case_id')} | "
        f"{row.get('aggregation_id')} | {row.get('aggregate_zone_id')} | "
        f"{row.get('model_id')}"
    )
    timestamps = frame["timestamp"].astype(str).tolist()
    series: list[dict[str, Any]] = []
    for predictor in predictors:
        if predictor not in frame.columns:
            continue
        series.append(
            {
                "name": f"{identity} | X: {predictor}",
                "identity": identity,
                "model_id": row.get("model_id"),
                "role": "predictor",
                "signal": predictor,
                "timestamp": timestamps,
                "value": pd.to_numeric(frame[predictor], errors="coerce").tolist(),
                "source_path": str(path),
            }
        )
    if target and target in frame.columns:
        series.append(
            {
                "name": f"{identity} | Y: {target}",
                "identity": identity,
                "model_id": row.get("model_id"),
                "role": "target",
                "signal": target,
                "timestamp": timestamps,
                "value": pd.to_numeric(frame[target], errors="coerce").tolist(),
                "source_path": str(path),
            }
        )
    return series


def load_dataset_preview(
    run_ref: dict[str, Any],
    *,
    max_datasets: int = 1,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Load one C4 regression-pair preview after explicit narrowing."""
    selected = filter_frame(dataset_catalog(run_ref), **filters)
    if "status" in selected.columns:
        selected = selected[selected["status"].astype(str) == "completed"]
    if len(selected) != max_datasets:
        raise ResultSelectionTooBroad(
            f"Dataset preview requires exactly {max_datasets} selected C4 dataset; "
            f"current filters resolve to {len(selected)}."
        )
    row = _clean_records(selected)[0]
    root = stage_roots(run_ref).get("C4")
    if root is None:
        return []
    model_root = _localize_artifact_path(row.get("output_root") or "", stage_root=root)
    path = model_root / "regression_pairs_preview.csv"
    frame = _read_csv(path)
    return _clean_records(frame.head(250))


def load_building_phvac_series(
    run_ref: dict[str, Any],
    *,
    split: str = "test",
    evaluation_modes_selected: Any = None,
    max_groups: int = 4,
    **filters: Any,
) -> list[dict[str, Any]]:
    """Load bounded building-level PHVAC reconstruction series after filters."""
    metrics = pd.DataFrame(
        building_phvac_metrics(
            run_ref,
            splits=[split],
            evaluation_modes_selected=evaluation_modes_selected,
            **filters,
        )
    )
    if metrics.empty:
        return []
    group_cols = [
        c
        for c in (
            "case_id",
            "aggregation_id",
            "weight_mode",
            "estimator_type",
            "requested_device",
            "split",
        )
        if c in metrics.columns
    ]
    unique = metrics[group_cols + ["predictions_path"]].drop_duplicates()
    if len(unique) > max_groups:
        raise ResultSelectionTooBroad(
            f"Building PHVAC plot resolves to {len(unique)} groups; narrow to at most "
            f"{max_groups}."
        )
    root = stage_roots(run_ref).get("C7")
    if root is None:
        return []
    series: list[dict[str, Any]] = []
    modes = set(_as_values(evaluation_modes_selected)) or {"oracle", "chained"}
    for row in _clean_records(unique):
        path = _localize_artifact_path(row.get("predictions_path") or "", stage_root=root)
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        timestamp = frame["timestamp"].astype(str).tolist()
        identity = " | ".join(str(row.get(c) or "") for c in group_cols)
        series.append(
            {
                "name": f"{identity} | building target",
                "identity": identity,
                "role": "target",
                "timestamp": timestamp,
                "value": pd.to_numeric(frame["building_target"], errors="coerce").tolist(),
                "source_path": str(path),
            }
        )
        for mode, column in (
            ("oracle", "building_prediction_oracle"),
            ("chained", "building_prediction_chained"),
        ):
            if mode not in modes or column not in frame.columns:
                continue
            series.append(
                {
                    "name": f"{identity} | {mode}",
                    "identity": identity,
                    "role": mode,
                    "timestamp": timestamp,
                    "value": pd.to_numeric(frame[column], errors="coerce").tolist(),
                    "source_path": str(path),
                }
            )
    return series


def build_model_bundle_export(
    run_ref: dict[str, Any],
    *,
    selection: dict[str, Any],
) -> tuple[bytes, str]:
    """Export exactly one selected C6 persisted model with lineage metadata."""
    filters = dict(selection.get("filters") or {})
    frame = filter_frame(training_catalog(run_ref), **filters)
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str) == "completed"]
    if len(frame) != 1:
        raise ResultSelectionTooBroad(
            "Model bundle export requires exactly one completed trained artifact; "
            f"current selection resolves to {len(frame)}."
        )
    row = _clean_records(frame)[0]
    root = stage_roots(run_ref).get("C6")
    if root is None:
        raise FileNotFoundError("C6 stage root is unavailable")
    artifact_dir = _localize_artifact_path(row.get("artifact_dir") or "", stage_root=root)
    model_manifest_path = artifact_dir / "model_manifest.json"
    training_manifest = artifact_dir.parent / "training_manifest.json"
    dataset_manifest_path = Path()
    dataset_rows = filter_frame(dataset_catalog(run_ref), **filters)
    if "model_id" in dataset_rows.columns:
        dataset_rows = dataset_rows[
            dataset_rows["model_id"].astype(str) == str(row.get("model_id") or "")
        ]
    if len(dataset_rows) == 1:
        c4_root = stage_roots(run_ref).get("C4")
        if c4_root is not None:
            dataset_output = _localize_artifact_path(
                _clean_records(dataset_rows)[0].get("output_root") or "",
                stage_root=c4_root,
            )
            candidate = dataset_output / "model_dataset_manifest.json"
            if candidate.is_file():
                dataset_manifest_path = candidate
    files: dict[str, bytes | str] = {}
    if artifact_dir.is_dir():
        for path in sorted(p for p in artifact_dir.iterdir() if p.is_file()):
            files[f"model_artifact/{path.name}"] = path.read_bytes()
    if training_manifest.is_file():
        files["training_manifest.json"] = training_manifest.read_bytes()
    if dataset_manifest_path.is_file():
        files["source_model_dataset_manifest.json"] = dataset_manifest_path.read_bytes()
    if model_manifest_path.is_file() and "model_artifact/model_manifest.json" not in files:
        files["model_artifact/model_manifest.json"] = model_manifest_path.read_bytes()
    files["selected_model_metadata.json"] = json.dumps(row, indent=2, default=str)
    files["selection_manifest.json"] = json.dumps(
        _selection_manifest(
            run_ref=run_ref,
            export_kind="selected_model_bundle",
            selection=selection,
            source_paths=[
                str(path)
                for path in (training_manifest, model_manifest_path, dataset_manifest_path)
                if str(path) and path.is_file()
            ],
        ),
        indent=2,
    )
    filename = (
        f"phase_c_model_{row.get('model_id')}_{row.get('estimator_type')}_"
        f"{run_ref['phase_c_run_id']}.zip"
    )
    return _zip_payload(files), filename


def build_plot_figure_export(
    figure: dict[str, Any],
    *,
    file_format: str,
    plot_key: str,
    run_id: str,
    run_ref: dict[str, Any] | None = None,
) -> tuple[bytes, str]:
    """Build a self-describing ZIP for the exact currently visible Plotly traces.

    The data file inside the ZIP is CSV or Parquet according to ``file_format``.
    Export rows are figure-backed, so legend-hidden traces are excluded and the
    downloaded artifact represents the graph snapshot the user is actually seeing.
    """
    normalized_format = str(file_format or "csv").lower().strip()
    if normalized_format not in {"csv", "parquet"}:
        raise ValueError("Plot data format must be 'csv' or 'parquet'.")

    traces = list((figure or {}).get("data") or [])
    rows: list[dict[str, Any]] = []
    trace_manifest: list[dict[str, Any]] = []
    visible_trace_names: list[str] = []
    hidden_trace_names: list[str] = []

    for trace_index, trace in enumerate(traces):
        visible_state = trace.get("visible", True)
        is_visible = not (
            visible_state is False or str(visible_state).lower() == "legendonly"
        )
        name = str(trace.get("name") or f"trace-{trace_index}")
        trace_type = str(trace.get("type") or "scatter")
        x_values = list(trace.get("x") or [])
        y_values = list(trace.get("y") or [])
        point_count = max(len(x_values), len(y_values))

        trace_manifest.append(
            {
                "trace_index": trace_index,
                "trace_name": name,
                "trace_type": trace_type,
                "visible": bool(is_visible),
                "point_count": int(point_count),
            }
        )

        if not is_visible:
            hidden_trace_names.append(name)
            continue

        visible_trace_names.append(name)
        if point_count == 0:
            continue

        for point_index in range(point_count):
            rows.append(
                {
                    "trace_index": trace_index,
                    "trace_name": name,
                    "trace_type": trace_type,
                    "point_index": point_index,
                    "x": x_values[point_index] if point_index < len(x_values) else None,
                    "y": y_values[point_index] if point_index < len(y_values) else None,
                }
            )

    if not visible_trace_names:
        raise ValueError("No visible plot traces are available to download.")
    if not rows:
        raise ValueError("The visible plot does not contain exportable x/y data.")

    frame = pd.DataFrame(rows)
    safe_plot_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(plot_key or "plot"))
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "phase_c"))

    layout = dict((figure or {}).get("layout") or {})
    xaxis = dict(layout.get("xaxis") or {})
    yaxis = dict(layout.get("yaxis") or {})
    plot_meta = dict(layout.get("meta") or {})
    snapshot = dict(plot_meta.get("phase_c_plot_export") or {})

    data_filename = (
        "data/plotted_data.csv"
        if normalized_format == "csv"
        else "data/plotted_data.parquet"
    )

    archive_files: dict[str, bytes | str] = {}
    if normalized_format == "csv":
        archive_files[data_filename] = frame.to_csv(index=False)
    else:
        parquet_buffer = BytesIO()
        frame.to_parquet(parquet_buffer, index=False)
        archive_files[data_filename] = parquet_buffer.getvalue()

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "export_type": "bgirs_phase_c_visible_plot_data",
        "phase": "C",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plot_key": str(plot_key or "plot"),
        "selected_format": normalized_format,
        "data_file": data_filename,
        "row_count": int(len(frame)),
        "columns": list(frame.columns),
        "phase_c_run_id": str(
            (run_ref or {}).get("phase_c_run_id") or run_id or "phase_c"
        ),
        "campaign_id": str((run_ref or {}).get("campaign_id") or ""),
        "source_campaign_manifest": str((run_ref or {}).get("manifest_path") or ""),
        "plot_snapshot": snapshot,
        "plot_axes": {
            "x_title": ((xaxis.get("title") or {}).get("text")),
            "y_title": ((yaxis.get("title") or {}).get("text")),
            "x_range": xaxis.get("range"),
            "y_range": yaxis.get("range"),
        },
        "all_traces": trace_manifest,
        "visible_trace_names": visible_trace_names,
        "hidden_trace_names": hidden_trace_names,
        "contract": "visible_plot_snapshot_equals_exported_data",
    }

    archive_files["selection_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    readme_lines = [
        "BGIRS Phase C — Plotted Data Export",
        "",
        f"Plot: {plot_key}",
        f"Phase C run: {manifest['phase_c_run_id']}",
        f"Campaign: {manifest['campaign_id'] or 'not recorded'}",
        f"Data format: {normalized_format.upper()}",
        f"Data file: {data_filename}",
        f"Rows: {len(frame)}",
        "",
        "This ZIP contains the data currently visible in the plotted graph.",
        "Traces hidden through the custom legend are excluded from plotted_data.",
        "selection_manifest.json records the plot snapshot, trace visibility,",
        "filter/selection context captured when the figure was built, and run lineage.",
        "",
        "Contract: visible plot snapshot = exported data.",
    ]
    archive_files["README.txt"] = "\n".join(readme_lines) + "\n"

    filename = f"{safe_run_id}__{safe_plot_key}__visible_plot_data_{normalized_format}.zip"
    return _zip_payload(archive_files), filename
