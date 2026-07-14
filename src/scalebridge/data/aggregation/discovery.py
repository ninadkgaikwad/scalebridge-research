# -*- coding: utf-8 -*-
"""Discovery utilities for ScaleBridge aggregation inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scalebridge.data.aggregation.models import (
    GenerationRunRef,
    RddVariableIntersection,
    SUCCESS_STATUSES,
)


DEFAULT_CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c_labpc_v2"


def resolve_repo_root() -> Path:
    """Resolve repo root from this module path.

    Module path:
        src/scalebridge/data/aggregation/discovery.py

    Repo root:
        parents[4]
    """
    return Path(__file__).resolve().parents[4]


def resolve_campaign_root(
    *,
    repo_root: Path,
    campaign_id: str,
    campaign_root: str | None = None,
    generated_data_root: str | None = None,
) -> Path:
    """Resolve a ScaleBridge campaign root.

    Default:
        <repo_root>/../../Data/ScaleBridge/campaigns/<campaign_id>
    """
    if campaign_root:
        return Path(campaign_root).expanduser().resolve()

    if generated_data_root:
        root = Path(generated_data_root).expanduser().resolve()
    else:
        root = (repo_root.parents[1] / "Data" / "ScaleBridge").resolve()

    return root / "campaigns" / campaign_id


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return payload


def discover_generation_runs(
    *,
    cases_root: Path,
    case_id: str | None = None,
    include_failed: bool = False,
) -> tuple[list[GenerationRunRef], list[dict[str, Any]]]:
    """Discover latest generation runs under generation/cases."""
    missing_rows: list[dict[str, Any]] = []
    run_refs: list[GenerationRunRef] = []

    case_dirs = [cases_root / case_id] if case_id else sorted(
        path for path in cases_root.iterdir() if path.is_dir()
    )

    for case_root in case_dirs:
        current_case_id = case_root.name

        if not case_root.is_dir():
            missing_rows.append(
                {
                    "case_id": current_case_id,
                    "run_id": "",
                    "missing_file": str(case_root),
                    "reason": "case directory does not exist",
                }
            )
            continue

        latest_path = case_root / "latest_run.json"
        if not latest_path.is_file():
            missing_rows.append(
                {
                    "case_id": current_case_id,
                    "run_id": "",
                    "missing_file": str(latest_path),
                    "reason": "latest_run.json not found",
                }
            )
            continue

        latest = load_json(latest_path)
        status = str(latest.get("status", "")).strip().casefold()
        run_id = str(latest.get("run_id", "")).strip()
        manifest_rel = str(latest.get("manifest_path", "")).strip()

        if not include_failed and status not in SUCCESS_STATUSES:
            continue

        if not run_id or not manifest_rel:
            missing_rows.append(
                {
                    "case_id": current_case_id,
                    "run_id": run_id,
                    "missing_file": str(latest_path),
                    "reason": "latest_run.json missing run_id or manifest_path",
                }
            )
            continue

        manifest_path = (case_root / manifest_rel).resolve()
        if not manifest_path.is_file():
            missing_rows.append(
                {
                    "case_id": current_case_id,
                    "run_id": run_id,
                    "missing_file": str(manifest_path),
                    "reason": "run_manifest.json not found from latest_run pointer",
                }
            )
            continue

        run_refs.append(
            GenerationRunRef(
                case_id=current_case_id,
                run_id=run_id,
                status=status,
                case_root=case_root,
                run_root=manifest_path.parent,
                manifest_path=manifest_path,
            )
        )

    return run_refs, missing_rows


def load_rdd_variable_intersection(case_root: Path) -> RddVariableIntersection:
    """Load optional rdd_probe/rdd_variable_intersection.json.

    This file may not exist for older generation campaigns. Missing is valid.
    """
    path = case_root / "rdd_probe" / "rdd_variable_intersection.json"

    if not path.is_file():
        return RddVariableIntersection(
            case_id=case_root.name,
            status="missing",
            path=None,
            requested_variable_count=None,
            rdd_available_variable_count=None,
            rdd_unavailable_variable_count=None,
            available_variables=(),
            unavailable_variables=(),
        )

    payload = load_json(path)

    available = tuple(str(item) for item in payload.get("available_variables", []))
    unavailable = tuple(str(item) for item in payload.get("unavailable_variables", []))

    return RddVariableIntersection(
        case_id=str(payload.get("case_id", case_root.name)),
        status="present",
        path=path,
        requested_variable_count=_optional_int(payload.get("requested_variable_count")),
        rdd_available_variable_count=_optional_int(
            payload.get("rdd_available_variable_count")
        ),
        rdd_unavailable_variable_count=_optional_int(
            payload.get("rdd_unavailable_variable_count")
        ),
        available_variables=available,
        unavailable_variables=unavailable,
    )


def _optional_int(value: Any) -> int | None:
    """Convert a value to int or None."""
    if value is None or value == "":
        return None
    return int(value)