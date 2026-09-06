from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import EXPECTED_PARTITIONS, EXPECTED_ROWS, CaseSpec, PaperConfig, SpatialCase, canonical_case_specs

META_COLUMNS = ("timestamp", "included", "partition", "window_id", "season")


@dataclass
class PhaseDTrajectory:
    case_name: str
    zone_ids: tuple[str, ...]
    dependency_mode: str
    timestamp: pd.Series
    included: np.ndarray
    partition: np.ndarray
    state: np.ndarray
    control: np.ndarray
    disturbance: np.ndarray
    target: np.ndarray
    state_columns: tuple[str, ...]
    control_columns: tuple[str, ...]
    disturbance_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    manifests: tuple[dict[str, Any], ...]
    frame: pd.DataFrame

    def mask(self, partition: str, included_only: bool = True) -> np.ndarray:
        mask = self.partition == partition
        if included_only:
            mask &= self.included
        return mask

    def split(self, partition: str) -> "PhaseDTrajectory":
        mask = self.mask(partition)
        f = self.frame.loc[mask].reset_index(drop=True)
        return _trajectory_from_frame(
            case_name=self.case_name,
            zone_ids=self.zone_ids,
            dependency_mode=self.dependency_mode,
            frame=f,
            manifests=self.manifests,
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _column_groups(manifest: dict[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "state": [],
        "control_input": [],
        "disturbance": [],
        "target": [],
        "metadata": [],
    }
    for item in manifest["final_columns"]:
        role = item["physical_role"]
        if role in groups:
            groups[role].append(item["name"])
    return groups


def load_manifest_only(config: PaperConfig, case_name: SpatialCase) -> tuple[CaseSpec, tuple[dict[str, Any], ...]]:
    spec = canonical_case_specs()[case_name]
    manifests = []
    for rel in spec.phase_d_paths:
        manifest_path = config.phase_d_case_root / rel / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing Phase D manifest: {manifest_path}")
        manifests.append(_read_json(manifest_path))
    validate_manifests(spec, tuple(manifests))
    return spec, tuple(manifests)


def validate_manifests(spec: CaseSpec, manifests: tuple[dict[str, Any], ...]) -> None:
    for manifest in manifests:
        if manifest.get("row_count") != EXPECTED_ROWS:
            raise ValueError(
                f"{spec.name}: expected {EXPECTED_ROWS} rows, found {manifest.get('row_count')}"
            )
        if manifest.get("partition_counts") != EXPECTED_PARTITIONS:
            raise ValueError(
                f"{spec.name}: unexpected partition counts {manifest.get('partition_counts')}"
            )
        tc = manifest.get("temporal_config", {})
        if tc.get("input_lag") != 1 or tc.get("target_horizon") != 1:
            raise ValueError(f"{spec.name}: Patch 1 requires l1_h1 data")
        if tc.get("policy_name") != "monthly_distributed_holdout":
            raise ValueError(f"{spec.name}: unexpected temporal policy {tc.get('policy_name')}")
        if manifest.get("mode") != spec.dependency_mode:
            raise ValueError(
                f"{spec.name}: expected mode {spec.dependency_mode}, got {manifest.get('mode')}"
            )


def _load_single(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = _read_json(path / "manifest.json")
    parquet = path / "data.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Missing Phase D parquet: {parquet}")
    try:
        frame = pd.read_parquet(parquet)
    except ImportError as exc:
        raise RuntimeError(
            "Reading Phase D data requires PyArrow/FastParquet. "
            "Run this inside the ScaleBridge development environment."
        ) from exc
    return frame, manifest


def _merge_independent(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) != 2:
        raise ValueError("identity_ind currently expects exactly Dining and Kitchen frames")
    left, right = frames
    for col in META_COLUMNS:
        if not left[col].equals(right[col]):
            raise ValueError(f"identity_ind metadata mismatch between zones: {col}")
    duplicate_meta = [c for c in META_COLUMNS if c in right.columns]
    duplicate_common = [c for c in right.columns if c == "outdoor_temperature__lag_0"]
    if "outdoor_temperature__lag_0" in left.columns and duplicate_common:
        if not left["outdoor_temperature__lag_0"].equals(right["outdoor_temperature__lag_0"]):
            raise ValueError("identity_ind outdoor temperature mismatch")
    right_payload = right.drop(columns=duplicate_meta + duplicate_common)
    return pd.concat([left.reset_index(drop=True), right_payload.reset_index(drop=True)], axis=1)


def load_case(config: PaperConfig, case_name: SpatialCase) -> PhaseDTrajectory:
    spec, _ = load_manifest_only(config, case_name)
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for rel in spec.phase_d_paths:
        frame, manifest = _load_single(config.phase_d_case_root / rel)
        frames.append(frame)
        manifests.append(manifest)

    if case_name == "identity_ind":
        frame = _merge_independent(frames)
    else:
        frame = frames[0]

    return _trajectory_from_frame(
        case_name=case_name,
        zone_ids=spec.zone_ids,
        dependency_mode=spec.dependency_mode,
        frame=frame,
        manifests=tuple(manifests),
    )


def _trajectory_from_frame(
    *,
    case_name: str,
    zone_ids: tuple[str, ...],
    dependency_mode: str,
    frame: pd.DataFrame,
    manifests: tuple[dict[str, Any], ...],
) -> PhaseDTrajectory:
    state_cols: list[str] = []
    control_cols: list[str] = []
    disturbance_cols: list[str] = []
    target_cols: list[str] = []
    for manifest in manifests:
        groups = _column_groups(manifest)
        for name, bucket in (
            ("state", state_cols),
            ("control_input", control_cols),
            ("disturbance", disturbance_cols),
            ("target", target_cols),
        ):
            for col in groups[name]:
                if col not in bucket:
                    bucket.append(col)

    # Preserve authoritative Phase D column order, but avoid duplicate outdoor T
    # when merging independent-zone silos.
    for cols in (state_cols, control_cols, disturbance_cols, target_cols):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{case_name}: missing expected columns: {missing}")

    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    included = frame["included"].astype(bool).to_numpy()
    partition = frame["partition"].astype(str).to_numpy()

    return PhaseDTrajectory(
        case_name=case_name,
        zone_ids=zone_ids,
        dependency_mode=dependency_mode,
        timestamp=timestamp,
        included=included,
        partition=partition,
        state=frame[state_cols].to_numpy(dtype=float),
        control=frame[control_cols].to_numpy(dtype=float),
        disturbance=frame[disturbance_cols].to_numpy(dtype=float),
        target=frame[target_cols].to_numpy(dtype=float),
        state_columns=tuple(state_cols),
        control_columns=tuple(control_cols),
        disturbance_columns=tuple(disturbance_cols),
        target_columns=tuple(target_cols),
        manifests=manifests,
        frame=frame,
    )


def numeric_audit(trajectory: PhaseDTrajectory) -> dict[str, Any]:
    report: dict[str, Any] = {
        "case": trajectory.case_name,
        "rows": len(trajectory.frame),
        "zones": list(trajectory.zone_ids),
        "dependency_mode": trajectory.dependency_mode,
        "partition_counts": {
            key: int(np.sum(trajectory.partition == key))
            for key in ("train", "validation", "test", "excluded")
        },
        "columns": {},
        "monotonic_timestamp": bool(trajectory.timestamp.is_monotonic_increasing),
    }
    for col in (
        list(trajectory.state_columns)
        + list(trajectory.control_columns)
        + list(trajectory.disturbance_columns)
        + list(trajectory.target_columns)
    ):
        values = pd.to_numeric(trajectory.frame[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        report["columns"][col] = {
            "finite_count": int(finite.sum()),
            "nonfinite_count": int((~finite).sum()),
            "min": float(np.nanmin(values)) if finite.any() else None,
            "max": float(np.nanmax(values)) if finite.any() else None,
            "mean": float(np.nanmean(values)) if finite.any() else None,
        }
    return report
