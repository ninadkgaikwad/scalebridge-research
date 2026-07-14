# -*- coding: utf-8 -*-
"""Canonical variable loaders for ScaleBridge aggregation.

This module reads variable-wise EnergyPlus generation outputs from:

    generation/cases/<case_id>/runs/<generation_run_id>/canonical/

Primary input:
    canonical/variable_manifest.csv
    canonical/variable_manifest.json
    canonical/variables/<variable_id>.parquet

The loader is intentionally independent of aggregation rules. It only resolves
and loads generated canonical variables so that later aggregation modules can
consume them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scalebridge.data.aggregation.discovery import load_json
from scalebridge.data.aggregation.models import GenerationRunRef


@dataclass(frozen=True)
class VariableManifestRecord:
    """One canonical generated variable artifact."""

    variable_id: str
    variable_name: str
    reporting_frequency: str
    row_count: int | None
    column_count: int | None
    raw_csv_deleted: bool | None
    canonical_parquet_path: Path
    legacy_pickle_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/CSV-friendly representation."""
        return {
            "variable_id": self.variable_id,
            "variable_name": self.variable_name,
            "reporting_frequency": self.reporting_frequency,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "raw_csv_deleted": self.raw_csv_deleted,
            "canonical_parquet_path": str(self.canonical_parquet_path),
            "legacy_pickle_path": str(self.legacy_pickle_path)
            if self.legacy_pickle_path
            else "",
        }


class CanonicalVariableLoader:
    """Load canonical long-form variable Parquet files for one generation run."""

    def __init__(self, *, run_ref: GenerationRunRef) -> None:
        self.run_ref = run_ref
        self.run_root = run_ref.run_root
        self.canonical_root = self.run_root / "canonical"
        self.variables_root = self.canonical_root / "variables"

        self.variable_manifest_csv_path = self.canonical_root / "variable_manifest.csv"
        self.variable_manifest_json_path = self.canonical_root / "variable_manifest.json"

        self._records: tuple[VariableManifestRecord, ...] | None = None

    @property
    def records(self) -> tuple[VariableManifestRecord, ...]:
        """Return cached variable manifest records."""
        if self._records is None:
            self._records = tuple(self._read_records())
        return self._records

    def list_available_variables(self) -> list[dict[str, Any]]:
        """Return available variables as dictionaries."""
        return [record.to_dict() for record in self.records]

    def get_record_by_id(self, variable_id: str) -> VariableManifestRecord:
        """Return a variable manifest record by variable_id."""
        target = variable_id.strip().casefold()
        for record in self.records:
            if record.variable_id.casefold() == target:
                return record

        available = ", ".join(record.variable_id for record in self.records)
        raise KeyError(
            f"Variable ID not found: {variable_id}. "
            f"Available variable_ids: {available}"
        )

    def get_record_by_name(self, variable_name: str) -> VariableManifestRecord:
        """Return a variable manifest record by exact variable_name."""
        target = variable_name.strip().casefold()
        for record in self.records:
            if record.variable_name.casefold() == target:
                return record

        available = ", ".join(record.variable_name for record in self.records)
        raise KeyError(
            f"Variable name not found: {variable_name}. "
            f"Available variable_names: {available}"
        )

    def load_variable_long_by_id(self, variable_id: str):
        """Load one canonical variable Parquet by variable_id.

        Returns
        -------
        pandas.DataFrame
            Expected canonical columns:
                parent_case_id
                variable_case_id
                timestamp_raw
                reporting_frequency
                key_value
                variable_name
                units
                semantic_role
                value
        """
        record = self.get_record_by_id(variable_id)
        return self._read_parquet(record.canonical_parquet_path)

    def load_variable_long_by_name(self, variable_name: str):
        """Load one canonical variable Parquet by variable_name."""
        record = self.get_record_by_name(variable_name)
        return self._read_parquet(record.canonical_parquet_path)

    def load_selected_variables(
        self,
        *,
        variable_ids: Iterable[str] | None = None,
        variable_names: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Load selected variables into a dictionary keyed by variable_id.

        Parameters
        ----------
        variable_ids:
            Variable IDs to load.
        variable_names:
            Variable names to load.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Mapping from variable_id to loaded long-form dataframe.
        """
        selected_records: list[VariableManifestRecord] = []

        if variable_ids:
            for variable_id in variable_ids:
                selected_records.append(self.get_record_by_id(variable_id))

        if variable_names:
            for variable_name in variable_names:
                selected_records.append(self.get_record_by_name(variable_name))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_records: list[VariableManifestRecord] = []
        for record in selected_records:
            if record.variable_id not in seen:
                seen.add(record.variable_id)
                unique_records.append(record)

        return {
            record.variable_id: self._read_parquet(record.canonical_parquet_path)
            for record in unique_records
        }

    def _read_records(self) -> list[VariableManifestRecord]:
        """Read variable manifest records from CSV first, JSON fallback."""
        raw_rows = self._read_variable_manifest_rows()
        records: list[VariableManifestRecord] = []

        for raw in raw_rows:
            variable_id = str(raw.get("variable_id", "")).strip()
            variable_name = str(raw.get("variable_name", "")).strip()

            if not variable_id:
                continue

            canonical_path = resolve_variable_parquet_path(
                run_root=self.run_root,
                manifest_path_value=str(raw.get("canonical_parquet_path", "")),
                variable_id=variable_id,
            )

            legacy_pickle_path = resolve_optional_legacy_pickle_path(
                run_root=self.run_root,
                manifest_path_value=str(raw.get("legacy_pickle_path", "")),
                variable_id=variable_id,
            )

            records.append(
                VariableManifestRecord(
                    variable_id=variable_id,
                    variable_name=variable_name,
                    reporting_frequency=str(raw.get("reporting_frequency", "")).strip(),
                    row_count=optional_int(raw.get("row_count")),
                    column_count=optional_int(raw.get("column_count")),
                    raw_csv_deleted=optional_bool(raw.get("raw_csv_deleted")),
                    canonical_parquet_path=canonical_path,
                    legacy_pickle_path=legacy_pickle_path,
                )
            )

        return records

    def _read_variable_manifest_rows(self) -> list[dict[str, Any]]:
        """Read raw variable manifest rows."""
        if self.variable_manifest_csv_path.is_file():
            with self.variable_manifest_csv_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                return list(csv.DictReader(stream))

        if self.variable_manifest_json_path.is_file():
            payload = load_json(self.variable_manifest_json_path)

            artifacts = payload.get("artifacts", [])
            if isinstance(artifacts, list):
                return [dict(item) for item in artifacts if isinstance(item, dict)]

            variables = payload.get("variables", [])
            if isinstance(variables, list):
                return [dict(item) for item in variables if isinstance(item, dict)]

        return []

    @staticmethod
    def _read_parquet(path: Path):
        """Read one Parquet file with pandas."""
        if not path.is_file():
            raise FileNotFoundError(f"Canonical variable parquet not found: {path}")

        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(
                "pandas is required to load canonical variable parquet files."
            ) from exc

        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read parquet file: {path}") from exc


def resolve_variable_parquet_path(
    *,
    run_root: Path,
    manifest_path_value: str,
    variable_id: str,
) -> Path:
    """Resolve a variable Parquet path portably.

    The manifest may contain:
        - an absolute path from the machine that generated the data
        - a relative path
        - only a file name

    For portability, prefer run-root-local canonical/variables/<file_name>.
    """
    value = manifest_path_value.strip()

    if value:
        candidate = Path(value)

        # If an absolute path exists on this machine, it is valid.
        if candidate.is_absolute() and candidate.is_file():
            return candidate

        # Most portable path: use only the file name under this run root.
        if candidate.name:
            local_by_name = run_root / "canonical" / "variables" / candidate.name
            if local_by_name.is_file():
                return local_by_name

        # If the manifest has a relative path, try run_root / relative path.
        if not candidate.is_absolute():
            local_relative = run_root / candidate
            if local_relative.is_file():
                return local_relative

    # Final deterministic fallback.
    return run_root / "canonical" / "variables" / f"{variable_id}.parquet"


def resolve_optional_legacy_pickle_path(
    *,
    run_root: Path,
    manifest_path_value: str,
    variable_id: str,
) -> Path | None:
    """Resolve optional per-variable legacy pickle path."""
    value = manifest_path_value.strip()

    if value:
        candidate = Path(value)

        if candidate.is_absolute() and candidate.is_file():
            return candidate

        if candidate.name:
            local_by_name = run_root / "legacy" / "per_variable_pickle" / candidate.name
            if local_by_name.is_file():
                return local_by_name

        if not candidate.is_absolute():
            local_relative = run_root / candidate
            if local_relative.is_file():
                return local_relative

    fallback = run_root / "legacy" / "per_variable_pickle" / f"{variable_id}.pickle"
    if fallback.is_file():
        return fallback

    return None


def optional_int(value: Any) -> int | None:
    """Convert optional integer-like value."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def optional_bool(value: Any) -> bool | None:
    """Convert optional boolean-like value."""
    if value is None:
        return None

    text = str(value).strip().casefold()
    if not text:
        return None

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n"}:
        return False

    return None