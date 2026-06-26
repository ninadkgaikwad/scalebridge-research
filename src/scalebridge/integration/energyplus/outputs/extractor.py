"""Extract requested EnergyPlus signals into canonical ScaleBridge artifacts.

The EnergyPlus ``eplusout.csv`` file is retained only as a raw diagnostic.
Canonical time series are read from ``eplusout.eso`` through opyplus and
written as typed, compressed Parquet files, one per reporting frequency.

Each Parquet file uses a long-form schema:

``timestamp, environment, reporting_frequency, key_value, variable_name,
units, semantic_role, value``.

This representation avoids ambiguous CSV column labels, supports selective
Parquet scans, and preserves the zone/key identity needed by aggregation.
Optional compatibility pickles reproduce the broad legacy dictionary shape
without requiring one EnergyPlus simulation per output variable.
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from scalebridge.integration.energyplus.manifests.models import (
    CaseSpec,
    OutputVariableRequest,
)
from scalebridge.integration.energyplus.outputs.eio import EioTable, parse_eio


CANONICAL_COLUMNS = (
    "timestamp",
    "environment",
    "reporting_frequency",
    "key_value",
    "variable_name",
    "units",
    "semantic_role",
    "value",
)


class CanonicalExtractionError(RuntimeError):
    """Raised when required simulation outputs cannot be canonicalized."""


class CanonicalExtractionResult(BaseModel):
    """Artifact paths and validation counts from one canonical extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    environment: str
    requested_signal_count: int = Field(ge=0)
    produced_signal_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    timestep_count: int = Field(ge=0)
    missing_required_signals: tuple[str, ...] = ()
    parquet_paths: dict[str, Path]
    metadata_path: Path
    eio_metadata_path: Path
    legacy_output_pickle_path: Path | None = None
    legacy_eio_pickle_path: Path | None = None


class EnergyPlusOutputExtractor:
    """Canonicalize one completed opyplus simulation.

    Parameters
    ----------
    eso_loader:
        Optional injected loader used by tests. Production uses
        ``opyplus.StandardOutput``.
    parquet_writer:
        Optional injected writer receiving ``(dataframe, destination)``.
        Production uses pandas with the pyarrow engine.
    """

    def __init__(
        self,
        *,
        eso_loader: Callable[[Path], Any] | None = None,
        parquet_writer: Callable[[pd.DataFrame, Path], None] | None = None,
    ) -> None:
        self._eso_loader = eso_loader or self._load_eso
        self._parquet_writer = parquet_writer or self._write_parquet

    def extract(
        self,
        *,
        case_spec: CaseSpec,
        simulation_directory: str | Path,
        canonical_directory: str | Path,
        legacy_directory: str | Path | None = None,
    ) -> CanonicalExtractionResult:
        """Extract, validate, and persist requested ESO and EIO outputs."""
        raw_root = Path(simulation_directory).expanduser().resolve()
        canonical_root = Path(canonical_directory).expanduser().resolve()
        legacy_root = (
            Path(legacy_directory).expanduser().resolve()
            if legacy_directory is not None
            else canonical_root.parent / "legacy"
        )

        eso_path = raw_root / "eplusout.eso"
        eio_path = raw_root / "eplusout.eio"
        if not eso_path.is_file():
            raise CanonicalExtractionError(f"ESO file does not exist: {eso_path}")
        if not eio_path.is_file():
            raise CanonicalExtractionError(f"EIO file does not exist: {eio_path}")

        canonical_root.mkdir(parents=True, exist_ok=True)
        standard_output = self._eso_loader(eso_path)
        start_year = case_spec.run_period.calendar_year
        if start_year is None:
            raise CanonicalExtractionError(
                "canonical datetime extraction requires run_period.calendar_year"
            )
        standard_output.create_datetime_index(start_year)

        environments = standard_output.get_environments()
        if not environments:
            raise CanonicalExtractionError("ESO contains no simulation environments")
        environment = tuple(environments)[-1]

        variable_catalog = standard_output.get_variables()
        requests_by_frequency = _group_requests(case_spec.output_variables)
        frames_by_frequency: dict[str, pd.DataFrame] = {}
        missing_required: list[str] = []
        produced_signal_count = 0

        for frequency, requests in requests_by_frequency.items():
            variables = list(variable_catalog.get(frequency, ()))
            source_frame = standard_output.get_data(environment, frequency=frequency)
            canonical_frame, missing, produced = _canonicalize_frequency(
                source_frame=source_frame,
                environment=environment,
                frequency=frequency,
                variables=variables,
                requests=requests,
            )
            frames_by_frequency[frequency] = canonical_frame
            missing_required.extend(missing)
            produced_signal_count += produced

        if missing_required:
            joined = ", ".join(missing_required)
            raise CanonicalExtractionError(f"required ESO signals are missing: {joined}")

        # Validate simulation metadata before committing canonical artifacts.
        eio_tables = parse_eio(eio_path)
        _validate_eio_calendar_year(eio_tables, expected_year=start_year)

        parquet_paths: dict[str, Path] = {}
        for frequency, frame in frames_by_frequency.items():
            destination = canonical_root / f"timeseries_{frequency}.parquet"
            self._parquet_writer(frame, destination)
            parquet_paths[frequency] = destination

        eio_metadata_path = canonical_root / "eio_tables.json"
        _write_json(
            eio_metadata_path,
            {
                "schema_version": "0.1.0",
                "source": str(eio_path),
                "table_count": len(eio_tables),
                "tables": {
                    name: table.to_serializable_dict()
                    for name, table in eio_tables.items()
                },
            },
        )

        row_count = sum(len(frame) for frame in frames_by_frequency.values())
        timestep_count = max(
            (
                frame["timestamp"].nunique()
                for frame in frames_by_frequency.values()
                if not frame.empty
            ),
            default=0,
        )
        metadata_path = canonical_root / "metadata.json"
        _write_json(
            metadata_path,
            {
                "schema_version": "0.1.0",
                "case_id": case_spec.case_id,
                "environment": environment,
                "requested_signal_count": len(case_spec.output_variables),
                "produced_signal_count": produced_signal_count,
                "row_count": row_count,
                "timestep_count": timestep_count,
                "parquet_files": {
                    frequency: path.name
                    for frequency, path in sorted(parquet_paths.items())
                },
                "canonical_columns": list(CANONICAL_COLUMNS),
            },
        )

        legacy_output_path: Path | None = None
        legacy_eio_path: Path | None = None
        if case_spec.write_legacy_pickles:
            legacy_root.mkdir(parents=True, exist_ok=True)
            legacy_output_path = legacy_root / "IDF_OutputVariables_DictDF.pickle"
            legacy_eio_path = legacy_root / "Eio_OutputFile.pickle"
            _write_legacy_output_pickle(frames_by_frequency, legacy_output_path)
            _write_legacy_eio_pickle(eio_tables, legacy_eio_path)

        return CanonicalExtractionResult(
            case_id=case_spec.case_id,
            environment=environment,
            requested_signal_count=len(case_spec.output_variables),
            produced_signal_count=produced_signal_count,
            row_count=row_count,
            timestep_count=timestep_count,
            missing_required_signals=(),
            parquet_paths=parquet_paths,
            metadata_path=metadata_path,
            eio_metadata_path=eio_metadata_path,
            legacy_output_pickle_path=legacy_output_path,
            legacy_eio_pickle_path=legacy_eio_path,
        )

    @staticmethod
    def _load_eso(path: Path) -> Any:
        """Load an ESO through the public opyplus output API."""
        import opyplus

        return opyplus.StandardOutput(str(path))

    @staticmethod
    def _write_parquet(frame: pd.DataFrame, destination: Path) -> None:
        """Write one compressed typed Parquet artifact."""
        try:
            frame.to_parquet(
                destination,
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
        except ImportError as exc:
            raise CanonicalExtractionError(
                "pyarrow is required for canonical Parquet output"
            ) from exc


def extract_canonical_outputs(
    *,
    case_spec: CaseSpec,
    simulation_directory: str | Path,
    canonical_directory: str | Path,
    legacy_directory: str | Path | None = None,
) -> CanonicalExtractionResult:
    """Convenience wrapper for the default production extractor."""
    return EnergyPlusOutputExtractor().extract(
        case_spec=case_spec,
        simulation_directory=simulation_directory,
        canonical_directory=canonical_directory,
        legacy_directory=legacy_directory,
    )


def _group_requests(
    requests: tuple[OutputVariableRequest, ...],
) -> dict[str, tuple[OutputVariableRequest, ...]]:
    """Group output requests by normalized reporting frequency."""
    grouped: dict[str, list[OutputVariableRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.reporting_frequency].append(request)
    return {frequency: tuple(items) for frequency, items in grouped.items()}


def _canonicalize_frequency(
    *,
    source_frame: pd.DataFrame,
    environment: str,
    frequency: str,
    variables: list[Any],
    requests: tuple[OutputVariableRequest, ...],
) -> tuple[pd.DataFrame, list[str], int]:
    """Select requested variables and convert one frequency to long form."""
    pieces: list[pd.DataFrame] = []
    missing_required: list[str] = []
    produced_refs: set[str] = set()
    columns_by_casefold = {str(column).casefold(): column for column in source_frame}

    for request in requests:
        matches = [
            variable
            for variable in variables
            if _variable_matches_request(variable, request)
        ]
        available_matches = [
            variable
            for variable in matches
            if str(variable.ref).casefold() in columns_by_casefold
        ]

        if not available_matches and request.required:
            missing_required.append(_request_identity(request))
            continue

        for variable in available_matches:
            source_column = columns_by_casefold[str(variable.ref).casefold()]
            piece = pd.DataFrame(
                {
                    "timestamp": source_frame.index,
                    "environment": environment,
                    "reporting_frequency": frequency,
                    "key_value": str(variable.key_value),
                    "variable_name": str(variable.name),
                    "units": str(variable.unit or ""),
                    "semantic_role": request.semantic_role,
                    "value": pd.to_numeric(source_frame[source_column], errors="coerce"),
                }
            )
            pieces.append(piece)
            produced_refs.add(str(variable.ref).casefold())

    if not pieces:
        return pd.DataFrame(columns=CANONICAL_COLUMNS), missing_required, 0

    canonical = pd.concat(pieces, ignore_index=True)
    canonical = canonical.loc[:, CANONICAL_COLUMNS]
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"])
    canonical["value"] = canonical["value"].astype("float64")
    return canonical, missing_required, len(produced_refs)


def _variable_matches_request(variable: Any, request: OutputVariableRequest) -> bool:
    """Return whether one ESO variable satisfies one declarative request."""
    if str(variable.name).casefold() != request.variable_name.casefold():
        return False
    return (
        request.key_value == "*"
        or str(variable.key_value).casefold() == request.key_value.casefold()
    )


def _request_identity(request: OutputVariableRequest) -> str:
    """Create an actionable identity for validation errors."""
    return (
        f"{request.key_value}|{request.variable_name}|"
        f"{request.reporting_frequency}"
    )


def _validate_eio_calendar_year(
    tables: dict[str, EioTable],
    *,
    expected_year: int,
) -> None:
    """Require EIO weather-run-period dates to match the scientific case."""
    environment_table = tables.get("Environment")
    if environment_table is None:
        raise CanonicalExtractionError("EIO is missing the Environment table")

    column_positions = {
        name.casefold(): index
        for index, name in enumerate(environment_table.columns)
    }
    required_columns = ("Environment Type", "Start Date", "End Date")
    missing_columns = [
        name
        for name in required_columns
        if name.casefold() not in column_positions
    ]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise CanonicalExtractionError(
            f"EIO Environment table is missing columns: {joined}"
        )

    weather_rows = [
        row
        for row in environment_table.rows
        if row[column_positions["environment type"]].casefold()
        == "weatherfilerunperiod"
    ]
    if not weather_rows:
        raise CanonicalExtractionError(
            "EIO Environment table contains no WeatherFileRunPeriod"
        )

    reported_years: set[int] = set()
    for row in weather_rows:
        for column_name in ("start date", "end date"):
            raw_date = row[column_positions[column_name]]
            try:
                reported_years.add(datetime.strptime(raw_date, "%m/%d/%Y").year)
            except ValueError as exc:
                raise CanonicalExtractionError(
                    f"EIO Environment date has unsupported format: {raw_date!r}"
                ) from exc

    if reported_years != {expected_year}:
        reported = ", ".join(str(year) for year in sorted(reported_years))
        raise CanonicalExtractionError(
            f"EIO run-period year {reported} does not match "
            f"CaseSpec calendar year {expected_year}"
        )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic UTF-8 JSON."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_legacy_output_pickle(
    frames_by_frequency: dict[str, pd.DataFrame],
    destination: Path,
) -> None:
    """Write variable-keyed DataFrames compatible with legacy aggregation."""
    combined = pd.concat(frames_by_frequency.values(), ignore_index=True)
    legacy: dict[str, object] = {
        "DateTime_List": sorted(combined["timestamp"].dropna().unique().tolist())
    }
    for variable_name, variable_frame in combined.groupby(
        "variable_name",
        sort=True,
    ):
        wide = variable_frame.pivot_table(
            index="timestamp",
            columns="key_value",
            values="value",
            aggfunc="first",
        ).reset_index()
        legacy[str(variable_name).replace(" ", "_")] = wide

    with destination.open("wb") as stream:
        pickle.dump(legacy, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _write_legacy_eio_pickle(
    tables: dict[str, EioTable],
    destination: Path,
) -> None:
    """Write the legacy EIO category-to-DataFrame dictionary."""
    payload = {name: table.to_dataframe() for name, table in tables.items()}
    with destination.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
