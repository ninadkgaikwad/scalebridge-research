# -*- coding: utf-8 -*-
"""Phase D D7 final Independent / Dependent 1 / Dependent 2 builders."""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .alignment import TimestampNormalizationConfig, load_and_align_paths
from .assembly import (
    AssemblyConfig,
    AssemblyResult,
    assemble_canonical_zone_table,
    required_phase_c_prediction_columns,
)
from .constants import ModelingSilo, PhaseDMode
from .discovery import discover_phase_d_sources
from .lineage import CounterpartResolution, resolve_all_to_one_counterpart
from .policies import (
    DEFAULT_MDH_TEST_FRACTION,
    DEFAULT_MDH_TRAIN_FRACTION,
    DEFAULT_MDH_VALIDATION_FRACTION,
    DEFAULT_SD_SEASON_OFFSET_DAYS,
    DEFAULT_SD_TEST_DAYS,
    DEFAULT_SD_TRAIN_DAYS,
    PolicyAssignmentDiagnostics,
    assign_chronological_holdout,
    assign_contiguous_identification,
    assign_custom_datetime_ranges,
    assign_monthly_distributed_holdout,
    assign_seasonal_block_holdout,
    assign_seasonal_distributed,
    assign_seasonal_holdout,
)
from .silo_contracts import (
    CONTROL_SIGNAL,
    QZIC_COMPONENTS,
    QZIR_COMPONENTS,
    SOLAR_DISTURBANCES,
    STATE_SIGNAL,
    VISIBLE_LIGHTING_COMPONENT,
    D6ContractError,
    HeatInputRepresentation,
    HeatRepresentationConfig,
    PhysicalRole,
    SiloProductContract,
    TemporalConfig,
    ZoneSignalAvailability,
)


D7_SCHEMA_VERSION = "phase_d_d7_final_builder_v1"
EXPECTED_PHASE_D_ROWS = 105120
DEFAULT_PARQUET_COMPRESSION = "zstd"
EXPECTED_TIMESTEP = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class FinalDatasetBuildResult:
    table: pd.DataFrame
    manifest: dict[str, Any]
    policy_diagnostics: PolicyAssignmentDiagnostics


@dataclass(frozen=True)
class CanonicalZoneData:
    aggregate_zone_id: str
    table: pd.DataFrame
    availability: ZoneSignalAvailability
    assembly_manifest: dict[str, Any]


def availability_from_canonical_table(
    aggregate_zone_id: str,
    table: pd.DataFrame,
) -> ZoneSignalAvailability:
    """Derive D6 usable signal availability from the validated D4 table."""

    required = {"timestamp", STATE_SIGNAL, "outdoor_temperature", CONTROL_SIGNAL}
    missing = required - set(table.columns)
    if missing:
        raise D6ContractError(
            f"Canonical zone table is missing required columns: {sorted(missing)}"
        )
    if table[CONTROL_SIGNAL].isna().any():
        raise D6ContractError(
            f"Aggregate zone {aggregate_zone_id!r} has missing QAC control values"
        )

    candidates = (
        *SOLAR_DISTURBANCES,
        *QZIC_COMPONENTS,
        *QZIR_COMPONENTS,
        VISIBLE_LIGHTING_COMPONENT,
        "zic",
        "zir",
    )
    available: list[str] = []
    for signal in candidates:
        if signal not in table.columns:
            continue
        if table[signal].notna().any():
            available.append(signal)

    return ZoneSignalAvailability(
        aggregate_zone_id=aggregate_zone_id,
        available_disturbances=tuple(available),
        qac_available=True,
    )


def _assert_same_timestamp_axis(tables: Mapping[str, pd.DataFrame]) -> pd.Series:
    if not tables:
        raise D6ContractError("At least one canonical zone table is required")
    first_name = next(iter(tables))
    reference = pd.to_datetime(tables[first_name]["timestamp"]).reset_index(drop=True)
    for zone_id, table in tables.items():
        timestamp = pd.to_datetime(table["timestamp"]).reset_index(drop=True)
        if len(timestamp) != len(reference) or not timestamp.equals(reference):
            raise D6ContractError(
                f"Timestamp axis mismatch between {first_name!r} and {zone_id!r}"
            )
    return reference


def _assert_same_outdoor_temperature(
    tables: Mapping[str, pd.DataFrame],
    *,
    tolerance: float = 1.0e-9,
) -> None:
    names = list(tables)
    if len(names) < 2:
        return
    reference = pd.to_numeric(
        tables[names[0]]["outdoor_temperature"], errors="raise"
    ).reset_index(drop=True)
    for name in names[1:]:
        other = pd.to_numeric(
            tables[name]["outdoor_temperature"], errors="raise"
        ).reset_index(drop=True)
        delta = (reference - other).abs()
        if delta.max(skipna=False) > tolerance:
            raise D6ContractError(
                f"Outdoor temperature differs between {names[0]!r} and {name!r}"
            )


def build_physical_table(
    contract: SiloProductContract,
    current_zone_tables: Mapping[str, pd.DataFrame],
    *,
    independent_zone_id: str | None = None,
    dependent_2_source_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the non-lagged wide physical table required by one D6 contract."""

    expected_current = {
        zone.aggregate_zone_id for zone in contract.current_zones
    }
    missing_current = expected_current - set(current_zone_tables)
    if missing_current:
        raise D6ContractError(
            f"Missing current canonical zone tables: {sorted(missing_current)}"
        )

    if contract.mode is PhaseDMode.INDEPENDENT:
        if independent_zone_id is None:
            raise D6ContractError("Independent builder requires independent_zone_id")
        selected_tables = {
            independent_zone_id: current_zone_tables[independent_zone_id]
        }
    else:
        selected_tables = {
            zone_id: current_zone_tables[zone_id]
            for zone_id in sorted(expected_current, key=str.casefold)
        }

    timestamp = _assert_same_timestamp_axis(selected_tables)
    _assert_same_outdoor_temperature(selected_tables)

    source_tables = dict(selected_tables)
    if contract.mode is PhaseDMode.DEPENDENT2:
        if dependent_2_source_table is None:
            raise D6ContractError("Dependent 2 requires its all-to-one source table")
        source_id = contract.dependent_2_source_zone.aggregate_zone_id
        source_table = dependent_2_source_table.reset_index(drop=True)
        source_timestamp = pd.to_datetime(source_table["timestamp"])
        if not source_timestamp.equals(timestamp):
            raise D6ContractError(
                "Dependent 2 all-to-one source timestamp axis differs from current run"
            )
        _assert_same_outdoor_temperature(
            {
                "current": next(iter(selected_tables.values())),
                source_id: source_table,
            }
        )
        source_tables[source_id] = source_table

    output = pd.DataFrame({"timestamp": timestamp})
    base_columns = contract.base_columns(
        independent_zone_id=independent_zone_id
    )

    for column in base_columns:
        if column.name == "outdoor_temperature":
            if contract.mode is PhaseDMode.DEPENDENT2:
                source_id = contract.dependent_2_source_zone.aggregate_zone_id
                source = source_tables[source_id]
            else:
                source = next(iter(selected_tables.values()))
            output[column.name] = source["outdoor_temperature"].reset_index(drop=True)
            continue

        if column.aggregate_zone_id is None:
            raise D6ContractError(
                f"Unexpected unqualified base column: {column.name}"
            )
        source = source_tables[column.aggregate_zone_id]
        if column.base_signal not in source.columns:
            raise D6ContractError(
                f"Source table {column.aggregate_zone_id!r} lacks "
                f"{column.base_signal!r}"
            )
        values = source[column.base_signal].reset_index(drop=True)
        if values.isna().any():
            raise D6ContractError(
                f"Included final signal {column.name!r} contains missing values"
            )
        output[column.name] = values

    return output


def _assign_policy(
    temporal: TemporalConfig,
    timestamps: pd.Series,
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    parameters = dict(temporal.policy_parameters)

    if temporal.policy_name == "monthly_distributed_holdout":
        return assign_monthly_distributed_holdout(
            timestamps,
            train_fraction=float(
                parameters.get("train_fraction", DEFAULT_MDH_TRAIN_FRACTION)
            ),
            test_fraction=float(
                parameters.get("test_fraction", DEFAULT_MDH_TEST_FRACTION)
            ),
            validation_fraction=float(
                parameters.get(
                    "validation_fraction", DEFAULT_MDH_VALIDATION_FRACTION
                )
            ),
        )

    if temporal.policy_name == "chronological_holdout":
        return assign_chronological_holdout(
            timestamps,
            train_fraction=float(
                parameters.get("train_fraction", DEFAULT_MDH_TRAIN_FRACTION)
            ),
            test_fraction=float(
                parameters.get("test_fraction", DEFAULT_MDH_TEST_FRACTION)
            ),
            validation_fraction=float(
                parameters.get(
                    "validation_fraction", DEFAULT_MDH_VALIDATION_FRACTION
                )
            ),
        )

    if temporal.policy_name == "seasonal_holdout":
        return assign_seasonal_holdout(
            timestamps,
            train_seasons=parameters.get("train_seasons", ("winter", "spring")),
            test_seasons=parameters.get("test_seasons", ("summer",)),
            validation_seasons=parameters.get("validation_seasons", ("fall",)),
        )

    if temporal.policy_name == "seasonal_distributed":
        return assign_seasonal_distributed(
            timestamps,
            season_offset_days=int(
                parameters.get(
                    "season_offset_days", DEFAULT_SD_SEASON_OFFSET_DAYS
                )
            ),
            train_days=int(parameters.get("train_days", DEFAULT_SD_TRAIN_DAYS)),
            test_days=int(parameters.get("test_days", DEFAULT_SD_TEST_DAYS)),
        )

    if temporal.policy_name == "seasonal_block_holdout":
        return assign_seasonal_block_holdout(
            timestamps,
            train_seasons=parameters.get(
                "train_seasons", ("winter", "spring", "fall")
            ),
            test_seasons=parameters.get("test_seasons", ("summer",)),
        )

    if temporal.policy_name == "contiguous_identification":
        return assign_contiguous_identification(
            timestamps,
            start_datetime=parameters.get("start_datetime"),
            train_days=int(parameters.get("train_days", DEFAULT_SD_TRAIN_DAYS)),
            test_days=int(parameters.get("test_days", DEFAULT_SD_TEST_DAYS)),
        )

    if temporal.policy_name == "custom_datetime_ranges":
        return assign_custom_datetime_ranges(
            timestamps,
            train_ranges=parameters.get("train_ranges"),
            test_ranges=parameters.get("test_ranges"),
        )

    raise D6ContractError(
        f"D7 builder does not implement policy {temporal.policy_name!r}"
    )


def expand_temporal_dataset(
    physical_table: pd.DataFrame,
    contract: SiloProductContract,
    *,
    independent_zone_id: str | None = None,
) -> tuple[pd.DataFrame, PolicyAssignmentDiagnostics]:
    """Expand one physical table to lagged inputs and future state targets."""

    base_contract_columns = contract.base_columns(
        independent_zone_id=independent_zone_id
    )
    expected = ["timestamp", *[item.name for item in base_contract_columns]]
    missing = set(expected) - set(physical_table.columns)
    if missing:
        raise D6ContractError(
            f"Physical table missing contract columns: {sorted(missing)}"
        )

    physical = physical_table[expected].copy().reset_index(drop=True)
    timestamps = pd.to_datetime(physical["timestamp"], errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise D6ContractError("Physical timestamps must be unique and increasing")

    policy_frame, diagnostics = _assign_policy(contract.temporal, timestamps)

    # Build temporal feature blocks in one batch. Repeated DataFrame column
    # insertion fragments pandas' internal blocks badly for wide Dep1/Dep2
    # realizations (especially many-zone buildings and larger ML lags).
    # The list order below intentionally preserves the locked D6 schema:
    # policy metadata, lag-major input columns, then horizon-major state targets.
    input_columns = [item.name for item in base_contract_columns]
    state_columns = [
        item.name
        for item in base_contract_columns
        if item.physical_role is PhysicalRole.STATE
    ]

    temporal_series: list[pd.Series] = []
    for lag in range(contract.temporal.input_lag):
        temporal_series.extend(
            physical[name]
            .shift(lag)
            .rename(f"{name}__lag_{lag}")
            for name in input_columns
        )

    for horizon in range(1, contract.temporal.target_horizon + 1):
        temporal_series.extend(
            physical[name]
            .shift(-horizon)
            .rename(f"{name}__target_{horizon}")
            for name in state_columns
        )

    if temporal_series:
        temporal_frame = pd.concat(temporal_series, axis=1)
        output = pd.concat(
            [policy_frame.reset_index(drop=True), temporal_frame.reset_index(drop=True)],
            axis=1,
        )
    else:
        output = policy_frame.copy()

    # Leakage-safe sample validity: every required lag and target must be
    # contiguous in time, included by the base policy, and in the same
    # partition as the anchor row.
    anchor_partition = policy_frame["partition"]
    anchor_included = policy_frame["included"].astype(bool)
    safe = anchor_included.copy()

    for lag in range(contract.temporal.input_lag):
        if lag == 0:
            continue
        expected_delta = EXPECTED_TIMESTEP * lag
        safe &= (timestamps - timestamps.shift(lag)).eq(expected_delta)
        safe &= policy_frame["included"].shift(lag, fill_value=False).astype(bool)
        safe &= anchor_partition.eq(anchor_partition.shift(lag))

    for horizon in range(1, contract.temporal.target_horizon + 1):
        expected_delta = EXPECTED_TIMESTEP * horizon
        safe &= (timestamps.shift(-horizon) - timestamps).eq(expected_delta)
        safe &= policy_frame["included"].shift(-horizon, fill_value=False).astype(bool)
        safe &= anchor_partition.eq(anchor_partition.shift(-horizon))

    unsafe = ~safe
    output.loc[unsafe, "included"] = False
    output.loc[unsafe, "partition"] = "excluded"
    output.loc[unsafe, "window_id"] = pd.NA

    expected_final_names = [
        item.name
        for item in contract.final_columns(
            independent_zone_id=independent_zone_id
        )
    ]
    if list(output.columns) != expected_final_names:
        raise D6ContractError(
            "Materialized D7 columns do not match D6 final-column contract"
        )

    return output, _post_temporal_diagnostics(
        output,
        diagnostics,
        contract.temporal.input_lag,
        contract.temporal.target_horizon,
    )


def _post_temporal_diagnostics(
    table: pd.DataFrame,
    base: PolicyAssignmentDiagnostics,
    input_lag: int,
    target_horizon: int,
) -> PolicyAssignmentDiagnostics:
    counts = {
        str(key): int(value)
        for key, value in table["partition"].value_counts(dropna=False).to_dict().items()
    }
    included = int(table["included"].sum())
    parameters = dict(base.parameters)
    parameters.update(
        {
            "input_lag": input_lag,
            "target_horizon": target_horizon,
            "leakage_safe_temporal_windows": True,
            "excluded_after_temporal_expansion": len(table) - included,
        }
    )
    return PolicyAssignmentDiagnostics(
        policy_name=base.policy_name,
        row_count=len(table),
        included_count=included,
        excluded_count=len(table) - included,
        partition_counts=counts,
        parameters=parameters,
    )


def build_final_dataset(
    contract: SiloProductContract,
    current_zone_tables: Mapping[str, pd.DataFrame],
    *,
    independent_zone_id: str | None = None,
    dependent_2_source_table: pd.DataFrame | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> FinalDatasetBuildResult:
    physical = build_physical_table(
        contract,
        current_zone_tables,
        independent_zone_id=independent_zone_id,
        dependent_2_source_table=dependent_2_source_table,
    )
    final, policy_diagnostics = expand_temporal_dataset(
        physical,
        contract,
        independent_zone_id=independent_zone_id,
    )
    del physical
    gc.collect()

    manifest = contract.to_manifest_contract(
        independent_zone_id=independent_zone_id
    )
    manifest.update(
        {
            "d7_schema_version": D7_SCHEMA_VERSION,
            "row_count": len(final),
            "included_row_count": int(final["included"].sum()),
            "excluded_row_count": int((~final["included"]).sum()),
            "partition_counts": {
                str(key): int(value)
                for key, value in final["partition"].value_counts().to_dict().items()
            },
            "first_timestamp": pd.Timestamp(final["timestamp"].iloc[0]).isoformat(),
            "last_timestamp": pd.Timestamp(final["timestamp"].iloc[-1]).isoformat(),
            "policy_assignment": policy_diagnostics.to_dict(),
            "provenance": dict(provenance or {}),
        }
    )
    return FinalDatasetBuildResult(
        table=final,
        manifest=manifest,
        policy_diagnostics=policy_diagnostics,
    )


def write_final_dataset(
    result: FinalDatasetBuildResult,
    *,
    silo_root: Path,
    contract: SiloProductContract,
    independent_zone_id: str | None = None,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> tuple[Path, Path]:
    """Atomically persist the only D7 time-series artifact plus its manifest."""

    relative = contract.relative_output_dir(
        independent_zone_id=independent_zone_id
    )
    output_dir = Path(silo_root) / contract.silo_folder_name / relative
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "data.parquet"
    manifest_path = output_dir / "manifest.json"
    temp_data = output_dir / "data.parquet.tmp"
    temp_manifest = output_dir / "manifest.json.tmp"

    result.table.to_parquet(
        temp_data,
        index=False,
        engine="pyarrow",
        compression=compression,
    )
    temp_manifest.write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    os.replace(temp_data, data_path)
    os.replace(temp_manifest, manifest_path)
    return data_path, manifest_path


def assemble_zone_in_memory(
    *,
    campaign_root: Path,
    matrix_run_id: str,
    aggregation_run_id: str,
    phase_c_campaign_run_id: str,
    aggregate_zone_id: str,
    phase_d_calendar_year: int,
    include_visible_lighting_in_zir: bool,
) -> CanonicalZoneData:
    """Reconstruct D4 directly from authoritative sources without persistence."""

    import pyarrow.parquet as pq

    discovery = discover_phase_d_sources(
        campaign_root=Path(campaign_root),
        matrix_run_id=matrix_run_id,
        aggregation_run_id=aggregation_run_id,
        phase_c_campaign_run_id=phase_c_campaign_run_id,
        aggregate_zone_id=aggregate_zone_id,
    )

    phase_c_schema = pq.ParquetFile(
        discovery.phase_c_zone.predictions_parquet_path
    ).schema_arrow
    phase_c_columns = required_phase_c_prediction_columns(
        discovery.phase_c_zone.applicable_models_path,
        discovery.phase_c_zone.unavailable_models_path,
        available_parquet_columns=set(phase_c_schema.names),
    )

    # D7 creates its own split/selection policy. Phase C split is used only as
    # the authoritative timestamp axis required by D3 alignment.
    aligned, alignment_diagnostics = load_and_align_paths(
        discovery.aggregation_zone.wide_parquet_path,
        discovery.phase_c_zone.predictions_parquet_path,
        discovery.phase_c_zone.split_assignments_parquet_path,
        TimestampNormalizationConfig(phase_d_calendar_year),
        phase_c_columns=phase_c_columns,
        split_columns=[],
    )

    result: AssemblyResult = assemble_canonical_zone_table(
        aligned,
        applicable_models_path=discovery.phase_c_zone.applicable_models_path,
        unavailable_models_path=discovery.phase_c_zone.unavailable_models_path,
        config=AssemblyConfig(
            include_visible_lighting_in_zir=include_visible_lighting_in_zir
        ),
    )
    del aligned
    gc.collect()

    if len(result.table) != EXPECTED_PHASE_D_ROWS:
        raise D6ContractError(
            f"Expected {EXPECTED_PHASE_D_ROWS} annual Phase D rows for "
            f"{aggregate_zone_id!r}; found {len(result.table)}"
        )

    availability = availability_from_canonical_table(
        aggregate_zone_id,
        result.table,
    )
    manifest = result.manifest_dict(
        aggregate_zone_id=aggregate_zone_id,
        include_signal_records=False,
    )
    manifest["alignment_diagnostics"] = alignment_diagnostics.to_dict()
    manifest["phase_d_calendar_year"] = phase_d_calendar_year
    manifest["persisted_intermediate"] = False

    return CanonicalZoneData(
        aggregate_zone_id=aggregate_zone_id,
        table=result.table,
        availability=availability,
        assembly_manifest=manifest,
    )


def build_contract(
    *,
    silo: ModelingSilo,
    mode: PhaseDMode,
    current_zones: tuple[ZoneSignalAvailability, ...],
    heat: HeatRepresentationConfig,
    input_lag: int,
    target_horizon: int,
    policy_name: str,
    policy_parameters: Mapping[str, Any],
    dependent_2_source_zone: ZoneSignalAvailability | None = None,
) -> SiloProductContract:
    temporal = TemporalConfig(
        silo=silo,
        input_lag=input_lag,
        target_horizon=target_horizon,
        policy_name=policy_name,
        policy_parameters=dict(policy_parameters),
    )
    return SiloProductContract(
        silo=silo,
        mode=mode,
        temporal=temporal,
        heat=heat,
        current_zones=current_zones,
        dependent_2_source_zone=dependent_2_source_zone,
    )


def resolve_dep2_for_build(
    *,
    campaign_root: Path,
    matrix_run_id: str,
    aggregation_run_id: str,
    phase_c_campaign_run_id: str,
    require_available: bool = True,
) -> CounterpartResolution:
    """Resolve D5 Dep2 lineage.

    D7 controlled validation keeps ``require_available=True``.  D8 production
    orchestration passes ``False`` so a legal D5 "unavailable" result produces
    ind/dep1 datasets while omitting dep2 rather than failing the whole
    aggregation run.
    """
    resolution = resolve_all_to_one_counterpart(
        campaign_root=campaign_root,
        matrix_run_id=matrix_run_id,
        aggregation_run_id=aggregation_run_id,
        phase_c_campaign_run_id=phase_c_campaign_run_id,
    )
    if (
        require_available
        and (
            not resolution.dependent_2_available
            or resolution.selected_lineage is None
        )
    ):
        raise D6ContractError(
            "Dependent 2 is unavailable: "
            f"status={resolution.status}, "
            f"phase_c_usable="
            f"{resolution.phase_c_usability.usable if resolution.phase_c_usability else False}"
        )
    return resolution
