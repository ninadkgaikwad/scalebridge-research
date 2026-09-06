
# -*- coding: utf-8 -*-
"""Phase D D4 canonical signal assembly and grouped heat-input construction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import DEFAULT_INCLUDE_VISIBLE_LIGHTING_IN_ZIR, NullableReason, PhaseDSignalStatus
from .models import ZoneSignalRecord
from .signals import (
    build_signal_registry,
    classify_phase_c_signal,
    group_components,
    SignalClassification,
)


class PhaseDAssemblyError(RuntimeError):
    """Raised when canonical Phase D signals cannot be assembled safely."""


@dataclass(frozen=True)
class AssemblyConfig:
    include_visible_lighting_in_zir: bool = DEFAULT_INCLUDE_VISIBLE_LIGHTING_IN_ZIR
    zero_absolute_tolerance: float = 1.0e-9


@dataclass(frozen=True)
class AssemblyDiagnostics:
    row_count: int
    canonical_column_count: int
    active_phase_c_signal_count: int
    nullable_complete_zero_count: int
    nullable_not_applicable_count: int
    constant_nonzero_count: int
    varying_count: int
    validation_failure_count: int
    zic_active_components: tuple[str, ...]
    zir_active_components: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["zic_active_components"] = list(self.zic_active_components)
        payload["zir_active_components"] = list(self.zir_active_components)
        return payload


@dataclass(frozen=True)
class AssemblyResult:
    table: pd.DataFrame
    signal_records: tuple[ZoneSignalRecord, ...]
    diagnostics: AssemblyDiagnostics

    def manifest_dict(
        self,
        *,
        aggregate_zone_id: str,
        include_signal_records: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "aggregate_zone_id": aggregate_zone_id,
            "row_count": int(len(self.table)),
            "canonical_columns": list(self.table.columns),
            "diagnostics": self.diagnostics.to_dict(),
        }
        if include_signal_records:
            payload["signal_records"] = [
                record.to_dict() for record in self.signal_records
            ]
        else:
            payload["signal_summary"] = {
                record.signal_name: {
                    "phase_d_status": record.phase_d_status.value,
                    "nullable": record.nullable,
                    "nullable_reason": record.nullable_reason.value,
                    "included_in_group": record.included_in_group,
                    "group_name": record.group_name,
                    "phase_c_reason_code": record.metadata.get(
                        "phase_c_reason_code"
                    ),
                }
                for record in self.signal_records
            }
        return payload


def _read_status_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model_id", "output_prediction_column", "applicable", "reason_code", "reason"}
    missing = required - set(frame.columns)
    if missing:
        raise PhaseDAssemblyError(
            f"Status table {path} is missing required columns: {sorted(missing)}"
        )
    return frame


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def required_phase_c_prediction_columns(
    applicable_models_path: str | Path,
    unavailable_models_path: str | Path,
    *,
    available_parquet_columns: set[str] | None = None,
) -> tuple[str, ...]:
    """Return the minimal Phase C prediction projection needed by D4.

    Applicable model outputs are required. Unavailable/all-zero outputs are
    represented from audit metadata and are not read into memory. Optional
    PHVAC and PHVAC-oracle columns are included only when physically present.
    """
    applicable_df = _read_status_table(applicable_models_path)
    unavailable_df = _read_status_table(unavailable_models_path)

    unavailable_ids = set(unavailable_df["model_id"].astype(str))
    columns: list[str] = []
    for _, row in applicable_df.iterrows():
        if not _boolean(row["applicable"]):
            continue
        column = str(row["output_prediction_column"])
        if available_parquet_columns is not None and column not in available_parquet_columns:
            raise PhaseDAssemblyError(
                f"Applicable prediction column is absent from Parquet schema: {column}"
            )
        columns.append(column)

    for optional in ("predicted_PHVAC", "predicted_PHVAC_oracle"):
        if available_parquet_columns is not None and optional in available_parquet_columns:
            if optional not in columns:
                columns.append(optional)

    return tuple(dict.fromkeys(columns))


def _model_status_maps(
    applicable_models_path: str | Path,
    unavailable_models_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    applicable_df = _read_status_table(applicable_models_path)
    unavailable_df = _read_status_table(unavailable_models_path)

    applicable: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, dict[str, Any]] = {}

    for _, row in applicable_df.iterrows():
        model_id = str(row["model_id"])
        if not _boolean(row["applicable"]):
            raise PhaseDAssemblyError(
                f"Applicable table contains model marked non-applicable: {model_id}"
            )
        applicable[model_id] = row.to_dict()

    for _, row in unavailable_df.iterrows():
        model_id = str(row["model_id"])
        if _boolean(row["applicable"]):
            raise PhaseDAssemblyError(
                f"Unavailable table contains model marked applicable: {model_id}"
            )
        unavailable[model_id] = row.to_dict()

    overlap = set(applicable) & set(unavailable)
    if overlap:
        raise PhaseDAssemblyError(
            f"Models appear in both applicable and unavailable tables: {sorted(overlap)}"
        )
    return applicable, unavailable


def _classify_applicable_series(
    values: pd.Series,
    *,
    zero_absolute_tolerance: float,
) -> SignalClassification:
    """Vectorized classification for an applicable annual prediction series."""
    numeric = pd.to_numeric(values, errors="coerce")
    missing_count = int(numeric.isna().sum())
    finite_count = int(numeric.notna().sum())
    if missing_count:
        return SignalClassification(
            status=PhaseDSignalStatus.VALIDATION_FAILURE,
            nullable=False,
            nullable_reason=NullableReason.NONE,
            finite_count=finite_count,
            missing_count=missing_count,
            minimum=None,
            maximum=None,
            mean=None,
            constant_value=None,
        )

    minimum = float(numeric.min())
    maximum = float(numeric.max())
    mean = float(numeric.mean())
    if max(abs(minimum), abs(maximum)) <= zero_absolute_tolerance:
        return SignalClassification(
            status=PhaseDSignalStatus.NULLABLE_COMPLETE_ZERO,
            nullable=True,
            nullable_reason=NullableReason.COMPLETE_ZERO_SIGNAL,
            finite_count=finite_count,
            missing_count=0,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            constant_value=0.0,
        )
    if abs(maximum - minimum) <= zero_absolute_tolerance:
        return SignalClassification(
            status=PhaseDSignalStatus.CONSTANT_NONZERO,
            nullable=False,
            nullable_reason=NullableReason.NONE,
            finite_count=finite_count,
            missing_count=0,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            constant_value=mean,
        )
    return SignalClassification(
        status=PhaseDSignalStatus.VARYING,
        nullable=False,
        nullable_reason=NullableReason.NONE,
        finite_count=finite_count,
        missing_count=0,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        constant_value=None,
    )


def _record(
    *,
    signal_name: str,
    definition: Any,
    classification: Any,
    included_in_group: bool,
    metadata: dict[str, Any],
) -> ZoneSignalRecord:
    return ZoneSignalRecord(
        signal_name=signal_name,
        source_phase=definition.source_phase.value,
        source_name=definition.source_name,
        units=definition.units,
        phase_d_status=classification.status,
        nullable=classification.nullable,
        nullable_reason=classification.nullable_reason,
        included_in_group=included_in_group,
        group_name=definition.group_name if included_in_group else None,
        minimum=classification.minimum,
        maximum=classification.maximum,
        mean=classification.mean,
        constant_value=classification.constant_value,
        finite_count=classification.finite_count,
        missing_count=classification.missing_count,
        metadata=metadata,
    )


def _safe_group_sum(
    table: pd.DataFrame,
    component_names: tuple[str, ...],
    *,
    group_name: str,
) -> tuple[pd.Series, tuple[str, ...]]:
    active = tuple(
        name
        for name in component_names
        if name in table.columns and not table[name].isna().all()
    )
    if not active:
        return pd.Series(pd.NA, index=table.index, dtype="Float64"), active

    missing_by_component = {
        name: int(table[name].isna().sum())
        for name in active
        if table[name].isna().any()
    }
    if missing_by_component:
        raise PhaseDAssemblyError(
            f"Active {group_name} components contain missing values: {missing_by_component}"
        )

    return table.loc[:, list(active)].sum(axis=1), active


def assemble_canonical_zone_table(
    aligned: pd.DataFrame,
    *,
    applicable_models_path: str | Path,
    unavailable_models_path: str | Path,
    config: AssemblyConfig = AssemblyConfig(),
) -> AssemblyResult:
    """Assemble one canonical Phase D zone table from D3-aligned inputs."""

    required_base = {"timestamp", "zone_temperature", "outdoor_temperature"}
    missing_base = required_base - set(aligned.columns)
    if missing_base:
        raise PhaseDAssemblyError(
            f"Aligned table is missing required Phase B columns: {sorted(missing_base)}"
        )
    if aligned["timestamp"].duplicated().any():
        raise PhaseDAssemblyError("Aligned table contains duplicate timestamps")
    if aligned[list(required_base)].isna().any().any():
        raise PhaseDAssemblyError("Aligned Phase B columns contain missing values")

    applicable, unavailable = _model_status_maps(
        applicable_models_path,
        unavailable_models_path,
    )
    registry = build_signal_registry(
        include_visible_lighting_in_zir=config.include_visible_lighting_in_zir
    )

    table = pd.DataFrame(index=aligned.index)
    table["timestamp"] = aligned["timestamp"]
    table["zone_temperature"] = aligned["zone_temperature"]
    table["outdoor_temperature"] = aligned["outdoor_temperature"]

    records: list[ZoneSignalRecord] = []
    phase_c_signal_names = [
        name
        for name, definition in registry.items()
        if definition.source_phase.value == "phase_c"
    ]

    for signal_name in phase_c_signal_names:
        definition = registry[signal_name]
        source_name = definition.source_name
        assert source_name is not None
        model_id = source_name.removeprefix("predicted_")

        status_row = applicable.get(model_id)
        unavailable_row = unavailable.get(model_id)

        if status_row is None and unavailable_row is None:
            # PHVAC is an auxiliary inference product and may not appear in the
            # heat-input applicability audit. Its presence in aligned data is authoritative.
            if signal_name == "phvac" and source_name in aligned.columns:
                status_row = {
                    "model_id": model_id,
                    "reason_code": "inference_auxiliary_available",
                    "reason": "auxiliary Phase C inference output is available",
                    "applicability_status": "available",
                }
            else:
                raise PhaseDAssemblyError(
                    f"No applicability record found for model {model_id} ({signal_name})"
                )

        phase_c_applicable = status_row is not None
        if phase_c_applicable and source_name not in aligned.columns:
            raise PhaseDAssemblyError(
                f"Applicable model {model_id} is missing prediction column {source_name}"
            )

        reason_code = str(
            (status_row if status_row is not None else unavailable_row).get(
                "reason_code", ""
            )
        ).lower()
        upstream_declares_complete_zero = (
            not phase_c_applicable
            and ("all_zero" in reason_code or "constant_zero" in reason_code)
        )

        if upstream_declares_complete_zero:
            classification = SignalClassification(
                status=PhaseDSignalStatus.NULLABLE_COMPLETE_ZERO,
                nullable=True,
                nullable_reason=NullableReason.COMPLETE_ZERO_SIGNAL,
                finite_count=0,
                missing_count=0,
                minimum=0.0,
                maximum=0.0,
                mean=0.0,
                constant_value=0.0,
            )
        elif not phase_c_applicable:
            classification = SignalClassification(
                status=PhaseDSignalStatus.NULLABLE_NOT_APPLICABLE,
                nullable=True,
                nullable_reason=NullableReason.PHASE_C_MODEL_NOT_APPLICABLE,
                finite_count=0,
                missing_count=len(aligned),
                minimum=None,
                maximum=None,
                mean=None,
                constant_value=None,
            )
        else:
            classification = _classify_applicable_series(
                aligned[source_name],
                zero_absolute_tolerance=config.zero_absolute_tolerance,
            )
        if classification.status is PhaseDSignalStatus.VALIDATION_FAILURE:
            raise PhaseDAssemblyError(
                f"Applicable model {model_id} failed Phase D classification: "
                f"missing_count={classification.missing_count}"
            )

        if classification.nullable:
            table[signal_name] = pd.Series(
                pd.NA, index=aligned.index, dtype="Float64"
            )
        else:
            table[signal_name] = pd.to_numeric(
                aligned[source_name], errors="coerce"
            )

        upstream = status_row if status_row is not None else unavailable_row
        metadata = {
            "model_id": model_id,
            "phase_c_reason_code": upstream.get("reason_code"),
            "phase_c_reason": upstream.get("reason"),
            "phase_c_applicability_status": upstream.get("applicability_status"),
            "stored_column": True,
            "column_storage_state": (
                "all_null" if classification.nullable else "populated"
            ),
        }
        records.append(
            _record(
                signal_name=signal_name,
                definition=definition,
                classification=classification,
                included_in_group=(
                    definition.group_name in {"zic", "zir"}
                    and not classification.nullable
                ),
                metadata=metadata,
            )
        )

    # Preserve optional oracle PHVAC separately as auxiliary provenance.
    if "predicted_PHVAC_oracle" in aligned.columns:
        oracle = pd.to_numeric(aligned["predicted_PHVAC_oracle"], errors="coerce")
        if oracle.isna().any():
            raise PhaseDAssemblyError(
                "predicted_PHVAC_oracle contains missing values"
            )
        table["phvac_oracle"] = oracle

    zic, zic_active = _safe_group_sum(
        table,
        group_components("zic"),
        group_name="zic",
    )
    zir, zir_active = _safe_group_sum(
        table,
        group_components(
            "zir",
            include_visible_lighting_in_zir=config.include_visible_lighting_in_zir,
        ),
        group_name="zir",
    )
    table["zic"] = zic
    table["zir"] = zir

    for passthrough in (
        "split",
        "split_index",
        "included",
        "exclusion_reason",
        "source_row_index",
    ):
        if passthrough in aligned.columns:
            table[passthrough] = aligned[passthrough]

    statuses = [record.phase_d_status for record in records]
    diagnostics = AssemblyDiagnostics(
        row_count=len(table),
        canonical_column_count=len(table.columns),
        active_phase_c_signal_count=sum(not record.nullable for record in records),
        nullable_complete_zero_count=statuses.count(
            PhaseDSignalStatus.NULLABLE_COMPLETE_ZERO
        ),
        nullable_not_applicable_count=statuses.count(
            PhaseDSignalStatus.NULLABLE_NOT_APPLICABLE
        ),
        constant_nonzero_count=statuses.count(
            PhaseDSignalStatus.CONSTANT_NONZERO
        ),
        varying_count=statuses.count(PhaseDSignalStatus.VARYING),
        validation_failure_count=statuses.count(
            PhaseDSignalStatus.VALIDATION_FAILURE
        ),
        zic_active_components=zic_active,
        zir_active_components=zir_active,
    )
    return AssemblyResult(
        table=table.reset_index(drop=True),
        signal_records=tuple(records),
        diagnostics=diagnostics,
    )
