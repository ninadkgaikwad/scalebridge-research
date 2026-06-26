"""Prepare one simulation-ready IDF from a ScaleBridge ``CaseSpec``.

The preparer copies scientific intent from ``CaseSpec`` into an IDF model:

1. set the run period;
2. set the number of timesteps per hour;
3. apply ordered schedule mutations;
4. replace existing ``Output:Variable`` records with the requested set;
5. configure ``Output:VariableDictionary`` when requested; and
6. save a new prepared IDF without modifying the source file.

The class depends only on the ``IdfBackend`` protocol. Production uses the
opyplus adapter, while tests use a small in-memory backend. This isolates
third-party API details and keeps preparation behavior independently testable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scalebridge.integration.energyplus.idf.backend import (
    IdfBackend,
    IdfBackendError,
    OpyplusIdfBackend,
)
from scalebridge.integration.energyplus.manifests.models import (
    CaseSpec,
    OutputVariableRequest,
    ScheduleOperation,
)


class IdfPreparationError(RuntimeError):
    """Raised when a case specification cannot be applied to its source IDF."""


class PreparedIdfResult(BaseModel):
    """Summary of a successfully prepared EnergyPlus IDF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    source_idf_path: Path
    prepared_idf_path: Path
    timestep_per_hour: int = Field(gt=0)
    output_variable_count: int = Field(ge=0)
    schedule_operation_count: int = Field(ge=0)
    variable_dictionary_requested: bool


class IdfPreparer:
    """Apply a ``CaseSpec`` to an IDF model through an ``IdfBackend``."""

    def __init__(self, backend: IdfBackend | None = None) -> None:
        """Initialize with an injected backend or the default opyplus adapter."""
        self._backend = backend or OpyplusIdfBackend()

    def prepare(
        self,
        case_spec: CaseSpec,
        destination_path: str | Path,
    ) -> PreparedIdfResult:
        """Create a prepared IDF without modifying the source model.

        Parameters
        ----------
        case_spec:
            Validated simulation case containing run, schedule, and output
            requirements.
        destination_path:
            Path for the newly prepared IDF.

        Returns
        -------
        PreparedIdfResult
            Paths and counts describing the completed preparation.
        """
        # ------------------------------------------------------------------
        # Phase 1: Validate portable filesystem boundaries.
        # ------------------------------------------------------------------
        source_path = case_spec.idf_path.expanduser().resolve()
        prepared_path = Path(destination_path).expanduser().resolve()

        if not source_path.is_file():
            raise IdfPreparationError(f"source IDF does not exist: {source_path}")
        if prepared_path.suffix.casefold() != ".idf":
            raise IdfPreparationError(
                f"prepared IDF must use the .idf extension: {prepared_path}"
            )
        if source_path == prepared_path:
            raise IdfPreparationError("prepared IDF path must differ from source IDF")

        prepared_path.parent.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Phase 2: Load and apply deterministic scientific configuration.
        # ------------------------------------------------------------------
        try:
            model = self._backend.load(source_path)
            timestep_per_hour = self._apply_run_configuration(model, case_spec)
            self._apply_schedule_operations(model, case_spec.schedule_operations)
            self._replace_output_variables(model, case_spec.output_variables)
            self._configure_variable_dictionary(
                model,
                requested=case_spec.request_variable_dictionary,
            )
            self._backend.save(model, prepared_path)
        except IdfBackendError as exc:
            raise IdfPreparationError(
                f"could not prepare IDF for case {case_spec.case_id}: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Phase 3: Verify that preparation produced a concrete output file.
        # ------------------------------------------------------------------
        if not prepared_path.is_file():
            raise IdfPreparationError(
                f"IDF backend did not produce prepared file: {prepared_path}"
            )

        return PreparedIdfResult(
            case_id=case_spec.case_id,
            source_idf_path=source_path,
            prepared_idf_path=prepared_path,
            timestep_per_hour=timestep_per_hour,
            output_variable_count=len(case_spec.output_variables),
            schedule_operation_count=len(case_spec.schedule_operations),
            variable_dictionary_requested=case_spec.request_variable_dictionary,
        )

    def _apply_run_configuration(self, model: Any, case_spec: CaseSpec) -> int:
        """Apply run-period boundaries and timestep frequency."""
        run_period = self._require_single_record(model, "RunPeriod")
        timestep = self._require_single_record(model, "TimeStep")
        timestep_per_hour = 60 // case_spec.timestep_minutes

        run_period_fields = {
            "begin_month": case_spec.run_period.start_month,
            "begin_day_of_month": case_spec.run_period.start_day,
            "end_month": case_spec.run_period.end_month,
            "end_day_of_month": case_spec.run_period.end_day,
        }
        if case_spec.run_period.calendar_year is not None:
            start_date = date(
                case_spec.run_period.calendar_year,
                case_spec.run_period.start_month,
                case_spec.run_period.start_day,
            )
            run_period_fields.update(
                {
                    "begin_year": case_spec.run_period.calendar_year,
                    "end_year": case_spec.run_period.calendar_year,
                    "day_of_week_for_start_day": start_date.strftime("%A"),
                }
            )

        self._backend.update_record(
            run_period,
            run_period_fields,
        )
        self._backend.update_record(
            timestep,
            {"number_of_timesteps_per_hour": timestep_per_hour},
        )
        return timestep_per_hour

    def _apply_schedule_operations(
        self,
        model: Any,
        operations: tuple[ScheduleOperation, ...],
    ) -> None:
        """Apply schedule operations in their declared order."""
        for operation in operations:
            if operation.operation == "add":
                self._add_schedule(model, operation)
            elif operation.operation in {"replace_fields", "rename"}:
                self._update_schedule(model, operation)
            elif operation.operation == "delete":
                self._delete_schedule(model, operation)
            else:  # pragma: no cover - Pydantic prevents unsupported operations.
                raise IdfPreparationError(
                    f"unsupported schedule operation: {operation.operation}"
                )

    def _add_schedule(self, model: Any, operation: ScheduleOperation) -> None:
        """Add a schedule and reject duplicate names."""
        existing = self._backend.find_named_record(
            model,
            operation.object_type,
            operation.schedule_name,
        )
        if existing is not None:
            raise IdfPreparationError(
                f"cannot add existing {operation.object_type} "
                f"schedule {operation.schedule_name!r}"
            )

        fields = {"name": operation.schedule_name, **operation.fields}
        self._backend.add_record(model, operation.object_type, fields)

    def _update_schedule(self, model: Any, operation: ScheduleOperation) -> None:
        """Update or rename one required existing schedule."""
        record = self._require_named_record(
            model,
            operation.object_type,
            operation.schedule_name,
        )
        self._backend.update_record(record, operation.fields)

    def _delete_schedule(self, model: Any, operation: ScheduleOperation) -> None:
        """Delete one required existing schedule."""
        record = self._require_named_record(
            model,
            operation.object_type,
            operation.schedule_name,
        )
        self._backend.delete_record(record)

    def _replace_output_variables(
        self,
        model: Any,
        requests: tuple[OutputVariableRequest, ...],
    ) -> None:
        """Replace existing output-variable objects with all case requests."""
        self._delete_all_records(model, "Output:Variable")

        for request in requests:
            self._backend.add_record(
                model,
                "Output:Variable",
                {
                    "key_value": request.key_value,
                    "variable_name": request.variable_name,
                    "reporting_frequency": request.reporting_frequency,
                },
            )

    def _configure_variable_dictionary(self, model: Any, *, requested: bool) -> None:
        """Replace variable-dictionary objects according to case configuration."""
        self._delete_all_records(model, "Output:VariableDictionary")
        if requested:
            self._backend.add_record(
                model,
                "Output:VariableDictionary",
                {
                    "key_field": "regular",
                    "sort_option": "Name",
                },
            )

    def _delete_all_records(self, model: Any, object_type: str) -> None:
        """Delete every record for an EnergyPlus object type."""
        # Copy before deletion to avoid mutating a live table during iteration.
        for record in list(self._backend.list_records(model, object_type)):
            self._backend.delete_record(record)

    def _require_single_record(self, model: Any, object_type: str) -> Any:
        """Return exactly one record or raise a preparation error."""
        records = self._backend.list_records(model, object_type)
        if len(records) != 1:
            raise IdfPreparationError(
                f"expected exactly one {object_type} record, found {len(records)}"
            )
        return records[0]

    def _require_named_record(
        self,
        model: Any,
        object_type: str,
        record_name: str,
    ) -> Any:
        """Return one required named record or raise a preparation error."""
        record = self._backend.find_named_record(model, object_type, record_name)
        if record is None:
            raise IdfPreparationError(
                f"{object_type} schedule {record_name!r} was not found"
            )
        return record


def prepare_idf(
    case_spec: CaseSpec,
    destination_path: str | Path,
    *,
    backend: IdfBackend | None = None,
) -> PreparedIdfResult:
    """Prepare one IDF using the default or an explicitly supplied backend."""
    return IdfPreparer(backend=backend).prepare(case_spec, destination_path)
