"""Backend boundary for loading, querying, modifying, and saving IDF models.

ScaleBridge keeps IDF preparation logic independent from a particular parser.
The ``IdfBackend`` protocol describes only the operations required by the
preparer. ``OpyplusIdfBackend`` implements that protocol with opyplus model,
table, query, record-update, and record-delete behavior.

The adapter imports opyplus lazily. Consequently, users can import ScaleBridge
and consume previously generated datasets without installing EnergyPlus IDF
tooling. Environments that generate simulations should install the optional
``scalebridge[energyplus]`` dependency set.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from scalebridge.integration.energyplus.idf.opyplus_compat import (
    install_opyplus_207_idd_compatibility,
)


class IdfBackendError(RuntimeError):
    """Base exception for IDF backend failures."""


class OpyplusNotInstalledError(IdfBackendError):
    """Raised when production IDF preparation is requested without opyplus."""


@runtime_checkable
class IdfBackend(Protocol):
    """Operations required by the backend-independent IDF preparer."""

    def load(self, source_path: Path) -> Any:
        """Load an IDF model from ``source_path``."""

    def save(self, model: Any, destination_path: Path) -> None:
        """Persist ``model`` as an IDF at ``destination_path``."""

    def list_records(self, model: Any, object_type: str) -> list[Any]:
        """Return every record for an EnergyPlus object type."""

    def find_named_record(
        self,
        model: Any,
        object_type: str,
        record_name: str,
    ) -> Any | None:
        """Find one case-insensitive named record or return ``None``."""

    def add_record(
        self,
        model: Any,
        object_type: str,
        fields: Mapping[str, Any],
    ) -> Any:
        """Add one EnergyPlus object record."""

    def update_record(self, record: Any, fields: Mapping[str, Any]) -> None:
        """Update fields on an existing record."""

    def delete_record(self, record: Any) -> None:
        """Delete an existing record."""


def normalize_object_type(object_type: str) -> str:
    """Convert EnergyPlus object names to opyplus table attribute names.

    Examples
    --------
    ``Schedule:Compact`` becomes ``Schedule_Compact`` and
    ``Output:Variable`` becomes ``Output_Variable``.
    """
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", object_type.strip())
    return normalized.strip("_")


def normalize_field_name(field_name: str) -> str:
    """Convert IDF labels to opyplus lowercase snake-case field names."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", field_name.strip())
    return normalized.strip("_").casefold()


class OpyplusIdfBackend:
    """Implement ``IdfBackend`` using opyplus EPM objects.

    Parameters
    ----------
    check_required:
        Whether opyplus should enforce required IDF fields while loading.
    check_length:
        Whether opyplus should enforce EnergyPlus field-length constraints.
    idd_or_version:
        Optional EnergyPlus version or IDD object passed to ``Epm.from_idf``.
    """

    def __init__(
        self,
        *,
        check_required: bool = True,
        check_length: bool = True,
        idd_or_version: str | Any | None = None,
    ) -> None:
        self._check_required = check_required
        self._check_length = check_length
        self._idd_or_version = idd_or_version

    def load(self, source_path: Path) -> Any:
        """Load an IDF through opyplus with the configured validation policy."""
        epm_class = self._get_epm_class()
        kwargs: dict[str, Any] = {
            "check_required": self._check_required,
            "check_length": self._check_length,
        }
        if self._idd_or_version is not None:
            kwargs["idd_or_version"] = self._idd_or_version

        try:
            return epm_class.from_idf(str(source_path), **kwargs)
        except Exception as exc:
            raise IdfBackendError(f"opyplus could not load IDF: {source_path}") from exc

    def save(self, model: Any, destination_path: Path) -> None:
        """Save an opyplus EPM model as an IDF."""
        try:
            model.save(str(destination_path))
        except Exception as exc:
            raise IdfBackendError(
                f"opyplus could not save prepared IDF: {destination_path}"
            ) from exc

    def list_records(self, model: Any, object_type: str) -> list[Any]:
        """Return records from the opyplus table for ``object_type``."""
        table = self._get_table(model, object_type)
        try:
            return list(table)
        except TypeError:
            # Older opyplus releases expose table records through ``all``.
            return list(table.all())

    def find_named_record(
        self,
        model: Any,
        object_type: str,
        record_name: str,
    ) -> Any | None:
        """Find one named opyplus record using case-insensitive comparison."""
        expected_name = record_name.casefold()
        matches = [
            record
            for record in self.list_records(model, object_type)
            if str(self._read_record_field(record, "name")).casefold() == expected_name
        ]

        if len(matches) > 1:
            raise IdfBackendError(
                f"multiple {object_type} records are named {record_name!r}"
            )
        return matches[0] if matches else None

    def add_record(
        self,
        model: Any,
        object_type: str,
        fields: Mapping[str, Any],
    ) -> Any:
        """Add one record through the public opyplus table API."""
        table = self._get_table(model, object_type)
        normalized_fields = {
            normalize_field_name(name): value for name, value in fields.items()
        }
        try:
            return table.add(**normalized_fields)
        except Exception as exc:
            raise IdfBackendError(
                f"could not add {object_type} record with fields {normalized_fields}"
            ) from exc

    def update_record(self, record: Any, fields: Mapping[str, Any]) -> None:
        """Update an opyplus record through its public update interface."""
        normalized_fields = {
            normalize_field_name(name): value for name, value in fields.items()
        }
        try:
            record.update(normalized_fields)
        except TypeError:
            # Some opyplus versions accept keyword arguments instead.
            try:
                record.update(**normalized_fields)
            except Exception as exc:
                raise IdfBackendError(
                    f"could not update IDF record with fields {normalized_fields}"
                ) from exc
        except Exception as exc:
            raise IdfBackendError(
                f"could not update IDF record with fields {normalized_fields}"
            ) from exc

    def delete_record(self, record: Any) -> None:
        """Delete one opyplus record."""
        try:
            record.delete()
        except Exception as exc:
            raise IdfBackendError("could not delete IDF record") from exc

    @staticmethod
    def _get_epm_class() -> Any:
        """Import and return ``opyplus.Epm`` only when generation is requested."""
        try:
            import opyplus as op
        except ImportError as exc:
            raise OpyplusNotInstalledError(
                "opyplus is required for IDF preparation. Install ScaleBridge "
                "with: python -m pip install -e '.[energyplus]'"
            ) from exc

        # Install the narrow process-local repair before Epm parses an IDD.
        install_opyplus_207_idd_compatibility()
        return op.Epm

    @staticmethod
    def _get_table(model: Any, object_type: str) -> Any:
        """Resolve an EnergyPlus object type to an opyplus model table."""
        table_name = normalize_object_type(object_type)
        try:
            return getattr(model, table_name)
        except AttributeError as exc:
            raise IdfBackendError(
                f"IDF model does not expose object type {object_type!r} "
                f"as table {table_name!r}"
            ) from exc

    @staticmethod
    def _read_record_field(record: Any, field_name: str) -> Any:
        """Read one record field across supported opyplus access styles."""
        normalized_name = normalize_field_name(field_name)
        try:
            return getattr(record, normalized_name)
        except AttributeError:
            try:
                return record[normalized_name]
            except Exception as exc:
                raise IdfBackendError(
                    f"IDF record does not expose field {normalized_name!r}"
                ) from exc
