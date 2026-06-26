"""Parse EnergyPlus EIO records into stable structured metadata.

EnergyPlus EIO files contain repeated table headers beginning with ``!`` and
data records beginning with a category token such as ``<Zone Information>``.
The legacy generator repeatedly rescanned the complete file for every header.
This parser performs one CSV-aware pass, preserves all values as emitted, and
can optionally recreate the legacy dictionary of pandas DataFrames.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class EioParsingError(RuntimeError):
    """Raised when an EIO file is absent or structurally inconsistent."""


@dataclass(frozen=True)
class EioTable:
    """One EIO category with its declared columns and emitted data rows."""

    category: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_serializable_dict(self) -> dict[str, object]:
        """Return a lossless JSON-compatible representation."""
        return {
            "category": self.category,
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the table to the DataFrame shape used by legacy workflows."""
        width = len(self.columns)
        normalized_rows = [
            (*row[:width], *("" for _ in range(max(0, width - len(row)))))
            for row in self.rows
        ]
        return pd.DataFrame(normalized_rows, columns=self.columns)


def parse_eio(path: str | Path) -> dict[str, EioTable]:
    """Parse an EnergyPlus EIO file into category-keyed tables."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise EioParsingError(f"EIO file does not exist: {source}")

    headers: dict[str, tuple[str, ...]] = {}
    rows: dict[str, list[tuple[str, ...]]] = {}

    with source.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for parsed_row in csv.reader(stream):
            fields = tuple(field.strip() for field in parsed_row)
            if not fields or not fields[0]:
                continue

            first = fields[0]
            if first.startswith("!"):
                category = _extract_category(first[1:].strip())
                if category is not None:
                    headers[category] = fields[1:]
                    rows.setdefault(category, [])
                continue

            category = _extract_category(first) or first
            if category in headers:
                rows[category].append(fields[1:])

    return {
        category: _normalize_table(
            category=category,
            columns=columns,
            rows=rows.get(category, []),
        )
        for category, columns in headers.items()
    }


def _extract_category(value: str) -> str | None:
    """Extract the text enclosed by the first angle-bracket pair."""
    start = value.find("<")
    end = value.find(">", start + 1)
    if start < 0 or end < 0:
        return None
    return value[start + 1 : end].strip()


def _normalize_table(
    *,
    category: str,
    columns: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> EioTable:
    """Normalize row widths without discarding non-empty EIO values."""
    trimmed_rows = [_trim_empty_overflow(row, len(columns)) for row in rows]
    normalized_width = max(
        [len(columns), *(len(row) for row in trimmed_rows)],
    )
    extra_count = normalized_width - len(columns)
    normalized_columns = columns + tuple(
        f"Undeclared Field {index}"
        for index in range(1, extra_count + 1)
    )
    normalized_rows = tuple(
        row + ("",) * (normalized_width - len(row))
        for row in trimmed_rows
    )
    return EioTable(
        category=category,
        columns=normalized_columns,
        rows=normalized_rows,
    )


def _trim_empty_overflow(
    row: tuple[str, ...],
    declared_width: int,
) -> tuple[str, ...]:
    """Remove only trailing empty fields beyond the declared table width."""
    values = list(row)
    while len(values) > declared_width and values[-1] == "":
        values.pop()
    return tuple(values)
