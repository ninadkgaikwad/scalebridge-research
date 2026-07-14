# -*- coding: utf-8 -*-
"""Writers for ScaleBridge aggregation outputs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    if not fieldnames:
        fieldnames = ["note"]
        rows = [{"note": "no rows"}]

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dataframe_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_dataframe_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write dataframe to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def copy_file_if_exists(source: Path, destination: Path) -> None:
    """Copy file if source exists."""
    if not source.is_file():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def make_safe_name(value: str) -> str:
    """Return filesystem-safe name."""
    safe = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in value
    )
    return safe.strip("_") or "unnamed"