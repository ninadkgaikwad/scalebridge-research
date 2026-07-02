"""
Pre-opyplus IDF normalization utilities.

Purpose
-------
Some DOE/PNNL prototype IDFs are accepted by EnergyPlus but fail earlier in
opyplus because opyplus validates object references while loading the IDF.

This module performs conservative text-level normalization before opyplus
loads the IDF. It does not modify the source IDF. It writes a normalized copy
and returns metadata describing what changed.

Current supported normalization
-------------------------------
1. If an IDF references Schedule Type Limits Name = "Control Type" but does
   not define ScheduleTypeLimits, Control Type, insert a valid definition.

Known case:
    ASHRAE901_ApartmentMidRise_STD2013_Seattle.idf

Observed opyplus error:
    FieldValidationError:
    No object found with any of given references:
    (('ScheduleTypeLimitsNames', 'control type'),).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class IdfNormalizationResult:
    """Result from normalizing an IDF before opyplus loads it."""

    source_idf_path: Path
    normalized_idf_path: Path
    changed: bool
    applied_patches: tuple[str, ...]


def normalize_idf_before_opyplus(
    source_idf_path: Path,
    normalized_idf_path: Path,
) -> IdfNormalizationResult:
    """Write a normalized copy of an IDF for opyplus loading.

    Parameters
    ----------
    source_idf_path:
        Original source IDF path. This file is never modified.
    normalized_idf_path:
        Destination path for the normalized copy.

    Returns
    -------
    IdfNormalizationResult
        Summary of whether the file changed and which patches were applied.
    """
    source_idf_path = Path(source_idf_path).expanduser().resolve()
    normalized_idf_path = Path(normalized_idf_path).expanduser().resolve()

    text = source_idf_path.read_text(encoding="utf-8", errors="ignore")

    applied_patches: list[str] = []

    text, changed = ensure_control_type_schedule_type_limits(text)
    if changed:
        applied_patches.append("inserted ScheduleTypeLimits: Control Type")

    normalized_idf_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_idf_path.write_text(text, encoding="utf-8")

    return IdfNormalizationResult(
        source_idf_path=source_idf_path,
        normalized_idf_path=normalized_idf_path,
        changed=bool(applied_patches),
        applied_patches=tuple(applied_patches),
    )


def ensure_control_type_schedule_type_limits(text: str) -> tuple[str, bool]:
    """Insert ScheduleTypeLimits, Control Type if referenced but missing.

    This fixes IDFs that contain a Schedule:Compact object such as:

        Schedule:Compact,
          ZONE CONTROL TYPE SCHED,
          Control Type,
          ...

    but do not define:

        ScheduleTypeLimits,
          Control Type,
          0,
          4,
          DISCRETE,
          Dimensionless;

    Returns
    -------
    tuple[str, bool]
        Updated text and whether a patch was inserted.
    """
    if not _references_control_type_schedule_limit(text):
        return text, False

    if _has_schedule_type_limits_control_type(text):
        return text, False

    insertion = """\
  ScheduleTypeLimits,
    Control Type,            !- Name
    0,                       !- Lower Limit Value
    4,                       !- Upper Limit Value
    DISCRETE,                !- Numeric Type
    Dimensionless;           !- Unit Type

"""

    insertion_position = _find_first_schedule_compact_position(text)
    if insertion_position is None:
        return insertion + text, True

    return text[:insertion_position] + insertion + text[insertion_position:], True


def _references_control_type_schedule_limit(text: str) -> bool:
    """Return True when an IDF references Schedule Type Limits = Control Type."""
    return bool(
        re.search(
            r"^\s*Control\s+Type\s*,",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )


def _has_schedule_type_limits_control_type(text: str) -> bool:
    """Return True when ScheduleTypeLimits named Control Type exists."""
    for block in _idf_object_blocks(text):
        clean_lines = [_strip_comment(line) for line in block.splitlines()]
        clean_lines = [line for line in clean_lines if line]

        if len(clean_lines) < 2:
            continue

        object_type = clean_lines[0].rstrip(",;").strip().lower()
        object_name = clean_lines[1].rstrip(",;").strip().lower()

        if object_type == "scheduletypeLimits".lower() and object_name == "control type":
            return True

    return False


def _find_first_schedule_compact_position(text: str) -> int | None:
    """Find the first Schedule:Compact object position for safe insertion."""
    match = re.search(
        r"^\s*Schedule:Compact\s*,",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return None
    return match.start()


def _idf_object_blocks(text: str):
    """Yield approximate IDF object blocks separated by semicolons.

    This is intentionally lightweight and robust enough for normalization checks.
    It does not attempt to fully parse IDF syntax.
    """
    current: list[str] = []

    for line in text.splitlines():
        current.append(line)

        if ";" in line:
            yield "\n".join(current)
            current = []

    if current:
        yield "\n".join(current)


def _strip_comment(line: str) -> str:
    """Remove IDF comments and surrounding whitespace."""
    return line.split("!")[0].strip()