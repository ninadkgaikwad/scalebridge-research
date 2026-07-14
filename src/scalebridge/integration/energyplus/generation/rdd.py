"""
EnergyPlus RDD parsing and requested-variable filtering utilities.

EnergyPlus writes an .rdd file when the IDF includes:

    Output:VariableDictionary,
        Regular;

The .rdd file lists the output variables that EnergyPlus can actually
produce for a specific model. This is important because the P1 variable
list is a maximum desired vocabulary, not every building/case can produce
every requested equipment-related variable.

Example .rdd rows:

    Var Type (reported time step),Var Report Type,Variable Name [Units]
    Zone,Average,Zone Air Temperature [C]
    Zone,Average,Zone Other Equipment Convective Heating Rate [W]
    HVAC,Average,Facility Total HVAC Electric Demand Power [W]

This module provides reusable parsing and matching logic so campaign
scripts do not need to know .rdd formatting details.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_UNIT_SUFFIX_RE = re.compile(r"\s*\[[^\]]*\]\s*$")
_UNIT_CAPTURE_RE = re.compile(r"\s*\[([^\]]*)\]\s*$")


@dataclass(frozen=True)
class RddVariable:
    """
    One variable entry from an EnergyPlus eplusout.rdd file.
    """

    var_type: str
    report_type: str
    variable_name: str
    variable_name_normalized: str
    units: str | None


def normalize_energyplus_variable_name(name: str) -> str:
    """
    Normalize an EnergyPlus report variable name for robust matching.

    This removes the trailing unit suffix, compresses whitespace, strips
    leading/trailing whitespace, and case-folds the result.

    Examples
    --------
    "Zone Air Temperature [C]" -> "zone air temperature"
    "  ZONE AIR TEMPERATURE  " -> "zone air temperature"
    """
    text = str(name).strip()
    text = _UNIT_SUFFIX_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def split_variable_name_and_units(raw_name: str) -> tuple[str, str | None]:
    """
    Split an RDD variable field into variable name and units.

    Examples
    --------
    "Zone Air Temperature [C]" -> ("Zone Air Temperature", "C")
    "Schedule Value []" -> ("Schedule Value", "")
    "Some Variable" -> ("Some Variable", None)
    """
    text = str(raw_name).strip()
    match = _UNIT_CAPTURE_RE.search(text)

    if match is None:
        return text, None

    units = match.group(1).strip()
    variable_name = text[: match.start()].strip()
    return variable_name, units


def parse_rdd_file(rdd_path: Path | str) -> list[RddVariable]:
    """
    Parse an EnergyPlus eplusout.rdd file.

    Parameters
    ----------
    rdd_path:
        Path to an EnergyPlus eplusout.rdd file.

    Returns
    -------
    list[RddVariable]
        Parsed RDD variable records.

    Raises
    ------
    FileNotFoundError
        If the RDD file does not exist.
    """
    path = Path(rdd_path)

    if not path.exists():
        raise FileNotFoundError(f"RDD file does not exist: {path}")

    variables: list[RddVariable] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            first = row[0].strip()

            if not first:
                continue

            if first.startswith("Program Version"):
                continue

            if first.startswith("Var Type"):
                continue

            if len(row) < 3:
                continue

            var_type = row[0].strip()
            report_type = row[1].strip()
            raw_variable_name = row[2].strip()

            if not raw_variable_name:
                continue

            variable_name, units = split_variable_name_and_units(raw_variable_name)

            variables.append(
                RddVariable(
                    var_type=var_type,
                    report_type=report_type,
                    variable_name=variable_name,
                    variable_name_normalized=normalize_energyplus_variable_name(
                        variable_name
                    ),
                    units=units,
                )
            )

    return variables


def available_rdd_variable_names(rdd_path: Path | str) -> set[str]:
    """
    Return normalized variable names available in an RDD file.
    """
    return {
        variable.variable_name_normalized
        for variable in parse_rdd_file(rdd_path)
    }


def get_requested_variable_name(
    variable_spec: Any,
    *,
    variable_name_attr: str = "variable_name",
) -> str:
    """
    Extract the EnergyPlus variable name from a requested variable spec.

    Supports:
      - dict-like specs with key `variable_name`
      - object/dataclass specs with attribute `.variable_name`

    Parameters
    ----------
    variable_spec:
        Requested variable spec object or dict.
    variable_name_attr:
        Field/attribute name containing the EnergyPlus variable name.
    """
    if isinstance(variable_spec, dict):
        value = variable_spec.get(variable_name_attr)
    else:
        value = getattr(variable_spec, variable_name_attr, None)

    if value is None:
        raise AttributeError(
            "Could not extract requested EnergyPlus variable name from "
            f"{type(variable_spec).__name__}. Expected field/attribute "
            f"{variable_name_attr!r}."
        )

    return str(value)


def filter_requested_variables_by_rdd(
    requested_variables: list[Any] | tuple[Any, ...],
    rdd_path: Path | str,
    *,
    variable_name_attr: str = "variable_name",
) -> tuple[list[Any], list[Any]]:
    """
    Split requested variables into RDD-available and RDD-unavailable lists.

    Parameters
    ----------
    requested_variables:
        Maximum desired variable list for a case.
    rdd_path:
        Path to the case-specific eplusout.rdd file.
    variable_name_attr:
        Field/attribute name containing the EnergyPlus variable name.

    Returns
    -------
    tuple[list[Any], list[Any]]
        available_variables, unavailable_variables

    Notes
    -----
    This function preserves the original variable spec objects. It only
    determines whether their requested EnergyPlus variable names appear in
    the case-specific RDD file.
    """
    available_names = available_rdd_variable_names(rdd_path)

    available: list[Any] = []
    unavailable: list[Any] = []

    for variable in requested_variables:
        variable_name = get_requested_variable_name(
            variable,
            variable_name_attr=variable_name_attr,
        )

        normalized = normalize_energyplus_variable_name(variable_name)

        if normalized in available_names:
            available.append(variable)
        else:
            unavailable.append(variable)

    return available, unavailable


def rdd_variables_to_manifest_rows(
    variables: list[RddVariable] | tuple[RddVariable, ...],
) -> list[dict[str, str | None]]:
    """
    Convert parsed RDD variables into JSON/CSV-friendly rows.
    """
    return [
        {
            "var_type": variable.var_type,
            "report_type": variable.report_type,
            "variable_name": variable.variable_name,
            "variable_name_normalized": variable.variable_name_normalized,
            "units": variable.units,
        }
        for variable in variables
    ]