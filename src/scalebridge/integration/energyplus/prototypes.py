"""Inventory PNNL commercial prototype IDFs and their matching EPW files.

This module treats the external ``Data`` directory as an immutable source
library. It discovers PNNL commercial prototype models, validates their naming
convention and embedded EnergyPlus version, maps U.S. prototype locations to
explicit weather filenames, calculates file hashes, and writes portable CSV
and JSON inventory artifacts.

The implementation deliberately avoids fuzzy weather-file matching. Prototype
location labels such as ``SanDiego`` do not always match EPW filename spelling,
so an explicit registry is used as the authoritative mapping.

No source IDF or EPW file is modified. Generated inventories are written below
``SCALEBRIDGE_GENERATED_DATA_ROOT/inventories`` by default.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final


EXTERNAL_DATA_ENV: Final = "SCALEBRIDGE_EXTERNAL_DATA_ROOT"
GENERATED_DATA_ENV: Final = "SCALEBRIDGE_GENERATED_DATA_ROOT"
SUPPORTED_ENERGYPLUS_VERSION: Final = "9.0"

# PNNL filenames encode the scientific model dimensions used by ScaleBridge.
PROTOTYPE_FILENAME_PATTERN: Final = re.compile(
    r"^ASHRAE901_(?P<building_type>[A-Za-z0-9]+)"
    r"_STD(?P<standard_year>\d{4})"
    r"_(?P<location>[A-Za-z0-9]+)\.idf$"
)

# The IDF version object and PNNL WeatherFile comment are read without parsing
# the complete model. This keeps inventory generation fast for hundreds of IDFs.
IDF_VERSION_PATTERN: Final = re.compile(
    r"^\s*Version\s*,\s*(?P<version>[0-9.]+)\s*;",
    flags=re.IGNORECASE | re.MULTILINE,
)
WEATHER_REFERENCE_PATTERN: Final = re.compile(
    r"^\s*!\s*WeatherFile\s*:\s*(?P<filename>[^\r\n]+?)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

# Explicit mappings replace fragile substring matching from the legacy scripts.
COMMERCIAL_TMY3_BY_LOCATION: Final[dict[str, str]] = {
    "Albuquerque": "USA_NM_Albuquerque.Intl.Sunport.723650_TMY3.epw",
    "Atlanta": "USA_GA_Atlanta-Hartsfield.Jackson.Intl.AP.722190_TMY3.epw",
    "Buffalo": "USA_NY_Buffalo.Niagara.Intl.AP.725280_TMY3.epw",
    "Denver": "USA_CO_Denver-Aurora-Buckley.AFB.724695_TMY3.epw",
    "ElPaso": "USA_TX_El.Paso.Intl.AP.722700_TMY3.epw",
    "Fairbanks": "USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw",
    "GreatFalls": "USA_MT_Great.Falls.Intl.AP.727750_TMY3.epw",
    "Honolulu": "USA_HI_Honolulu.Intl.AP.911820_TMY3.epw",
    "InternationalFalls": "USA_MN_International.Falls.Intl.AP.727470_TMY3.epw",
    "NewYork": "USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw",
    "PortAngeles": "USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw",
    "Rochester": "USA_MN_Rochester.Intl.AP.726440_TMY3.epw",
    "SanDiego": "USA_CA_San.Deigo-Brown.Field.Muni.AP.722904_TMY3.epw",
    "Seattle": "USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw",
    "Tampa": "USA_FL_Tampa-MacDill.AFB.747880_TMY3.epw",
    "Tucson": "USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw",
}


class PrototypeInventoryError(RuntimeError):
    """Raised when data roots or prototype inventory inputs are invalid."""


@dataclass(frozen=True)
class PrototypeInventoryRecord:
    """One prototype-to-weather inventory record.

    Paths are stored relative to the external data root so inventories remain
    portable across the laptop, compute PCs, and Kamiak.
    """

    building_type: str
    standard_year: int
    location: str
    idf_relative_path: str
    epw_relative_path: str | None
    idf_filename: str
    epw_filename: str | None
    embedded_energyplus_version: str | None
    referenced_weather_filename: str | None
    idf_sha256: str
    epw_sha256: str | None
    status: str
    issues: tuple[str, ...]

    def to_serializable_dict(self) -> dict[str, object]:
        """Return a JSON/CSV-compatible representation of this record."""
        payload = asdict(self)
        payload["issues"] = list(self.issues)
        return payload


@dataclass(frozen=True)
class PrototypeInventoryResult:
    """Paths and summary counts produced by one inventory scan."""

    external_data_root: Path
    prototype_root: Path
    weather_root: Path
    record_count: int
    status_counts: dict[str, int]
    csv_path: Path
    json_path: Path


def resolve_external_data_root(explicit_root: str | Path | None = None) -> Path:
    """Resolve and validate the immutable external data root.

    Parameters
    ----------
    explicit_root:
        Optional path overriding ``SCALEBRIDGE_EXTERNAL_DATA_ROOT``.
    """
    root_value = (
        explicit_root
        or os.environ.get(EXTERNAL_DATA_ENV)
        or _discover_repository_relative_data_root()
    )
    if not root_value:
        raise PrototypeInventoryError(
            f"{EXTERNAL_DATA_ENV} is not set and no external root was provided"
        )

    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise PrototypeInventoryError(f"external data root does not exist: {root}")
    return root


def resolve_generated_data_root(explicit_root: str | Path | None = None) -> Path:
    """Resolve and create the ScaleBridge generated-data root."""
    root_value = (
        explicit_root
        or os.environ.get(GENERATED_DATA_ENV)
        or resolve_external_data_root() / "ScaleBridge"
    )
    if not root_value:
        raise PrototypeInventoryError(
            f"{GENERATED_DATA_ENV} is not set and no generated root was provided"
        )

    root = Path(root_value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _discover_repository_relative_data_root() -> Path | None:
    """Discover the shared ``Data`` directory two levels above the repository."""
    repository_root = Path(__file__).resolve().parents[4]
    candidate = repository_root.parents[1] / "Data"
    return candidate if candidate.is_dir() else None


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_idf_metadata(idf_path: Path) -> tuple[str | None, str | None]:
    """Extract embedded EnergyPlus version and referenced weather filename."""
    text = idf_path.read_text(encoding="utf-8", errors="replace")
    version_match = IDF_VERSION_PATTERN.search(text)
    weather_match = WEATHER_REFERENCE_PATTERN.search(text)
    version = version_match.group("version").rstrip(".") if version_match else None
    weather = weather_match.group("filename").strip() if weather_match else None
    return version, weather


def scan_pnnl_commercial_prototypes(
    *,
    external_data_root: str | Path | None = None,
    standard_year: int = 2013,
    expected_energyplus_version: str = SUPPORTED_ENERGYPLUS_VERSION,
) -> list[PrototypeInventoryRecord]:
    """Scan one ASHRAE prototype year and validate its weather relationships.

    Parameters
    ----------
    external_data_root:
        Optional external data root override.
    standard_year:
        ASHRAE standard year encoded in the prototype directory and filenames.
    expected_energyplus_version:
        Required major/minor IDF version for campaign eligibility.
    """
    data_root = resolve_external_data_root(external_data_root)
    prototype_root = (
        data_root
        / "Commercial_Prototypes"
        / "ASHRAE"
        / f"90_1_{standard_year}"
    )
    weather_root = data_root / "TMY3_WeatherFiles_Commercial"

    if not prototype_root.is_dir():
        raise PrototypeInventoryError(
            f"prototype directory does not exist: {prototype_root}"
        )
    if not weather_root.is_dir():
        raise PrototypeInventoryError(f"weather directory does not exist: {weather_root}")

    records: list[PrototypeInventoryRecord] = []
    epw_hashes: dict[Path, str] = {}

    # ----------------------------------------------------------------------
    # Scan every IDF in deterministic filename order.
    # ----------------------------------------------------------------------
    for idf_path in sorted(prototype_root.glob("*.idf"), key=lambda path: path.name):
        filename_match = PROTOTYPE_FILENAME_PATTERN.fullmatch(idf_path.name)
        if filename_match is None:
            records.append(
                _invalid_filename_record(idf_path=idf_path, data_root=data_root)
            )
            continue

        building_type = filename_match.group("building_type")
        file_year = int(filename_match.group("standard_year"))
        location = filename_match.group("location")
        embedded_version, referenced_weather = extract_idf_metadata(idf_path)
        issues: list[str] = []

        # ------------------------------------------------------------------
        # Validate filename dimensions and EnergyPlus schema version.
        # ------------------------------------------------------------------
        if file_year != standard_year:
            issues.append(
                f"filename standard year {file_year} does not match {standard_year}"
            )
        if embedded_version is None:
            issues.append("missing EnergyPlus Version object")
        elif embedded_version != expected_energyplus_version:
            issues.append(
                f"EnergyPlus version {embedded_version} does not match "
                f"{expected_energyplus_version}"
            )

        # ------------------------------------------------------------------
        # Resolve weather using the explicit location registry.
        # ------------------------------------------------------------------
        expected_epw_filename = COMMERCIAL_TMY3_BY_LOCATION.get(location)
        epw_path = (
            weather_root / expected_epw_filename
            if expected_epw_filename is not None
            else None
        )

        if expected_epw_filename is None:
            issues.append(f"no commercial EPW registry entry for {location}")
        elif not epw_path.is_file():
            issues.append(f"registered EPW file is missing: {expected_epw_filename}")

        if referenced_weather is None:
            issues.append("missing PNNL WeatherFile comment")
        elif (
            expected_epw_filename is not None
            and referenced_weather.casefold() != expected_epw_filename.casefold()
        ):
            issues.append(
                f"IDF weather reference {referenced_weather!r} does not match "
                f"registry {expected_epw_filename!r}"
            )

        status = _classify_inventory_status(
            issues=issues,
            embedded_version=embedded_version,
            expected_version=expected_energyplus_version,
            expected_epw_filename=expected_epw_filename,
            epw_path=epw_path,
            referenced_weather=referenced_weather,
        )

        records.append(
            PrototypeInventoryRecord(
                building_type=building_type,
                standard_year=file_year,
                location=location,
                idf_relative_path=idf_path.relative_to(data_root).as_posix(),
                epw_relative_path=(
                    epw_path.relative_to(data_root).as_posix()
                    if epw_path is not None and epw_path.is_file()
                    else None
                ),
                idf_filename=idf_path.name,
                epw_filename=(
                    expected_epw_filename
                    if epw_path is not None and epw_path.is_file()
                    else None
                ),
                embedded_energyplus_version=embedded_version,
                referenced_weather_filename=referenced_weather,
                idf_sha256=sha256_file(idf_path),
                epw_sha256=_cached_file_hash(epw_path, epw_hashes),
                status=status,
                issues=tuple(issues),
            )
        )

    return records


def write_prototype_inventory(
    records: list[PrototypeInventoryRecord],
    *,
    external_data_root: str | Path | None = None,
    generated_data_root: str | Path | None = None,
    inventory_name: str = "pnnl_ashrae_2013_commercial",
    standard_year: int = 2013,
) -> PrototypeInventoryResult:
    """Write inventory records to CSV and JSON with a validation summary."""
    data_root = resolve_external_data_root(external_data_root)
    generated_root = resolve_generated_data_root(generated_data_root)
    inventory_root = generated_root / "inventories"
    inventory_root.mkdir(parents=True, exist_ok=True)

    csv_path = inventory_root / f"{inventory_name}.csv"
    json_path = inventory_root / f"{inventory_name}.json"
    status_counts = dict(sorted(Counter(record.status for record in records).items()))

    # ----------------------------------------------------------------------
    # Write a flat CSV suitable for manual inspection and spreadsheet use.
    # ----------------------------------------------------------------------
    fieldnames = [
        "building_type",
        "standard_year",
        "location",
        "idf_relative_path",
        "epw_relative_path",
        "idf_filename",
        "epw_filename",
        "embedded_energyplus_version",
        "referenced_weather_filename",
        "idf_sha256",
        "epw_sha256",
        "status",
        "issues",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record.to_serializable_dict()
            row["issues"] = " | ".join(record.issues)
            writer.writerow(row)

    # ----------------------------------------------------------------------
    # Write structured JSON with source roots, summary counts, and records.
    # ----------------------------------------------------------------------
    payload = {
        "schema_version": "0.1.0",
        "external_data_root": str(data_root),
        "record_count": len(records),
        "status_counts": status_counts,
        "records": [record.to_serializable_dict() for record in records],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    prototype_root = (
        data_root
        / "Commercial_Prototypes"
        / "ASHRAE"
        / f"90_1_{standard_year}"
    )
    weather_root = data_root / "TMY3_WeatherFiles_Commercial"

    return PrototypeInventoryResult(
        external_data_root=data_root,
        prototype_root=prototype_root,
        weather_root=weather_root,
        record_count=len(records),
        status_counts=status_counts,
        csv_path=csv_path,
        json_path=json_path,
    )


def build_and_write_pnnl_inventory(
    *,
    external_data_root: str | Path | None = None,
    generated_data_root: str | Path | None = None,
    standard_year: int = 2013,
) -> PrototypeInventoryResult:
    """Scan and persist one complete PNNL commercial prototype inventory."""
    records = scan_pnnl_commercial_prototypes(
        external_data_root=external_data_root,
        standard_year=standard_year,
    )
    return write_prototype_inventory(
        records,
        external_data_root=external_data_root,
        generated_data_root=generated_data_root,
        inventory_name=f"pnnl_ashrae_{standard_year}_commercial",
        standard_year=standard_year,
    )


def _invalid_filename_record(
    *,
    idf_path: Path,
    data_root: Path,
) -> PrototypeInventoryRecord:
    """Create an inventory record for an IDF outside the naming convention."""
    return PrototypeInventoryRecord(
        building_type="",
        standard_year=0,
        location="",
        idf_relative_path=idf_path.relative_to(data_root).as_posix(),
        epw_relative_path=None,
        idf_filename=idf_path.name,
        epw_filename=None,
        embedded_energyplus_version=None,
        referenced_weather_filename=None,
        idf_sha256=sha256_file(idf_path),
        epw_sha256=None,
        status="invalid_filename",
        issues=("filename does not match the PNNL ASHRAE prototype convention",),
    )


def _cached_file_hash(path: Path | None, cache: dict[Path, str]) -> str | None:
    """Return a cached SHA-256 digest for an existing file."""
    if path is None or not path.is_file():
        return None
    if path not in cache:
        cache[path] = sha256_file(path)
    return cache[path]


def _classify_inventory_status(
    *,
    issues: list[str],
    embedded_version: str | None,
    expected_version: str,
    expected_epw_filename: str | None,
    epw_path: Path | None,
    referenced_weather: str | None,
) -> str:
    """Classify the primary validation status for one inventory record."""
    if embedded_version != expected_version:
        return "version_mismatch"
    if expected_epw_filename is None or epw_path is None or not epw_path.is_file():
        return "missing_weather"
    if (
        referenced_weather is None
        or referenced_weather.casefold() != expected_epw_filename.casefold()
    ):
        return "weather_reference_mismatch"
    return "eligible" if not issues else "invalid"
