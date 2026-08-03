"""Source discovery for general EnergyPlus Generation campaigns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from scalebridge.integration.energyplus.prototypes import (
    PROTOTYPE_FILENAME_PATTERN,
    extract_idf_metadata,
    resolve_external_data_root,
    sha256_file,
)

SUPPORTED_ASHRAE_YEARS = (2013, 2016, 2019)
EPW_LOCATION_PATTERN = re.compile(r"^LOCATION,(?P<city>[^,]*),(?P<state>[^,]*),(?P<country>[^,]*)")

@dataclass(frozen=True)
class BuildingSource:
    source_id: str
    name: str
    building_type: str
    source_location: str | None
    standard_year: int | None
    idf_path: Path
    idf_sha256: str
    embedded_energyplus_version: str | None

@dataclass(frozen=True)
class WeatherSource:
    source_id: str
    name: str
    epw_path: Path
    epw_sha256: str
    city: str | None = None
    state: str | None = None
    country: str | None = None


def ashrae_prototype_root(*, standard_year: int, external_data_root=None) -> Path:
    if standard_year not in SUPPORTED_ASHRAE_YEARS:
        raise ValueError(f"Unsupported ASHRAE year: {standard_year}")
    return resolve_external_data_root(external_data_root) / "Commercial_Prototypes" / "ASHRAE" / f"90_1_{standard_year}"


def commercial_weather_root(*, external_data_root=None) -> Path:
    return resolve_external_data_root(external_data_root) / "TMY3_WeatherFiles_Commercial"


def discover_ashrae_buildings(*, standard_year: int, external_data_root=None) -> tuple[BuildingSource, ...]:
    root = ashrae_prototype_root(standard_year=standard_year, external_data_root=external_data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"ASHRAE prototype directory does not exist: {root}")
    rows=[]
    for path in sorted(root.rglob("*.idf"), key=lambda item: item.name.casefold()):
        match=PROTOTYPE_FILENAME_PATTERN.fullmatch(path.name)
        if match:
            building_type=match.group("building_type")
            source_location=match.group("location")
            file_year=int(match.group("standard_year"))
        else:
            building_type=path.stem
            source_location=None
            file_year=standard_year
        version,_=extract_idf_metadata(path)
        rows.append(BuildingSource(
            source_id=path.relative_to(root).as_posix(),
            name=path.stem,
            building_type=building_type,
            source_location=source_location,
            standard_year=file_year,
            idf_path=path.resolve(),
            idf_sha256=sha256_file(path),
            embedded_energyplus_version=version,
        ))
    return tuple(rows)


def _epw_metadata(path: Path):
    try:
        first=path.open('r', encoding='utf-8', errors='replace').readline().strip()
    except OSError:
        return None,None,None
    m=EPW_LOCATION_PATTERN.match(first)
    if not m: return None,None,None
    return tuple((m.group(k).strip() or None) for k in ('city','state','country'))


def discover_commercial_weather(*, external_data_root=None) -> tuple[WeatherSource, ...]:
    root=commercial_weather_root(external_data_root=external_data_root)
    if not root.is_dir(): raise FileNotFoundError(f"Commercial weather directory does not exist: {root}")
    rows=[]
    for path in sorted(root.rglob('*.epw'), key=lambda item:item.name.casefold()):
        city,state,country=_epw_metadata(path)
        rows.append(WeatherSource(
            source_id=path.relative_to(root).as_posix(), name=path.stem,
            epw_path=path.resolve(), epw_sha256=sha256_file(path),
            city=city,state=state,country=country,
        ))
    return tuple(rows)
