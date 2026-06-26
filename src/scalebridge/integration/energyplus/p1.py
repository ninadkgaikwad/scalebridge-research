"""P1 annual EnergyPlus campaign definitions.

P1 uses all 16 ASHRAE 90.1-2013 commercial prototype building types across
four climate-specific prototype locations: Seattle (4C), Tucson (2B), Tampa
(2A), and Buffalo (5A). This produces 64 distinct annual cases.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from scalebridge.integration.energyplus.manifests.models import (
    CaseSpec,
    OutputVariableRequest,
    RunPeriod,
)
from scalebridge.integration.energyplus.prototypes import (
    resolve_external_data_root,
    resolve_generated_data_root,
    scan_pnnl_commercial_prototypes,
)


P1_CAMPAIGN_ID: Final = "p1_ashrae2013_one_zone_64"
P1_BUILDING_TYPES: Final = (
    "ApartmentHighRise",
    "ApartmentMidRise",
    "Hospital",
    "HotelLarge",
    "HotelSmall",
    "OfficeLarge",
    "OfficeMedium",
    "OfficeSmall",
    "OutPatientHealthCare",
    "RestaurantFastFood",
    "RestaurantSitDown",
    "RetailStandalone",
    "RetailStripmall",
    "SchoolPrimary",
    "SchoolSecondary",
    "Warehouse",
)
P1_CLIMATES: Final = {
    "Seattle": {"category": "cool_marine", "climate_zone": "4C"},
    "Tucson": {"category": "hot_dry", "climate_zone": "2B"},
    "Tampa": {"category": "hot_humid", "climate_zone": "2A"},
    "Buffalo": {"category": "cold", "climate_zone": "5A"},
}
P1_REQUIRED_VARIABLE_NAMES: Final = (
    "Schedule Value",
    "Facility Total HVAC Electric Demand Power",
    "Site Diffuse Solar Radiation Rate per Area",
    "Site Direct Solar Radiation Rate per Area",
    "Site Outdoor Air Drybulb Temperature",
    "Site Solar Altitude Angle",
    "Surface Inside Face Internal Gains Radiation Heat Gain Rate",
    "Surface Inside Face Lights Radiation Heat Gain Rate",
    "Surface Inside Face Solar Radiation Heat Gain Rate",
    "Surface Inside Face Temperature",
    "Zone Windows Total Transmitted Solar Radiation Rate",
    "Zone Air Temperature",
    "Zone People Convective Heating Rate",
    "Zone Lights Convective Heating Rate",
    "Zone Electric Equipment Convective Heating Rate",
    "Zone Gas Equipment Convective Heating Rate",
    "Zone Other Equipment Convective Heating Rate",
    "Zone Hot Water Equipment Convective Heating Rate",
    "Zone Steam Equipment Convective Heating Rate",
    "Zone People Radiant Heating Rate",
    "Zone Lights Radiant Heating Rate",
    "Zone Electric Equipment Radiant Heating Rate",
    "Zone Gas Equipment Radiant Heating Rate",
    "Zone Other Equipment Radiant Heating Rate",
    "Zone Hot Water Equipment Radiant Heating Rate",
    "Zone Steam Equipment Radiant Heating Rate",
    "Zone Lights Visible Radiation Heating Rate",
    "Zone Total Internal Convective Heating Rate",
    "Zone Total Internal Radiant Heating Rate",
    "Zone Total Internal Total Heating Rate",
    "Zone Total Internal Visible Radiation Heating Rate",
    "Zone Air System Sensible Cooling Rate",
    "Zone Air System Sensible Heating Rate",
    "System Node Temperature",
    "System Node Mass Flow Rate",
)


class P1CampaignError(RuntimeError):
    """Raised when the source inventory cannot produce the locked P1 campaign."""


@dataclass(frozen=True)
class P1CampaignManifestResult:
    """Paths and counts for a persisted P1 campaign specification."""

    campaign_id: str
    case_count: int
    json_path: Path
    csv_path: Path


def p1_output_variables() -> tuple[OutputVariableRequest, ...]:
    """Return the locked P1 EnergyPlus output-variable requests."""
    return tuple(
        OutputVariableRequest(
            variable_name=name,
            reporting_frequency="timestep",
            required=True,
            semantic_role=_semantic_role(name),
        )
        for name in P1_REQUIRED_VARIABLE_NAMES
    )


def build_p1_case_specs(
    *,
    external_data_root: str | Path | None = None,
    write_legacy_pickles: bool = False,
) -> tuple[CaseSpec, ...]:
    """Build and validate all 64 annual P1 case specifications."""
    data_root = resolve_external_data_root(external_data_root)
    records = scan_pnnl_commercial_prototypes(
        external_data_root=data_root,
        standard_year=2013,
    )
    selected = {
        (record.building_type, record.location): record
        for record in records
        if record.status == "eligible"
        and record.building_type in P1_BUILDING_TYPES
        and record.location in P1_CLIMATES
    }

    expected = {
        (building_type, location)
        for building_type in P1_BUILDING_TYPES
        for location in P1_CLIMATES
    }
    missing = sorted(expected - set(selected))
    if missing:
        raise P1CampaignError(f"P1 source cases are missing: {missing}")

    requests = p1_output_variables()
    cases: list[CaseSpec] = []
    for building_type in P1_BUILDING_TYPES:
        for location, climate in P1_CLIMATES.items():
            record = selected[(building_type, location)]
            if record.epw_relative_path is None or record.epw_sha256 is None:
                raise P1CampaignError(
                    f"P1 case has no validated EPW: {building_type}/{location}"
                )
            cases.append(
                CaseSpec(
                    case_name=f"p1_{building_type}_{location}_2013",
                    building_type=building_type,
                    prototype_standard="ASHRAE 90.1",
                    prototype_year="2013",
                    climate_zone=climate["climate_zone"],
                    weather_location=location,
                    idf_path=data_root / record.idf_relative_path,
                    epw_path=data_root / record.epw_relative_path,
                    idf_sha256=record.idf_sha256,
                    epw_sha256=record.epw_sha256,
                    run_period=RunPeriod(
                        start_month=1,
                        start_day=1,
                        end_month=12,
                        end_day=31,
                        calendar_year=2013,
                    ),
                    timestep_minutes=5,
                    output_variables=requests,
                    energyplus_version="9.0.1",
                    write_legacy_pickles=write_legacy_pickles,
                    preserve_raw_outputs=True,
                    tags={
                        "paper": "P1",
                        "campaign_id": P1_CAMPAIGN_ID,
                        "climate_category": climate["category"],
                    },
                )
            )
    return tuple(cases)


def write_p1_campaign_manifest(
    cases: tuple[CaseSpec, ...],
    *,
    generated_data_root: str | Path | None = None,
) -> P1CampaignManifestResult:
    """Persist portable JSON and CSV manifests before campaign execution."""
    root = resolve_generated_data_root(generated_data_root)
    destination = root / "campaigns" / P1_CAMPAIGN_ID
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "case_specs.json"
    csv_path = destination / "case_specs.csv"

    json_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "campaign_id": P1_CAMPAIGN_ID,
                "case_count": len(cases),
                "cases": [
                    {
                        "case_id": case.case_id,
                        **case.model_dump(
                            mode="json",
                            exclude_none=True,
                            exclude={"output_variables"},
                        ),
                        "output_variable_count": len(case.output_variables),
                    }
                    for case in cases
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "case_id",
            "case_name",
            "building_type",
            "climate_zone",
            "weather_location",
            "idf_path",
            "epw_path",
            "timestep_minutes",
            "output_variable_count",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "case_name": case.case_name,
                    "building_type": case.building_type,
                    "climate_zone": case.climate_zone,
                    "weather_location": case.weather_location,
                    "idf_path": case.idf_path,
                    "epw_path": case.epw_path,
                    "timestep_minutes": case.timestep_minutes,
                    "output_variable_count": len(case.output_variables),
                }
            )
    return P1CampaignManifestResult(
        campaign_id=P1_CAMPAIGN_ID,
        case_count=len(cases),
        json_path=json_path,
        csv_path=csv_path,
    )


def _semantic_role(name: str) -> str:
    """Assign a stable broad P1 role without altering EnergyPlus names."""
    lowered = name.casefold()
    if "temperature" in lowered:
        return "temperature"
    if "solar" in lowered:
        return "solar"
    if "mass flow" in lowered:
        return "mass_flow"
    if "schedule" in lowered:
        return "schedule"
    if "hvac electric" in lowered:
        return "hvac_electric_power"
    if "cooling rate" in lowered:
        return "hvac_cooling_rate"
    if "heating rate" in lowered or "heat gain rate" in lowered:
        return "heat_gain_rate"
    return "other"
