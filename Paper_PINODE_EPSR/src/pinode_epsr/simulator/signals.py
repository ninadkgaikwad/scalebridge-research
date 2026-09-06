from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class SignalSpec:
    alias: str
    variable_name: str
    key: str
    required: bool
    category: str
    units_hint: str
    description: str


@dataclass(frozen=True)
class ZoneRuntimeSpec:
    zone_token: str
    zone_name: str
    unitary_name: str
    fan_name: str
    heating_coil_name: str
    cooling_coil_name: str

    return_node: str
    mixed_node: str
    cool_out_node: str
    heat_out_node: str
    supply_outlet_node: str
    zone_supply_node: str


def environment_signal_specs() -> List[SignalSpec]:
    """
    Broad environment history profile.

    Only dry-bulb is required. Other quantities are optional and are recorded
    automatically when EnergyPlus exposes them for the current model.
    """
    return [
        SignalSpec(
            "outdoor_drybulb_c",
            "Site Outdoor Air Drybulb Temperature",
            "Environment",
            True,
            "environment",
            "C",
            "Outdoor dry-bulb temperature",
        ),
        SignalSpec(
            "outdoor_wetbulb_c",
            "Site Outdoor Air Wetbulb Temperature",
            "Environment",
            False,
            "environment",
            "C",
            "Outdoor wet-bulb temperature",
        ),
        SignalSpec(
            "outdoor_humidity_ratio",
            "Site Outdoor Air Humidity Ratio",
            "Environment",
            False,
            "environment",
            "kgWater/kgDryAir",
            "Outdoor humidity ratio",
        ),
        SignalSpec(
            "outdoor_relative_humidity_pct",
            "Site Outdoor Air Relative Humidity",
            "Environment",
            False,
            "environment",
            "%",
            "Outdoor relative humidity",
        ),
        SignalSpec(
            "outdoor_barometric_pressure_pa",
            "Site Outdoor Air Barometric Pressure",
            "Environment",
            False,
            "environment",
            "Pa",
            "Outdoor barometric pressure",
        ),
        SignalSpec(
            "wind_speed_m_s",
            "Site Wind Speed",
            "Environment",
            False,
            "environment",
            "m/s",
            "Site wind speed",
        ),
        SignalSpec(
            "direct_solar_w_m2",
            "Site Direct Solar Radiation Rate per Area",
            "Environment",
            False,
            "environment",
            "W/m2",
            "Direct solar radiation",
        ),
        SignalSpec(
            "diffuse_solar_w_m2",
            "Site Diffuse Solar Radiation Rate per Area",
            "Environment",
            False,
            "environment",
            "W/m2",
            "Diffuse solar radiation",
        ),
    ]


def zone_signal_specs(spec: ZoneRuntimeSpec) -> List[SignalSpec]:
    out: List[SignalSpec] = []

    # Zone thermodynamic/comfort signals.
    out.extend([
        SignalSpec(
            "zone_temperature_c",
            "Zone Air Temperature",
            spec.zone_name,
            True,
            "zone",
            "C",
            "Zone mean air temperature",
        ),
        SignalSpec(
            "zone_mean_radiant_temperature_c",
            "Zone Mean Radiant Temperature",
            spec.zone_name,
            False,
            "zone",
            "C",
            "Zone mean radiant temperature",
        ),
        SignalSpec(
            "zone_operative_temperature_c",
            "Zone Operative Temperature",
            spec.zone_name,
            False,
            "zone",
            "C",
            "Zone operative temperature",
        ),
        SignalSpec(
            "zone_relative_humidity_pct",
            "Zone Air Relative Humidity",
            spec.zone_name,
            False,
            "zone",
            "%",
            "Zone air relative humidity",
        ),
        SignalSpec(
            "zone_heating_setpoint_c",
            "Zone Thermostat Heating Setpoint Temperature",
            spec.zone_name,
            False,
            "zone",
            "C",
            "Current zone heating setpoint",
        ),
        SignalSpec(
            "zone_cooling_setpoint_c",
            "Zone Thermostat Cooling Setpoint Temperature",
            spec.zone_name,
            False,
            "zone",
            "C",
            "Current zone cooling setpoint",
        ),
        SignalSpec(
            "zone_people_total_heating_rate_w",
            "Zone People Total Heating Rate",
            spec.zone_name,
            False,
            "zone_gains",
            "W",
            "Total people heat gain",
        ),
        SignalSpec(
            "zone_lights_total_heating_rate_w",
            "Zone Lights Total Heating Rate",
            spec.zone_name,
            False,
            "zone_gains",
            "W",
            "Total lighting heat gain",
        ),
        SignalSpec(
            "zone_electric_equipment_total_heating_rate_w",
            "Zone Electric Equipment Total Heating Rate",
            spec.zone_name,
            False,
            "zone_gains",
            "W",
            "Total electric equipment heat gain",
        ),
        SignalSpec(
            "zone_air_system_sensible_heating_rate_w",
            "Zone Air System Sensible Heating Rate",
            spec.zone_name,
            False,
            "zone_hvac",
            "W",
            "Zone air-system sensible heating rate",
        ),
        SignalSpec(
            "zone_air_system_sensible_cooling_rate_w",
            "Zone Air System Sensible Cooling Rate",
            spec.zone_name,
            False,
            "zone_hvac",
            "W",
            "Zone air-system sensible cooling rate",
        ),
    ])

    # Full known PSZ air path from the actuator forensics.
    nodes = [
        ("return", spec.return_node),
        ("mixed", spec.mixed_node),
        ("cool_out", spec.cool_out_node),
        ("heat_out", spec.heat_out_node),
        ("supply_outlet", spec.supply_outlet_node),
        ("zone_supply", spec.zone_supply_node),
    ]

    for prefix, node in nodes:
        out.extend([
            SignalSpec(
                f"{prefix}_temperature_c",
                "System Node Temperature",
                node,
                True,
                "air_path_node",
                "C",
                f"{prefix} node temperature",
            ),
            SignalSpec(
                f"{prefix}_mass_flow_kg_s",
                "System Node Mass Flow Rate",
                node,
                True,
                "air_path_node",
                "kg/s",
                f"{prefix} node air mass flow",
            ),
            SignalSpec(
                f"{prefix}_humidity_ratio",
                "System Node Humidity Ratio",
                node,
                True,
                "air_path_node",
                "kgWater/kgDryAir",
                f"{prefix} node humidity ratio",
            ),
            SignalSpec(
                f"{prefix}_pressure_pa",
                "System Node Pressure",
                node,
                False,
                "air_path_node",
                "Pa",
                f"{prefix} node pressure",
            ),
        ])

    # HVAC component/intermediate signals.
    out.extend([
        SignalSpec(
            "fan_mass_flow_kg_s",
            "Fan Air Mass Flow Rate",
            spec.fan_name,
            True,
            "fan",
            "kg/s",
            "Fan air mass flow rate",
        ),
        SignalSpec(
            "fan_electric_power_w",
            "Fan Electricity Rate",
            spec.fan_name,
            True,
            "fan",
            "W",
            "Fan electrical power",
        ),
        SignalSpec(
            "heating_coil_rate_w",
            "Heating Coil Heating Rate",
            spec.heating_coil_name,
            True,
            "heating_coil",
            "W",
            "Heating coil thermal rate",
        ),
        SignalSpec(
            "cooling_coil_total_rate_w",
            "Cooling Coil Total Cooling Rate",
            spec.cooling_coil_name,
            True,
            "cooling_coil",
            "W",
            "DX cooling coil total cooling rate",
        ),
        SignalSpec(
            "cooling_coil_sensible_rate_w",
            "Cooling Coil Sensible Cooling Rate",
            spec.cooling_coil_name,
            False,
            "cooling_coil",
            "W",
            "DX cooling coil sensible cooling rate",
        ),
        SignalSpec(
            "unitary_part_load_ratio",
            "Unitary System Part Load Ratio",
            spec.unitary_name,
            False,
            "unitary",
            "1",
            "Unitary system part-load ratio",
        ),
        SignalSpec(
            "unitary_fan_part_load_ratio",
            "Unitary System Fan Part Load Ratio",
            spec.unitary_name,
            False,
            "unitary",
            "1",
            "Unitary fan part-load ratio",
        ),
        SignalSpec(
            "dx_speed_ratio",
            "Unitary System DX Coil Speed Ratio",
            spec.unitary_name,
            False,
            "unitary",
            "1",
            "DX coil speed ratio",
        ),
        SignalSpec(
            "dx_cycling_ratio",
            "Unitary System DX Coil Cycling Ratio",
            spec.unitary_name,
            False,
            "unitary",
            "1",
            "DX coil cycling ratio",
        ),
    ])

    return out
