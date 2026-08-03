# -*- coding: utf-8 -*-
"""Semantic-to-physical signal catalog for Stage B aggregation outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalDefinition:
    semantic_name: str
    canonical_column: str
    category: str
    expected_units: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def accepted_columns(self) -> tuple[str, ...]:
        return (self.canonical_column, *self.aliases)


_BASE = [
    SignalDefinition("site_direct_solar", "Site_Direct_Solar_Radiation_Rate_per_Area_", "solar_predictor", "W/m2"),
    SignalDefinition("site_diffuse_solar", "Site_Diffuse_Solar_Radiation_Rate_per_Area_", "solar_predictor", "W/m2"),
    SignalDefinition("solar_altitude_angle", "Site_Solar_Altitude_Angle_", "solar_predictor", "deg"),
    SignalDefinition("window_transmitted_solar", "Zone_Windows_Total_Transmitted_Solar_Radiation_Rate_", "solar_target", "W"),
    SignalDefinition("inside_face_solar_gain", "Surface_Inside_Face_Solar_Radiation_Heat_Gain_Rate_", "solar_target", "W"),
    SignalDefinition("zone_air_temperature", "Zone_Air_Temperature_", "hvac_predictor", "C"),
    SignalDefinition("system_node_temperature", "System_Node_Temperature_", "hvac_predictor", "C", ("System_Node_Temperature",)),
    SignalDefinition("system_node_mass_flow_rate", "System_Node_Mass_Flow_Rate", "hvac_predictor", "kg/s", ("System_Node_Mass_Flow_Rate_",)),
    SignalDefinition("zone_sensible_heating", "Zone_Air_System_Sensible_Heating_Rate_", "hvac_target_candidate", "W"),
    SignalDefinition("zone_sensible_cooling", "Zone_Air_System_Sensible_Cooling_Rate_", "hvac_target_candidate", "W"),
    SignalDefinition("facility_hvac_electric_demand", "Facility_Total_HVAC_Electric_Demand_Power_", "hvac_target_candidate", "W"),
]

_SOURCE_NAMES = {
    "People": "People", "Lights": "Lights",
    "ElectricEquipment": "Electric_Equipment", "GasEquipment": "Gas_Equipment",
    "OtherEquipment": "Other_Equipment", "HotWaterEquipment": "Hot_Water_Equipment",
    "SteamEquipment": "Steam_Equipment",
}

for source, target_token in _SOURCE_NAMES.items():
    key = source.lower()
    _BASE.extend([
        SignalDefinition(f"schedule_{key}", f"Schedule_Value_{source}", "schedule", "fraction"),
        SignalDefinition(f"target_convective_{key}", f"Zone_{target_token}_Convective_Heating_Rate_", "internal_gain_target", "W"),
        SignalDefinition(f"target_radiant_{key}", f"Zone_{target_token}_Radiant_Heating_Rate_", "internal_gain_target", "W"),
    ])
_BASE.append(SignalDefinition("target_visible_lights", "Zone_Lights_Visible_Radiation_Heating_Rate_", "internal_gain_target", "W"))

SIGNAL_DEFINITIONS = tuple(_BASE)
SIGNAL_BY_SEMANTIC = {item.semantic_name: item for item in SIGNAL_DEFINITIONS}


def get_signal_definition(semantic_name: str) -> SignalDefinition:
    try:
        return SIGNAL_BY_SEMANTIC[semantic_name]
    except KeyError as exc:
        raise KeyError(f"Unknown semantic signal: {semantic_name}") from exc


def resolve_present_column(semantic_name: str, available_columns: set[str]) -> str | None:
    definition = get_signal_definition(semantic_name)
    for column in definition.accepted_columns:
        if column in available_columns:
            return column
    return None
