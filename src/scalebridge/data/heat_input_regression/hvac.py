# -*- coding: utf-8 -*-
"""Deterministic QHVAC and PHVAC feature/target construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scalebridge.data.heat_input_regression.signal_catalog import resolve_present_column

AIR_HEAT_CAPACITY_KJ_PER_KG_K = 1.005
HVAC_PREDICTOR_OUTPUT_COLUMN = "derived_QHVAC_X_W"
HVAC_TARGET_OUTPUT_COLUMN = "derived_QHVAC_Y_W"
PHVAC_PREDICTOR_OUTPUT_COLUMN = "derived_PHVAC_X_abs_QHVAC_W"
PHVAC_TARGET_OUTPUT_COLUMN = "derived_PHVAC_Y_allocated_W"
SUPPORTED_HVAC_TARGET_METHODS = (
    "signed_zone_sensible",
    "absolute_zone_sensible",
)


def build_hvac_predictor(
    frame: pd.DataFrame,
    *,
    air_heat_capacity_kj_per_kg_k: float = AIR_HEAT_CAPACITY_KJ_PER_KG_K,
) -> tuple[pd.Series, dict[str, Any]]:
    """Build QHVAC_X = 1000 * c_a * mdot * (T_supply - T_zone)."""
    columns = {str(column) for column in frame.columns}
    mass_flow_column = resolve_present_column("system_node_mass_flow_rate", columns)
    supply_temperature_column = resolve_present_column("system_node_temperature", columns)
    zone_temperature_column = resolve_present_column("zone_air_temperature", columns)
    missing = [
        name
        for name, column in (
            ("system_node_mass_flow_rate", mass_flow_column),
            ("system_node_temperature", supply_temperature_column),
            ("zone_air_temperature", zone_temperature_column),
        )
        if column is None
    ]
    if missing:
        raise KeyError(f"Cannot build HVAC predictor; missing signals: {missing}")
    mass_flow = pd.to_numeric(frame[mass_flow_column], errors="coerce")
    supply_temperature = pd.to_numeric(frame[supply_temperature_column], errors="coerce")
    zone_temperature = pd.to_numeric(frame[zone_temperature_column], errors="coerce")
    predictor = 1000.0 * float(air_heat_capacity_kj_per_kg_k) * mass_flow * (
        supply_temperature - zone_temperature
    )
    predictor = pd.Series(predictor, index=frame.index, name=HVAC_PREDICTOR_OUTPUT_COLUMN, dtype="float64")
    return predictor, {
        "feature_name": HVAC_PREDICTOR_OUTPUT_COLUMN,
        "feature_family": "hvac",
        "formula": "1000 * c_a * mdot * (T_supply - T_zone)",
        "units": "W",
        "air_heat_capacity_kj_per_kg_k": float(air_heat_capacity_kj_per_kg_k),
        "source_columns": [mass_flow_column, supply_temperature_column, zone_temperature_column],
        "sign_convention": "positive when supply temperature exceeds zone temperature",
    }


def build_hvac_target(
    frame: pd.DataFrame,
    *,
    method: str = "signed_zone_sensible",
) -> tuple[pd.Series, dict[str, Any]]:
    """Build the zone sensible QHVAC target; facility power is not a QAC target."""
    if method not in SUPPORTED_HVAC_TARGET_METHODS:
        raise ValueError(f"Unsupported HVAC target method: {method}")
    columns = {str(column) for column in frame.columns}
    heating_column = resolve_present_column("zone_sensible_heating", columns)
    cooling_column = resolve_present_column("zone_sensible_cooling", columns)
    if heating_column is None or cooling_column is None:
        raise KeyError("Zone sensible heating and cooling targets are required")
    heating = pd.to_numeric(frame[heating_column], errors="coerce")
    cooling = pd.to_numeric(frame[cooling_column], errors="coerce")
    if method == "signed_zone_sensible":
        target = heating - cooling
        formula = "zone_sensible_heating - zone_sensible_cooling"
        sign_convention = "positive heating, negative cooling"
    else:
        target = heating.abs() + cooling.abs()
        formula = "abs(zone_sensible_heating) + abs(zone_sensible_cooling)"
        sign_convention = "nonnegative sensible magnitude"
    target = pd.Series(target, index=frame.index, name=HVAC_TARGET_OUTPUT_COLUMN, dtype="float64")
    return target, {
        "feature_name": HVAC_TARGET_OUTPUT_COLUMN,
        "feature_family": "hvac_target",
        "target_method": method,
        "formula": formula,
        "units": "W",
        "source_columns": [heating_column, cooling_column],
        "sign_convention": sign_convention,
    }


def build_phvac_features(
    frame: pd.DataFrame,
    *,
    qhvac_target: pd.Series,
    aggregate_zone_count: int,
) -> tuple[pd.Series, pd.Series, dict[str, Any], dict[str, Any]]:
    """Build x=abs(QHVAC) and y=P_HVAC,building/n for one aggregate zone."""
    if aggregate_zone_count <= 0:
        raise ValueError("aggregate_zone_count must be positive")
    columns = {str(column) for column in frame.columns}
    electric_column = resolve_present_column("facility_hvac_electric_demand", columns)
    if electric_column is None:
        raise KeyError("Facility HVAC electric-demand target is absent")
    predictor = pd.Series(
        pd.to_numeric(qhvac_target, errors="coerce").abs(),
        index=frame.index,
        name=PHVAC_PREDICTOR_OUTPUT_COLUMN,
        dtype="float64",
    )
    facility_power = pd.to_numeric(frame[electric_column], errors="coerce")
    target = pd.Series(
        facility_power / float(aggregate_zone_count),
        index=frame.index,
        name=PHVAC_TARGET_OUTPUT_COLUMN,
        dtype="float64",
    )
    predictor_metadata = {
        "feature_name": PHVAC_PREDICTOR_OUTPUT_COLUMN,
        "feature_family": "hvac_power_predictor",
        "formula": f"abs({HVAC_TARGET_OUTPUT_COLUMN})",
        "units": "W",
        "source_columns": [HVAC_TARGET_OUTPUT_COLUMN],
        "input_transform": "absolute_value",
        "dependency_model_id": "QAC",
    }
    target_metadata = {
        "feature_name": PHVAC_TARGET_OUTPUT_COLUMN,
        "feature_family": "hvac_power_target",
        "formula": f"{electric_column} / aggregate_zone_count",
        "units": "W",
        "source_columns": [electric_column],
        "aggregate_zone_count": int(aggregate_zone_count),
        "target_allocation": "equal_across_aggregate_zones",
        "building_reconstruction": "sum_across_aggregate_zones",
    }
    return predictor, target, predictor_metadata, target_metadata
