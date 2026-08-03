# -*- coding: utf-8 -*-
"""Authoritative registry of candidate Stage C heat-input regression models."""

from __future__ import annotations

from scalebridge.models.heat_input_regression.specifications import (
    HeatInputModelSpecification,
)

INTERNAL_SOURCES = (
    ("People", "P"),
    ("Lights", "L"),
    ("ElectricEquipment", "EE"),
    ("GasEquipment", "GE"),
    ("OtherEquipment", "OE"),
    ("HotWaterEquipment", "HWE"),
    ("SteamEquipment", "SE"),
)


def build_default_registry() -> tuple[HeatInputModelSpecification, ...]:
    specs: list[HeatInputModelSpecification] = [
        HeatInputModelSpecification(
            model_id="QSol1",
            display_name="Window transmitted solar",
            source_family="Solar",
            component="solar",
            predictor_kind="ghi",
            predictor_semantic_names=(
                "site_direct_solar",
                "site_diffuse_solar",
                "solar_altitude_angle",
            ),
            target_semantic_name="window_transmitted_solar",
            output_prediction_column="predicted_QSol1",
            expected_predictor_units="W/m2",
            fit_intercept=False,
        ),
        HeatInputModelSpecification(
            model_id="QSol2",
            display_name="Inside-face solar gain",
            source_family="Solar",
            component="solar",
            predictor_kind="ghi",
            predictor_semantic_names=(
                "site_direct_solar",
                "site_diffuse_solar",
                "solar_altitude_angle",
            ),
            target_semantic_name="inside_face_solar_gain",
            output_prediction_column="predicted_QSol2",
            expected_predictor_units="W/m2",
            fit_intercept=False,
        ),
    ]
    for source, suffix in INTERNAL_SOURCES:
        source_key = source.lower()
        specs.append(
            HeatInputModelSpecification(
                model_id=f"QZic_{suffix}",
                display_name=f"{source} convective gain",
                source_family=source,
                component="convective",
                predictor_kind="corrected_schedule",
                predictor_semantic_names=(
                    f"schedule_{source_key}",
                    f"static_level_{source_key}",
                ),
                target_semantic_name=f"target_convective_{source_key}",
                output_prediction_column=f"predicted_QZic_{suffix}",
                supported_predictor_methods=(
                    "aggregate_average",
                    "contribution_sum",
                ),
                default_predictor_method="aggregate_average",
                fit_intercept=False,
            )
        )
        specs.append(
            HeatInputModelSpecification(
                model_id=f"QZir_{suffix}",
                display_name=f"{source} radiant gain",
                source_family=source,
                component="radiant",
                predictor_kind="corrected_schedule",
                predictor_semantic_names=(
                    f"schedule_{source_key}",
                    f"static_level_{source_key}",
                ),
                target_semantic_name=f"target_radiant_{source_key}",
                output_prediction_column=f"predicted_QZir_{suffix}",
                supported_predictor_methods=(
                    "aggregate_average",
                    "contribution_sum",
                ),
                default_predictor_method="aggregate_average",
                fit_intercept=False,
            )
        )
    specs.append(
        HeatInputModelSpecification(
            model_id="QZivr_L",
            display_name="Lights visible gain",
            source_family="Lights",
            component="visible",
            predictor_kind="corrected_schedule",
            predictor_semantic_names=("schedule_lights", "static_level_lights"),
            target_semantic_name="target_visible_lights",
            output_prediction_column="predicted_QZivr_L",
            supported_predictor_methods=("aggregate_average", "contribution_sum"),
            default_predictor_method="aggregate_average",
            fit_intercept=False,
        )
    )
    specs.append(
        HeatInputModelSpecification(
            model_id="QAC",
            display_name="HVAC sensible heat input",
            source_family="HVAC",
            component="hvac",
            predictor_kind="hvac_thermodynamic",
            predictor_semantic_names=(
                "system_node_mass_flow_rate",
                "system_node_temperature",
                "zone_air_temperature",
            ),
            target_semantic_name="derived_signed_zone_sensible_hvac",
            output_prediction_column="predicted_QAC",
            expected_predictor_units="W",
            fit_intercept=False,
            input_transform="identity",
            model_role="hvac_thermal_delivery",
        )
    )
    specs.append(
        HeatInputModelSpecification(
            model_id="PHVAC",
            display_name="Allocated aggregate-zone HVAC electric power",
            source_family="HVAC",
            component="hvac_power",
            predictor_kind="hvac_power_from_qhvac",
            predictor_semantic_names=("derived_signed_zone_sensible_hvac",),
            target_semantic_name="facility_hvac_electric_demand",
            output_prediction_column="predicted_PHVAC",
            expected_predictor_units="W",
            fit_intercept=True,
            input_transform="absolute_value",
            model_role="hvac_electric_power",
            dependency_model_id="QAC",
            target_allocation="equal_across_aggregate_zones",
        )
    )
    return tuple(specs)


DEFAULT_MODEL_REGISTRY = build_default_registry()


def get_model_specification(model_id: str) -> HeatInputModelSpecification:
    for spec in DEFAULT_MODEL_REGISTRY:
        if spec.model_id == model_id:
            return spec
    raise KeyError(f"Unknown heat-input model_id: {model_id}")


def list_model_specifications() -> tuple[HeatInputModelSpecification, ...]:
    return DEFAULT_MODEL_REGISTRY
