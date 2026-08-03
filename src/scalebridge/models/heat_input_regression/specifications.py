# -*- coding: utf-8 -*-
"""Immutable specifications for Stage C heat-input regression relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Component = Literal[
    "solar",
    "convective",
    "radiant",
    "visible",
    "hvac",
    "hvac_power",
]
PredictorKind = Literal[
    "ghi",
    "corrected_schedule",
    "hvac_thermodynamic",
    "hvac_power_from_qhvac",
]


@dataclass(frozen=True)
class HeatInputModelSpecification:
    """Describe one physical regression relationship independently of model form."""

    model_id: str
    display_name: str
    source_family: str
    component: Component
    predictor_kind: PredictorKind
    predictor_semantic_names: tuple[str, ...]
    target_semantic_name: str
    output_prediction_column: str
    supported_predictor_methods: tuple[str, ...] = ()
    default_predictor_method: str = ""
    expected_predictor_units: str = ""
    expected_target_units: str = "W"

    # Authoritative physical/model policy propagated through C4-C8.
    fit_intercept: bool = False
    input_transform: str = "identity"
    model_role: str = "heat_input_component"
    dependency_model_id: str = ""
    target_allocation: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "source_family": self.source_family,
            "component": self.component,
            "predictor_kind": self.predictor_kind,
            "predictor_semantic_names": list(self.predictor_semantic_names),
            "target_semantic_name": self.target_semantic_name,
            "output_prediction_column": self.output_prediction_column,
            "supported_predictor_methods": list(self.supported_predictor_methods),
            "default_predictor_method": self.default_predictor_method,
            "expected_predictor_units": self.expected_predictor_units,
            "expected_target_units": self.expected_target_units,
            "fit_intercept": self.fit_intercept,
            "input_transform": self.input_transform,
            "model_role": self.model_role,
            "dependency_model_id": self.dependency_model_id,
            "target_allocation": self.target_allocation,
        }
