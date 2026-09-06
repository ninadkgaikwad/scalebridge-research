# -*- coding: utf-8 -*-
"""Portable BGIRS definition envelope for Phase D Thermal-Model Data."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_CAMPAIGN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$"
_SEASONS = {"winter", "spring", "summer", "fall"}
_ML_POLICIES = {
    "monthly_distributed_holdout",
    "chronological_holdout",
    "seasonal_holdout",
}
_OB_POLICIES = {
    "seasonal_distributed",
    "seasonal_block_holdout",
    "contiguous_identification",
    "custom_datetime_ranges",
}


class PhaseDRunnerConfig(BaseModel):
    """Typed configuration that maps one-to-one to the general Phase D runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_root: str = Field(min_length=1)
    output_root: str | None = None
    matrix_run_id: str = Field(min_length=1)
    phase_c_campaign_run_id: str = Field(min_length=1)

    aggregation_ids: tuple[str, ...] = ()
    weight_modes: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    max_aggregation_runs: int | None = Field(default=None, ge=1)

    phase_d_calendar_year: int = 2001
    heat_representation: Literal["grouped", "components"] = "grouped"
    qzivr_separate: bool = False

    ml_policies: tuple[str, ...] = ("monthly_distributed_holdout",)
    ml_input_lags: tuple[int, ...] = (12,)
    ml_target_horizons: tuple[int, ...] = (6,)
    ml_train_fraction: float = 0.70
    ml_test_fraction: float = 0.15
    ml_validation_fraction: float = 0.15
    ml_sh_train_seasons: tuple[str, ...] = ("winter", "spring")
    ml_sh_test_seasons: tuple[str, ...] = ("summer",)
    ml_sh_validation_seasons: tuple[str, ...] = ("fall",)

    ob_policies: tuple[str, ...] = ("seasonal_distributed",)
    sd_season_offset_days: int = Field(default=0, ge=0)
    sd_train_days: int = Field(default=21, ge=1)
    sd_test_days: int = Field(default=7, ge=1)
    sbh_train_seasons: tuple[str, ...] = ("winter", "spring", "fall")
    sbh_test_seasons: tuple[str, ...] = ("summer",)
    ci_start_datetime: str | None = None
    ci_train_days: int = Field(default=21, ge=1)
    ci_test_days: int = Field(default=7, ge=1)
    cdr_train_ranges: tuple[str, ...] = ()
    cdr_test_ranges: tuple[str, ...] = ()
    parquet_compression: str = "zstd"

    mlflow_enabled: bool = False
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    mlflow_strict: bool = False

    @model_validator(mode="after")
    def _validate_runner_contract(self):
        for name, values in (
            ("aggregation_ids", self.aggregation_ids),
            ("weight_modes", self.weight_modes),
            ("case_ids", self.case_ids),
            ("ml_policies", self.ml_policies),
            ("ml_input_lags", self.ml_input_lags),
            ("ml_target_horizons", self.ml_target_horizons),
            ("ob_policies", self.ob_policies),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} cannot contain duplicates")

        if not self.ml_policies:
            raise ValueError("Select at least one ML/SciML policy")
        if not self.ob_policies:
            raise ValueError("Select at least one Optimization/Bayesian policy")
        if set(self.ml_policies) - _ML_POLICIES:
            raise ValueError("Unsupported ML/SciML policy selected")
        if set(self.ob_policies) - _OB_POLICIES:
            raise ValueError("Unsupported Optimization/Bayesian policy selected")
        if any(value < 1 for value in self.ml_input_lags):
            raise ValueError("Every ML input lag must be >= 1")
        if not self.ml_target_horizons:
            raise ValueError("Select at least one ML target horizon")
        if any(value < 1 for value in self.ml_target_horizons):
            raise ValueError("Every ML target horizon must be >= 1")

        if {"monthly_distributed_holdout", "chronological_holdout"} & set(self.ml_policies):
            fractions = (
                self.ml_train_fraction,
                self.ml_test_fraction,
                self.ml_validation_fraction,
            )
            if any(value <= 0.0 or value >= 1.0 for value in fractions):
                raise ValueError("ML train/test/validation fractions must each be between 0 and 1")
            if abs(sum(fractions) - 1.0) > 1.0e-12:
                raise ValueError("ML train/test/validation fractions must sum to 1")

        if "seasonal_holdout" in self.ml_policies:
            self._validate_seasons(
                self.ml_sh_train_seasons,
                self.ml_sh_test_seasons,
                self.ml_sh_validation_seasons,
                label="ML seasonal holdout",
            )
        if "seasonal_distributed" in self.ob_policies:
            if self.sd_season_offset_days + self.sd_train_days + self.sd_test_days > 90:
                raise ValueError("Seasonal Distributed offset + train days + test days must fit the shortest 90-day meteorological season")
        if "seasonal_block_holdout" in self.ob_policies:
            self._validate_seasons(
                self.sbh_train_seasons,
                self.sbh_test_seasons,
                label="Opt/Bayes seasonal block holdout",
            )

        if "custom_datetime_ranges" in self.ob_policies:
            if not self.cdr_train_ranges or not self.cdr_test_ranges:
                raise ValueError(
                    "Custom Datetime Ranges requires at least one train range and one test range"
                )
        if not self.parquet_compression.strip():
            raise ValueError("Parquet compression must not be empty")
        return self

    @staticmethod
    def _validate_seasons(*groups: tuple[str, ...], label: str) -> None:
        seen: set[str] = set()
        for group in groups:
            if not group:
                raise ValueError(f"{label} season groups must not be empty")
            normalized = [str(value).strip().lower() for value in group]
            invalid = sorted(set(normalized) - _SEASONS)
            if invalid:
                raise ValueError(f"{label} contains unsupported seasons: {invalid}")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{label} contains duplicate seasons")
            overlap = seen & set(normalized)
            if overlap:
                raise ValueError(f"{label} season groups overlap: {sorted(overlap)}")
            seen.update(normalized)


class PhaseDCampaignDefinition(BaseModel):
    """Saved BGIRS Phase D definition backed by the general campaign runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "0.1.0"
    phase_d_campaign_id: str = Field(pattern=_CAMPAIGN_ID_PATTERN)
    parent_generation_campaign_id: str = Field(pattern=_CAMPAIGN_ID_PATTERN)
    parent_phase_c_run_key: str = Field(min_length=3)
    machine_id: str = Field(min_length=1)
    display_name: str | None = None
    notes: str | None = None
    runner_config: PhaseDRunnerConfig

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
