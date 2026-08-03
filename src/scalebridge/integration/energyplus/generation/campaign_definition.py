"""Portable definitions for user-built Generation campaigns."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class SourceFileRef(BaseModel):
    model_config=ConfigDict(extra='forbid', frozen=True)
    source_id: str
    name: str
    path: str
    sha256: str
    building_type: str | None=None
    source_location: str | None=None
    standard_year: int | None=None

class GenerationCampaignDefinition(BaseModel):
    model_config=ConfigDict(extra='forbid', frozen=True)
    schema_version: str='0.2.0'
    campaign_id: str=Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$')
    source_mode: Literal['ashrae_library','uploaded_zip']
    machine_id: str=Field(min_length=1)
    ashrae_year: int | None=None
    buildings: tuple[SourceFileRef,...]=Field(min_length=1)
    weather_files: tuple[SourceFileRef,...]=Field(min_length=1)
    signal_profile: str='p1_generation_fixed_v1'
    case_limit: int | None=Field(default=None, ge=1)
    variable_limit: int | None=Field(default=None, ge=1)
    parallel_variable_workers: int=Field(default=1, ge=1)
    write_legacy_pickles: bool=True
    rerun_completed: bool=False
    generated_data_root: str | None=None
    mlflow_enabled: bool=True
    mlflow_tracking_uri: str='http://127.0.0.1:5000'
    mlflow_experiment_name: str | None=None
    mlflow_strict: bool=False

    @model_validator(mode='after')
    def validate_source(self):
        if self.source_mode=='ashrae_library' and self.ashrae_year is None:
            raise ValueError('ashrae_year is required for ashrae_library')
        return self

    @property
    def case_count(self):
        n=len(self.buildings)*len(self.weather_files)
        return min(n,self.case_limit) if self.case_limit else n
