"""Typed records for the BGIRS Phase A Generation workspace."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class GenerationDatasetProfile:
    role_id: str
    label: str
    campaign_id: str
    description: str
    intended_use: str
    expected_case_count: int | None = None
    expected_latest_run_count: int | None = None
    expected_rdd_manifest_count: int | None = None
    expected_parquet_count: int | None = None
    expected_pickle_count: int | None = None
    expected_traceback_count: int | None = None
    expected_mlflow_run_count: int | None = None
    buildings: tuple[str, ...] = ()
    climates: tuple[str, ...] = ()

@dataclass(frozen=True)
class GenerationCampaignSummary:
    campaign_id: str
    campaign_root: Path
    dataset_role: str
    label: str
    exists: bool
    detected_case_count: int = 0
    latest_run_count: int = 0
    rdd_manifest_count: int = 0
    parquet_count: int = 0
    pickle_count: int = 0
    traceback_count: int = 0
    validation_status: str = "unknown"
    messages: tuple[str, ...] = field(default_factory=tuple)

    def as_row(self) -> dict[str, object]:
        return {
            "dataset_role": self.label,
            "campaign_id": self.campaign_id,
            "case_count": self.detected_case_count,
            "latest_run_count": self.latest_run_count,
            "rdd_manifest_count": self.rdd_manifest_count,
            "parquet_count": self.parquet_count,
            "pickle_count": self.pickle_count,
            "traceback_count": self.traceback_count,
            "validation_status": self.validation_status,
        }
