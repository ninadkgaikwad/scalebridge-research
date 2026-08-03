"""Validation rules for known BGIRS Generation dataset roles."""
from __future__ import annotations
from scalebridge.dashapp.schemas.pipeline import GenerationCampaignSummary, GenerationDatasetProfile

def classify_generation_campaign(profile: GenerationDatasetProfile, counts: dict[str, int | bool]) -> tuple[str, tuple[str, ...]]:
    if not bool(counts.get("exists")):
        return "not_found", ("Campaign folder was not found on this machine.",)
    checks=(
        ("case count", profile.expected_case_count, int(counts["detected_case_count"])),
        ("latest_run.json count", profile.expected_latest_run_count, int(counts["latest_run_count"])),
        ("RDD manifest count", profile.expected_rdd_manifest_count, int(counts["rdd_manifest_count"])),
        ("parquet count", profile.expected_parquet_count, int(counts["parquet_count"])),
        ("pickle count", profile.expected_pickle_count, int(counts["pickle_count"])),
        ("traceback count", profile.expected_traceback_count, int(counts["traceback_count"])),
    )
    mismatches=[f"Expected {name} {expected}, detected {actual}." for name, expected, actual in checks if expected is not None and expected != actual]
    if mismatches:
        return "incomplete", tuple(mismatches)
    return "complete", ("All configured artifact-count checks match the known dataset profile.",)

def build_summary(profile: GenerationDatasetProfile, campaign_root, counts: dict[str, int | bool]) -> GenerationCampaignSummary:
    status, messages=classify_generation_campaign(profile, counts)
    return GenerationCampaignSummary(
        campaign_id=profile.campaign_id,
        campaign_root=campaign_root,
        dataset_role=profile.role_id,
        label=profile.label,
        exists=bool(counts["exists"]),
        detected_case_count=int(counts["detected_case_count"]),
        latest_run_count=int(counts["latest_run_count"]),
        rdd_manifest_count=int(counts["rdd_manifest_count"]),
        parquet_count=int(counts["parquet_count"]),
        pickle_count=int(counts["pickle_count"]),
        traceback_count=int(counts["traceback_count"]),
        validation_status=status,
        messages=messages,
    )
