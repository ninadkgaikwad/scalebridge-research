"""Known Generation dataset roles and read-only campaign discovery."""
from __future__ import annotations
from pathlib import Path
from scalebridge.dashapp.adapters.generation import discover_generation_campaign_ids, scan_generation_campaign
from scalebridge.dashapp.schemas.pipeline import GenerationCampaignSummary, GenerationDatasetProfile
from scalebridge.dashapp.services.system.live_settings import generated_root
from scalebridge.dashapp.validation.campaigns import build_summary

TESTING_CAMPAIGN_ID="p1_ashrae2013_one_zone_compact_4b4c_labpc_test_1B_RDD_1W_v3"
FULL_P1_CAMPAIGN_ID="p1_compact_4b4c_labpc_1w_v1"

KNOWN_GENERATION_PROFILES=(
    GenerationDatasetProfile(
        role_id="testing_generation_dataset", label="Testing Generation Dataset",
        campaign_id=TESTING_CAMPAIGN_ID,
        description="Validated one-building, one-climate RDD-filtered Generation dataset: RestaurantFastFood/Buffalo.",
        intended_use="Testing, smoke checks, fixtures, parser validation, and first visualization development.",
        expected_case_count=1, expected_latest_run_count=1, expected_rdd_manifest_count=1,
        expected_parquet_count=29, expected_pickle_count=29, expected_traceback_count=0,
        buildings=("RestaurantFastFood",), climates=("Buffalo",),
    ),
    GenerationDatasetProfile(
        role_id="full_paper_p1_generation_dataset", label="Full Paper P1 Generation Dataset",
        campaign_id=FULL_P1_CAMPAIGN_ID,
        description="Validated compact P1 Generation campaign covering four buildings and four climates.",
        intended_use="Paper-scale upstream source for Aggregation and cross-building/cross-climate result inspection.",
        expected_case_count=16, expected_latest_run_count=16, expected_rdd_manifest_count=16,
        expected_parquet_count=440, expected_pickle_count=440, expected_traceback_count=0,
        expected_mlflow_run_count=16,
        buildings=("RestaurantFastFood","OfficeSmall","RetailStripmall","ApartmentMidRise"),
        climates=("Buffalo","Seattle","Tampa","Tucson"),
    ),
)

def campaigns_root() -> Path:
    return generated_root() / "campaigns"

def known_profile_rows() -> list[dict[str, object]]:
    rows=[]
    for p in KNOWN_GENERATION_PROFILES:
        rows.append({"dataset_role":p.label,"campaign_id":p.campaign_id,"buildings":len(p.buildings),"climates":len(p.climates),"expected_cases":p.expected_case_count,"expected_parquet":p.expected_parquet_count,"intended_use":p.intended_use})
    return rows

def inspect_known_campaigns() -> list[GenerationCampaignSummary]:
    root=campaigns_root()
    return [build_summary(p, root / p.campaign_id, scan_generation_campaign(root / p.campaign_id)) for p in KNOWN_GENERATION_PROFILES]

def discover_campaign_rows() -> list[dict[str, object]]:
    root=campaigns_root()
    known={p.campaign_id:p for p in KNOWN_GENERATION_PROFILES}
    ids=sorted(set(discover_generation_campaign_ids(root)) | set(known), key=str.lower)
    rows=[]
    for cid in ids:
        counts=scan_generation_campaign(root/cid)
        profile=known.get(cid)
        if profile:
            summary=build_summary(profile, root/cid, counts)
            role=profile.label; status=summary.validation_status
        else:
            role="Custom / Discovered Generation Campaign"
            status="discovered" if counts["exists"] else "not_found"
        rows.append({"campaign_id":cid,"dataset_role":role,"campaign_root":str(root/cid),"case_count":counts["detected_case_count"],"latest_run_count":counts["latest_run_count"],"rdd_manifest_count":counts["rdd_manifest_count"],"parquet_count":counts["parquet_count"],"pickle_count":counts["pickle_count"],"traceback_count":counts["traceback_count"],"validation_status":status})
    return rows
