"""Unit tests for read-only Generation campaign profiles."""
from pathlib import Path
from scalebridge.dashapp.services.generation.profiles import KNOWN_GENERATION_PROFILES
from scalebridge.dashapp.validation.campaigns.generation import classify_generation_campaign

def test_known_generation_profiles_are_authoritative():
    by_id={p.role_id:p for p in KNOWN_GENERATION_PROFILES}
    assert by_id["testing_generation_dataset"].expected_parquet_count==29
    assert by_id["full_paper_p1_generation_dataset"].expected_case_count==16
    assert by_id["full_paper_p1_generation_dataset"].expected_parquet_count==440

def test_complete_testing_profile_classification():
    profile=KNOWN_GENERATION_PROFILES[0]
    status,_=classify_generation_campaign(profile,{"exists":True,"detected_case_count":1,"latest_run_count":1,"rdd_manifest_count":1,"parquet_count":29,"pickle_count":29,"traceback_count":0})
    assert status=="complete"

def test_missing_campaign_is_not_found():
    profile=KNOWN_GENERATION_PROFILES[0]
    status,_=classify_generation_campaign(profile,{"exists":False,"detected_case_count":0,"latest_run_count":0,"rdd_manifest_count":0,"parquet_count":0,"pickle_count":0,"traceback_count":0})
    assert status=="not_found"
