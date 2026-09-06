"""Shared downstream lifecycle matching run_p1_compact_campaign.py."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json, os

from scalebridge.integration.energyplus.generation.variable_wise import generate_variable_wise_case
from scalebridge.integration.energyplus.idf.pre_opyplus_normalization import normalize_idf_before_opyplus
from scalebridge.integration.energyplus.generation.rdd import filter_requested_variables_by_rdd, get_requested_variable_name
from scalebridge.integration.energyplus.generation.rdd_probe import run_energyplus_rdd_probe

SUCCESS_STATUSES={'completed','completed_with_warnings'}

@dataclass(frozen=True)
class CampaignRunResult:
    campaign_id: str
    selected_count: int
    completed_count: int
    skipped_count: int
    failed_count: int
    return_code: int


def campaign_case_collection_name(campaign_id): return str(Path('campaigns')/campaign_id/'generation')
def case_generation_root(generated_data_root,case_collection_name,case_id): return Path(generated_data_root)/case_collection_name/'cases'/str(case_id)
def normalized_idf_path_for_case(generated_data_root,campaign_id,case_id): return Path(generated_data_root)/'campaigns'/campaign_id/'normalization'/'idfs'/str(case_id)/'normalized.idf'

def latest_run_succeeded(case_root):
    p=Path(case_root)/'latest_run.json'
    if not p.is_file(): return False
    try: return str(json.loads(p.read_text(encoding='utf-8')).get('status','')).lower() in SUCCESS_STATUSES
    except Exception: return False

def case_weather_path(case_spec):
    for name in ('weather_path','epw_path','weather_file','epw_file'):
        value=getattr(case_spec,name,None)
        if value: return Path(value)
    raise AttributeError('CaseSpec contains no weather path')

def write_rdd_intersection_manifest(*,output_path,case_id,rdd_path,requested_variables,available_variables,unavailable_variables):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    payload={'case_id':str(case_id),'rdd_path':str(rdd_path),'requested_variable_count':len(requested_variables),'rdd_available_variable_count':len(available_variables),'rdd_unavailable_variable_count':len(unavailable_variables),'available_variables':[get_requested_variable_name(v) for v in available_variables],'unavailable_variables':[get_requested_variable_name(v) for v in unavailable_variables]}
    output_path.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')

def validate_case_generation_outputs(*,case_root,expected_variable_count,require_legacy_pickles):
    case_root=Path(case_root); latest=case_root/'latest_run.json'
    if not latest.exists(): raise RuntimeError(f'Missing latest_run.json: {case_root}')
    payload=json.loads(latest.read_text(encoding='utf-8')); run_id=payload.get('run_id')
    if not run_id: raise RuntimeError(f'latest_run.json lacks run_id: {latest}')
    run_root=case_root/'runs'/str(run_id)
    parquet=list((run_root/'canonical'/'variables').glob('*.parquet')) if (run_root/'canonical'/'variables').exists() else []
    pickles=list((run_root/'legacy'/'per_variable_pickle').glob('*.pickle')) if (run_root/'legacy'/'per_variable_pickle').exists() else []
    traces=[p for p in run_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.txt','.log'} and 'trace' in p.name.lower()]
    failures=[]
    if len(parquet)!=expected_variable_count: failures.append(f'canonical parquet count mismatch: expected {expected_variable_count}, found {len(parquet)}')
    if require_legacy_pickles and len(pickles)!=expected_variable_count: failures.append(f'legacy pickle count mismatch: expected {expected_variable_count}, found {len(pickles)}')
    if traces: failures.append('traceback files found: '+', '.join(str(p) for p in traces[:5]))
    if failures: raise RuntimeError('Case generation output validation failed.\n'+'\n'.join('- '+x for x in failures))

def normalize_cases(*,selected_cases,campaign_id,generated_data_root):
    records=[]
    total=len(selected_cases)
    for index,case in enumerate(selected_cases,1):
        destination=normalized_idf_path_for_case(generated_data_root,campaign_id,case.case_id)
        print(
            f'[prepare {index}/{total}] Normalizing IDF: '
            f'{case.building_type} / {case.weather_location}',
            flush=True,
        )
        print(f'  source_idf: {case.idf_path}',flush=True)
        print(f'  normalized_idf: {destination}',flush=True)
        norm=normalize_idf_before_opyplus(
            source_idf_path=Path(case.idf_path),
            normalized_idf_path=destination,
        )
        tags=dict(case.tags)
        tags['idf_pre_opyplus_normalization']='true'
        tags['idf_patches']='; '.join(norm.applied_patches) if norm.applied_patches else 'none'
        records.append((case.model_copy(update={'idf_path':norm.normalized_idf_path,'tags':tags}),norm))
        print(
            f'[prepare {index}/{total}] IDF normalization complete; '
            f'patches: {tags["idf_patches"]}',
            flush=True,
        )
    return records

def run_generation_campaign(*,selected_cases:Sequence, campaign_id:str, machine_id:str, generated_data_root:Path, variable_limit:int|None, parallel_variable_workers:int, write_legacy_pickles:bool, rerun_completed:bool, mlflow_tracker=None, dry_run=False):
    collection=campaign_case_collection_name(campaign_id)
    print('='*100,flush=True)
    print('GENERATION CAMPAIGN STARTUP',flush=True)
    print('='*100,flush=True)
    print(f'campaign_id: {campaign_id}',flush=True)
    print(f'selected_case_count: {len(selected_cases)}',flush=True)
    print('Preparing normalized IDFs before EnergyPlus execution...',flush=True)
    records=normalize_cases(selected_cases=selected_cases,campaign_id=campaign_id,generated_data_root=generated_data_root)
    print('='*100,flush=True); print('GENERAL GENERATION CAMPAIGN PLAN',flush=True); print('='*100,flush=True)
    print(f'campaign_id: {campaign_id}',flush=True); print(f'generated_data_root: {generated_data_root}',flush=True); print(f'selected_case_count: {len(records)}',flush=True); print(f'variable_limit: {variable_limit if variable_limit is not None else "all"}',flush=True); print(f'parallel_variable_workers: {parallel_variable_workers}',flush=True)
    for i,(case,norm) in enumerate(records,1): print(f'{i},{case.building_type},{case.weather_location},{case.case_id},{norm.normalized_idf_path}',flush=True)
    if dry_run: print('Dry run complete. No EnergyPlus simulations were launched.',flush=True); return CampaignRunResult(campaign_id,len(records),0,0,0,0)
    completed=skipped=failed=0
    for i,(case,norm) in enumerate(records,1):
        root=case_generation_root(generated_data_root,collection,case.case_id)
        print('\n'+'='*100,flush=True); print(f'[{i}/{len(records)}] {case.building_type} / {case.weather_location}',flush=True); print(f'case_id: {case.case_id}',flush=True); print(f'case_root: {root}',flush=True)
        if not rerun_completed and latest_run_succeeded(root): print('Skipping: latest_run.json indicates completed status.',flush=True); skipped+=1; continue
        requested=list(case.output_variables[:variable_limit] if variable_limit else case.output_variables)
        try:
            probe_root=root/'rdd_probe'; probe=run_energyplus_rdd_probe(source_idf_path=Path(case.idf_path),weather_path=case_weather_path(case),output_dir=probe_root)
            available,unavailable=filter_requested_variables_by_rdd(requested,probe.rdd_path,variable_name_attr='variable_name')
            write_rdd_intersection_manifest(output_path=probe_root/'rdd_variable_intersection.json',case_id=case.case_id,rdd_path=probe.rdd_path,requested_variables=requested,available_variables=available,unavailable_variables=unavailable)
            print(f'requested_variable_count: {len(requested)}',flush=True); print(f'rdd_available_variable_count: {len(available)}',flush=True); print(f'rdd_unavailable_variable_count: {len(unavailable)}',flush=True)
            if not available: raise RuntimeError('RDD filtering produced zero available output variables')
            generate_variable_wise_case(case_spec=case,generated_data_root=generated_data_root,campaign_id=campaign_id,case_collection_name=collection,machine_id=machine_id,selected_output_variables=available,delete_raw_csv=True,mlflow_tracker=mlflow_tracker,short_work_root=os.environ.get('SCALEBRIDGE_EPLUS_WORK_ROOT'),parallel_variable_workers=parallel_variable_workers)
            validate_case_generation_outputs(case_root=root,expected_variable_count=len(available),require_legacy_pickles=write_legacy_pickles)
            completed+=1; print('COMPLETED',flush=True)
        except Exception as exc:
            failed+=1; print('FAILED',flush=True); print(f'{type(exc).__name__}: {exc}',flush=True)
    print('\n'+'='*100,flush=True); print('GENERATION CAMPAIGN SUMMARY',flush=True); print(f'campaign_id: {campaign_id}',flush=True); print(f'selected_cases: {len(records)}',flush=True); print(f'completed_or_launched: {completed}',flush=True); print(f'skipped: {skipped}',flush=True); print(f'failed: {failed}',flush=True)
    return CampaignRunResult(campaign_id,len(records),completed,skipped,failed,1 if failed else 0)
