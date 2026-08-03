"""Build general CaseSpec matrices from explicit IDF and EPW selections."""
from __future__ import annotations
from pathlib import Path
from scalebridge.integration.energyplus.manifests.models import CaseSpec, RunPeriod
from scalebridge.integration.energyplus.p1 import p1_output_variables
from .campaign_definition import GenerationCampaignDefinition


def build_generation_case_specs(definition: GenerationCampaignDefinition) -> tuple[CaseSpec,...]:
    requests=p1_output_variables()
    cases=[]
    for building in definition.buildings:
        for weather in definition.weather_files:
            idf=Path(building.path).expanduser().resolve()
            epw=Path(weather.path).expanduser().resolve()
            if not idf.is_file(): raise FileNotFoundError(f'IDF does not exist: {idf}')
            if not epw.is_file(): raise FileNotFoundError(f'EPW does not exist: {epw}')
            cases.append(CaseSpec(
                case_name=f'{definition.campaign_id}_{building.name}_{weather.name}',
                building_type=building.building_type or building.name,
                prototype_standard='ASHRAE 90.1' if definition.source_mode=='ashrae_library' else None,
                prototype_year=str(building.standard_year) if building.standard_year else None,
                climate_zone=None,
                weather_location=weather.name,
                idf_path=idf, epw_path=epw,
                idf_sha256=building.sha256, epw_sha256=weather.sha256,
                run_period=RunPeriod(start_month=1,start_day=1,end_month=12,end_day=31,calendar_year=2013),
                timestep_minutes=5,
                output_variables=requests,
                energyplus_version='9.0.1',
                write_legacy_pickles=definition.write_legacy_pickles,
                preserve_raw_outputs=True,
                tags={
                    'campaign_id':definition.campaign_id,
                    'source_mode':definition.source_mode,
                    'source_idf_name':building.name,
                    'source_weather_name':weather.name,
                    'source_prototype_location':building.source_location or 'not_applicable',
                    'signal_profile':definition.signal_profile,
                },
            ))
    cases.sort(key=lambda c:(str(c.building_type),str(c.weather_location),c.case_id))
    if definition.case_limit is not None: cases=cases[:definition.case_limit]
    return tuple(cases)
