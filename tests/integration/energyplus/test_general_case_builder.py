from pathlib import Path
from scalebridge.integration.energyplus.generation.campaign_definition import GenerationCampaignDefinition,SourceFileRef
from scalebridge.integration.energyplus.generation.case_builder import build_generation_case_specs
from scalebridge.integration.energyplus.prototypes import sha256_file

def test_builder_crosses_every_idf_and_epw(tmp_path):
 idfs=[]; epws=[]
 for n in ('A','B'):
  p=tmp_path/f'{n}.idf'; p.write_text('Version,9.0;',encoding='utf-8'); idfs.append(SourceFileRef(source_id=p.name,name=p.stem,path=str(p),sha256=sha256_file(p),building_type=p.stem))
 for n in ('W1','W2','W3'):
  p=tmp_path/f'{n}.epw'; p.write_text('LOCATION,X,Y,Z\n',encoding='utf-8'); epws.append(SourceFileRef(source_id=p.name,name=p.stem,path=str(p),sha256=sha256_file(p)))
 d=GenerationCampaignDefinition(campaign_id='abc_campaign',source_mode='uploaded_zip',machine_id='test',buildings=tuple(idfs),weather_files=tuple(epws),mlflow_enabled=False)
 cases=build_generation_case_specs(d)
 assert len(cases)==6
 assert {c.building_type for c in cases}=={'A','B'}
 assert {c.weather_location for c in cases}=={'W1','W2','W3'}
 assert all(len(c.output_variables)==35 for c in cases)
