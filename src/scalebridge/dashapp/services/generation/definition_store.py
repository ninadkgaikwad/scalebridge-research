from __future__ import annotations
from pathlib import Path
import json
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root
from scalebridge.integration.energyplus.generation.campaign_definition import GenerationCampaignDefinition

def definition_root():
    p=resolve_generated_data_root()/ 'campaign_definitions'/'generation'; p.mkdir(parents=True,exist_ok=True); return p

def definition_path(campaign_id): return definition_root()/f'{campaign_id}.json'
def list_definitions():
    rows=[]
    for p in sorted(definition_root().glob('*.json')):
        try:
            d=GenerationCampaignDefinition.model_validate(json.loads(p.read_text(encoding='utf-8')))
            rows.append({'campaign_id':d.campaign_id,'source_mode':d.source_mode,'case_count':d.case_count,'machine_id':d.machine_id,'path':str(p)})
        except Exception: continue
    return rows

def load_definition(campaign_id):
    p=definition_path(campaign_id); return GenerationCampaignDefinition.model_validate(json.loads(p.read_text(encoding='utf-8')))
def save_definition(definition):
    p=definition_path(definition.campaign_id); p.write_text(json.dumps(definition.model_dump(mode='json'),indent=2,sort_keys=True)+'\n',encoding='utf-8'); return p
