"""Run a saved BGIRS Generation campaign definition."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from scalebridge.integration.energyplus.generation.campaign_definition import GenerationCampaignDefinition
from scalebridge.integration.energyplus.generation.case_builder import build_generation_case_specs
from scalebridge.integration.energyplus.generation.campaign_runner import run_generation_campaign
from scalebridge.integration.energyplus.prototypes import resolve_generated_data_root

def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign-definition',required=True); p.add_argument('--dry-run',action='store_true'); return p.parse_args()

def create_mlflow(defn):
    if not defn.mlflow_enabled: return None
    import mlflow; mlflow.set_tracking_uri(defn.mlflow_tracking_uri)
    from scalebridge.tracking.mlflow import MLflowGenerationTracker
    return MLflowGenerationTracker(experiment_name=defn.mlflow_experiment_name or f'{defn.campaign_id}_generation',strict=defn.mlflow_strict)

def main():
    args=parse_args(); path=Path(args.campaign_definition).resolve(); defn=GenerationCampaignDefinition.model_validate(json.loads(path.read_text(encoding='utf-8')))
    root=resolve_generated_data_root(defn.generated_data_root); cases=build_generation_case_specs(defn); tracker=None
    if not args.dry_run:
        try: tracker=create_mlflow(defn)
        except Exception:
            if defn.mlflow_strict: raise
            print('WARNING: MLflow unavailable; continuing without MLflow.',flush=True)
    result=run_generation_campaign(selected_cases=cases,campaign_id=defn.campaign_id,machine_id=defn.machine_id,generated_data_root=root,variable_limit=defn.variable_limit,parallel_variable_workers=defn.parallel_variable_workers,write_legacy_pickles=defn.write_legacy_pickles,rerun_completed=defn.rerun_completed,mlflow_tracker=tracker,dry_run=args.dry_run)
    return result.return_code
if __name__=='__main__': raise SystemExit(main())
