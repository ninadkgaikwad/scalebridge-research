from __future__ import annotations
import argparse, json
from pathlib import Path
from scalebridge.data.thermal_modeling.alignment import TimestampNormalizationConfig, load_and_align_paths
from scalebridge.data.thermal_modeling.discovery import discover_phase_d_sources

def main()->int:
 p=argparse.ArgumentParser()
 p.add_argument('--campaign-root',required=True); p.add_argument('--matrix-run-id',required=True)
 p.add_argument('--aggregation-run-id',required=True); p.add_argument('--phase-c-campaign-run-id',required=True)
 p.add_argument('--aggregate-zone-id',required=True); p.add_argument('--phase-d-calendar-year',type=int,default=2001)
 p.add_argument('--output-json',required=True)
 a=p.parse_args()
 d=discover_phase_d_sources(campaign_root=Path(a.campaign_root),matrix_run_id=a.matrix_run_id,aggregation_run_id=a.aggregation_run_id,phase_c_campaign_run_id=a.phase_c_campaign_run_id,aggregate_zone_id=a.aggregate_zone_id)
 aligned,diag=load_and_align_paths(d.aggregation_zone.wide_parquet_path,d.phase_c_zone.predictions_parquet_path,d.phase_c_zone.split_assignments_parquet_path,TimestampNormalizationConfig(a.phase_d_calendar_year))
 payload={'zone':a.aggregate_zone_id,'diagnostics':diag.to_dict(),'columns':list(aligned.columns),'first_timestamp':aligned.timestamp.iloc[0].isoformat(),'last_timestamp':aligned.timestamp.iloc[-1].isoformat()}
 out=Path(a.output_json); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(payload,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
