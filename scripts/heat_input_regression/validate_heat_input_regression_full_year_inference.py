# -*- coding: utf-8 -*-
"""Independently validate a Stage C8 full-year inference run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[2]; SRC_ROOT=REPO_ROOT/"src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0,str(SRC_ROOT))
from scalebridge.inference.heat_input_regression import validate_zone_inference_artifact


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--inference-root",required=True); p.add_argument("--prediction-atol",type=float,default=1e-12); p.add_argument("--prediction-rtol",type=float,default=1e-12); args=p.parse_args()
    root=Path(args.inference_root).resolve(); manifests=sorted(root.rglob("annual_component_predictions_manifest.json"))
    print("="*100); print("SCALEBRIDGE HEAT-INPUT REGRESSION FULL-YEAR INFERENCE VALIDATION"); print("="*100); print(f"inference_root: {root}"); print(f"inference_artifact_count: {len(manifests)}")
    results=[]; diagnostics=[]
    for i,path in enumerate(manifests,1):
        payload=json.loads(path.read_text(encoding="utf-8")); print(f"[{i}/{len(manifests)}] {payload.get('aggregate_zone_id')} | components={payload.get('component_count')}")
        result,checks=validate_zone_inference_artifact(path,prediction_atol=args.prediction_atol,prediction_rtol=args.prediction_rtol); results.append(result)
        for row in checks: diagnostics.append({"manifest_path":str(path),"aggregate_zone_id":result.get("aggregate_zone_id",""),**row})
    pd.DataFrame(results).to_csv(root/"inference_validation_results.csv",index=False); pd.DataFrame(diagnostics).to_csv(root/"inference_validation_diagnostics.csv",index=False)
    failed=sum(r["status"]!="passed" for r in results); summary={"schema_version":"0.1.0","stage":"C8_validation","created_at_utc":datetime.now(timezone.utc).isoformat(),"inference_root":str(root),"inference_artifact_count":len(manifests),"passed_artifact_count":len(manifests)-failed,"failed_artifact_count":failed,"validation_status":"passed" if manifests and failed==0 else "failed","prediction_atol":args.prediction_atol,"prediction_rtol":args.prediction_rtol}
    (root/"inference_validation_manifest.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print("\n"+"="*100); print("INFERENCE VALIDATION SUMMARY"); print("="*100); print(f"passed_artifact_count: {summary['passed_artifact_count']}"); print(f"failed_artifact_count: {failed}"); print(f"validation_status: {summary['validation_status']}")
    if summary["validation_status"]!="passed": raise SystemExit(1)

if __name__=="__main__": main()
