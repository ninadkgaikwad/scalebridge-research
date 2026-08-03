# -*- coding: utf-8 -*-
"""Run Stage C8 full-year component inference from validated C7 artifacts."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scalebridge.inference.heat_input_regression import discover_evaluation_artifacts, run_zone_inference, build_building_phvac_inference


def _safe(value: object) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in str(value)).strip("_")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root", required=True)
    p.add_argument("--feature-root", default=None, help="C2 feature run root; enables zero-model-zone preservation and applicability coverage.")
    p.add_argument("--dataset-root", default=None, help="C4 dataset run root recorded as provenance.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--inference-run-id", required=True)
    p.add_argument("--model-id", action="append", default=None)
    p.add_argument("--aggregate-zone-id", action="append", default=None)
    p.add_argument("--estimator-type", action="append", default=None)
    p.add_argument("--requested-device", action="append", default=None)
    p.add_argument("--max-artifacts", type=int, default=None)
    p.add_argument("--preview-rows", type=int, default=100)
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def _feature_zones(feature_root: Path | None) -> dict[tuple[str, str, str, str], dict]:
    zones: dict[tuple[str, str, str, str], dict] = {}
    if feature_root is None:
        return zones
    for manifest_path in sorted(feature_root.rglob("zone_feature_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = (
            str(payload.get("case_id", "")),
            str(payload.get("aggregation_id", "")),
            str(payload.get("weight_mode", "")),
            str(payload.get("aggregate_zone_id", "")),
        )
        payload["_manifest_path"] = str(manifest_path)
        payload["_zone_root"] = str(manifest_path.parent)
        zones[key] = payload
    return zones


def _write_zero_component_zone(*, key, feature_manifest: dict, inference_root: Path, run_id: str, preview_rows: int, overwrite: bool) -> dict:
    zone_root = Path(feature_manifest["_zone_root"])
    outputs = feature_manifest.get("outputs", {}) or {}
    feature_path = Path(outputs.get("derived_features_parquet", zone_root / "derived_heat_input_features.parquet"))
    frame = pd.read_parquet(feature_path, columns=["timestamp_raw", "timestamp"])
    out_dir = inference_root / "cases" / _safe(key[0]) / _safe(key[1]) / _safe(key[2]) / _safe(key[3])
    manifest_path = out_dir / "annual_component_predictions_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"C8 output exists: {manifest_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "annual_component_predictions.parquet"
    preview_path = out_dir / "annual_component_predictions_preview.csv"
    registry_path = out_dir / "component_prediction_registry.csv"
    applicability_path = zone_root / "model_applicability_snapshot.csv"
    component_applicability_path = out_dir / "component_applicability.csv"
    frame.to_parquet(predictions_path, index=False)
    frame.head(preview_rows).to_csv(preview_path, index=False)
    pd.DataFrame(columns=["model_id", "output_prediction_column", "prediction_units"]).to_csv(registry_path, index=False)
    if applicability_path.is_file():
        pd.read_csv(applicability_path).to_csv(component_applicability_path, index=False)
    else:
        pd.DataFrame().to_csv(component_applicability_path, index=False)
    manifest = {
        "schema_version": "0.2.0", "stage": "C8", "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "inference_run_id": run_id,
        "case_id": key[0], "aggregation_id": key[1], "weight_mode": key[2], "aggregate_zone_id": key[3],
        "row_count": int(len(frame)), "component_count": 0, "zero_applicable_components": True,
        "source_feature_manifest": feature_manifest["_manifest_path"],
        "outputs": {"annual_component_predictions": str(predictions_path), "annual_component_predictions_preview": str(preview_path), "component_prediction_registry": str(registry_path), "component_applicability": str(component_applicability_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"case_id": key[0], "aggregation_id": key[1], "weight_mode": key[2], "aggregate_zone_id": key[3], "row_count": len(frame), "component_count": 0, "status": "completed", "zero_applicable_components": True, "output_dir": str(out_dir), "manifest_path": str(manifest_path)}


def main():
    args = parse_args(); started = time.perf_counter()
    evaluation_root = Path(args.evaluation_root).resolve()
    feature_root = Path(args.feature_root).resolve() if args.feature_root else None
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None
    inference_root = Path(args.output_root).resolve() / args.inference_run_id
    inference_root.mkdir(parents=True, exist_ok=True)
    refs = discover_evaluation_artifacts(evaluation_root, model_ids=args.model_id, aggregate_zone_ids=args.aggregate_zone_id, estimator_types=args.estimator_type, requested_devices=args.requested_device, max_artifacts=args.max_artifacts)
    groups = defaultdict(list)
    for ref in refs:
        groups[ref.zone_key].append(ref)
    feature_zones = _feature_zones(feature_root)
    selected_zone_keys = set(feature_zones) if feature_zones else set(groups)
    print("="*100); print("SCALEBRIDGE HEAT-INPUT REGRESSION FULL-YEAR INFERENCE"); print("="*100)
    print(f"evaluation_root: {evaluation_root}")
    print(f"feature_root: {feature_root or ''}")
    print(f"inference_root: {inference_root}")
    print(f"selected_evaluation_artifact_count: {len(refs)}")
    print(f"selected_zone_count: {len(selected_zone_keys)}")
    selected_columns = ["case_id", "aggregation_id", "weight_mode", "aggregate_zone_id", "model_id", "output_prediction_column", "evaluation_manifest"]
    selected = [{"case_id": r.case_id, "aggregation_id": r.aggregation_id, "weight_mode": r.weight_mode, "aggregate_zone_id": r.aggregate_zone_id, "model_id": r.model_id, "output_prediction_column": r.output_prediction_column, "evaluation_manifest": str(r.evaluation_manifest_path)} for r in refs]
    pd.DataFrame(selected, columns=selected_columns).to_csv(inference_root / "selected_evaluation_artifacts.csv", index=False)
    results=[]; failures=[]; zero_component_zone_count=0
    for i,key in enumerate(sorted(selected_zone_keys),1):
        zone_refs = groups.get(key, [])
        print(f"[{i}/{len(selected_zone_keys)}] {key[3]} | components={len(zone_refs)}")
        try:
            if zone_refs:
                result = run_zone_inference(zone_refs, inference_root=inference_root, inference_run_id=args.inference_run_id, preview_rows=args.preview_rows, overwrite_existing=args.overwrite_existing)
                row = result.row
                # Preserve the complete C1 applicability contract beside C8 predictions.
                if key in feature_zones:
                    source = Path(feature_zones[key]["_zone_root"]) / "model_applicability_snapshot.csv"
                    target = Path(row["output_dir"]) / "component_applicability.csv"
                    if source.is_file():
                        pd.read_csv(source).to_csv(target, index=False)
                results.append(row)
            else:
                zero_component_zone_count += 1
                results.append(_write_zero_component_zone(key=key, feature_manifest=feature_zones[key], inference_root=inference_root, run_id=args.inference_run_id, preview_rows=args.preview_rows, overwrite=args.overwrite_existing))
        except Exception as exc:
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            row={"case_id":key[0],"aggregation_id":key[1],"weight_mode":key[2],"aggregate_zone_id":key[3],"component_count":len(zone_refs),"status":"failed","error_type":type(exc).__name__,"error_message":str(exc),"traceback":traceback.format_exc()}
            results.append(row); failures.append(row)
            if not args.continue_on_error:
                raise
    result_columns = ["case_id", "aggregation_id", "weight_mode", "aggregate_zone_id", "row_count", "component_count", "status", "zero_applicable_components", "output_dir", "manifest_path", "error_type", "error_message"]
    pd.DataFrame(results).reindex(columns=result_columns).to_csv(inference_root / "inference_results.csv", index=False)
    pd.DataFrame(failures).reindex(columns=result_columns).to_csv(inference_root / "inference_failures.csv", index=False)
    completed=sum(r.get("status")=="completed" for r in results)
    building_phvac_index = build_building_phvac_inference(inference_root)
    manifest={"schema_version":"0.2.0","stage":"C8","status":"completed" if not failures else "completed_with_failures","created_at_utc":datetime.now(timezone.utc).isoformat(),"inference_run_id":args.inference_run_id,"evaluation_root":str(evaluation_root),"feature_root":str(feature_root or ""),"dataset_root":str(dataset_root or ""),"inference_root":str(inference_root),"selected_evaluation_artifact_count":len(refs),"selected_zone_count":len(selected_zone_keys),"completed_zone_count":completed,"failed_zone_count":len(failures),"zero_component_zone_count":zero_component_zone_count,"runtime_seconds":time.perf_counter()-started,"building_phvac_reconstruction_index":str(building_phvac_index or "")}
    (inference_root/"inference_run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    print("\n"+"="*100); print("INFERENCE SUMMARY"); print("="*100); print(f"completed_zone_count: {completed}"); print(f"zero_component_zone_count: {zero_component_zone_count}"); print(f"failed_zone_count: {len(failures)}"); print(f"inference_root: {inference_root}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
