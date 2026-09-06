# -*- coding: utf-8 -*-
"""Validate the campaign-level Phase D D8.4 MLflow run against local manifests."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-root", required=True, type=Path)
    p.add_argument("--phase-d-run-id", required=True)
    return p


def main():
    args=parser().parse_args()
    root=args.campaign_root/"phase_d"/"campaign_runs"/args.phase_d_run_id
    manifest=json.loads((root/"phase_d_campaign_run_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("mlflow_enabled"):
        raise SystemExit("Phase D manifest says MLflow was not enabled")
    run_id=manifest.get("mlflow_run_id")
    if not run_id:
        raise SystemExit("Phase D manifest has no mlflow_run_id")

    import mlflow
    from scalebridge.tracking.mlflow.semantic import configure_mlflow_tracking
    configure_mlflow_tracking()
    run=mlflow.get_run(run_id)
    tags=run.data.tags
    params=run.data.params
    metrics=run.data.metrics
    required_tags={
        "campaign_id": manifest["campaign_id"],
        "matrix_run_id": manifest["matrix_run_id"],
        "phase_c_campaign_run_id": manifest["phase_c_campaign_run_id"],
        "phase_d_run_id": manifest["phase_d_run_id"],
        "pipeline_stage": "phase_d",
    }
    for k,v in required_tags.items():
        if str(tags.get(k)) != str(v):
            raise SystemExit(f"MLflow tag mismatch {k}: {tags.get(k)!r} != {v!r}")
    for k in ("matrix_run_id","phase_c_campaign_run_id","phase_d_run_id"):
        if str(params.get(k)) != str(manifest[k]):
            raise SystemExit(f"MLflow param mismatch {k}")
    metric_pairs={
        "dataset_count": manifest["dataset_count"],
        "completed_aggregation_run_count": manifest["completed_aggregation_run_count"],
        "failed_aggregation_run_count": manifest["failed_aggregation_run_count"],
    }
    for k,v in metric_pairs.items():
        if float(metrics.get(k, -1)) != float(v):
            raise SystemExit(f"MLflow metric mismatch {k}: {metrics.get(k)} != {v}")
    print(f"mlflow_run_id={run_id}")
    print(f"dataset_count={manifest['dataset_count']}")
    print("PHASE_D_MLFLOW_VALIDATED")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
