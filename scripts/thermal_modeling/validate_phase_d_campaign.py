# -*- coding: utf-8 -*-
"""Validate a completed D8 Phase D campaign realization."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--campaign-root", required=True, type=Path)
    p.add_argument("--phase-d-run-id", required=True)
    args=p.parse_args()
    campaign_root=args.campaign_root.expanduser().resolve()
    phase_root=campaign_root/"phase_d"
    run_root=phase_root/"campaign_runs"/args.phase_d_run_id

    manifest=json.loads((run_root/"phase_d_campaign_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="completed", manifest
    assert manifest["failed_aggregation_run_count"]==0, manifest
    assert manifest["intermediate_time_series_persisted"] is False

    with (run_root/"aggregation_run_registry.csv").open("r",encoding="utf-8-sig",newline="") as f:
        agg=list(csv.DictReader(f))
    with (run_root/"dataset_registry.csv").open("r",encoding="utf-8-sig",newline="") as f:
        ds=list(csv.DictReader(f))

    assert len(agg)==manifest["selected_aggregation_run_count"]
    assert len(ds)==manifest["dataset_count"]

    # Every final time series must be exactly a silo data.parquet with adjacent manifest.
    all_parquets=list(phase_root.rglob("*.parquet"))
    bad=[p for p in all_parquets if p.name!="data.parquet" or "silos" not in p.parts]
    assert not bad, f"Unexpected Phase D Parquets: {bad[:10]}"
    missing_manifest=[p for p in all_parquets if not p.with_name("manifest.json").is_file()]
    assert not missing_manifest, f"Missing manifests: {missing_manifest[:10]}"

    tmp=list(phase_root.rglob("*.tmp"))
    assert not tmp, f"Temporary files remain: {tmp[:10]}"

    families=sorted({r["aggregation_family"] for r in agg})
    weights=sorted({r["weight_mode"] for r in agg})
    levels=sorted({r["aggregation_level"] for r in agg})
    print(f"aggregation_runs={len(agg)} datasets={len(ds)}")
    print(f"aggregation_levels={levels}")
    print(f"aggregation_families={families}")
    print(f"weight_modes={weights}")
    print(f"phase_d_parquets={len(all_parquets)} unexpected_parquets=0")
    print("D8_VALIDATION_COMPLETE")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
