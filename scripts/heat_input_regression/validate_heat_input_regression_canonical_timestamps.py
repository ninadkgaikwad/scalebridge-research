# -*- coding: utf-8 -*-
"""Validate canonical timestamp axes for a Stage C2 feature run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-root', type=Path, required=True)
    p.add_argument('--expected-row-count', type=int, default=None)
    p.add_argument('--expected-cadence-seconds', type=float, default=300.0)
    args=p.parse_args()
    manifests=sorted(args.feature_root.rglob('zone_feature_manifest.json'))
    if not manifests: raise ValueError(f'No zone manifests under {args.feature_root}')
    rows=[]; failures=[]
    for mpath in manifests:
        payload=json.loads(mpath.read_text(encoding='utf-8'))
        parquet=Path(payload['outputs']['derived_features_parquet'])
        frame=pd.read_parquet(parquet)
        ts=pd.to_datetime(frame['timestamp'], errors='coerce')
        deltas=ts.diff().dropna().dt.total_seconds()
        row={
            'aggregate_zone_id':payload.get('aggregate_zone_id',''),
            'row_count':len(frame),
            'unparsed_timestamp_count':int(ts.isna().sum()),
            'duplicate_timestamp_count':int(ts.duplicated().sum()),
            'timestamp_monotonic':bool(ts.is_monotonic_increasing),
            'noncanonical_cadence_count':int((deltas != args.expected_cadence_seconds).sum()),
            'removed_duplicate_row_count':payload.get('canonical_timestamp_metadata',{}).get('removed_duplicate_row_count',''),
        }
        checks=[row['unparsed_timestamp_count']==0,row['duplicate_timestamp_count']==0,row['timestamp_monotonic'],row['noncanonical_cadence_count']==0]
        if args.expected_row_count is not None: checks.append(row['row_count']==args.expected_row_count)
        row['status']='passed' if all(checks) else 'failed'
        if row['status']=='failed': failures.append(row)
        rows.append(row)
    out=args.feature_root/'canonical_timestamp_validation.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    print('='*100); print('C2 CANONICAL TIMESTAMP VALIDATION'); print('='*100)
    print(pd.DataFrame(rows).to_string(index=False)); print(f'output: {out}')
    print(f'passed_zone_count: {len(rows)-len(failures)}'); print(f'failed_zone_count: {len(failures)}')
    return 0 if not failures else 1
if __name__=='__main__': raise SystemExit(main())
