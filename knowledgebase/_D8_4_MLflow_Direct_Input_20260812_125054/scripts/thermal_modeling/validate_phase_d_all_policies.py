# -*- coding: utf-8 -*-
"""Validate that one D8 campaign run realized the complete Phase D policy catalog."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

EXPECTED = {
    "monthly_distributed_holdout": "mdh",
    "chronological_holdout": "ch",
    "seasonal_holdout": "sh",
    "seasonal_distributed": "sd",
    "seasonal_block_holdout": "sbh",
    "contiguous_identification": "ci",
    "custom_datetime_ranges": "cdr",
}
ML = {"monthly_distributed_holdout", "chronological_holdout", "seasonal_holdout"}
OB = {"seasonal_distributed", "seasonal_block_holdout", "contiguous_identification", "custom_datetime_ranges"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-root", required=True, type=Path)
    p.add_argument("--phase-d-run-id", required=True)
    args = p.parse_args()

    campaign_root = args.campaign_root.expanduser().resolve()
    run_root = campaign_root / "phase_d" / "campaign_runs" / args.phase_d_run_id
    summary = json.loads((run_root / "phase_d_campaign_run_manifest.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("failed_aggregation_run_count") != 0:
        raise SystemExit(f"Phase D campaign is not cleanly completed: {summary}")

    with (run_root / "dataset_registry.csv").open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("dataset_registry.csv is empty")

    observed = {row["policy_name"] for row in rows}
    if observed != set(EXPECTED):
        raise SystemExit(f"Policy catalog mismatch: observed={sorted(observed)} expected={sorted(EXPECTED)}")

    by_run: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        policy = row["policy_name"]
        token = row["policy_token"]
        if EXPECTED[policy] != token:
            raise SystemExit(f"Wrong token for {policy}: {token}")
        data_path = Path(row["data_path"])
        if data_path.name != "data.parquet" or data_path.parent.name != token:
            raise SystemExit(f"Policy folder/path mismatch: {data_path}")
        silo = row["silo"]
        if policy in ML and silo != "ml_sciml":
            raise SystemExit(f"ML policy {policy} written to wrong silo {silo}")
        if policy in OB and silo != "opt_bayes":
            raise SystemExit(f"OB policy {policy} written to wrong silo {silo}")
        manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
        assignment = manifest.get("policy_assignment") or {}
        if assignment.get("policy_name") != policy:
            raise SystemExit(f"Manifest policy assignment mismatch: {row['manifest_path']}")
        by_run[row["aggregation_run_id"]].add(policy)

    for aggregation_run_id, policies in by_run.items():
        if policies != set(EXPECTED):
            raise SystemExit(
                f"Aggregation run {aggregation_run_id} missing policies: "
                f"{sorted(set(EXPECTED)-policies)}"
            )

    print(f"aggregation_runs={len(by_run)}")
    print(f"datasets={len(rows)}")
    print(f"ml_policies={sorted(ML)}")
    print(f"ob_policies={sorted(OB)}")
    print("policy_tokens=" + str(EXPECTED))
    print("ALL_PHASE_D_POLICIES_VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
