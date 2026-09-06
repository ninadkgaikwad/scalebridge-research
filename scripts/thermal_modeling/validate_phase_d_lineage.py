# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scalebridge.data.thermal_modeling.lineage import (
    resolve_all_to_one_counterpart,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--phase-c-campaign-run-id", required=True)
    parser.add_argument("--aggregation-run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = resolve_all_to_one_counterpart(
        campaign_root=args.campaign_root,
        matrix_run_id=args.matrix_run_id,
        aggregation_run_id=args.aggregation_run_id,
        phase_c_campaign_run_id=args.phase_c_campaign_run_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        f"run={result.current_lineage.aggregation_run_id} "
        f"aggregation_id={result.current_lineage.aggregation_id} "
        f"zones={result.current_lineage.aggregate_zone_count} "
        f"source_zones={result.current_lineage.source_zone_count} "
        f"status={result.status} "
        f"selected={result.selected_aggregation_run_id} "
        f"phase_c_usable="
        f"{result.phase_c_usability.usable if result.phase_c_usability else False} "
        f"dep2_available={result.dependent_2_available}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
