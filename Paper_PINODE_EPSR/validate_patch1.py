from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PaperConfig, canonical_case_specs
from .data import load_case, load_manifest_only, numeric_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PINODE/EPSR Patch 1 data contracts")
    parser.add_argument("--full", action="store_true", help="also read Parquet and run numerical audit")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args()

    cfg = PaperConfig.from_environment()
    report: dict[str, object] = {
        "campaign_id": cfg.campaign_id,
        "case_id": cfg.case_id,
        "dt_seconds": cfg.dt_seconds,
        "cases": {},
    }

    for name in canonical_case_specs():
        spec, manifests = load_manifest_only(cfg, name)
        case_report: dict[str, object] = {
            "dependency_mode": spec.dependency_mode,
            "zones": list(spec.zone_ids),
            "phase_d_paths": list(spec.phase_d_paths),
            "aliases": list(spec.all_to_one_aliases),
            "manifest_rows": [m["row_count"] for m in manifests],
            "partition_counts": [m["partition_counts"] for m in manifests],
        }
        if args.full:
            trajectory = load_case(cfg, name)
            case_report["numeric_audit"] = numeric_audit(trajectory)
        report["cases"][name] = case_report

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
