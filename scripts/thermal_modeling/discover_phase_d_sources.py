# -*- coding: utf-8 -*-
"""CLI for Phase D D2 source discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scalebridge.data.thermal_modeling.discovery import discover_phase_d_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve Phase B and Phase C sources for one Phase D zone."
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--aggregation-run-id", required=True)
    parser.add_argument("--phase-c-campaign-run-id", required=True)
    parser.add_argument("--aggregate-zone-id", required=True)
    parser.add_argument("--output-json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = discover_phase_d_sources(
        campaign_root=Path(args.campaign_root),
        matrix_run_id=args.matrix_run_id,
        aggregation_run_id=args.aggregation_run_id,
        phase_c_campaign_run_id=args.phase_c_campaign_run_id,
        aggregate_zone_id=args.aggregate_zone_id,
    )
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
