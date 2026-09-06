# -*- coding: utf-8 -*-
"""Run a saved general ScaleBridge Phase B Aggregation campaign definition."""
from __future__ import annotations

import argparse
from pathlib import Path

from scalebridge.data.aggregation.campaign_definition import (
    load_aggregation_campaign_definition,
)
from scalebridge.data.aggregation.campaign_runner import run_aggregation_campaign


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-definition",
        required=True,
        help="Path to AggregationCampaignDefinition JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover parent Generation runs and build/write exact Aggregation plans "
            "and matrix preview, but do not execute aggregation engine runs."
        ),
    )
    parser.add_argument(
        "--matrix-run-id",
        default=None,
        help="Optional explicit matrix/execution ID; otherwise generated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    definition_path = Path(args.campaign_definition).expanduser().resolve()
    definition = load_aggregation_campaign_definition(definition_path)
    result = run_aggregation_campaign(
        definition=definition,
        definition_path=definition_path,
        dry_run=args.dry_run,
        matrix_run_id=args.matrix_run_id,
    )
    return result.return_code


if __name__ == "__main__":
    raise SystemExit(main())
