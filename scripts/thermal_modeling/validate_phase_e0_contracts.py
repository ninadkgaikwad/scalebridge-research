# -*- coding: utf-8 -*-
"""Validate one final Phase D realization against the E0-2 canonical contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scalebridge.data.thermal_modeling.phase_e_adapter import (
    load_phase_e_data_contract,
    validate_materialized_columns,
    validate_partition_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a final Phase D manifest/data.parquet against E0-2."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--validate-parquet",
        action="store_true",
        help="Validate adjacent data.parquet columns and partition labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_phase_e_data_contract(args.manifest)

    if args.validate_parquet:
        import pyarrow.parquet as pq

        parquet_path = args.manifest.parent / "data.parquet"
        schema = pq.ParquetFile(parquet_path).schema_arrow
        validate_materialized_columns(contract, schema.names)

        # Read only the partition column for the outer-policy vocabulary check.
        partitions = (
            pq.read_table(parquet_path, columns=["partition"])
            .column("partition")
            .to_pylist()
        )
        validate_partition_values(contract, partitions)

    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
