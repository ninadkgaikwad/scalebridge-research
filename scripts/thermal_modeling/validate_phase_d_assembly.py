from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


from scalebridge.data.thermal_modeling.alignment import (
    TimestampNormalizationConfig,
    load_and_align_paths,
)
from scalebridge.data.thermal_modeling.assembly import (
    AssemblyConfig,
    assemble_canonical_zone_table,
    required_phase_c_prediction_columns,
)
from scalebridge.data.thermal_modeling.discovery import discover_phase_d_sources


SPLIT_COLUMNS = (
    "split",
    "split_index",
    "included",
    "exclusion_reason",
    "source_row_index",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run memory-conscious Phase D D4 canonical signal assembly "
            "for one aggregate zone."
        )
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--matrix-run-id", required=True)
    parser.add_argument("--aggregation-run-id", required=True)
    parser.add_argument("--phase-c-campaign-run-id", required=True)
    parser.add_argument("--aggregate-zone-id", required=True)
    parser.add_argument("--phase-d-calendar-year", type=int, default=2001)
    parser.add_argument(
        "--exclude-visible-lighting-from-zir",
        action="store_true",
    )
    parser.add_argument(
        "--validation-level",
        choices=("standard", "full"),
        default="standard",
        help=(
            "standard writes a compact manifest; full writes complete "
            "per-signal records. Scientific validation remains enabled."
        ),
    )
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument(
        "--output-table-parquet",
        help=(
            "Optional controlled-validation assembled table. Omit for the "
            "production storage policy, where D4 remains in memory until the "
            "final Phase D silo writer consumes it."
        ),
    )
    parser.add_argument("--output-preview-parquet")
    parser.add_argument("--preview-rows", type=int, default=100)
    parser.add_argument(
        "--parquet-compression",
        choices=("zstd", "snappy", "gzip", "brotli", "none"),
        default="zstd",
    )
    return parser


def _compression(value: str) -> str | None:
    return None if value == "none" else value


def main() -> int:
    import pyarrow.parquet as pq

    args = build_parser().parse_args()
    if args.preview_rows < 1:
        raise ValueError("--preview-rows must be at least 1")

    discovery = discover_phase_d_sources(
        campaign_root=Path(args.campaign_root),
        matrix_run_id=args.matrix_run_id,
        aggregation_run_id=args.aggregation_run_id,
        phase_c_campaign_run_id=args.phase_c_campaign_run_id,
        aggregate_zone_id=args.aggregate_zone_id,
    )

    phase_c_schema = pq.ParquetFile(
        discovery.phase_c_zone.predictions_parquet_path
    ).schema_arrow
    phase_c_columns = required_phase_c_prediction_columns(
        discovery.phase_c_zone.applicable_models_path,
        discovery.phase_c_zone.unavailable_models_path,
        available_parquet_columns=set(phase_c_schema.names),
    )
    split_schema = pq.ParquetFile(
        discovery.phase_c_zone.split_assignments_parquet_path
    ).schema_arrow
    split_columns = tuple(
        column for column in SPLIT_COLUMNS if column in split_schema.names
    )

    aligned, alignment_diagnostics = load_and_align_paths(
        discovery.aggregation_zone.wide_parquet_path,
        discovery.phase_c_zone.predictions_parquet_path,
        discovery.phase_c_zone.split_assignments_parquet_path,
        TimestampNormalizationConfig(args.phase_d_calendar_year),
        phase_c_columns=phase_c_columns,
        split_columns=split_columns,
    )

    result = assemble_canonical_zone_table(
        aligned,
        applicable_models_path=discovery.phase_c_zone.applicable_models_path,
        unavailable_models_path=discovery.phase_c_zone.unavailable_models_path,
        config=AssemblyConfig(
            include_visible_lighting_in_zir=(
                not args.exclude_visible_lighting_from_zir
            )
        ),
    )
    del aligned
    gc.collect()

    payload = result.manifest_dict(
        aggregate_zone_id=args.aggregate_zone_id,
        include_signal_records=args.validation_level == "full",
    )
    payload["alignment_diagnostics"] = alignment_diagnostics.to_dict()
    payload["phase_d_calendar_year"] = args.phase_d_calendar_year
    payload["include_visible_lighting_in_zir"] = (
        not args.exclude_visible_lighting_from_zir
    )
    payload["validation_level"] = args.validation_level
    payload["parquet_projection"] = {
        "phase_b_columns": [
            "timestamp_raw",
            "Zone_Air_Temperature_",
            "Site_Outdoor_Air_Drybulb_Temperature_",
        ],
        "phase_c_columns": ["timestamp", *phase_c_columns],
        "split_columns": ["timestamp", *split_columns],
    }
    payload["first_timestamp"] = result.table["timestamp"].iloc[0].isoformat()
    payload["last_timestamp"] = result.table["timestamp"].iloc[-1].isoformat()

    manifest_path = Path(args.output_manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    compression = _compression(args.parquet_compression)
    if args.output_table_parquet:
        table_path = Path(args.output_table_parquet)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        result.table.to_parquet(
            table_path,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        payload["assembled_table_parquet_path"] = str(table_path)
    else:
        payload["assembled_table_parquet_path"] = None

    if args.output_preview_parquet:
        preview_path = Path(args.output_preview_parquet)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        result.table.head(args.preview_rows).to_parquet(
            preview_path,
            index=False,
            engine="pyarrow",
            compression=compression,
        )
        payload["preview_parquet_path"] = str(preview_path)
        payload["preview_rows"] = min(args.preview_rows, len(result.table))
    else:
        payload["preview_parquet_path"] = None
        payload["preview_rows"] = 0

    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    diagnostics = result.diagnostics
    print(
        " ".join(
            (
                f"zone={args.aggregate_zone_id}",
                f"rows={diagnostics.row_count}",
                f"columns={diagnostics.canonical_column_count}",
                f"active={diagnostics.active_phase_c_signal_count}",
                f"nullable_zero={diagnostics.nullable_complete_zero_count}",
                f"nullable_na={diagnostics.nullable_not_applicable_count}",
                f"zic_components={len(diagnostics.zic_active_components)}",
                f"zir_components={len(diagnostics.zir_active_components)}",
                f"validation_failures={diagnostics.validation_failure_count}",
                "status=passed",
            )
        )
    )

    del result
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
