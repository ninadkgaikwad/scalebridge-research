from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_validator_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "thermal_modeling"
        / "validate_phase_d_assembly.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_phase_d_assembly",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_args() -> list[str]:
    return [
        "--campaign-root", "campaign",
        "--matrix-run-id", "matrix",
        "--aggregation-run-id", "aggregation",
        "--phase-c-campaign-run-id", "phase-c",
        "--aggregate-zone-id", "ZoneA",
        "--output-manifest-json", "manifest.json",
    ]


def test_d4_storage_defaults_do_not_require_intermediate_parquet():
    module = _load_validator_module()
    args = module.build_parser().parse_args(_required_args())
    assert args.output_table_parquet is None
    assert args.output_preview_parquet is None


def test_d4_storage_allows_explicit_parquet_debug_outputs():
    module = _load_validator_module()
    args = module.build_parser().parse_args(
        _required_args()
        + [
            "--output-table-parquet", "assembly.parquet",
            "--output-preview-parquet", "preview.parquet",
        ]
    )
    assert args.output_table_parquet == "assembly.parquet"
    assert args.output_preview_parquet == "preview.parquet"
