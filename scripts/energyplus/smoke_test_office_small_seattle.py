"""Run a real two-day OfficeSmall Seattle EnergyPlus smoke test.

This script exercises the current ScaleBridge generation foundation against:

- the native PNNL ASHRAE 90.1-2013 OfficeSmall Seattle IDF;
- the matching Seattle TMY3 EPW;
- EnergyPlus 9.0.1 discovered by opyplus 2.0.7;
- the complete ScaleBridge generation orchestrator;
- canonical Parquet and EIO metadata extraction; and
- optional legacy pickle compatibility output.

The source files remain unchanged. Outputs are written beneath
``SCALEBRIDGE_GENERATED_DATA_ROOT/smoke_tests/office_small_seattle``.
"""

from __future__ import annotations

import json
from pathlib import Path

import opyplus

from scalebridge.integration.energyplus import (
    CaseSpec,
    EnergyPlusGenerationOrchestrator,
    OutputVariableRequest,
    RunPeriod,
    RunStatus,
)
from scalebridge.integration.energyplus.prototypes import (
    COMMERCIAL_TMY3_BY_LOCATION,
    resolve_external_data_root,
    resolve_generated_data_root,
    sha256_file,
)


def build_smoke_case(data_root: Path) -> CaseSpec:
    """Build the native EnergyPlus 9.0 OfficeSmall Seattle smoke case."""
    idf_path = (
        data_root
        / "Commercial_Prototypes"
        / "ASHRAE"
        / "90_1_2013"
        / "ASHRAE901_OfficeSmall_STD2013_Seattle.idf"
    )
    epw_path = (
        data_root
        / "TMY3_WeatherFiles_Commercial"
        / COMMERCIAL_TMY3_BY_LOCATION["Seattle"]
    )

    if not idf_path.is_file():
        raise FileNotFoundError(f"OfficeSmall Seattle IDF not found: {idf_path}")
    if not epw_path.is_file():
        raise FileNotFoundError(f"Seattle EPW not found: {epw_path}")

    return CaseSpec(
        case_name="pnnl_office_small_2013_seattle_smoke",
        building_type="OfficeSmall",
        prototype_standard="ASHRAE 90.1",
        prototype_year="2013",
        climate_zone="4C",
        weather_location="Seattle",
        idf_path=idf_path,
        epw_path=epw_path,
        idf_sha256=sha256_file(idf_path),
        epw_sha256=sha256_file(epw_path),
        run_period=RunPeriod(
            start_month=1,
            start_day=1,
            end_month=1,
            end_day=2,
            calendar_year=2013,
        ),
        timestep_minutes=5,
        output_variables=(
            OutputVariableRequest(
                variable_name="Zone Air Temperature",
                semantic_role="zone_temperature",
            ),
            OutputVariableRequest(
                variable_name="Site Outdoor Air Drybulb Temperature",
                semantic_role="outdoor_temperature",
            ),
            OutputVariableRequest(
                variable_name="Zone Air System Sensible Heating Rate",
                semantic_role="hvac_heating_rate",
            ),
            OutputVariableRequest(
                variable_name="Zone Air System Sensible Cooling Rate",
                semantic_role="hvac_cooling_rate",
            ),
        ),
        energyplus_version="9.0.1",
        write_legacy_pickles=True,
        preserve_raw_outputs=True,
        tags={"purpose": "real_integration_smoke_test"},
    )


def main() -> None:
    """Prepare, execute, validate, and summarize the real smoke test."""
    # ----------------------------------------------------------------------
    # Phase 1: Resolve configured roots and validate EnergyPlus discovery.
    # ----------------------------------------------------------------------
    data_root = resolve_external_data_root()
    generated_root = resolve_generated_data_root()
    energyplus_root = Path(opyplus.get_eplus_base_dir_path((9, 0, 1)))

    # ----------------------------------------------------------------------
    # Phase 2: Build the case and isolated smoke-test run directories.
    # ----------------------------------------------------------------------
    case_spec = build_smoke_case(data_root)
    case_root = (
        generated_root
        / "smoke_tests"
        / "office_small_seattle"
        / case_spec.case_id
    )

    # ----------------------------------------------------------------------
    # Phase 3: Execute the complete loop-safe generation workflow.
    # ----------------------------------------------------------------------
    result = EnergyPlusGenerationOrchestrator(
        generated_data_root=generated_root,
        case_collection_name="smoke_tests",
    ).generate(
        case_spec,
        campaign_id="office_small_seattle_smoke",
        case_root=case_root,
    )

    # ----------------------------------------------------------------------
    # Phase 4: Print a compact human-readable smoke-test summary.
    # ----------------------------------------------------------------------
    metadata_path = result.canonical_output_paths.get("metadata.json")
    canonical_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path is not None
        else {}
    )
    completed_statuses = {
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_WITH_WARNINGS,
    }
    summary = {
        "case_id": case_spec.case_id,
        "run_id": result.run_id,
        "energyplus_root": str(energyplus_root),
        "prepared_idf": str(result.artifact_root / "inputs" / "prepared.idf"),
        "simulation_directory": str(result.artifact_root / "raw"),
        "manifest": str(result.manifest_path),
        "status": result.status.value,
        "completed_successfully": result.status in completed_statuses,
        "warnings": result.warning_count,
        "severe_errors": result.severe_count,
        "fatal_errors": result.fatal_count,
        "runtime_seconds": result.runtime_seconds,
        "failure_message": result.error_message,
        "canonical_parquet": {
            name: str(path)
            for name, path in result.canonical_output_paths.items()
            if path.suffix.casefold() == ".parquet"
        },
        "canonical_row_count": canonical_metadata.get("row_count", 0),
        "produced_signal_count": result.produced_signal_count,
        "timestep_count": result.timestep_count or 0,
        "legacy_output_pickle": _optional_path_string(
            result.compatibility_output_paths.get(
                "IDF_OutputVariables_DictDF.pickle",
            ),
        ),
        "legacy_eio_pickle": _optional_path_string(
            result.compatibility_output_paths.get("Eio_OutputFile.pickle"),
        ),
        "legacy_pickles_created": (
            "IDF_OutputVariables_DictDF.pickle"
            in result.compatibility_output_paths
            and "Eio_OutputFile.pickle" in result.compatibility_output_paths
        ),
    }
    summary_path = result.artifact_root / "smoke_test_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest_summary_path = case_root / "latest_smoke_test_summary.json"
    latest_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    print(f"Latest summary: {latest_summary_path}")

    if result.status not in completed_statuses:
        raise SystemExit(1)


def _optional_path_string(path: Path | None) -> str | None:
    """Return a filesystem path string while preserving missing values."""
    return str(path) if path is not None else None


if __name__ == "__main__":
    main()
