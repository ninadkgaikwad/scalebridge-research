"""
Run the compact/diverse P1 EnergyPlus campaign.

Compact campaign identity
-------------------------
Campaign:
    p1_ashrae2013_one_zone_compact_4b4c

Buildings:
    RestaurantFastFood
    OfficeSmall
    RetailStripmall
    ApartmentMidRise

Default weather locations:
    All weather locations already present in the P1 ASHRAE 2013 case specs.
    For the current P1 setup this should be the selected four climate zones.

Data-root policy
----------------
This runner does NOT write repo-root scratch folders.

Generated campaign data goes under:

    <SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/generation/...

Normalized pre-opyplus IDFs go under:

    <SCALEBRIDGE_GENERATED_DATA_ROOT>/campaigns/<campaign_id>/normalization/idfs/<case_id>/normalized.idf

Expected local environment variable:

    SCALEBRIDGE_GENERATED_DATA_ROOT =
    C:\\Users\\ninad\\Dropbox\\NinadGaikwad_PhD\\Gaikwad_Research\\From_WSU_OneDrive\\BuildingModelingProject_Condensed\\Data\\ScaleBridge

Why this runner exists
----------------------
The full P1 campaign is:

    16 PNNL commercial prototype buildings
    x 4 climate/weather locations
    x 35 EnergyPlus output variables

The compact campaign is designed as a fast but still diverse first paper-scale
campaign before expanding to all 64 building-weather cases.

Important normalization
-----------------------
ApartmentMidRise was found to reference "Control Type" as a Schedule Type
Limits Name without defining ScheduleTypeLimits, Control Type. EnergyPlus may
tolerate the original IDF, but opyplus fails while loading it.

This runner writes a normalized copy of each selected source IDF under the
ScaleBridge data root and updates the CaseSpec to point to the normalized copy
before calling the existing variable-wise generation function.

Original DOE/PNNL IDFs are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys
from typing import Iterable, Sequence

from scalebridge.integration.energyplus import build_p1_case_specs
from scalebridge.integration.energyplus.generation.variable_wise import (
    generate_variable_wise_case,
)
from scalebridge.integration.energyplus.idf.pre_opyplus_normalization import (
    normalize_idf_before_opyplus,
)


DEFAULT_CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c"

DEFAULT_BUILDING_TYPES: tuple[str, ...] = (
    "RestaurantFastFood",
    "OfficeSmall",
    "RetailStripmall",
    "ApartmentMidRise",
)

SUCCESS_STATUSES = {
    "completed",
    "completed_with_warnings",
}


@dataclass(frozen=True)
class CompactCaseRecord:
    """Selected compact campaign case plus normalization metadata."""

    case_spec: object
    source_idf_path: Path
    normalized_idf_path: Path
    idf_patches: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact/diverse P1 EnergyPlus variable-wise campaign."
    )

    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help=f"Campaign ID. Default: {DEFAULT_CAMPAIGN_ID}",
    )
    parser.add_argument(
        "--machine-id",
        required=True,
        help="Machine identifier, for example laptop, home-pc, lab-pc, kamiak.",
    )
    parser.add_argument(
        "--building-type",
        action="append",
        choices=DEFAULT_BUILDING_TYPES,
        help=(
            "Building type to include. Can be repeated. "
            "Default: all compact building types."
        ),
    )
    parser.add_argument(
        "--weather-location",
        action="append",
        help=(
            "Weather location to include. Can be repeated. "
            "Default: all weather locations available in selected compact cases."
        ),
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional limit after filtering and sorting.",
    )
    parser.add_argument(
        "--variable-limit",
        type=int,
        default=None,
        help=(
            "Optional number of output variables per case. "
            "Use for smoke tests. Omit for all variables."
        ),
    )
    parser.add_argument(
        "--parallel-variable-workers",
        type=int,
        default=1,
        help="Number of variable-wise workers per case.",
    )
    parser.add_argument(
        "--write-legacy-pickles",
        action="store_true",
        help="Also write legacy per-variable pickle artifacts.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun cases even if latest_run.json indicates success.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected cases and exit without running EnergyPlus.",
    )
    parser.add_argument(
        "--generated-data-root",
        default=None,
        help=(
            "Generated data root. If omitted, uses SCALEBRIDGE_GENERATED_DATA_ROOT "
            "or ../../Data/ScaleBridge relative to repo root."
        ),
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=None,
        help="MLflow experiment name. Default: <campaign-id>_generation.",
    )
    parser.add_argument(
        "--mlflow-strict",
        action="store_true",
        help="Fail if MLflow logging fails.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    generated_data_root = resolve_generated_data_root(args.generated_data_root)
    campaign_id = args.campaign_id
    case_collection_name = campaign_case_collection_name(campaign_id)

    building_types = (
        tuple(args.building_type)
        if args.building_type
        else DEFAULT_BUILDING_TYPES
    )

    source_cases = build_p1_case_specs(
        write_legacy_pickles=args.write_legacy_pickles,
    )

    selected_cases = select_cases(
        source_cases=source_cases,
        building_types=building_types,
        weather_locations=tuple(args.weather_location) if args.weather_location else None,
        case_limit=args.case_limit,
    )

    compact_records = normalize_selected_cases(
        selected_cases=selected_cases,
        campaign_id=campaign_id,
        generated_data_root=generated_data_root,
    )

    print_compact_campaign_plan(
        records=compact_records,
        campaign_id=campaign_id,
        generated_data_root=generated_data_root,
        variable_limit=args.variable_limit,
        parallel_variable_workers=args.parallel_variable_workers,
        write_legacy_pickles=args.write_legacy_pickles,
    )

    if args.dry_run:
        print()
        print("Dry run complete. No EnergyPlus simulations were launched.")
        return 0

    mlflow_tracker = None
    if not args.disable_mlflow:
        mlflow_tracker = create_mlflow_tracker(args, campaign_id)

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    for index, record in enumerate(compact_records, start=1):
        case_spec = record.case_spec
        case_root = case_generation_root(
                                            generated_data_root=generated_data_root,
                                            case_collection_name=case_collection_name,
                                            case_id=case_spec.case_id,
                                        )

        print()
        print("=" * 100)
        print(
            f"[{index}/{len(compact_records)}] "
            f"{case_spec.building_type} / {case_spec.weather_location}"
        )
        print(f"case_id: {case_spec.case_id}")
        print(f"case_root: {case_root}")
        print(f"source_idf: {record.source_idf_path}")
        print(f"normalized_idf: {record.normalized_idf_path}")
        print(f"idf_patches: {record.idf_patches or '(none)'}")

        if not args.rerun_completed and latest_run_succeeded(case_root):
            print("Skipping: latest_run.json indicates completed status.")
            skipped_count += 1
            continue

        selected_variables = select_output_variables(
            case_spec.output_variables,
            variable_limit=args.variable_limit,
        )

        try:
            generate_variable_wise_case(
                                            case_spec=case_spec,
                                            generated_data_root=generated_data_root,
                                            campaign_id=campaign_id,
                                            case_collection_name=case_collection_name,
                                            machine_id=args.machine_id,
                                            selected_output_variables=selected_variables,
                                            delete_raw_csv=True,
                                            mlflow_tracker=mlflow_tracker,
                                            short_work_root=os.environ.get("SCALEBRIDGE_EPLUS_WORK_ROOT"),
                                            parallel_variable_workers=args.parallel_variable_workers,
                                        )
            completed_count += 1

        except Exception as exc:
            failed_count += 1
            print()
            print("FAILED")
            print(f"{type(exc).__name__}: {exc}")

            if args.mlflow_strict:
                raise

    print()
    print("=" * 100)
    print("COMPACT CAMPAIGN SUMMARY")
    print(f"campaign_id: {campaign_id}")
    print(f"selected_cases: {len(compact_records)}")
    print(f"completed_or_launched: {completed_count}")
    print(f"skipped: {skipped_count}")
    print(f"failed: {failed_count}")

    return 1 if failed_count else 0


def select_cases(
    source_cases: Iterable[object],
    building_types: tuple[str, ...],
    weather_locations: tuple[str, ...] | None,
    case_limit: int | None,
) -> list[object]:
    building_order = {name: index for index, name in enumerate(building_types)}

    selected = [
        case
        for case in source_cases
        if case.building_type in building_types
        and (weather_locations is None or case.weather_location in weather_locations)
    ]

    selected.sort(
        key=lambda case: (
            building_order.get(case.building_type, 999),
            str(case.weather_location),
            str(case.case_id),
        )
    )

    if not selected:
        available = sorted({case.building_type for case in source_cases})
        raise RuntimeError(
            "No cases selected for compact campaign. "
            f"Requested building types: {list(building_types)}. "
            f"Available building types: {available}"
        )

    # Only require all compact buildings when running the full filtered campaign.
    # For smoke tests, --case-limit intentionally allows a subset such as 1 case.
    if case_limit is None:
        missing_buildings = sorted(
            set(building_types) - {case.building_type for case in selected}
        )
        if missing_buildings:
            available = sorted({case.building_type for case in source_cases})
            raise RuntimeError(
                "Missing compact building types after filtering: "
                f"{missing_buildings}. Available building types: {available}"
            )

    if case_limit is not None:
        if case_limit <= 0:
            raise ValueError("--case-limit must be positive when provided.")
        selected = selected[:case_limit]

    return selected


def normalize_selected_cases(
    selected_cases: Sequence[object],
    campaign_id: str,
    generated_data_root: Path,
) -> list[CompactCaseRecord]:
    records: list[CompactCaseRecord] = []

    for case_spec in selected_cases:
        normalized_idf_path = normalized_idf_path_for_case(
            generated_data_root=generated_data_root,
            campaign_id=campaign_id,
            case_id=case_spec.case_id,
        )

        normalization = normalize_idf_before_opyplus(
            source_idf_path=Path(case_spec.idf_path),
            normalized_idf_path=normalized_idf_path,
        )

        patched_case_spec = case_spec.model_copy(
            update={
                "idf_path": normalization.normalized_idf_path,
                "tags": build_case_tags(
                    case_spec=case_spec,
                    campaign_id=campaign_id,
                    idf_patches=normalization.applied_patches,
                ),
            }
        )

        records.append(
            CompactCaseRecord(
                case_spec=patched_case_spec,
                source_idf_path=normalization.source_idf_path,
                normalized_idf_path=normalization.normalized_idf_path,
                idf_patches=normalization.applied_patches,
            )
        )

    return records


def normalized_idf_path_for_case(
    generated_data_root: Path,
    campaign_id: str,
    case_id: str,
) -> Path:
    return (
        generated_data_root
        / "campaigns"
        / campaign_id
        / "normalization"
        / "idfs"
        / str(case_id)
        / "normalized.idf"
    )


def build_case_tags(
    case_spec: object,
    campaign_id: str,
    idf_patches: tuple[str, ...],
) -> dict[str, str]:
    original_tags = getattr(case_spec, "tags", None)

    if isinstance(original_tags, dict):
        tags = dict(original_tags)
    else:
        tags = {}

    tags["campaign_id"] = campaign_id
    tags["campaign_variant"] = "compact_4b4c"
    tags["compact_building_set"] = ",".join(DEFAULT_BUILDING_TYPES)
    tags["idf_pre_opyplus_normalization"] = "true"
    tags["idf_patches"] = "; ".join(idf_patches) if idf_patches else "none"

    return tags


def select_output_variables(
    output_variables: Sequence[object],
    variable_limit: int | None,
) -> tuple[object, ...]:
    if variable_limit is None:
        return tuple(output_variables)

    if variable_limit <= 0:
        raise ValueError("--variable-limit must be positive when provided.")

    return tuple(output_variables[:variable_limit])


def latest_run_succeeded(case_root: Path) -> bool:
    latest_run_path = case_root / "latest_run.json"
    if not latest_run_path.is_file():
        return False

    try:
        payload = json.loads(latest_run_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    status = str(payload.get("status", "")).lower()
    return status in SUCCESS_STATUSES

def campaign_case_collection_name(campaign_id: str) -> str:
    return str(Path("campaigns") / campaign_id / "generation")

def case_generation_root(
    generated_data_root: Path,
    case_collection_name: str,
    case_id: str,
) -> Path:
    return (
        generated_data_root
        / case_collection_name
        / "cases"
        / str(case_id)
    )


def resolve_generated_data_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()

    env_value = os.environ.get("SCALEBRIDGE_GENERATED_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root.parent.parent / "Data" / "ScaleBridge").resolve()


def create_mlflow_tracker(args: argparse.Namespace, campaign_id: str):
    experiment_name = args.mlflow_experiment_name or f"{campaign_id}_generation"

    try:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    except Exception as exc:
        if args.mlflow_strict:
            raise
        print()
        print("WARNING: Could not configure MLflow tracking URI.")
        print(f"{type(exc).__name__}: {exc}")
        print("Continuing without MLflow.")
        return None

    try:
        from scalebridge.tracking.mlflow import MLflowGenerationTracker
    except Exception as exc:
        if args.mlflow_strict:
            raise
        print()
        print("WARNING: Could not import MLflowGenerationTracker.")
        print(f"{type(exc).__name__}: {exc}")
        print("Continuing without MLflow.")
        return None

    try:
        return MLflowGenerationTracker(
            experiment_name=experiment_name,
            strict=args.mlflow_strict,
        )
    except TypeError:
        try:
            return MLflowGenerationTracker(
                experiment_name=experiment_name,
            )
        except TypeError:
            return MLflowGenerationTracker()


def print_compact_campaign_plan(
    records: Sequence[CompactCaseRecord],
    campaign_id: str,
    generated_data_root: Path,
    variable_limit: int | None,
    parallel_variable_workers: int,
    write_legacy_pickles: bool,
) -> None:
    print("=" * 100)
    print("P1 COMPACT CAMPAIGN PLAN")
    print("=" * 100)
    print(f"campaign_id: {campaign_id}")
    print(f"generated_data_root: {generated_data_root}")
    print(f"selected_case_count: {len(records)}")
    print(f"variable_limit: {variable_limit if variable_limit is not None else 'all'}")
    print(f"parallel_variable_workers: {parallel_variable_workers}")
    print(f"write_legacy_pickles: {write_legacy_pickles}")
    print()
    print("Selected cases:")
    print(
        "index,building_type,weather_location,case_id,"
        "output_variable_count,normalized_idf,patches"
    )

    for index, record in enumerate(records, start=1):
        case_spec = record.case_spec
        variable_count = (
            min(len(case_spec.output_variables), variable_limit)
            if variable_limit is not None
            else len(case_spec.output_variables)
        )

        patches = "; ".join(record.idf_patches) if record.idf_patches else "none"

        print(
            f"{index},"
            f"{case_spec.building_type},"
            f"{case_spec.weather_location},"
            f"{case_spec.case_id},"
            f"{variable_count},"
            f"{record.normalized_idf_path},"
            f"{patches}"
        )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))