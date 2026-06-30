"""Run all P1 cases or one machine's fixed share of the campaign.

Without ``--machine-number``, one machine executes all 64 cases. When the
option is set to 1, 2, 3, or 4, the machine executes its 16 non-overlapping
cases. All execution is sequential and uses no distributed-computing
framework, central scheduler, or worker service.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from scalebridge.integration.energyplus import (
    EnergyPlusGenerationOrchestrator,
    P1_CAMPAIGN_ID,
    RunStatus,
    build_p1_case_specs,
    resolve_generated_data_root,
)
from scalebridge.integration.energyplus.generation.variable_wise import (
    generate_variable_wise_case,
    resolve_parallel_variable_workers,
    safe_variable_id,
)
from scalebridge.tracking.mlflow import MLflowGenerationTracker


MACHINE_COUNT = 4
SUCCESS_STATUSES = {
    RunStatus.COMPLETED.value,
    RunStatus.COMPLETED_WITH_WARNINGS.value,
}


def main() -> int:
    """Run all cases or one machine's share and summarize their outcomes."""
    arguments = _parse_arguments()
    all_cases = build_p1_case_specs(
        write_legacy_pickles=arguments.write_legacy_pickles
    )
    all_cases = _filter_cases(
        all_cases,
        building_type=arguments.building_type,
        weather_location=arguments.weather_location,
    )
    if not all_cases:
        raise SystemExit(
            "No cases matched the requested filters: "
            f"building_type={arguments.building_type!r}, "
            f"weather_location={arguments.weather_location!r}"
        )
    if arguments.machine_number is None:
        selected_cases = all_cases
        allocation = "all cases on one machine"
    else:
        selected_cases = all_cases[
            arguments.machine_number - 1 :: MACHINE_COUNT
        ]
        allocation = (
            f"machine {arguments.machine_number}/{MACHINE_COUNT}"
        )
    if arguments.case_limit is not None:
        selected_cases = selected_cases[: arguments.case_limit]

    generated_root = resolve_generated_data_root()
    print(f"Campaign: {P1_CAMPAIGN_ID}")
    print(f"Allocation: {allocation}")
    print(f"Machine ID: {arguments.machine_id}")
    print(f"Selected cases: {len(selected_cases)}")
    print(f"Generated root: {generated_root}")
    print(f"Generation mode: {arguments.generation_mode}")
    if arguments.variable_limit is not None:
        print(f"Variable limit: {arguments.variable_limit}")

    if arguments.generation_mode == "variable-wise":
        preview_requests = _selected_output_variables(
            selected_cases[0].output_variables,
            variable_limit=arguments.variable_limit,
        ) if selected_cases else ()
        resolved_workers = resolve_parallel_variable_workers(
            machine_id=arguments.machine_id,
            requested=arguments.parallel_variable_workers,
            variable_count=len(preview_requests),
        )
        print(f"Parallel variable workers: {resolved_workers}")

    if arguments.dry_run:
        for case in selected_cases:
            print(
                f"DRY RUN: {case.case_id} "
                f"{case.building_type}/{case.weather_location}"
            )
            if arguments.generation_mode == "variable-wise":
                requests = _selected_output_variables(
                    case.output_variables,
                    variable_limit=arguments.variable_limit,
                )
                for index, request in enumerate(requests, start=1):
                    print(
                        f"  VARIABLE {index}: "
                        f"{safe_variable_id(request)} | "
                        f"{request.variable_name} | "
                        f"{request.reporting_frequency}"
                    )
        return 0

    mlflow_tracker = MLflowGenerationTracker(
        experiment_name=arguments.mlflow_experiment_name,
        artifact_subdir=arguments.mlflow_artifact_subdir,
        enabled=not arguments.disable_mlflow,
        strict=arguments.mlflow_strict,
    )

    print(
        "MLflow generation tracking: "
        + ("enabled" if mlflow_tracker.enabled else "disabled")
    )

    orchestrator = EnergyPlusGenerationOrchestrator(
        generated_data_root=generated_root,
        case_collection_name=f"campaigns/{P1_CAMPAIGN_ID}/generation",
        machine_id=arguments.machine_id,
        mlflow_tracker=mlflow_tracker,
    )
    status_counts: Counter[str] = Counter()

    for position, case in enumerate(selected_cases, start=1):
        case_root = _case_root(generated_root, case.case_id)
        if not arguments.rerun_completed and _latest_run_succeeded(case_root):
            status_counts["skipped_completed"] += 1
            print(
                f"[{position}/{len(selected_cases)}] SKIP: "
                f"{case.building_type}/{case.weather_location}"
            )
            continue

        print(
            f"[{position}/{len(selected_cases)}] RUN: "
            f"{case.building_type}/{case.weather_location}"
        )

        if arguments.generation_mode == "standard":
            result = orchestrator.generate(case, campaign_id=P1_CAMPAIGN_ID)
        else:
            result = generate_variable_wise_case(
                case,
                generated_data_root=generated_root,
                case_collection_name=f"campaigns/{P1_CAMPAIGN_ID}/generation",
                machine_id=arguments.machine_id,
                campaign_id=P1_CAMPAIGN_ID,
                selected_output_variables=_selected_output_variables(
                    case.output_variables,
                    variable_limit=arguments.variable_limit,
                ),
                delete_raw_csv=True,
                mlflow_tracker=mlflow_tracker,
                short_work_root=os.environ.get("SCALEBRIDGE_EPLUS_WORK_ROOT"),
                parallel_variable_workers=arguments.parallel_variable_workers,
            )
        status_counts[result.status.value] += 1

        print(
            f"[{position}/{len(selected_cases)}] "
            f"{result.status.value.upper()}: {result.case_id} "
            f"({result.runtime_seconds:.1f} seconds)"
        )

    print("Summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    unsuccessful = (
        status_counts[RunStatus.FAILED.value]
        + status_counts[RunStatus.INVALID.value]
    )
    return 1 if unsuccessful else 0


def _parse_arguments() -> argparse.Namespace:
    """Parse the small command interface used on one or four machines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--machine-number",
        type=int,
        choices=range(1, MACHINE_COUNT + 1),
        help="Run share 1, 2, 3, or 4; omit to run all 64 cases.",
    )
    parser.add_argument(
        "--machine-id",
        required=True,
        help="Stable descriptive name such as laptop or home-pc.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Run only the first N selected cases for controlled testing.",
    )
    parser.add_argument(
        "--building-type",
        help=(
            "Optional case filter, for example OfficeSmall. "
            "Case-sensitive match against CaseSpec.building_type."
        ),
    )
    parser.add_argument(
        "--weather-location",
        help=(
            "Optional case filter, for example Seattle. "
            "Case-sensitive match against CaseSpec.weather_location."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected cases without running EnergyPlus.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run cases again even when their latest attempt succeeded.",
    )
    parser.add_argument(
        "--write-legacy-pickles",
        action="store_true",
        help="Also create the two legacy aggregation pickle files.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow generation tracking for this run.",
    )
    parser.add_argument(
        "--mlflow-strict",
        action="store_true",
        help="Fail the campaign if MLflow logging fails.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default=f"{P1_CAMPAIGN_ID}_generation",
        help="MLflow experiment name for P1 generation runs.",
    )
    parser.add_argument(
        "--mlflow-artifact-subdir",
        default=P1_CAMPAIGN_ID,
        help=(
            "Semantic artifact subdirectory under "
            "SCALEBRIDGE_GENERATED_DATA_ROOT/mlflow_artifacts."
        ),
    )
    parser.add_argument(
        "--generation-mode",
        choices=("standard", "variable-wise"),
        default="standard",
        help=(
            "Generation strategy. Use standard for one all-variable EnergyPlus "
            "run per case, or variable-wise for one EnergyPlus run per requested "
            "output variable."
        ),
    )
    parser.add_argument(
        "--variable-limit",
        type=int,
        help=(
            "For variable-wise generation, run only the first N requested "
            "variables. Intended for smoke testing."
        ),
    )
    parser.add_argument(
        "--parallel-variable-workers",
        type=int,
        help=(
            "Number of concurrent EnergyPlus variable-wise workers. "
            "If omitted, a machine-aware default is used."
        ),
    )
    arguments = parser.parse_args()
    if arguments.case_limit is not None and arguments.case_limit < 1:
        parser.error("--case-limit must be at least 1")
    if arguments.variable_limit is not None and arguments.variable_limit < 1:
        parser.error("--variable-limit must be at least 1")
    if arguments.variable_limit is not None and arguments.generation_mode != "variable-wise":
        parser.error("--variable-limit requires --generation-mode variable-wise")
    if (
        arguments.parallel_variable_workers is not None
        and arguments.parallel_variable_workers < 1
    ):
        parser.error("--parallel-variable-workers must be at least 1")
    if (
        arguments.parallel_variable_workers is not None
        and arguments.generation_mode != "variable-wise"
    ):
        parser.error(
            "--parallel-variable-workers requires --generation-mode variable-wise"
        )
    return arguments

def _filter_cases(
    cases,
    *,
    building_type: str | None,
    weather_location: str | None,
):
    """Filter campaign cases before machine allocation."""
    filtered = tuple(cases)

    if building_type:
        filtered = tuple(
            case for case in filtered if case.building_type == building_type
        )

    if weather_location:
        filtered = tuple(
            case for case in filtered if case.weather_location == weather_location
        )

    return filtered

def _selected_output_variables(
    output_variables,
    *,
    variable_limit: int | None,
):
    """Return the output-variable subset used by the selected generation mode."""
    if variable_limit is None:
        return tuple(output_variables)
    return tuple(output_variables[:variable_limit])

def _case_root(generated_root: Path, case_id: str) -> Path:
    """Return the shared case directory used by the orchestrator."""
    return (
        generated_root
        / "campaigns"
        / P1_CAMPAIGN_ID
        / "generation"
        / "cases"
        / case_id
    )


def _latest_run_succeeded(case_root: Path) -> bool:
    """Return whether the latest persisted run completed successfully."""
    pointer_path = case_root / "latest_run.json"
    if not pointer_path.is_file():
        return False
    try:
        payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") in SUCCESS_STATUSES


if __name__ == "__main__":
    raise SystemExit(main())