"""Run all P1 cases or one machine's fixed share of the campaign.

Without ``--machine-number``, one machine executes all 64 cases. When the
option is set to 1, 2, 3, or 4, the machine executes its 16 non-overlapping
cases. All execution is sequential and uses no distributed-computing
framework, central scheduler, or worker service.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scalebridge.integration.energyplus import (
    EnergyPlusGenerationOrchestrator,
    P1_CAMPAIGN_ID,
    RunStatus,
    build_p1_case_specs,
    resolve_generated_data_root,
)


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

    if arguments.dry_run:
        for case in selected_cases:
            print(
                f"DRY RUN: {case.case_id} "
                f"{case.building_type}/{case.weather_location}"
            )
        return 0

    orchestrator = EnergyPlusGenerationOrchestrator(
        generated_data_root=generated_root,
        case_collection_name=f"campaigns/{P1_CAMPAIGN_ID}/generation",
        machine_id=arguments.machine_id,
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
        result = orchestrator.generate(case, campaign_id=P1_CAMPAIGN_ID)
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
    arguments = parser.parse_args()
    if arguments.case_limit is not None and arguments.case_limit < 1:
        parser.error("--case-limit must be at least 1")
    return arguments


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
