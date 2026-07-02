from __future__ import annotations

import csv
import shutil
import traceback
from pathlib import Path
from datetime import datetime

from scalebridge.integration.energyplus import build_p1_case_specs
from scalebridge.integration.energyplus.idf import IdfPreparer
from scalebridge.integration.energyplus.simulation import EnergyPlusRunner


TARGET_BUILDINGS = (
    "RestaurantFastFood",
    "OfficeSmall",
    "SchoolPrimary",
    "ApartmentMidRise",
)

TARGET_WEATHER = "Seattle"

OUTPUT_ROOT = Path("scratch") / "p1_compact_seattle_1day_opyplus_smoke"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
SUMMARY_TXT = OUTPUT_ROOT / "summary.txt"


def make_one_day_case(case):
    """Return a one-day version of a CaseSpec.

    Keeps source IDF/EPW unchanged. The prepared IDF written by IdfPreparer
    should receive the one-day run period.
    """
    run_period = case.run_period

    if hasattr(run_period, "model_copy"):
        one_day_run_period = run_period.model_copy(
            update={
                "start_month": 1,
                "start_day": 1,
                "end_month": 1,
                "end_day": 1,
                "calendar_year": 2013,
            }
        )
    else:
        one_day_run_period = {
            **dict(run_period),
            "start_month": 1,
            "start_day": 1,
            "end_month": 1,
            "end_day": 1,
            "calendar_year": 2013,
        }

    # Use only the first requested variable to keep the smoke test small.
    # The objective here is EnergyPlus/opyplus run viability, not data coverage.
    one_variable = tuple(case.output_variables[:1])

    return case.model_copy(
        update={
            "run_period": one_day_run_period,
            "output_variables": one_variable,
            "write_legacy_pickles": False,
            "preserve_raw_outputs": True,
        }
    )


def main():
    cases = build_p1_case_specs(write_legacy_pickles=False)

    selected = [
        case
        for case in cases
        if case.building_type in TARGET_BUILDINGS
        and case.weather_location == TARGET_WEATHER
    ]

    selected_by_building = {case.building_type: case for case in selected}

    missing = [b for b in TARGET_BUILDINGS if b not in selected_by_building]
    if missing:
        print("Missing requested building/weather cases:")
        for item in missing:
            print("  " + item)
        print()
        print("Available Seattle buildings:")
        for case in sorted(
            [c for c in cases if c.weather_location == TARGET_WEATHER],
            key=lambda c: c.building_type,
        ):
            print("  " + case.building_type)
        raise SystemExit(2)

    idf_preparer = IdfPreparer()
    runner = EnergyPlusRunner(beat_frequency_seconds=5)

    rows = []

    for index, building in enumerate(TARGET_BUILDINGS, start=1):
        case = selected_by_building[building]
        smoke_case = make_one_day_case(case)

        case_root = OUTPUT_ROOT / f"{index:02d}_{building}"
        prepared_idf = case_root / "inputs" / "prepared_1day.idf"
        output_dir = case_root / "opyplus_output"

        if case_root.exists():
            shutil.rmtree(case_root)
        prepared_idf.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 100)
        print(f"[{index}/{len(TARGET_BUILDINGS)}] {building} / {TARGET_WEATHER}")
        print(f"source idf: {case.idf_path}")
        print(f"epw:        {case.epw_path}")
        print(f"output:     {output_dir}")

        started = datetime.now()

        row = {
            "building_type": building,
            "weather_location": TARGET_WEATHER,
            "status": "failed",
            "exit_code": "",
            "warnings": "",
            "severe_errors": "",
            "fatal_errors": "",
            "runtime_seconds": "",
            "prepared_idf": str(prepared_idf),
            "output_dir": str(output_dir),
            "err_file": str(output_dir / "eplusout.err"),
            "csv_file_exists": "",
            "eso_file_exists": "",
            "message": "",
        }

        try:
            preparation = idf_preparer.prepare(
                smoke_case,
                prepared_idf,
            )

            result = runner.run(
                idf_path=preparation.prepared_idf_path,
                epw_path=case.epw_path,
                output_directory=output_dir,
            )

            elapsed = (datetime.now() - started).total_seconds()

            row.update(
                {
                    "status": "completed" if result.completed_successfully else "failed",
                    "exit_code": getattr(result, "exit_code", ""),
                    "warnings": getattr(result, "warning_count", ""),
                    "severe_errors": getattr(result, "severe_count", ""),
                    "fatal_errors": getattr(result, "fatal_count", ""),
                    "runtime_seconds": elapsed,
                    "csv_file_exists": (output_dir / "eplusout.csv").is_file(),
                    "eso_file_exists": (output_dir / "eplusout.eso").is_file(),
                    "message": getattr(result, "failure_message", "") or "",
                }
            )

            print(f"status:     {row['status']}")
            print(f"warnings:   {row['warnings']}")
            print(f"severe:     {row['severe_errors']}")
            print(f"fatal:      {row['fatal_errors']}")
            print(f"runtime_s:  {elapsed:.1f}")
            print(f"csv exists: {row['csv_file_exists']}")
            print(f"eso exists: {row['eso_file_exists']}")

        except Exception as exc:
            elapsed = (datetime.now() - started).total_seconds()
            tb_path = case_root / "traceback.txt"
            tb_path.write_text(traceback.format_exc(), encoding="utf-8")

            row.update(
                {
                    "status": "exception",
                    "runtime_seconds": elapsed,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

            print("status: exception")
            print(row["message"])
            print(f"traceback: {tb_path}")

        rows.append(row)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("P1 Compact Seattle 1-Day opyplus Smoke Test")
    lines.append("=" * 80)
    lines.append(f"Output root: {OUTPUT_ROOT.resolve()}")
    lines.append(f"Summary CSV: {SUMMARY_CSV.resolve()}")
    lines.append("")
    for row in rows:
        lines.append(
            f"{row['building_type']}: "
            f"{row['status']} | "
            f"warnings={row['warnings']} | "
            f"severe={row['severe_errors']} | "
            f"fatal={row['fatal_errors']} | "
            f"runtime_s={row['runtime_seconds']}"
        )
        if row["message"]:
            lines.append(f"  message: {row['message']}")

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("=" * 100)
    print("SUMMARY")
    print(SUMMARY_TXT.read_text(encoding="utf-8"))
    print(f"CSV: {SUMMARY_CSV.resolve()}")
    print(f"TXT: {SUMMARY_TXT.resolve()}")

    failed = [row for row in rows if row["status"] != "completed"]
    severe_or_fatal = [
        row
        for row in rows
        if str(row["severe_errors"]) not in ("", "0")
        or str(row["fatal_errors"]) not in ("", "0")
    ]

    if failed or severe_or_fatal:
        raise SystemExit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
