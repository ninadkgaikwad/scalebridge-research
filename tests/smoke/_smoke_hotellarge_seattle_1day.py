from __future__ import annotations

import csv
import shutil
import traceback
from pathlib import Path
from datetime import datetime

from scalebridge.integration.energyplus import build_p1_case_specs
from scalebridge.integration.energyplus.idf import IdfPreparer
from scalebridge.integration.energyplus.simulation import EnergyPlusRunner


TARGET_BUILDING = "HotelLarge"
TARGET_WEATHER = "Seattle"

OUTPUT_ROOT = Path("scratch") / "p1_compact_hotellarge_seattle_1day_smoke"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
SUMMARY_TXT = OUTPUT_ROOT / "summary.txt"


def make_one_day_case(case):
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
        if case.building_type == TARGET_BUILDING
        and case.weather_location == TARGET_WEATHER
    ]

    if not selected:
        print("Could not find target case.")
        print("Available Seattle cases:")
        for case in sorted(
            [c for c in cases if c.weather_location == TARGET_WEATHER],
            key=lambda c: c.building_type,
        ):
            print("  " + case.building_type)
        raise SystemExit(2)

    case = selected[0]
    smoke_case = make_one_day_case(case)

    case_root = OUTPUT_ROOT / f"{TARGET_BUILDING}"
    prepared_idf = case_root / "inputs" / "prepared_1day.idf"
    output_dir = case_root / "opyplus_output"

    if case_root.exists():
        shutil.rmtree(case_root)

    prepared_idf.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    idf_preparer = IdfPreparer()
    runner = EnergyPlusRunner(beat_frequency_seconds=5)

    print("=" * 100)
    print(f"{TARGET_BUILDING} / {TARGET_WEATHER}")
    print(f"source idf: {case.idf_path}")
    print(f"epw:        {case.epw_path}")
    print(f"output:     {output_dir}")

    started = datetime.now()

    row = {
        "building_type": TARGET_BUILDING,
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

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    lines = []
    lines.append("HotelLarge Seattle 1-Day opyplus Smoke Test")
    lines.append("=" * 80)
    lines.append(f"Output root: {OUTPUT_ROOT.resolve()}")
    lines.append(f"Summary CSV: {SUMMARY_CSV.resolve()}")
    lines.append("")
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

    print(SUMMARY_TXT.read_text(encoding="utf-8"))
    print(f"CSV: {SUMMARY_CSV.resolve()}")
    print(f"TXT: {SUMMARY_TXT.resolve()}")

    if row["status"] != "completed":
        raise SystemExit(1)

    if str(row["severe_errors"]) not in ("", "0") or str(row["fatal_errors"]) not in ("", "0"):
        raise SystemExit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
