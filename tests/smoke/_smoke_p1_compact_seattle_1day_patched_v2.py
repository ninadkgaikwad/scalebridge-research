
from __future__ import annotations

import csv
import re
import shutil
import traceback
from pathlib import Path
from datetime import datetime

from scalebridge.integration.energyplus import build_p1_case_specs
from scalebridge.integration.energyplus.idf import IdfPreparer
from scalebridge.integration.energyplus.simulation import EnergyPlusRunner


TARGET_BUILDINGS = (
    "SchoolPrimary",
    "ApartmentMidRise",
)

TARGET_WEATHER = "Seattle"

OUTPUT_ROOT = Path("scratch") / "p1_compact_seattle_1day_patched_smoke_v2"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
SUMMARY_TXT = OUTPUT_ROOT / "summary.txt"


def object_blocks(text: str):
    current = []
    for line in text.splitlines():
        current.append(line)
        if ";" in line:
            yield "\n".join(current)
            current = []
    if current:
        yield "\n".join(current)


def strip_comments(line: str) -> str:
    return line.split("!")[0].strip()


def has_schedule_type_limits_control_type(text: str) -> bool:
    for block in object_blocks(text):
        clean_lines = [strip_comments(line) for line in block.splitlines()]
        clean_lines = [line for line in clean_lines if line]
        if not clean_lines:
            continue

        object_type = clean_lines[0].rstrip(",;").strip().lower()
        if object_type != "scheduletypeLimits".lower():
            continue

        if len(clean_lines) > 1:
            name = clean_lines[1].rstrip(",;").strip().lower()
            if name == "control type":
                return True

    return False


def references_control_type_schedule_limit(text: str) -> bool:
    return bool(re.search(r"^\s*Control\s+Type\s*,", text, flags=re.IGNORECASE | re.MULTILINE))


def ensure_control_type_schedule_limit(text: str) -> tuple[str, bool]:
    if not references_control_type_schedule_limit(text):
        return text, False

    if has_schedule_type_limits_control_type(text):
        return text, False

    insertion = """
  ScheduleTypeLimits,
    Control Type,            !- Name
    0,                       !- Lower Limit Value
    4,                       !- Upper Limit Value
    DISCRETE,                !- Numeric Type
    Dimensionless;           !- Unit Type

"""

    marker = "  Schedule:Compact,"
    pos = text.find(marker)
    if pos >= 0:
        text = text[:pos] + insertion + text[pos:]
    else:
        text = insertion + text

    return text, True


def patch_building_max_warmup_days(text: str, max_warmup_days: int = 100) -> tuple[str, bool]:
    blocks = list(object_blocks(text))
    patched_blocks = []
    changed = False

    for block in blocks:
        lines = block.splitlines()
        if not lines:
            patched_blocks.append(block)
            continue

        clean_first = strip_comments(lines[0]).rstrip(",;").lower()
        if clean_first != "building":
            patched_blocks.append(block)
            continue

        field_index = 0
        new_lines = []

        for line in lines:
            no_comment = line.split("!")[0]
            if "," in no_comment or ";" in no_comment:
                field_index += 1

                # Field index 8 means Maximum Number of Warmup Days:
                # 1 object type, 2 name, 3 north axis, 4 terrain,
                # 5 load tol, 6 temp tol, 7 solar distribution, 8 max warmup.
                if field_index == 8:
                    comment = ""
                    if "!" in line:
                        comment = " !" + line.split("!", 1)[1]
                    suffix = "," if "," in no_comment else ";"
                    indent = re.match(r"^\s*", line).group(0)
                    line = f"{indent}{max_warmup_days}{suffix}{comment}"
                    changed = True

            new_lines.append(line)

        patched_blocks.append("\n".join(new_lines))

    if not changed:
        return text, False

    return "\n\n".join(patched_blocks) + "\n", True


def make_one_day_case(case, patched_idf_path: Path):
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
            "idf_path": patched_idf_path,
            "run_period": one_day_run_period,
            "output_variables": one_variable,
            "write_legacy_pickles": False,
            "preserve_raw_outputs": True,
        }
    )


def patch_source_idf(case, case_root: Path) -> tuple[Path, list[str]]:
    source_path = Path(case.idf_path)
    text = source_path.read_text(encoding="utf-8", errors="ignore")

    changes = []

    text, changed = ensure_control_type_schedule_limit(text)
    if changed:
        changes.append("inserted ScheduleTypeLimits: Control Type")

    text, changed = patch_building_max_warmup_days(text, max_warmup_days=100)
    if changed:
        changes.append("set Building maximum warmup days to 100")

    patched_path = case_root / "inputs" / "source_patched_for_smoke.idf"
    patched_path.parent.mkdir(parents=True, exist_ok=True)
    patched_path.write_text(text, encoding="utf-8")

    return patched_path, changes


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
        raise RuntimeError(f"Missing cases: {missing}")

    idf_preparer = IdfPreparer()
    runner = EnergyPlusRunner(beat_frequency_seconds=5)

    rows = []

    for index, building in enumerate(TARGET_BUILDINGS, start=1):
        case = selected_by_building[building]

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
            "patches": "",
            "patched_idf": "",
            "prepared_idf": str(prepared_idf),
            "output_dir": str(output_dir),
            "err_file": str(output_dir / "eplusout.err"),
            "csv_file_exists": "",
            "eso_file_exists": "",
            "message": "",
        }

        try:
            patched_source_idf, changes = patch_source_idf(case, case_root)
            smoke_case = make_one_day_case(case, patched_source_idf)

            row["patched_idf"] = str(patched_source_idf)
            row["patches"] = "; ".join(changes)

            print(f"patches:    {row['patches'] or '(none)'}")

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
    lines.append("P1 Compact Seattle 1-Day Patched opyplus Smoke Test v2")
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
            f"runtime_s={row['runtime_seconds']} | "
            f"patches={row['patches']}"
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
