from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


CAMPAIGN_ID = "p1_ashrae2013_one_zone_compact_4b4c_labpc_full_v2"

generated_root = Path(os.environ.get(
    "SCALEBRIDGE_GENERATED_DATA_ROOT",
    r"F:\Dropbox\NinadGaikwad_PhD\Gaikwad_Research\From_WSU_OneDrive\BuildingModelingProject_Condensed\Data\ScaleBridge",
))

campaign_cases_root = (
    generated_root
    / "campaigns"
    / CAMPAIGN_ID
    / "generation"
    / "cases"
)

eplus_work_root = Path(os.environ.get(
    "SCALEBRIDGE_EPLUS_WORK_ROOT",
    r"D:\ScaleBridge_EPlus_Work",
))

out_dir = Path.cwd() / "audit_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

traceback_csv = out_dir / f"{CAMPAIGN_ID}_traceback_audit.csv"
case_csv = out_dir / f"{CAMPAIGN_ID}_case_completion_audit.csv"
work_csv = out_dir / f"{CAMPAIGN_ID}_d_work_folder_audit.csv"
summary_txt = out_dir / f"{CAMPAIGN_ID}_failure_summary.txt"


def read_text_safe(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="replace")
        except Exception:
            pass
    return ""


def classify_traceback(text: str) -> str:
    t = text.lower()

    if "no space left on device" in t or "errno 28" in t:
        return "disk_or_temp_no_space"

    if "permissionerror" in t and "being used by another process" in t:
        return "file_lock_permission_error"

    if "did not produce" in t and "eplusout.csv" in t:
        return "missing_eplusout_csv"

    if "energyplusexecutionerror" in t:
        return "energyplus_execution_error"

    if "filenotfounderror" in t:
        return "file_not_found"

    if "permissionerror" in t:
        return "permission_error"

    return "other_traceback"


def extract_last_exception(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if (
            "Error:" in line
            or "Exception:" in line
            or line.startswith("OSError:")
            or line.startswith("PermissionError:")
            or line.startswith("FileNotFoundError:")
        ):
            return line[:500]
    return lines[-1][:500] if lines else ""


def extract_variable_name(text: str, path: Path) -> str:
    patterns = [
        r"variable_idfs[\\/](?P<name>[^\\/]+)\.idf",
        r"variable_csv[\\/](?P<name>[^\\/]+)\.csv",
        r"D:[\\/]ScaleBridge_EPlus_Work[\\/][^\\/]+[\\/]v\d+[\\/](?P<name>eplusout)\.csv",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group("name")

    # fallback from nearby path names
    for part in reversed(path.parts):
        low = part.lower()
        if low.endswith(".idf"):
            return part[:-4]
        if low.endswith(".csv"):
            return part[:-4]

    return ""


def extract_work_run_and_vdir(text: str) -> tuple[str, str]:
    m = re.search(
        r"D:[\\/]ScaleBridge_EPlus_Work[\\/](?P<run>epvwr_[^\\/]+)[\\/](?P<vdir>v\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group("run"), m.group("vdir")
    return "", ""


def is_traceback_file(path: Path) -> bool:
    if not path.is_file():
        return False

    name = path.name.lower()
    if "trace" in name and path.suffix.lower() in {".txt", ".log"}:
        return True

    if path.suffix.lower() in {".txt", ".log"}:
        text = read_text_safe(path)
        return "Traceback (most recent call last):" in text

    return False


def count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


trace_rows = []
case_rows = []
work_rows = []

failure_counter = Counter()
case_failure_counter = Counter()
variable_failure_counter = Counter()
run_failure_counter = Counter()

if not campaign_cases_root.exists():
    raise SystemExit(f"Campaign cases root does not exist: {campaign_cases_root}")

case_dirs = sorted([p for p in campaign_cases_root.iterdir() if p.is_dir()])

for case_dir in case_dirs:
    case_id = case_dir.name
    latest_path = case_dir / "latest_run.json"
    latest_run_id = ""

    if latest_path.exists():
        try:
            latest = json.loads(read_text_safe(latest_path))
            latest_run_id = str(latest.get("run_id", ""))
        except Exception:
            latest_run_id = ""

    runs_root = case_dir / "runs"
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()]) if runs_root.exists() else []

    for run_dir in run_dirs:
        run_id = run_dir.name

        trace_files = [p for p in run_dir.rglob("*") if is_traceback_file(p)]

        parquet_count = count_files(run_dir / "canonical" / "variables", "*.parquet")
        pickle_count = count_files(run_dir / "legacy" / "per_variable_pickle", "*.pickle")
        raw_csv_count = count_files(run_dir / "raw" / "variable_csv", "*.csv")

        manifest_path = run_dir / "canonical" / "variable_manifest.json"
        metadata_path = run_dir / "canonical" / "metadata.json"

        case_rows.append({
            "case_id": case_id,
            "run_id": run_id,
            "is_latest_run": str(run_id == latest_run_id),
            "traceback_count": len(trace_files),
            "parquet_count": parquet_count,
            "pickle_count": pickle_count,
            "raw_csv_count": raw_csv_count,
            "manifest_exists": str(manifest_path.exists()),
            "metadata_exists": str(metadata_path.exists()),
            "complete_35_parquet": str(parquet_count == 35),
            "complete_35_pickle": str(pickle_count == 35),
            "run_dir": str(run_dir),
        })

        for tf in trace_files:
            text = read_text_safe(tf)
            category = classify_traceback(text)
            last_exception = extract_last_exception(text)
            variable_name = extract_variable_name(text, tf)
            work_run_id, work_vdir = extract_work_run_and_vdir(text)

            failure_counter[category] += 1
            case_failure_counter[case_id] += 1
            if variable_name:
                variable_failure_counter[variable_name] += 1
            run_failure_counter[run_id] += 1

            trace_rows.append({
                "case_id": case_id,
                "run_id": run_id,
                "is_latest_run": str(run_id == latest_run_id),
                "traceback_file": str(tf),
                "failure_category": category,
                "variable_name": variable_name,
                "work_run_id_in_trace": work_run_id,
                "work_vdir_in_trace": work_vdir,
                "last_exception": last_exception,
            })


# Audit D:\ScaleBridge_EPlus_Work
if eplus_work_root.exists():
    for run_dir in sorted([p for p in eplus_work_root.iterdir() if p.is_dir()]):
        run_id = run_dir.name
        for vdir in sorted([p for p in run_dir.iterdir() if p.is_dir() and re.match(r"v\d+", p.name.lower())]):
            eplusout_csv = vdir / "eplusout.csv"
            eplusout_err = vdir / "eplusout.err"
            eplusout_eio = vdir / "eplusout.eio"

            err_text = read_text_safe(eplusout_err) if eplusout_err.exists() else ""
            fatal_count = len(re.findall(r"\*\*\s*Fatal\s*\*\*", err_text, flags=re.IGNORECASE))
            severe_count = len(re.findall(r"\*\*\s*Severe\s*\*\*", err_text, flags=re.IGNORECASE))
            warning_count = len(re.findall(r"\*\*\s*Warning\s*\*\*", err_text, flags=re.IGNORECASE))

            work_rows.append({
                "work_run_id": run_id,
                "vdir": vdir.name,
                "eplusout_csv_exists": str(eplusout_csv.exists()),
                "eplusout_csv_size": eplusout_csv.stat().st_size if eplusout_csv.exists() else 0,
                "eplusout_err_exists": str(eplusout_err.exists()),
                "eplusout_eio_exists": str(eplusout_eio.exists()),
                "fatal_count": fatal_count,
                "severe_count": severe_count,
                "warning_count": warning_count,
                "work_dir": str(vdir),
            })


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


write_csv(traceback_csv, trace_rows)
write_csv(case_csv, case_rows)
write_csv(work_csv, work_rows)

with summary_txt.open("w", encoding="utf-8") as f:
    f.write(f"Campaign traceback/work audit\n")
    f.write(f"campaign_id: {CAMPAIGN_ID}\n")
    f.write(f"campaign_cases_root: {campaign_cases_root}\n")
    f.write(f"eplus_work_root: {eplus_work_root}\n\n")

    f.write("Overall counts\n")
    f.write(f"case_count: {len(case_dirs)}\n")
    f.write(f"run_count_in_campaign_folders: {len(case_rows)}\n")
    f.write(f"traceback_count: {len(trace_rows)}\n")
    f.write(f"d_work_variable_folder_count: {len(work_rows)}\n\n")

    f.write("Failure categories\n")
    for k, v in failure_counter.most_common():
        f.write(f"{k}: {v}\n")

    f.write("\nTop cases by traceback count\n")
    for k, v in case_failure_counter.most_common(20):
        f.write(f"{k}: {v}\n")

    f.write("\nTop runs by traceback count\n")
    for k, v in run_failure_counter.most_common(20):
        f.write(f"{k}: {v}\n")

    f.write("\nTop variables by traceback count\n")
    for k, v in variable_failure_counter.most_common(50):
        f.write(f"{k}: {v}\n")

    incomplete = [
        r for r in case_rows
        if r["is_latest_run"] == "True"
        and (r["complete_35_parquet"] != "True" or r["complete_35_pickle"] != "True" or int(r["traceback_count"]) > 0)
    ]

    f.write("\nLatest runs incomplete or with tracebacks\n")
    for r in incomplete:
        f.write(
            f"{r['case_id']} | {r['run_id']} | "
            f"parquet={r['parquet_count']} pickle={r['pickle_count']} "
            f"tracebacks={r['traceback_count']}\n"
        )

    missing_csv = [
        r for r in work_rows
        if r["eplusout_csv_exists"] != "True"
    ]
    f.write("\nD-work v-folders missing eplusout.csv\n")
    for r in missing_csv[:200]:
        f.write(f"{r['work_run_id']} | {r['vdir']} | {r['work_dir']}\n")

print("AUDIT COMPLETE")
print(f"traceback_csv: {traceback_csv}")
print(f"case_csv:      {case_csv}")
print(f"work_csv:      {work_csv}")
print(f"summary_txt:   {summary_txt}")
