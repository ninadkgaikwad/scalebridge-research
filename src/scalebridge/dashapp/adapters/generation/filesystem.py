"""Read-only filesystem inspection for EnergyPlus Generation campaigns."""
from __future__ import annotations
from pathlib import Path

TRACEBACK_PATTERNS = ("traceback.txt", "trace*.txt", "trace*.log")

def count_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    try:
        return sum(1 for p in root.rglob(pattern) if p.is_file())
    except OSError:
        return 0

def discover_generation_campaign_ids(campaigns_root: Path) -> list[str]:
    """Return campaigns that contain a Generation cases directory."""
    if not campaigns_root.is_dir():
        return []
    result=[]
    try:
        for child in campaigns_root.iterdir():
            if child.is_dir() and (child / "generation" / "cases").is_dir():
                result.append(child.name)
    except OSError:
        return []
    return sorted(result, key=str.lower)

def scan_generation_campaign(campaign_root: Path) -> dict[str, int | bool]:
    """Count Generation artifacts without opening parquet payloads."""
    case_root=campaign_root / "generation" / "cases"
    case_count=0
    if case_root.is_dir():
        try:
            case_count=sum(1 for p in case_root.iterdir() if p.is_dir())
        except OSError:
            case_count=0
    traceback_count=sum(count_files(campaign_root, pattern) for pattern in TRACEBACK_PATTERNS)
    return {
        "exists": campaign_root.is_dir(),
        "detected_case_count": case_count,
        "latest_run_count": count_files(case_root, "latest_run.json"),
        "rdd_manifest_count": count_files(case_root, "rdd_variable_intersection.json"),
        "parquet_count": count_files(case_root, "*.parquet"),
        "pickle_count": count_files(case_root, "*.pickle"),
        "traceback_count": traceback_count,
    }
